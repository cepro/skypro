import logging
from datetime import timedelta

import numpy as np
import pulp
import pandas as pd
from skypro.common.rate_utils.to_dfs import VolRatesForEnergyFlows
from skypro.common.timeutils.math import floor_day
from skypro.common.timeutils.math_wallclock import add_wallclock_days
from skypro.common.timeutils.timeseries import get_step_size

from skypro.commands.simulator.algorithms.rate_management import run_osam_calcs_for_day, add_total_vol_rates_to_df

from skypro.common.cli_utils.cli_utils import get_user_ack_of_warning_or_exit
from skypro.commands.simulator.config.config import Optimiser as OptimiserConfig, OptimiserBlocks, Bess


class Optimiser:
    """
    This is a linear programming optimiser that will find the best battery actions for the given rates / conditions.
    """
    def __init__(
        self,
        algo_config: OptimiserConfig,
        bess_config: Bess,
        final_vol_rates: VolRatesForEnergyFlows,
        df: pd.DataFrame,
    ):
        """
        See _preprocess_input_data for required columns in `df`
        """

        self._algo_config = algo_config
        self._bess_config = bess_config
        self._final_vol_rates = final_vol_rates
        self._df_in = self._preprocess_input_data(df.copy())

    def run(self) -> pd.DataFrame:
        """
        Optimises the entire time range given in self._df_in.
        It does this by making multiple calls to self._run_one_optimisation and stacking the results together, with the
        duration of each optimisation block defined by configuration.
        The end of each optimisation block should be dropped as it won't be accurate because the optimiser doesn't know how to
        value the energy in the battery at the end of the optimisation (at the moment it just drains the battery on the
        last day).
        """

        init_soe = self._bess_config.energy_capacity / 2  # assume the battery is half full at the beginning of the optimisation
        n_timeslots_with_nan_pricing = 0

        self._df_in = self._df_in.sort_index()

        df_out = pd.DataFrame()

        step_size = get_step_size(self._df_in.index)

        current_start_t = self._df_in.index[0]
        end_t = self._df_in.index[-1]
        while current_start_t < end_t:
            current_end_t = add_wallclock_days(floor_day(current_start_t), self._algo_config.blocks.duration_days) - step_size
            current_end_to_use_t = add_wallclock_days(floor_day(current_start_t), self._algo_config.blocks.used_duration_days) - step_size  # we only use the first part of each optimisation
            if current_end_t > end_t:
                current_end_t = end_t
            if current_end_to_use_t > end_t:
                current_end_to_use_t = end_t

            # Create a smaller dataframe for the optimisation of the number of days in the block
            block_df_in = self._df_in.loc[current_start_t:current_end_t]

            # Calculate the OSAM flows and NCSP for the upcoming day
            block_df_in, _ = run_osam_calcs_for_day(block_df_in, current_start_t)

            # We know the NCSP for the upcoming day, but not for any days thereafter, so we estimate that it will stay
            # the same for the following days so that we can optimise multiple days in one block.
            # If this is a significant source of error then the simulation could be configured to only use a single day
            # from the optimisation block. E.g. blocks.duration_days = 3, blocks.used_duration_hh = 1
            block_df_in["osam_ncsp"] = block_df_in["osam_ncsp"].ffill()

            # Calculate the total market and internal volume rates in p/kWh for the whole block
            block_df_in = add_total_vol_rates_to_df(
                df=block_df_in,
                index_to_add_for=block_df_in.index,
                mkt_vol_rates=self._final_vol_rates,
                live_or_final="final"
            )

            logging.info(f"Optimising range {block_df_in.index[0]} -> {block_df_in.index[-1]}...")
            block_df_out, block_num_nan = self._run_one_optimisation(
                df_in=block_df_in,
                init_soe=init_soe,
                block_config=self._algo_config.blocks
            )
            n_timeslots_with_nan_pricing += block_num_nan

            # Only keep the time range that we 'should use' (this is discussed above)
            block_df_out_to_use = block_df_out.loc[:current_end_to_use_t]

            # Create a single dataframe with the results of all the individual optimisations
            df_out = pd.concat([df_out, block_df_out_to_use], axis=0)

            # Prepare for the next iteration
            current_start_t = current_end_to_use_t + step_size
            if len(block_df_out) > len(block_df_out_to_use):
                init_soe = block_df_out.iloc[len(block_df_out_to_use)]["soe"]
            else:
                init_soe = np.nan  # this must be the last iteration

        if n_timeslots_with_nan_pricing > 0:
            get_user_ack_of_warning_or_exit(
                f"{n_timeslots_with_nan_pricing} time slots had NaN pricing data and could not be optimised"
            )

        return df_out

    def _preprocess_input_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess input data to calculate derived microgrid flows and constraints.

        This method calculates:
        1. Basic energy flows (solar to load, excess solar, unmet load)
        2. Battery charge/discharge limits considering grid constraints
        3. Minimum charge/discharge requirements for constraint management
        """
        required_columns = [
            'solar', 'load', 'bess_max_charge', 'bess_max_discharge', 'time_into_sp'
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Calculate some of the microgrid flows - at the moment this is the only algo that uses these values, but in
        # the future it may make sense to pass these values in rather than have each algo calculate them independently.
        df["solar_to_load"] = df[["solar", "load"]].min(axis=1)
        df["load_not_supplied_by_solar"] = df["load"] - df["solar_to_load"]
        df["solar_not_supplying_load"] = df["solar"] - df["solar_to_load"]
        # When charging we must use excess solar first:
        df["max_charge_from_grid"] = np.maximum(df["bess_max_charge"] - df["solar_not_supplying_load"], 0)
        # When discharging we must send power to microgrid load first:
        df["max_discharge_to_grid"] = np.maximum(df["bess_max_discharge"] - df["load_not_supplied_by_solar"], 0)

        if self._algo_config.blocks.do_active_export_constraint_management:
            df["min_charge"] = df[df["bess_max_discharge"] < 0]["bess_max_discharge"] * -1
            df["min_charge"] = df["min_charge"].fillna(0)
            # The min_charge constraint is currently applied to the 'charge from solar' flow, as it was used to
            # manage excess solar power. If the min charge is a floating point error away from solar_not_supplying_load
            # then make them equal to avoid constraint issues.
            close_idx = df.index[np.isclose(df["min_charge"], df["solar_not_supplying_load"])]
            df.loc[close_idx, "min_charge"] = df.loc[close_idx, "solar_not_supplying_load"]
        else:
            df["min_charge"] = 0.0

        if self._algo_config.blocks.do_active_import_constraint_management:
            df["min_discharge"] = df[df["bess_max_charge"] < 0]["bess_max_charge"] * -1
            df["min_discharge"] = df["min_discharge"].fillna(0)
            # The min_discharge constraint is currently applied to the 'discharge to load' flow, as it was used to
            # manage excess load. If the min discharge is a floating point error away from load_not_supplied_by_solar
            # then make them equal to avoid constraint issues.
            close_idx = df.index[np.isclose(df["min_discharge"], df["load_not_supplied_by_solar"])]
            df.loc[close_idx, "min_discharge"] = df.loc[close_idx, "load_not_supplied_by_solar"]
        else:
            df["min_discharge"] = 0.0

        return df

    def _run_one_optimisation(
        self,
        df_in: pd.DataFrame,
        init_soe: float,
        block_config: OptimiserBlocks,
    ) -> (pd.DataFrame, int):
        """
        Uses the pulp library to optimise the battery schedule as a linear programming optimisation problem.
        This is currently a 'perfect hindsight' view because in practice we wouldn't know the imbalance pricing or
        microgrid load and solar generation ahead of time.
        It also returns the number of timeslots that had nan pricing for logging/warning purposes.

        max_avg_cycles_per_day is applied to the entire optimisation block - so the *average* cycles per day of the
        whole block will not exceed `max_avg_cycles_per_day`, but any given day may exceed `max_avg_cycles_per_day`.
        """
        problem = pulp.LpProblem(name="MicrogridProblem", sense=pulp.LpMinimize)

        lp_var_bess_soe = []
        lp_var_bess_discharges = []
        lp_var_bess_discharges_to_load = []
        lp_var_bess_discharges_to_grid = []
        lp_var_bess_charges = []
        lp_var_bess_charges_from_solar = []
        lp_var_bess_charges_from_grid = []
        lp_var_bess_is_charging = []
        lp_costs = []

        # We use indexes rather than datetimes to represent each time slot
        timeslots = range(0, len(df_in))

        n_timeslots_with_nan_pricing = 0

        for timeslot_idx in timeslots:

            lp_var_bess_soe.append(
                pulp.LpVariable(
                    name=f"bess_soe_{timeslot_idx}",
                    lowBound=0.0,
                    upBound=self._bess_config.energy_capacity
                )
            )
            lp_var_bess_charges_from_solar.append(
                pulp.LpVariable(
                    name=f"solar_to_batt_{timeslot_idx}",
                    lowBound=0,
                    upBound=df_in.iloc[timeslot_idx]["solar_not_supplying_load"]
                )
            )
            lp_var_bess_charges_from_grid.append(
                pulp.LpVariable(
                    name=f"grid_to_batt_{timeslot_idx}",
                    lowBound=0.0,
                    upBound=df_in.iloc[timeslot_idx]["max_charge_from_grid"]
                )
            )
            lp_var_bess_discharges_to_load.append(
                pulp.LpVariable(
                    name=f"batt_to_load_{timeslot_idx}",
                    lowBound=0,
                    upBound=df_in.iloc[timeslot_idx]["load_not_supplied_by_solar"]
                )
            )
            lp_var_bess_discharges_to_grid.append(
                pulp.LpVariable(
                    name=f"batt_to_grid_{timeslot_idx}",
                    lowBound=0.0,
                    upBound=df_in.iloc[timeslot_idx]["max_discharge_to_grid"],
                )
            )

            # These totals of charge and discharge are just defined for convenience
            lp_var_bess_charges.append(
                pulp.LpVariable(
                    name=f"bess_charge_{timeslot_idx}",
                    lowBound=0.0,
                )
            )
            lp_var_bess_discharges.append(
                pulp.LpVariable(
                    name=f"bess_discharge_{timeslot_idx}",
                    lowBound=0.0,
                )
            )

            # This binary var is used to make charge and discharging mutually exclusive for each time period
            lp_var_bess_is_charging.append(
                pulp.LpVariable(
                    name=f"bess_is_charging_{timeslot_idx}",
                    cat=pulp.LpBinary
                )
            )

            # Get the rates from the input dataframe, and check they are not nan - if they are nan then don't allow any
            # activity in this period.
            mkt_rate_final_grid_to_batt = df_in.iloc[timeslot_idx]["mkt_vol_rate_final_grid_to_batt"]
            int_rate_final_solar_to_batt = df_in.iloc[timeslot_idx]["int_vol_rate_final_solar_to_batt"]
            mkt_rate_final_batt_to_grid = df_in.iloc[timeslot_idx]["mkt_vol_rate_final_batt_to_grid"]
            int_rate_final_batt_to_load = df_in.iloc[timeslot_idx]["int_vol_rate_final_batt_to_load"]
            if np.any(np.isnan([
                mkt_rate_final_grid_to_batt,
                int_rate_final_solar_to_batt,
                mkt_rate_final_batt_to_grid,
                int_rate_final_batt_to_load
            ])):
                # the costs function throws an exception when these are NaN, so set to zero but disallow any activity
                # by adding constraints
                mkt_rate_final_grid_to_batt = 0
                int_rate_final_solar_to_batt = 0
                mkt_rate_final_batt_to_grid = 0
                int_rate_final_batt_to_load = 0
                problem += lp_var_bess_charges_from_solar[timeslot_idx] == 0
                problem += lp_var_bess_charges_from_grid[timeslot_idx] == 0
                problem += lp_var_bess_discharges_to_load[timeslot_idx] == 0
                problem += lp_var_bess_discharges_to_grid[timeslot_idx] == 0

                n_timeslots_with_nan_pricing += 1

            lp_costs.append(
                lp_var_bess_charges_from_grid[timeslot_idx] * mkt_rate_final_grid_to_batt +
                lp_var_bess_charges_from_solar[timeslot_idx] * int_rate_final_solar_to_batt +
                lp_var_bess_discharges_to_grid[timeslot_idx] * mkt_rate_final_batt_to_grid +
                lp_var_bess_discharges_to_load[timeslot_idx] * int_rate_final_batt_to_load
            )

        # Calculate any limits on the 'optional' actions (i.e. those which are not required for active grid constraint management)
        optional_action_limit_df = pd.DataFrame(index=df_in.index, columns=["charge", "discharge"])
        if block_config.no_optional_charging_in_lowest_priced_quantile is not None:
            for _, df_day in df_in.groupby(df_in.index.date):
                df_lowest = self._get_lowest_valued_rows(block_config.no_optional_charging_in_lowest_priced_quantile, df_day, "mkt_vol_rate_final_grid_to_batt")
                optional_action_limit_df.loc[df_lowest.index, "charge"] = 0.0
        # Constraints to prevent activity in the first ten minutes (if that's what is configured)
        if block_config.no_optional_actions_in_first_ten_mins_except_for_period is not None:
            is_in_first_ten_mins = df_in["time_into_sp"] < timedelta(minutes=10)
            is_exempt = df_in.apply(lambda row: block_config.no_optional_actions_in_first_ten_mins_except_for_period.contains(row.name), axis=1)
            no_actions = is_in_first_ten_mins & ~is_exempt
            optional_action_limit_df.loc[no_actions] = 0.0

        for timeslot_idx in timeslots:

            # Constraints to define that all the flows are positive - prevent the optimiser from using a negative
            problem += lp_var_bess_charges_from_solar[timeslot_idx] >= 0.0
            problem += lp_var_bess_charges_from_grid[timeslot_idx] >= 0.0
            problem += lp_var_bess_discharges_to_load[timeslot_idx] >= 0.0
            problem += lp_var_bess_discharges_to_grid[timeslot_idx] >= 0.0

            # Constraints to define the total of all charge flows and total of all discharge flows. This is just for
            # convenience as the totals are used a few times later on.
            problem += lp_var_bess_charges[timeslot_idx] == lp_var_bess_charges_from_solar[timeslot_idx] + lp_var_bess_charges_from_grid[timeslot_idx]
            problem += lp_var_bess_discharges[timeslot_idx] == lp_var_bess_discharges_to_load[timeslot_idx] + lp_var_bess_discharges_to_grid[timeslot_idx]

            # Constraints for maximum charge/discharge rates AND make charge and discharge mutually exclusive
            problem += lp_var_bess_charges[timeslot_idx] <= (df_in.iloc[timeslot_idx]["bess_max_charge"] * lp_var_bess_is_charging[timeslot_idx])
            problem += lp_var_bess_discharges[timeslot_idx] <= (df_in.iloc[timeslot_idx]["bess_max_discharge"] * (1 - lp_var_bess_is_charging[timeslot_idx]))

            # Constraints for minimum charge/discharge rates - for when doing 'active constraint management'
            problem += lp_var_bess_charges[timeslot_idx] >= df_in.iloc[timeslot_idx]["min_charge"]
            problem += lp_var_bess_discharges[timeslot_idx] >= df_in.iloc[timeslot_idx]["min_discharge"]

            # Apply any constraints on optional actions (if configured)
            charge_limit = optional_action_limit_df.iloc[timeslot_idx]["charge"]
            if not np.isnan(charge_limit):
                # Force the charge and discharge level to the limit for this time slot, unless the battery is required to be doing
                # active constraint management - in which case allow the battery to do the constraint management but nothing else (this is not optional).
                if df_in.iloc[timeslot_idx]["min_charge"] > 0:
                    charge_limit = df_in.iloc[timeslot_idx]["min_charge"]
                problem += lp_var_bess_charges[timeslot_idx] <= charge_limit
            discharge_limit = optional_action_limit_df.iloc[timeslot_idx]["discharge"]
            if not np.isnan(discharge_limit):
                if df_in.iloc[timeslot_idx]["min_discharge"] > 0:
                    discharge_limit = df_in.iloc[timeslot_idx]["min_discharge"]
                problem += lp_var_bess_discharges[timeslot_idx] <= discharge_limit

        # Apply cycling constraint to all timeslots
        if block_config.max_avg_cycles_per_day:
            days_in_block = (df_in.index[-1] - df_in.index[0]).total_seconds() / (3600 * 24)
            problem += pulp.lpSum(lp_var_bess_discharges) <= (block_config.max_avg_cycles_per_day * days_in_block * self._bess_config.energy_capacity)

        # The objective function is the sum of costs across all timeslots, which will be minimised
        problem += pulp.lpSum(lp_costs)

        # Set the initial state of energy
        problem += lp_var_bess_soe[0] == init_soe

        # Don't allow any battery activity in the last period as this requires more complicated constraints to make it
        # work (the end of each optimisation is dropped anyway as the individual optimisations runs are combined)
        problem += lp_var_bess_charges_from_solar[-1] == 0
        problem += lp_var_bess_charges_from_grid[-1] == 0
        problem += lp_var_bess_discharges_to_load[-1] == 0
        problem += lp_var_bess_discharges_to_grid[-1] == 0

        # Constraint to define how the SoE changes across the timeslots. This loop starts from the second timeslot.
        for timeslot_idx in timeslots[1:]:
            problem += (
                lp_var_bess_soe[timeslot_idx] == lp_var_bess_soe[timeslot_idx - 1]
                + lp_var_bess_charges_from_solar[timeslot_idx - 1] * self._bess_config.charge_efficiency
                + lp_var_bess_charges_from_grid[timeslot_idx - 1] * self._bess_config.charge_efficiency
                - lp_var_bess_discharges_to_load[timeslot_idx - 1]
                - lp_var_bess_discharges_to_grid[timeslot_idx - 1]
            )

        status = problem.solve(pulp.PULP_CBC_CMD(
            msg=False,
            gapRel=block_config.max_optimal_tolerance,
            timeLimit=block_config.max_computation_secs
        ))
        if status != 1:
            raise RuntimeError("Failed to solve optimisation problem")

        df_sol = _get_solution_as_dataframe(problem.variables(), df_in.index)

        self._ensure_merit_order_of_charge_and_discharge(df_sol)

        # Create a dataframe to return with just the required info
        df_ret = pd.DataFrame(index=df_sol.index)
        df_ret["soe"] = df_sol["bess_soe"]
        df_ret["energy_delta"] = (
            df_sol["solar_to_batt"] + df_sol["grid_to_batt"]
            - df_sol["batt_to_grid"] - df_sol["batt_to_load"]
        )
        df_ret["bess_losses"] = (
            (df_sol["solar_to_batt"] + df_sol["grid_to_batt"]) * (1 - self._bess_config.charge_efficiency)
        )

        return df_ret, n_timeslots_with_nan_pricing

    @staticmethod
    def _get_lowest_valued_rows(quantile: float, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """This returns the rows of `df` which have the lowest values for `col`"""
        threshold = df[col].quantile(quantile)
        lowest_df = df[df[col] <= threshold]
        num_rows = int(np.floor(len(df) * quantile))
        if len(lowest_df) > num_rows:
            # The above may return too many rows if there are multiple rows with the same price at the quantile boundary
            lowest_df = lowest_df.sort_values(col)
            lowest_df = lowest_df.head(num_rows)
        return lowest_df

    def _ensure_merit_order_of_charge_and_discharge(self, df_sol: pd.DataFrame) -> None:
        """
        When charging we must always use solar first, before grid power. And when discharging we must always supply to
        on-site loads before the power goes out to grid.
        The optimiser will likely always prefer this anyway because the prices are better that way round anyway, but
        there are not yet optimisation constraints to ensure it, so it's checked here.
        If this were wrong then we would just end up with suboptimal solution, but the reported figures should be
        correct as the reported microgrid flows are calculated outside of this module - this module just returns
        the battery charge and discharge energies.
        """

        tolerance = 0.01

        # Check that we always charge from solar 'first', before charging from grid
        when_charging_from_grid = df_sol[df_sol["grid_to_batt"] > 0]
        check = (
            (when_charging_from_grid["solar_to_batt"] - self._df_in["solar_not_supplying_load"])
            > tolerance
        ).sum()
        assert check == 0, "Optimisation internal error - add constraint for energy merit order on charge"

        # Check that we always discharge to onsite load 'first', before discharging to grid
        when_discharging_to_grid = df_sol[df_sol["batt_to_grid"] > 0]
        check = (
                (when_discharging_to_grid["batt_to_load"] - self._df_in["load_not_supplied_by_solar"])
                > tolerance
        ).sum()
        assert check == 0, "Optimisation internal error - add constraint for energy merit order on discharge"


def _get_solution_as_dataframe(problem_variables, time_index: pd.Series) -> pd.DataFrame:
    """
    Convert the pulp solution variables into a dataframe.
    Pulp returns the variables as a list, with each variable named, e.g. the var named "bess_soe_23" would be the SoE at
    timeslot index 23.
    """

    # Sometimes (but not always) pulp includes a dummy variable in the output - not sure why.
    if problem_variables[0].name == "__dummy":
        problem_variables = problem_variables[1:]

    variables_data = {
        "var_name": [v.name for v in problem_variables],
        "var_value": [v.varValue for v in problem_variables]
    }
    df_sol = pd.DataFrame(variables_data)
    df_sol[["var_name", "timeslot_index"]] = df_sol["var_name"].str.rsplit(pat="_", n=1, expand=True)
    df_sol["timeslot_index"] = df_sol["timeslot_index"].astype(int)
    df_sol = pd.pivot(df_sol, index="timeslot_index", columns="var_name",
                      values="var_value")  # TODO: agg func to assert
    df_sol = df_sol.sort_index()
    df_sol["time"] = time_index[df_sol.index]
    df_sol = df_sol.set_index("time")
    return df_sol
