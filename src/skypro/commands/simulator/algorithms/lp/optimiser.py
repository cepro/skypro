import logging
from datetime import timedelta

import numpy as np
import pulp
import pandas as pd

from skypro.cli_utils.cli_utils import get_user_ack_of_warning_or_exit
from skypro.commands.simulator.config.config_common import Optimiser as OptimiserConfig


class Optimiser:
    def __init__(self, config: OptimiserConfig, df: pd.DataFrame, battery_energy_capacity: float, battery_charge_efficiency: float):
        self._config = config
        self._df_in = df.copy()
        self._battery_energy_capacity = battery_energy_capacity
        self._battery_charge_efficiency = battery_charge_efficiency

        # Calculate some of the microgrid flows - at the moment this is the only algo that uses these values, but in
        # the future it may make sense to pass these values in rather than have each algo calculate them independently.
        self._df_in["solar_to_load"] = self._df_in[["solar", "load"]].min(axis=1)
        self._df_in["load_not_supplied_by_solar"] = self._df_in["load"] - self._df_in["solar_to_load"]
        self._df_in["solar_not_supplying_load"] = self._df_in["solar"] - self._df_in["solar_to_load"]
        # When charging we must use excess solar first:
        self._df_in["max_charge_from_grid"] = np.maximum(self._df_in["bess_max_charge"] - self._df_in["solar_not_supplying_load"], 0)
        # When discharging we must send power to microgrid load first:
        self._df_in["max_discharge_to_load"] = np.maximum(self._df_in["bess_max_discharge"] - self._df_in["load_not_supplied_by_solar"], 0)

    def run(self) -> pd.DataFrame:
        """
        Optimises the entire time range given in self._df_in.
        It does this by making multiple calls to self._run_one_optimisation and stacking the results together, with the
        duration of each optimisation block defined by configuration.
        The end of each optimisation should be dropped as it won't be accurate because the optimiser doesn't know how to
        value the energy in the battery at the end of the optimisation (at the moment it just drains the battery on the
        last day).
        """

        init_soe = self._battery_energy_capacity / 2
        n_timeslots_with_nan_pricing = 0

        self._df_in = self._df_in.sort_index()
        n_timeslots = len(self._df_in)
        # The native Python timedelta class seems more reliable that the pandas equivalent
        period = timedelta(seconds=pd.to_timedelta(self._df_in.index.freq).total_seconds())

        block_duration = self._config.blocks.duration_hh * timedelta(minutes=30)
        block_duration_to_use = self._config.blocks.used_duration_hh * timedelta(minutes=30)

        n_timeslots_per_block = int(block_duration / period)
        n_timeslots_per_block_to_use = int(block_duration_to_use / period)

        df_out = pd.DataFrame()

        current_start_index = 0
        while current_start_index < n_timeslots:
            current_end_index = current_start_index + n_timeslots_per_block
            if current_end_index > n_timeslots:
                current_end_index = n_timeslots

            # Create a smaller dataframe for the optimisation of the smaller period of time
            sub_opt_df_in = self._df_in.iloc[current_start_index:current_end_index+1]

            logging.info(f"Optimising range {sub_opt_df_in.index[0]} -> {sub_opt_df_in.index[-1]}...")
            sub_opt_df_out, sub_opt_num_nan = self._run_one_optimisation(df_in=sub_opt_df_in, init_soe=init_soe)
            n_timeslots_with_nan_pricing += sub_opt_num_nan

            # Drop the end of the optimisation solution (this is discussed above)
            sub_opt_df_out_to_use = sub_opt_df_out.iloc[0:n_timeslots_per_block_to_use]

            # Create a single dataframe with the results of all the individual optimisations
            df_out = pd.concat([df_out, sub_opt_df_out_to_use], axis=0)

            # Prepare for the next iteration
            current_start_index += n_timeslots_per_block_to_use
            if len(sub_opt_df_out) > len(sub_opt_df_out_to_use):
                init_soe = sub_opt_df_out.iloc[len(sub_opt_df_out_to_use)]["soe"]
            else:
                init_soe = np.nan  # this must be the last iteration

        if n_timeslots_with_nan_pricing > 0:
            get_user_ack_of_warning_or_exit(
                f"{n_timeslots_with_nan_pricing} time slots had NaN pricing data and could not be optimised"
            )

        return df_out

    def _run_one_optimisation(self, df_in: pd.DataFrame, init_soe: float) -> (pd.DataFrame, int):
        """
        Uses the pulp library to optimise the battery schedule as a linear programming optimisation problem.
        This is currently a 'perfect hindsight' view because in practice we wouldn't know the imbalance pricing or
        microgrid load and solar generation ahead of time.
        It also returns the number of timeslots that had nan pricing for logging/warning purposes.
        """
        problem = pulp.LpProblem(name="MicrogridProblem", sense=pulp.LpMinimize)

        lp_var_bess_soe = []
        lp_var_bess_discharges_to_load = []
        lp_var_bess_discharges_to_grid = []
        lp_var_bess_charges_from_solar = []
        lp_var_bess_charges_from_grid = []
        lp_var_bess_is_charging = []
        lp_costs = []

        # We use indexes rather than datetimes to represent each time slot
        timeslots = range(0, len(df_in))

        n_timeslots_with_nan_pricing = 0

        for t in timeslots:

            lp_var_bess_soe.append(
                pulp.LpVariable(
                    name=f"bess_soe_{t}",
                    lowBound=0.0,
                    upBound=self._battery_energy_capacity
                )
            )
            lp_var_bess_charges_from_solar.append(
                pulp.LpVariable(
                    name=f"bess_charge_from_solar_{t}",
                    lowBound=0.0,
                    upBound=df_in.iloc[t]["solar_not_supplying_load"]
                )
            )
            lp_var_bess_charges_from_grid.append(
                pulp.LpVariable(
                    name=f"bess_charge_from_grid_{t}",
                    lowBound=0.0,
                    upBound=df_in.iloc[t]["max_charge_from_grid"]
                )
            )
            lp_var_bess_discharges_to_load.append(
                pulp.LpVariable(
                    name=f"bess_discharge_to_load_{t}",
                    lowBound=0.0,
                    upBound=df_in.iloc[t]["load_not_supplied_by_solar"]
                )
            )
            lp_var_bess_discharges_to_grid.append(
                pulp.LpVariable(
                    name=f"bess_discharge_to_grid_{t}",
                    lowBound=0.0,
                    upBound=df_in.iloc[t]["max_discharge_to_load"]
                )
            )

            # This binary var is used to make charge and discharging mutually exclusive for each time period
            lp_var_bess_is_charging.append(
                pulp.LpVariable(
                    name=f"bess_is_charging_{t}",
                    cat=pulp.LpBinary
                )
            )

            # Get the rates from the input dataframe, and check they are not nan - if they are then don't allow any
            # activity in this period.
            rate_final_bess_charge_from_grid = df_in.iloc[t]["rate_final_bess_charge_from_grid"]
            int_rate_final_bess_charge_from_solar = df_in.iloc[t]["int_rate_final_bess_charge_from_solar"]
            rate_final_bess_discharge_to_grid = df_in.iloc[t]["rate_final_bess_discharge_to_grid"]
            int_rate_final_bess_discharge_to_load = df_in.iloc[t]["int_rate_final_bess_discharge_to_load"]
            if np.any(np.isnan([
                rate_final_bess_charge_from_grid,
                int_rate_final_bess_charge_from_solar,
                rate_final_bess_discharge_to_grid,
                int_rate_final_bess_discharge_to_load
            ])):
                # the costs function throws an exception when these are NaN, so set to zero but disallow any activity
                # by adding constraints
                rate_final_bess_charge_from_grid = 0
                int_rate_final_bess_charge_from_solar = 0
                rate_final_bess_discharge_to_grid = 0
                int_rate_final_bess_discharge_to_load = 0
                problem += lp_var_bess_charges_from_solar[t] == 0
                problem += lp_var_bess_charges_from_grid[t] == 0
                problem += lp_var_bess_discharges_to_load[t] == 0
                problem += lp_var_bess_discharges_to_grid[t] == 0

                n_timeslots_with_nan_pricing += 1

            lp_costs.append(
                lp_var_bess_charges_from_grid[t] * rate_final_bess_charge_from_grid +
                lp_var_bess_charges_from_solar[t] * int_rate_final_bess_charge_from_solar +
                lp_var_bess_discharges_to_grid[t] * rate_final_bess_discharge_to_grid +
                lp_var_bess_discharges_to_load[t] * int_rate_final_bess_discharge_to_load
            )

        for t in timeslots:

            # Constraints to define that all the flows are positive - prevent the optimiser from using a negative
            problem += lp_var_bess_charges_from_solar[t] >= 0.0
            problem += lp_var_bess_charges_from_grid[t] >= 0.0
            problem += lp_var_bess_discharges_to_load[t] >= 0.0
            problem += lp_var_bess_discharges_to_grid[t] >= 0.0
            # Constraints for maximum charge/discharge rates AND make charge and discharge mutually exclusive
            problem += lp_var_bess_charges_from_solar[t] + lp_var_bess_charges_from_grid[t] <= (
                    df_in.iloc[t]["bess_max_charge"] * lp_var_bess_is_charging[t])
            problem += lp_var_bess_discharges_to_load[t] + lp_var_bess_discharges_to_grid[t] <= (
                    df_in.iloc[t]["bess_max_discharge"] * (1 - lp_var_bess_is_charging[t]))

        # The objective function is the sum of costs across all timeslots, which will be minimised
        problem += pulp.lpSum(lp_costs)

        # Set the initial state of energy
        problem += lp_var_bess_soe[0] == init_soe

        # Don't allow any battery activity in the last period as this requires more complicated constraints to make it
        # work, instead just ignore the last period and drop it from the results.
        problem += lp_var_bess_charges_from_solar[-1] == 0
        problem += lp_var_bess_charges_from_grid[-1] == 0
        problem += lp_var_bess_discharges_to_load[-1] == 0
        problem += lp_var_bess_discharges_to_grid[-1] == 0

        # Constraint to define how the SoE changes across the timeslots. This loop starts from the second timeslot.
        for t in timeslots[1:]:
            problem += (
                lp_var_bess_soe[t] == lp_var_bess_soe[t - 1]
                + lp_var_bess_charges_from_solar[t - 1] * self._battery_charge_efficiency
                + lp_var_bess_charges_from_grid[t - 1] * self._battery_charge_efficiency
                - lp_var_bess_discharges_to_load[t - 1]
                - lp_var_bess_discharges_to_grid[t - 1]
            )

        status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
        if status != 1:
            raise RuntimeError("Failed to solve optimisation problem")

        df_sol = _get_solution_as_dataframe(problem.variables(), df_in.index)
        df_sol = df_sol.iloc[:-1, :]  # Drop the last row because the last time slot is not actually optimised

        self._ensure_merit_order_of_charge_and_discharge(df_sol)

        # Create a dataframe to return with just the required info
        df_ret = pd.DataFrame(index=df_sol.index)
        df_ret["soe"] = df_sol["bess_soe"]
        df_ret["energy_delta"] = (
            df_sol["bess_charge_from_solar"] + df_sol["bess_charge_from_grid"]
            - df_sol["bess_discharge_to_grid"] - df_sol["bess_discharge_to_load"]
        )
        df_ret["bess_losses"] = (
            (df_sol["bess_charge_from_solar"] + df_sol["bess_charge_from_grid"]) * (1 - self._battery_charge_efficiency)
        )

        # TODO: tidy up code generally - a good point to clean up interface to multiple algos?

        return df_ret, n_timeslots_with_nan_pricing

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
        when_charging_from_grid = df_sol[df_sol["bess_charge_from_grid"] > 0]
        check = (
            (when_charging_from_grid["bess_charge_from_solar"] - self._df_in["solar_not_supplying_load"])
            > tolerance
        ).sum()
        assert check == 0, "Optimisation internal error - add constraint for energy merit order on charge"

        # Check that we always discharge to onsite load 'first', before discharging to grid
        when_discharging_to_grid = df_sol[df_sol["bess_discharge_to_grid"] > 0]
        check = (
                (when_discharging_to_grid["bess_discharge_to_load"] - self._df_in["load_not_supplied_by_solar"])
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
