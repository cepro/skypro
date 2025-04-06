import logging
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from simt_common.rates.microgrid import VolRatesForEnergyFlows, get_vol_rates_dfs
from simt_common.rates.rates import OSAMFlatVolRate
from simt_common.timeutils.math import floor_hh
from simt_common.timeutils.timeseries import get_step_size

from skypro.commands.simulator.algorithms.lp.optimiser import run_one_optimisation
from skypro.commands.simulator.algorithms.price_curve.algo import is_first_timeslot_of_month, is_first_timeslot_of_day
from skypro.commands.simulator.algorithms.rate_management import run_osam_calcs_for_day


from skypro.commands.simulator.algorithms.utils import get_energy, get_power
from skypro.commands.simulator.config.config import Bess as BessConfig, MonteCarloAlgo as Config, Optimiser


# TODO: split imports from price_curve into utility module?


class MonteCarloAlgo:
    def __init__(
            self,
            algo_config: Config,
            bess_config: BessConfig,
            live_vol_rates: VolRatesForEnergyFlows,
            df: pd.DataFrame
    ):
        self._algo_config = algo_config
        self._bess_config = bess_config
        self._live_vol_rates = live_vol_rates

        self._df = df.copy()

        ref_prices = pd.read_csv("/Users/marcuswood/Desktop/all/repos/skypro-cli/reference_price.csv")
        ref_prices.index = pd.to_datetime(ref_prices["time"], utc=True)
        ref_prices = ref_prices.drop("time", axis=1)
        ref_prices = ref_prices.ffill().bfill()
        self._df["ref_bid_price"] = ref_prices["bid"]
        self._df["ref_offer_price"] = ref_prices["offer"]

        self._time_step = timedelta(seconds=pd.to_timedelta(self._df.index.freq).total_seconds())
        self._time_steps_per_sp = int(timedelta(minutes=30) / self._time_step)
        # time_step_hours = time_step.total_seconds() / 3600

        # We have multiple rows per SP, each row within the same SP has the same reference price
        self._df["ref_bid_price"] = self._df["ref_bid_price"].ffill(limit=self._time_steps_per_sp-1)
        self._df["ref_offer_price"] = self._df["ref_offer_price"].ffill(limit=self._time_steps_per_sp-1)

    def run(self) -> pd.DataFrame:

        # These vars keep track of the previous settlement periods values
        last_soe = self._bess_config.energy_capacity / 2  # initial SoE is 50%
        last_energy_delta = 0
        last_bess_losses = 0
        end_t = self._df.index[-1]



        # Run through each row (where each row represents a time step) and apply the strategy
        for t in self._df.index:

            if is_first_timeslot_of_month(t):
                # Show the user the progress with a log of each month
                print(f"Simulating {t.date()}...")

            if (t == self._df.index[0]) or is_first_timeslot_of_day(t):
                # If this is the first timestep of the day then calculate the rates for the coming day.
                # This is done on each day in turn because OSAM rates vary day-by-day depending on historical volumes.
                self._df, todays_index = run_osam_calcs_for_day(self._df, t)

                self._df = add_vol_rates_to_df(
                    df=self._df,
                    index_to_add_for=todays_index,
                    mkt_vol_rates=self._live_vol_rates,
                    live_or_final="live"
                )

                # TODO: this is duplicated with price curve
                # This algo also uses the last live rate from the previous SP to inform actions, so make that available
                # on each df row:
                # TODO: this shifts all rows for every day, it may be a speed improvement to make it so only the days
                #  data is shifted
                cols_to_shift = [
                    "mkt_vol_rate_live_grid_to_batt",
                    "mkt_vol_rate_live_batt_to_grid",
                    "imbalance_volume_live",
                ]
                for col in cols_to_shift:
                    self._df[f"prev_sp_{col}"] = self._df[col].shift(self._time_steps_per_sp).bfill(limit=self._time_steps_per_sp-1)

            # Set the `soe` column to the value at the start of this time step (the previous value plus the energy
            # transferred in the previous time step)
            soe = last_soe + last_energy_delta - last_bess_losses
            self._df.loc[t, "soe"] = soe

            mkt_vol_rate_grid_to_batt = None
            mkt_vol_rate_batt_to_grid = None

            if not np.isnan(self._df.loc[t, "mkt_vol_rate_live_grid_to_batt"]) and \
               not np.isnan(self._df.loc[t, "mkt_vol_rate_live_batt_to_grid"]):
                # If we have predictions then use them
                mkt_vol_rate_grid_to_batt = self._df.loc[t, "mkt_vol_rate_live_grid_to_batt"]
                mkt_vol_rate_batt_to_grid = self._df.loc[t, "mkt_vol_rate_live_batt_to_grid"]
            elif self._df.loc[t, "time_into_sp"] < timedelta(minutes=10) and \
                 not np.isnan(self._df.loc[t, "prev_sp_mkt_vol_rate_live_grid_to_batt"]) and \
                 not np.isnan(self._df.loc[t, "prev_sp_mkt_vol_rate_live_batt_to_grid"]) and \
                 not np.isnan(self._df.loc[t, "prev_sp_imbalance_volume_live"]) and \
                 abs(self._df.loc[t, "prev_sp_imbalance_volume_live"]) * 1e3 >= 150000:  # TODO: configurable

                mkt_vol_rate_grid_to_batt = self._df.loc[t, "prev_sp_mkt_vol_rate_live_grid_to_batt"]
                mkt_vol_rate_batt_to_grid = self._df.loc[t, "prev_sp_mkt_vol_rate_live_batt_to_grid"]

            if mkt_vol_rate_grid_to_batt is not None:
                opt_end = t + timedelta(hours=36)
                if opt_end > end_t:
                    opt_end = end_t

                energy_delta = self._monte_carlo(
                    index=pd.date_range(start=t, end=opt_end, freq=self._time_step),
                    init_soe=soe,
                    init_price_prediction_batt_to_grid=mkt_vol_rate_batt_to_grid,
                    init_price_prediction_grid_to_batt=mkt_vol_rate_batt_to_grid,
                )
            else:
                energy_delta = 0.0

            # Cap the SoE at the physical limits of the battery
            if soe + energy_delta > self._bess_config.energy_capacity:
                energy_delta = self._bess_config.energy_capacity - soe
            elif soe + energy_delta < 0:
                energy_delta = -soe

            # Apply a charge efficiency
            if energy_delta > 0:
                bess_losses = energy_delta * (1 - self._bess_config.charge_efficiency)
            else:
                bess_losses = 0

            self._df.loc[t, "power"] = get_power(energy_delta, self._time_step)
            self._df.loc[t, "energy_delta"] = energy_delta
            self._df.loc[t, "bess_losses"] = bess_losses

            # Save for next iteration...
            last_soe = soe
            last_energy_delta = energy_delta
            last_bess_losses = bess_losses

        return self._df[["soe", "energy_delta", "bess_losses"]]

    def _monte_carlo(
            self,
            index: pd.DatetimeIndex,
            init_soe: float,
            init_price_prediction_grid_to_batt: float,  # TODO: names standardised
            init_price_prediction_batt_to_grid: float
    ) -> float:
        """
        Runs a monte-carlo analysis of optimisations and returns the energy delta should be set at index[0]
        :param index is the time index to analyse over
        :param num_scenarios is the number of simulated scenarios which are optimised
        """

        common_df = self._get_common_df(self._algo_config, index)

        opt_dfs: List[pd.DataFrame] = []  # This holds a dataframe for each optimisation run in the monte-carlo analysis

        t_start = index[0]
        t_first_sp_end = floor_hh(t_start) + timedelta(minutes=30)

        system_state_periods = [
            timedelta(hours=8),
            timedelta(hours=6),
            timedelta(hours=4),
            timedelta(hours=2),
        ]
        system_state_dfs: List[pd.DataFrame] = []
        for period in system_state_periods:
            wave = _square_wave(
                index=index,
                wave_period=period,
                amplitude=1.0
            )
            system_state_dfs.append((wave * -0.5 + 0.5).to_frame(name="system_state"))
            system_state_dfs.append((wave * 0.5 + 0.5).to_frame(name="system_state"))

        # Work out the long and the short prices for each flow of interest
        prices_df = pd.DataFrame(index=index)

        for flow, direction in [("grid_to_batt", 1), ("batt_to_grid", -1)]:
            prices_df[f"mkt_vol_rate_{flow}_predicted_when_long"] = self._df[f"mkt_vol_rate_{flow}_non_imbalance"] + self._df["ref_bid_price"] * direction
            prices_df[f"mkt_vol_rate_{flow}_predicted_when_short"] = self._df[f"mkt_vol_rate_{flow}_non_imbalance"] + self._df["ref_offer_price"] * direction

        # import plotly.express as px
        # px.line(prices_df).show()
        # px.line(self._df[["mkt_vol_rate_grid_to_batt_non_imbalance"]])

        for system_state_df in system_state_dfs:
            opt_df = common_df.copy()
            for flow in ["grid_to_batt", "batt_to_grid"]:
                opt_df.loc[system_state_df["system_state"] == 0, f"mkt_vol_rate_{flow}"] = prices_df[f"mkt_vol_rate_{flow}_predicted_when_long"]
                opt_df.loc[system_state_df["system_state"] == 1, f"mkt_vol_rate_{flow}"] = prices_df[f"mkt_vol_rate_{flow}_predicted_when_short"]
                opt_df["int_vol_rate_solar_to_batt"] = 0  # TODO:
                opt_df["int_vol_rate_batt_to_load"] = 0  # TODO:

            # if init_price_prediction is not None:
            # Use a real-time price prediction for the first settlement period, rather than modelling it as stochastic
            opt_df.loc[(opt_df.index >= t_start) & (opt_df.index < t_first_sp_end), f"mkt_vol_rate_grid_to_batt"] = init_price_prediction_grid_to_batt
            opt_df.loc[(opt_df.index >= t_start) & (opt_df.index < t_first_sp_end), f"mkt_vol_rate_batt_to_grid"] = init_price_prediction_batt_to_grid

            opt_dfs.append(opt_df)

        # plt_df = pd.DataFrame(index=index)
        # for col in [
        #     "mkt_vol_rate_grid_to_batt_predicted_when_long",
        #     "mkt_vol_rate_grid_to_batt_predicted_when_short",
        #     "mkt_vol_rate_batt_to_grid_predicted_when_long",
        #     "mkt_vol_rate_batt_to_grid_predicted_when_short"
        # ]:
        #     plt_df[col] = prices_df[col]
        # for i, opt_df in enumerate(opt_dfs):
        #     plt_df[f"{i}_mkt_vol_rate_grid_to_batt"] = opt_df[f"mkt_vol_rate_grid_to_batt"]
        #     plt_df[f"{i}_mkt_vol_rate_batt_to_grid"] = opt_df[f"mkt_vol_rate_batt_to_grid"]
        # px.line(plt_df).show()

        logging.info(f"Monte-carlo optimising range {index[0]} -> {index[-1]}...")

        # Run all the monte-carlo options for this time-step in parallel if possible
        results = []
        with ProcessPoolExecutor(max_workers=8) as executor:
            future_results = []
            for i, opt_df in enumerate(opt_dfs):
                future_results.append(
                    executor.submit(
                        run_one_optimisation,
                        df_in=opt_df,
                        init_soe=init_soe,
                        block_config=self._algo_config.optimiser.blocks,
                        bess_config=self._bess_config,
                    )
                )
            for future in future_results:
                results.append(future.result())

        #
        # sol_dfs = []
        # for i, opt_df in enumerate(opt_dfs):
        #     logging.info(f"Optimising {opt_df.index[0]} -> {opt_df.index[-1]} MC {i+1}/{len(opt_dfs)}...")
        #     sol_df, _ = run_one_optimisation(  # TODO: number of nans is returned
        #         df_in=opt_df,
        #         init_soe=init_soe,
        #         block_config=self._algo_config.optimiser.blocks,
        #         bess_config=self._bess_config,
        #     )
        #     sol_dfs.append(sol_df)

        total_energy_delta = 0.0
        for sol_df, _ in results: # TODO: number of nans is returned
            first_sp_sol_df = sol_df[(sol_df.index >= t_start) & (sol_df.index < t_first_sp_end)]
            total_energy_delta += first_sp_sol_df["energy_delta"].sum()
        sp_avg_energy_delta = total_energy_delta / len(opt_dfs)

        num_timeslots_left_in_this_sp = len(first_sp_sol_df)
        energy_in_this_timestep = sp_avg_energy_delta * (num_timeslots_left_in_this_sp / self._time_steps_per_sp)

        return energy_in_this_timestep

    def _get_common_df(self, config: Config, index: pd.DatetimeIndex):
        # TODO: this is copied and edited from optimiser.py: consolidate!
        common_df = pd.DataFrame(index=index)

        for col in ["time_into_sp", "bess_max_charge", "bess_max_discharge"]:
            common_df[col] = self._df[col]

        # Calculate some of the microgrid flows - at the moment this is the only algo that uses these values, but in
        # the future it may make sense to pass these values in rather than have each algo calculate them independently.
        common_df["solar_to_load"] = self._df[["solar", "load"]].min(axis=1)
        common_df["load_not_supplied_by_solar"] = self._df["load"] - common_df["solar_to_load"]
        common_df["solar_not_supplying_load"] = self._df["solar"] - common_df["solar_to_load"]
        # When charging we must use excess solar first:
        common_df["max_charge_from_grid"] = np.maximum(self._df["bess_max_charge"] - common_df["solar_not_supplying_load"], 0)
        # When discharging we must send power to microgrid load first:
        common_df["max_discharge_to_grid"] = np.maximum(self._df["bess_max_discharge"] - common_df["load_not_supplied_by_solar"], 0)

        if config.optimiser.blocks.do_active_export_constraint_management:
            common_df["min_charge"] = self._df[self._df["bess_max_discharge"] < 0]["bess_max_discharge"] * -1
            common_df["min_charge"] = common_df["min_charge"].fillna(0)
            # The min_charge constraint is currently applied to the 'charge from solar' flow, as it was used to
            # manage excess solar power. If the min charge is a floating point error away from solar_not_supplying_load
            # then make them equal to avoid constraint issues.
            close_idx = common_df.index[np.isclose(common_df["min_charge"], common_df["solar_not_supplying_load"])]
            common_df.loc[close_idx, "min_charge"] = common_df.loc[close_idx, "solar_not_supplying_load"]
        else:
            common_df["min_charge"] = 0.0

        if config.optimiser.blocks.do_active_import_constraint_management:
            common_df["min_discharge"] = self._df[self._df["bess_max_charge"] < 0]["bess_max_charge"] * -1
            common_df["min_discharge"] = common_df["min_discharge"].fillna(0)
            # The min_discharge constraint is currently applied to the 'discharge to load' flow, as it was used to
            # manage excess load. If the min discharge is a floating point error away from load_not_supplied_by_solar
            # then make them equal to avoid constraint issues.
            close_idx = common_df.index[np.isclose(common_df["min_discharge"], common_df["load_not_supplied_by_solar"])]
            common_df.loc[close_idx, "min_discharge"] = common_df.loc[close_idx, "load_not_supplied_by_solar"]
        else:
            common_df["min_discharge"] = 0.0

        return common_df

def _square_wave(index: pd.DatetimeIndex, wave_period: timedelta, amplitude: float) -> pd.Series:

    cycle_length = int(np.floor(wave_period / get_step_size(index)))  # Number of rows for one complete cycle

    positions = np.arange(len(index))
    square_wave = np.where((positions % cycle_length) < cycle_length / 2, amplitude, -amplitude)

    return pd.Series(index=index, data=square_wave)


# TODO: this is copied and edited from rate_mangaement.py: consolidate
# TODO: these should return sub-dataframes that are appended, rather than returning the whole df?
def add_vol_rates_to_df(
    df: pd.DataFrame,
    index_to_add_for: pd.DatetimeIndex,
    mkt_vol_rates: VolRatesForEnergyFlows,
    live_or_final: str,  # TODO: this isn't helpful for the LP optimiser? as it's neither live nor final
) -> pd.DataFrame:
    """
    Adds the total market and internal p/kWh rates for each flow to the dataframe for the period specified by
    `index_to_add_for` and returns the dataframe.
    """
    df = df.copy()

    # Inform any OSAM rate objects about the NCSP for today
    for rate in mkt_vol_rates.grid_to_batt:
        if isinstance(rate, OSAMFlatVolRate):
            rate.add_ncsp(df.loc[index_to_add_for, "osam_ncsp"])

    # Next we can calculate the individual p/kWh rates that apply for today
    mkt_vol_rates_dfs, int_vol_rates_dfs = get_vol_rates_dfs(index_to_add_for, mkt_vol_rates, log=False)

    # Then we sum up the individual rates to create a total for each flow
    for set_name, vol_rates_df in mkt_vol_rates_dfs.items():
        df.loc[index_to_add_for, f"mkt_vol_rate_{live_or_final}_{set_name}"] = vol_rates_df.sum(axis=1, skipna=False)
    for set_name, vol_rates_df in int_vol_rates_dfs.items():
        df.loc[index_to_add_for, f"int_vol_rate_{live_or_final}_{set_name}"] = vol_rates_df.sum(axis=1, skipna=False)

    # TODO: this is a bit of a hack to get to 'non-imbalance' rates separated out
    for set_name in ["grid_to_batt", "batt_to_grid"]:
        df[f"mkt_vol_rate_{set_name}_non_imbalance"] = mkt_vol_rates_dfs[set_name].drop(["imbalance", "gimbalance", "StatkraftMultiplier"], axis=1, errors="ignore").sum(axis=1)
    #
    # breakpoint()
    # import plotly.express as px
    # px.line(mkt_vol_rates_dfs["grid_to_batt"]).show()

    return df
