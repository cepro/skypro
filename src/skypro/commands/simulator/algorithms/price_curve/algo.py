import logging
from datetime import timedelta, datetime

import numpy as np
import pandas as pd
import pytz
from simt_common.rates.microgrid import get_rates_dfs, RatesForEnergyFlows
from simt_common.rates.osam import calculate_osam_ncsp
from simt_common.rates.rates import OSAMRate
from simt_common.timeutils.math import add_wallclock_days

from skypro.commands.simulator.algorithms.peak import get_peak_approach_energies, get_peak_power
from skypro.commands.simulator.algorithms.microgrid import get_microgrid_algo_energy
from skypro.commands.simulator.algorithms.system_state import get_system_state, SystemState
from skypro.commands.simulator.algorithms.utils import get_power, cap_power, get_energy
from skypro.commands.simulator.cartesian import Curve, Point
from skypro.commands.simulator.config import get_relevant_niv_config, PriceCurveAlgo as PriceCurveAlgoConfig, Bess as BessConfig
from skypro.commands.simulator.microgrid import calculate_microgrid_flows


class PriceCurveAlgo:
    def __init__(
            self,
            algo_config: PriceCurveAlgoConfig,
            bess_config: BessConfig,
            live_rates: RatesForEnergyFlows,
            df: pd.DataFrame
    ):
        """
        df columns:
        - `solar`
        - `load`
        """
        self._algo_config = algo_config
        self._bess_config = bess_config
        self._live_rates = live_rates

        self._df = df.copy()

    def run(self):

        # These vars keep track of the previous settlement periods values
        last_soe = self._bess_config.energy_capacity / 2  # initial SoE is 50%
        last_energy_delta = 0
        last_bess_losses = 0
        num_skipped_periods = 0

        time_step = timedelta(seconds=pd.to_timedelta(self._df.index.freq).total_seconds())
        time_steps_per_sp = int(timedelta(minutes=30)/ time_step)
        time_step_hours = time_step.total_seconds() / 3600

        # Run through each row (where each row represents a time step) and apply the strategy
        for t in self._df.index:

            if is_first_timeslot_of_month(t):
                # Show the user the progress with a log of each month
                print(f"Simulating {t.date()}...")

            if (t == self._df.index[0]) or is_first_timeslot_of_day(t):
                # If this is the first timestep of the day then calculate the rates for the coming day.
                # This is done on each day in turn because OSAM rates vary day-by-day depending on historical volumes.
                self.calculate_rates_for_day(t)

                # This algo also uses the last live rate from the previous SP to inform actions, so make that available
                # on each df row:
                # TODO: this shifts all rows for every day, it may be a speed improvement to make it so only the days
                #  data is shifted
                cols_to_shift = [
                    "rate_live_bess_charge_from_grid",
                    "rate_live_bess_discharge_to_grid",
                    "imbalance_volume_live",
                ]
                for col in cols_to_shift:
                    self._df[f"prev_sp_{col}"] = self._df[col].shift(time_steps_per_sp).bfill(limit=time_steps_per_sp-1)

            # Set the `soe` column to the value at the start of this time step (the previous value plus the energy
            # transferred in the previous time step)
            soe = last_soe + last_energy_delta - last_bess_losses
            self._df.loc[t, "soe"] = soe

            # Select the appropriate NIV chasing configuration for this time of day
            niv_config = get_relevant_niv_config(self._algo_config.niv_chase_periods, t).niv

            system_state = get_system_state(self._df, t, niv_config.volume_cutoff_for_prediction)

            peak_power = get_peak_power(
                peak_config=self._algo_config.peak,
                t=t,
                time_step=time_step,
                soe=soe,
                bess_max_power_discharge=self._df.loc[t, "bess_max_power_discharge"],
                microgrid_residual_power=self._df.loc[t, "microgrid_residual_power"],
                system_state=system_state
            )
            if peak_power is not None:
                power = peak_power

            else:
                target_energy_delta = 0

                red_approach_energy, amber_approach_energy = get_peak_approach_energies(
                    t=t,
                    time_step=time_step,
                    soe=soe,
                    charge_efficiency=self._bess_config.charge_efficiency,
                    peak_config=self._algo_config.peak,
                    is_long=system_state == SystemState.LONG
                )

                self._df.loc[t, "red_approach_distance"] = red_approach_energy
                self._df.loc[t, "amber_approach_distance"] = amber_approach_energy

                if not np.isnan(self._df.loc[t, "rate_live_bess_charge_from_grid"]) and \
                        not np.isnan(self._df.loc[t, "rate_live_bess_discharge_to_grid"]) and \
                        not np.isnan(self._df.loc[t, "imbalance_volume_live"]):

                    # If we have predictions then use them
                    target_energy_delta = get_target_energy_delta_from_shifted_curves(
                        df_in=self._df,
                        t=t,
                        charge_rate_col="rate_live_bess_charge_from_grid",
                        discharge_rate_col="rate_live_bess_discharge_to_grid",
                        imbalance_volume_col="imbalance_volume_live",
                        soe=soe,
                        battery_charge_efficiency=self._bess_config.charge_efficiency,
                        niv_config=niv_config
                    )

                elif self._df.loc[t, "time_into_sp"] < timedelta(minutes=10) and \
                        not np.isnan(self._df.loc[t, "prev_sp_rate_live_bess_charge_from_grid"]) and \
                        not np.isnan(self._df.loc[t, "prev_sp_rate_live_bess_discharge_to_grid"]) and \
                        not np.isnan(self._df.loc[t, "prev_sp_imbalance_volume_live"]):

                    # If we don't have predictions yet, then in the first 10mins of the SP we can use the previous SP's
                    # imbalance data to inform the activity

                    # MWh to kWh
                    if abs(self._df.loc[
                               t, "prev_sp_imbalance_volume_live"]) * 1e3 >= niv_config.volume_cutoff_for_prediction:
                        target_energy_delta = get_target_energy_delta_from_shifted_curves(
                            df_in=self._df,
                            t=t,
                            charge_rate_col="prev_sp_rate_live_bess_charge_from_grid",
                            discharge_rate_col="prev_sp_rate_live_bess_discharge_to_grid",
                            imbalance_volume_col="prev_sp_imbalance_volume_live",
                            soe=soe,
                            battery_charge_efficiency=self._bess_config.charge_efficiency,
                            niv_config=niv_config
                        )
                else:
                    # TODO: this isn't very helpful, it would be more interesting to report how many settlement periods
                    #       are skipped
                    num_skipped_periods += 1

                if self._algo_config.microgrid:
                    system_state = SystemState.UNKNOWN
                    if self._algo_config.microgrid.imbalance_control:
                        system_state = get_system_state(self._df, t,
                                                        self._algo_config.microgrid.imbalance_control.niv_cutoff_for_system_state_assumption)
                    microgrid_algo_energy = get_microgrid_algo_energy(
                        config=self._algo_config.microgrid,
                        microgrid_residual_energy=self._df.loc[t, "microgrid_residual_power"] * time_step_hours,
                        system_state=system_state
                    )
                else:
                    microgrid_algo_energy = 0.0

                if red_approach_energy > 0:
                    target_energy_delta = max(red_approach_energy, amber_approach_energy, target_energy_delta)
                elif amber_approach_energy > 0:
                    target_energy_delta = max(amber_approach_energy, target_energy_delta)
                else:
                    target_energy_delta = target_energy_delta + microgrid_algo_energy

                power = get_power(target_energy_delta, time_step)  # TODO: check this tme step

            power = cap_power(power, self._df.loc[t, "bess_max_power_charge"], self._df.loc[t, "bess_max_power_discharge"])
            energy_delta = get_energy(power, time_step)

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

            self._df.loc[t, "power"] = power
            self._df.loc[t, "energy_delta"] = energy_delta
            self._df.loc[t, "bess_losses"] = bess_losses

            # Save for next iteration...
            last_soe = soe
            last_energy_delta = energy_delta
            last_bess_losses = bess_losses

        if num_skipped_periods > 0:
            time_step_minutes = time_step.total_seconds() / 60
            logging.info(
                f"Skipped {num_skipped_periods}/{len(self._df)} {time_step_minutes} minute periods (probably due to "
                f"missing imbalance data)")

        return self._df[["soe", "energy_delta", "bess_losses", "red_approach_distance", "amber_approach_distance"]]

    def calculate_rates_for_day(self, t):
        """
        Adds the rates for the day starting at `t` to the dataframe
        """

        has_osam_rates = False
        for _, rates in self._live_rates.get_all_sets_named():
            for rate in rates:
                if isinstance(rate, OSAMRate):
                    has_osam_rates = True
                    break

        end_of_today = add_wallclock_days(t, 1)
        start_of_yesterday = add_wallclock_days(t, -1)
        todays_index = self._df.loc[t:end_of_today].iloc[:-1].index

        mg_flow_calc_start = start_of_yesterday
        if mg_flow_calc_start < self._df.index[0]:
            mg_flow_calc_start = self._df.index[0]
        if mg_flow_calc_start < t:
            # To calculate OSAM rates we first need to work out the microgrid energy flows for yesterday given the
            # simulated actions
            df_with_mg_flows = calculate_microgrid_flows(self._df.loc[mg_flow_calc_start:t])

            # The below loc command doesn't work unless all the columns are already present.
            match_columns(self._df, df_with_mg_flows)
            self._df.loc[mg_flow_calc_start:t] = calculate_microgrid_flows(self._df.loc[mg_flow_calc_start:t])
        else:
            logging.info("Couldn't calculate microgrid flows as there's no data")

        if has_osam_rates:
            # Next we can calculate the OSAM NCSP factor for today
            # TODO: this isn't working - it looks like the input df is empty?
            self._df.loc[todays_index, "osam_ncsp"] = calculate_osam_ncsp(
                df=self._df,
                index_to_calc_for=todays_index,
                imp_bp_col="grid_import",
                exp_bp_col="grid_export",
                imp_stor_col="bess_charge",
                exp_stor_col="bess_discharge",
                imp_gen_col=None,
                exp_gen_col="solar",
            )

            # Inform any OSAM rate objects about the NCSP for today
            for _, rates in self._live_rates.get_all_sets_named():
                for rate in rates:
                    if isinstance(rate, OSAMRate):
                        rate.add_ncsp(self._df.loc[todays_index, "osam_ncsp"])

        # Next we can calculate the individual p/kWh rates that apply for today
        live_ext_rates_dfs, live_int_rates_dfs = get_rates_dfs(todays_index, self._live_rates, log=False)

        # Then we sum up the individual rates to create a total for each flow
        for set_name, rates_df in live_ext_rates_dfs.items():
            self._df.loc[todays_index, f"rate_live_{set_name}"] = rates_df.sum(axis=1, skipna=False)
        for set_name, rates_df in live_int_rates_dfs.items():
            self._df.loc[todays_index, f"int_rate_live_{set_name}"] = rates_df.sum(axis=1, skipna=False)


def get_target_energy_delta_from_shifted_curves(
        df_in: pd.DataFrame,
        t: datetime,
        charge_rate_col: str,
        discharge_rate_col: str,
        imbalance_volume_col: str,
        soe: float,
        battery_charge_efficiency: float,
        niv_config,
):

    shifted_rate_charge_from_grid, shifted_rate_discharge_to_grid = shift_rates(
        original_import_rate=df_in.loc[t, charge_rate_col],
        original_export_rate=-df_in.loc[t, discharge_rate_col],
        imbalance_volume=df_in.loc[t, imbalance_volume_col],
        rate_shift_long=niv_config.curve_shift_long,
        rate_shift_short=niv_config.curve_shift_short
    )

    target_energy_delta = get_target_energy_delta_from_curves(
        charge_curve=niv_config.charge_curve,
        discharge_curve=niv_config.discharge_curve,
        import_rate=shifted_rate_charge_from_grid,
        export_rate=shifted_rate_discharge_to_grid,
        soe=soe,
        battery_charge_efficiency=battery_charge_efficiency
    )
    return target_energy_delta


def shift_rates(
        original_import_rate: float,
        original_export_rate: float,
        imbalance_volume: float,
        rate_shift_long: float,
        rate_shift_short: float
) -> (float, float):
    """
    Shifts the original import and export rates and returns the shifted rates.
    """

    is_long = imbalance_volume < 0

    if is_long:
        shifted_import_rate = original_import_rate - rate_shift_long
        shifted_export_rate = original_export_rate - rate_shift_long
    else:
        shifted_import_rate = original_import_rate + rate_shift_short
        shifted_export_rate = original_export_rate + rate_shift_short

    return shifted_import_rate, shifted_export_rate


def get_target_energy_delta_from_curves(
        charge_curve: Curve,
        discharge_curve: Curve,
        import_rate: float,
        export_rate: float,
        soe: float,
        battery_charge_efficiency: float
) -> float:
    """
    Checks the charge/discharge curves to see if we should be charging/discharging and to what extent.
    Returns the kWh that we should/charge discharge at this price - which may not be practically achievable, depending
    on timeframes, site limits etc.
    """

    target_energy_delta = 0

    charge_distance = charge_curve.vertical_distance(Point(import_rate, soe))
    if charge_distance > 0:
        target_energy_delta = charge_distance / battery_charge_efficiency
    else:
        discharge_distance = discharge_curve.vertical_distance(Point(export_rate, soe))
        if discharge_distance < 0:
            target_energy_delta = discharge_distance

    return target_energy_delta


def match_columns(target_df, source_df):
    """
    Makes sure that all the columns in `source_df` are also present in `target_df`, by creating the columns with NaN
    values if they are not present.
    """
    for col in source_df.columns:
        if col not in target_df:
            target_df[col] = np.nan


def is_first_timeslot_of_day(t: pd.Timestamp) -> bool:
    t = t.astimezone(pytz.timezone("Europe/London"))
    return t.time().hour == 0 and t.time().minute == 0


def is_first_timeslot_of_month(t: pd.Timestamp) -> bool:
    t = t.astimezone(pytz.timezone("Europe/London"))
    return t.day == 1 and t.time().hour == 0 and t.time().minute == 0
