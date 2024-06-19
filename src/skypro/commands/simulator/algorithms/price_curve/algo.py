import logging
from datetime import timedelta, datetime
from typing import List, Optional

import numpy as np
import pandas as pd
from simt_common.timeutils import ClockTimePeriod
from simt_common.timeutils.hh_math import floor_hh

from skypro.cli_utils.cli_utils import get_user_ack_of_warning_or_exit
from skypro.commands.simulator.algorithms.approach import get_red_approach_energy, get_amber_approach_energy
from skypro.commands.simulator.algorithms.spread.algo_2 import get_spread_algo_energy
from skypro.commands.simulator.cartesian import Curve, Point
from skypro.commands.simulator.config.config import get_relevant_niv_config
import skypro.commands.simulator.config as config


def run_price_curve_imbalance_algorithm(
        df_in: pd.DataFrame,
        battery_energy_capacity: float,
        battery_charge_efficiency: float,
        niv_chase_periods: List[config.NivPeriod],
        full_discharge_period: Optional[ClockTimePeriod],
) -> pd.DataFrame:

    # Create a separate dataframe for outputs
    df_out = pd.DataFrame(index=df_in.index)

    # These vars keep track of the previous settlement periods values
    last_soe = battery_energy_capacity / 2  # initial SoE is 50%
    last_energy_delta = 0
    last_bess_losses = 0
    num_skipped_periods = 0

    time_step = pd.to_timedelta(df_in.index.freq)

    # The settlement period is calculated by rounding down to the nearest half-hour
    df_out["sp"] = df_in.index.to_series().apply(lambda t: floor_hh(t))
    df_out["time_into_sp"] = df_in.index.to_series() - df_out["sp"]
    df_out["time_left_of_sp"] = timedelta(minutes=30) - df_out["time_into_sp"]

    # Run through each row (where each row represents a time step) and apply the strategy
    for t in df_in.index:

        # Show the user some progress status
        if (t == df_in.index[0]) or (t.date().day == 1 and t.time().hour == 0 and t.time().minute == 0):
            print(f"Simulating {t.date()}...")

        # Set the `soe` column to the value at the start of this time step (the previous value plus the energy
        # transferred in the previous time step)
        soe = last_soe + last_energy_delta - last_bess_losses
        df_out.loc[t, "soe"] = soe

        # Select the appropriate NIV chasing configuration for this time of day
        niv_config = get_relevant_niv_config(niv_chase_periods, t).niv

        if full_discharge_period and full_discharge_period.contains(t):
            # The configuration may specify that we ignore the charge/discharge curves and do a full discharge
            # for a certain period - probably a DUoS red band
            power = -df_in.loc[t, "bess_max_power_discharge"]

        else:
            target_energy_delta = 0

            # TODO: include the volume size threshold to match with other algo
            imbalance_volume_assumed = df_in.loc[t, "imbalance_volume_predicted"]
            if np.isnan(imbalance_volume_assumed):
                imbalance_volume_assumed = df_in.loc[t, "prev_sp_imbalance_volume_final"]
            if np.isnan(imbalance_volume_assumed):
                imbalance_volume_assumed = np.nan

            red_approach_energy = get_red_approach_energy(t, soe)
            amber_approach_energy = get_amber_approach_energy(t, soe, imbalance_volume_assumed)

            df_out.loc[t, "red_approach_distance"] = red_approach_energy
            df_out.loc[t, "amber_approach_distance"] = amber_approach_energy

            if not np.isnan(df_in.loc[t, "rate_predicted_bess_charge_from_grid"]) and \
                not np.isnan(df_in.loc[t, "rate_predicted_bess_discharge_to_grid"]) and \
                not np.isnan(df_in.loc[t, "imbalance_volume_predicted"]):

                # If we have predictions then use them
                target_energy_delta = get_target_energy_delta_from_shifted_curves(
                    df_in=df_in,
                    t=t,
                    charge_rate_col="rate_predicted_bess_charge_from_grid",
                    discharge_rate_col="rate_predicted_bess_discharge_to_grid",
                    imbalance_volume_col="imbalance_volume_predicted",
                    soe=soe,
                    battery_charge_efficiency=battery_charge_efficiency,
                    niv_config=niv_config
                )

            elif df_out.loc[t, "time_into_sp"] < timedelta(minutes=10) and \
                    not np.isnan(df_in.loc[t, "prev_sp_rate_final_bess_charge_from_grid"]) and \
                    not np.isnan(df_in.loc[t, "prev_sp_rate_final_bess_discharge_to_grid"]) and \
                    not np.isnan(df_in.loc[t, "prev_sp_imbalance_volume_final"]):

                # If we don't have predictions yet, then in the first 10mins of the SP we can use the previous SP's
                # imbalance data to inform the activity

                # MWh to kWh
                if abs(df_in.loc[t, "prev_sp_imbalance_volume_final"]) * 1e3 >= niv_config.volume_cutoff_for_prediction:
                    target_energy_delta = get_target_energy_delta_from_shifted_curves(
                        df_in=df_in,
                        t=t,
                        charge_rate_col="prev_sp_rate_final_bess_charge_from_grid",
                        discharge_rate_col="prev_sp_rate_final_bess_discharge_to_grid",
                        imbalance_volume_col="prev_sp_imbalance_volume_final",
                        soe=soe,
                        battery_charge_efficiency=battery_charge_efficiency,
                        niv_config=niv_config
                    )
            else:
                # TODO: this isn't very helpful, it would be more interesting to report how many settlement periods
                #       are skipped
                num_skipped_periods += 1

            if red_approach_energy > 0:
                target_energy_delta = max(red_approach_energy, amber_approach_energy, target_energy_delta)
            elif amber_approach_energy > 0:
                target_energy_delta = max(amber_approach_energy, target_energy_delta)

            power = get_power(target_energy_delta, df_out.loc[t, "time_left_of_sp"])

        power = cap_power(power, df_in.loc[t, "bess_max_power_charge"], df_in.loc[t, "bess_max_power_discharge"])
        energy_delta = get_energy(power, time_step)

        # Cap the SoE at the physical limits of the battery
        if soe + energy_delta > battery_energy_capacity:
            energy_delta = battery_energy_capacity - soe
        elif soe + energy_delta < 0:
            energy_delta = -soe

        # Apply a charge efficiency
        if energy_delta > 0:
            bess_losses = energy_delta * (1 - battery_charge_efficiency)
        else:
            bess_losses = 0

        df_out.loc[t, "power"] = power
        df_out.loc[t, "energy_delta"] = energy_delta
        df_out.loc[t, "bess_losses"] = bess_losses

        # Save for next iteration...
        last_soe = soe
        last_energy_delta = energy_delta
        last_bess_losses = bess_losses

    if num_skipped_periods > 0:
        time_step_minutes = time_step.total_seconds() / 60
        logging.info(f"Skipped {num_skipped_periods}/{len(df_in)} {time_step_minutes} minute periods (probably due to "
                     f"missing imbalance data)")

    return df_out[["soe", "energy_delta", "bess_losses", "red_approach_distance", "amber_approach_distance"]]


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


def get_capped_power(target_energy_delta: float, df_in, df_out, t) -> float:
    """
    Returns the power level for the given target energy delta, accounting for the charge and discharge power constraints
    that apply to the given timestep
    """
    power = get_power(target_energy_delta, df_out.loc[t, "time_left_of_sp"])
    power = cap_power(power, df_in.loc[t, "bess_max_power_charge"], df_in.loc[t, "bess_max_power_discharge"])
    return power


def get_energy(power: float, duration: timedelta) -> float:
    """
    Returns the energy, in kWh, given a power in kW and duration.
    """
    return power * get_hours(duration)


def get_power(energy: float, duration: timedelta) -> float:
    """
    Returns the power, in kW, given an energy in kWh and duration.
    """
    return energy / get_hours(duration)


def cap_power(power: float, max_charge: float, max_discharge) -> float:
    """
    Caps the power at the batteries capabilities
    """
    if power > max_charge:
        return max_charge
    elif power < -max_discharge:
        return -max_discharge
    return power


def get_hours(duration: timedelta) -> float:
    """
    Returns the duration in number of hours, with decimal places if required.
    """
    return duration.total_seconds() / 3600
