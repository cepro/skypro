import logging
from dataclasses import dataclass
from datetime import timedelta, datetime, time, date
from typing import Optional, List

import numpy as np
import pandas as pd
import plotly.express as px

from simt_common.timeutils import ClockTimePeriod
from simt_common.timeutils.days import Days
from simt_common.timeutils.hh_math import floor_hh

from skypro.commands.simulator.algorithms.approach import get_red_approach_energy, get_amber_approach_energy
from skypro.commands.simulator.cartesian import Point, Curve



def run_spread_based_algo_2(
        df_in: pd.DataFrame,
        battery_energy_capacity: float,
        battery_charge_efficiency: float,
        full_discharge_period: Optional[ClockTimePeriod],
) -> pd.DataFrame:

    # Create a separate dataframe for outputs
    df_out = pd.DataFrame(index=df_in.index)

    # These vars keep track of the previous settlement periods values
    last_soe = battery_energy_capacity / 2  # initial SoE is 50%
    last_energy_delta = 0
    num_skipped_periods = 0

    time_step = pd.to_timedelta(df_in.index.freq)

    # TODO: the surrounding 'harness' code should be brought out to be shared with all algos

    df_out["prev_sp_imbalance_price_long"] = df_in["prev_sp_imbalance_price_final"][df_in["prev_sp_imbalance_volume_final"] < 0]
    df_out["prev_sp_imbalance_price_short"] = df_in["prev_sp_imbalance_price_final"][df_in["prev_sp_imbalance_volume_final"] > 0]

    RECENT_SPAN = 4

    df_out["recent_imbalance_price_long"] = df_out["prev_sp_imbalance_price_long"].rolling(window=RECENT_SPAN, min_periods=1).mean().ffill()
    df_out["recent_imbalance_price_short"] = df_out["prev_sp_imbalance_price_short"].rolling(window=RECENT_SPAN, min_periods=1).mean().ffill()

    df_out["rate_bess_charge_from_grid_non_imbalance"] = df_in["rate_bess_charge_from_grid_non_imbalance"]
    df_out["rate_bess_discharge_to_grid_non_imbalance"] = df_in["rate_bess_discharge_to_grid_non_imbalance"]

    notional_spread_short = (
            df_in[f"rate_predicted_bess_charge_from_grid"] -
            ((df_out[f"recent_imbalance_price_long"] + df_in["rate_bess_charge_from_grid_non_imbalance"]) / battery_charge_efficiency)
    )[df_in[f"imbalance_volume_predicted"] > 0]

    notional_spread_long = -(
            (-df_out[f"recent_imbalance_price_short"] + df_in["rate_bess_discharge_to_grid_non_imbalance"]) -
            (df_in[f"rate_predicted_bess_discharge_to_grid"])
    )[df_in[f"imbalance_volume_predicted"] < 0]

    df_out["notional_spread"] = pd.concat([notional_spread_long, notional_spread_short])
    df_out["notional_spread_short"] = notional_spread_short
    df_out["notional_spread_long"] = notional_spread_long

    STEPS_PER_SP = 3  # TODO: link up
    df_out["prev_sp_notional_spread"] = df_out["notional_spread"].shift(STEPS_PER_SP).bfill(limit=STEPS_PER_SP-1)

    # Run through each row (where each row represents a time step) and apply the strategy
    for t in df_in.index:

        # Show the user some progress status
        if (t == df_in.index[0]) or (t.date().day == 1 and t.time().hour == 0 and t.time().minute == 0):
            print(f"Simulating {t.date()}...")

        # Set the `soe` column to the value at the start of this time step (the previous value plus the energy
        # transferred in the previous time step)
        if last_energy_delta > 0:
            soe = last_soe + (last_energy_delta * battery_charge_efficiency)  # Apply a charge efficiency
        else:
            soe = last_soe + last_energy_delta
        df_out.loc[t, "soe"] = soe

        # if t > datetime.fromisoformat("2024-05-01T15:00:00+00:00"):
        #     breakpoint()

        if full_discharge_period and full_discharge_period.contains(t):
            # The configuration may specify that we ignore the charge/discharge curves and do a full discharge
            # for a certain period - probably a DUoS red band
            power = -df_in.loc[t, "bess_max_power_discharge"]

        else:
            target_energy_delta = 0

            # ALGO ALGO ALGO ALGO ALGO ALGO ALGO ALGO ALGO

            imbalance_volume_assumed = df_in.loc[t, "imbalance_volume_predicted"]
            # TODO: optionally only allow this for the first 10mins? df_in.loc[t, "time_into_sp"] < timedelta(minutes=10) and
            if np.isnan(imbalance_volume_assumed) and \
                    abs(df_in.loc[t, "prev_sp_imbalance_volume_final"]) > 150:
                imbalance_volume_assumed = df_in.loc[t, "prev_sp_imbalance_volume_final"]

            df_out.loc[t, "imbalance_volume_assumed"] = imbalance_volume_assumed

            # if np.isnan(imbalance_volume_assumed):
            #     imbalance_volume_assumed = np.nan

            red_approach_energy = get_red_approach_energy(t, soe)
            amber_approach_energy = get_amber_approach_energy(t, soe, imbalance_volume_assumed)
            spread_algo_energy = get_spread_algo_energy(t, df_out, imbalance_volume_assumed)

            df_out.loc[t, "red_approach_distance"] = red_approach_energy
            df_out.loc[t, "amber_approach_distance"] = amber_approach_energy
            df_out.loc[t, "spread_algo_energy"] = spread_algo_energy

            if red_approach_energy > 0:
                target_energy_delta = max(red_approach_energy, amber_approach_energy, spread_algo_energy)
            elif amber_approach_energy > 0:
                target_energy_delta = max(amber_approach_energy, spread_algo_energy)
            else:
                target_energy_delta = spread_algo_energy


            # END END END END END END END END END

            power = get_power(target_energy_delta, time_step)

        power = cap_power(power, df_in.loc[t, "bess_max_power_charge"], df_in.loc[t, "bess_max_power_discharge"])
        energy_delta = get_energy(power, time_step)

        # Cap the SoE at the physical limits of the battery
        if soe + energy_delta > battery_energy_capacity:
            energy_delta = battery_energy_capacity - soe
        elif soe + energy_delta < 0:
            energy_delta = -soe

        df_out.loc[t, "power"] = power
        df_out.loc[t, "energy_delta"] = energy_delta

        # Save for next iteration...
        last_soe = soe
        last_energy_delta = energy_delta

    if num_skipped_periods > 0:
        time_step_minutes = time_step.total_seconds() / 60
        logging.info(f"Skipped {num_skipped_periods}/{len(df_in)} {time_step_minutes} minute periods (probably due to "
                     f"missing imbalance data)")

    fig = px.line(
        df_out[[
            "prev_sp_imbalance_price_long",
            "prev_sp_imbalance_price_short",
            "recent_imbalance_price_long",
            "recent_imbalance_price_short",
            "rate_bess_charge_from_grid_non_imbalance",
            "rate_bess_discharge_to_grid_non_imbalance",
            "notional_spread_short",
            "notional_spread_long",
            "notional_spread",
        ]],
        # markers=True,
        line_shape='hv',
    )
    fig.update_traces(connectgaps=True)
    fig.show()


    return df_out[["soe", "energy_delta", "notional_spread", "red_approach_distance", "amber_approach_distance", "spread_algo_energy"]]


def get_spread_algo_energy(t, df_out, imbalance_volume_assumed) -> float:

    if np.isnan(imbalance_volume_assumed):
        return 0

    is_currently_short = imbalance_volume_assumed > 0
    notional_spread_assumed = df_out.loc[t, "notional_spread"]
    if np.isnan(notional_spread_assumed):
        notional_spread_assumed = df_out.loc[t, "prev_sp_notional_spread"]
    if np.isnan(notional_spread_assumed):
        notional_spread_assumed = np.nan

    if notional_spread_assumed > 5:
        if is_currently_short:
            return -50
        else:
            return 100
    return 0

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
