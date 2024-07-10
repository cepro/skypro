import logging
from datetime import timedelta, date

import numpy as np
import pandas as pd

from skypro.commands.simulator.algorithms.peak import get_peak_approach_energies, get_peak_power
from skypro.commands.simulator.algorithms.microgrid import get_microgrid_algo_energy
from skypro.commands.simulator.algorithms.system_state import SystemState, get_system_state
from skypro.commands.simulator.algorithms.utils import get_power, cap_power, get_energy
from skypro.commands.simulator.config.config import SpreadAlgo


def run_spread_based_algo(
        df_in: pd.DataFrame,
        battery_energy_capacity: float,
        battery_charge_efficiency: float,
        config: SpreadAlgo,
) -> pd.DataFrame:

    # TODO: the surrounding 'harness' code should be brought out to be shared with all algos

    # Create a separate dataframe for working values
    df = pd.DataFrame(index=df_in.index)

    # These vars keep track of the previous settlement periods values
    last_soe = battery_energy_capacity / 2  # initial SoE is 50%
    last_energy_delta = 0
    last_bess_losses = 0
    num_skipped_periods = 0

    time_step = pd.to_timedelta(df_in.index.freq)
    time_step_hours = time_step.total_seconds() / 3600
    steps_per_sp = int(timedelta(minutes=30) / time_step)

    # The energy that the spread algo tries to charge/discharge when the spread is good
    charge_energy = config.fixed_action.charge_power * time_step_hours
    discharge_energy = config.fixed_action.discharge_power * time_step_hours

    # Calculate the "notional spread" for each time period
    df["prev_sp_imbalance_price_long"] = df_in["prev_sp_imbalance_price_final"][
        df_in["prev_sp_imbalance_volume_final"] < 0
    ]
    df["prev_sp_imbalance_price_short"] = df_in["prev_sp_imbalance_price_final"][
        df_in["prev_sp_imbalance_volume_final"] > 0
    ]

    df["recent_imbalance_price_long"] = df["prev_sp_imbalance_price_long"].rolling(
        window=config.recent_pricing_span,
        min_periods=1
    ).mean().ffill()
    df["recent_imbalance_price_short"] = df["prev_sp_imbalance_price_short"].rolling(
        window=config.recent_pricing_span,
        min_periods=1
    ).mean().ffill()

    df["rate_bess_charge_from_grid_non_imbalance"] = df_in["rate_bess_charge_from_grid_non_imbalance"]
    df["rate_bess_discharge_to_grid_non_imbalance"] = df_in["rate_bess_discharge_to_grid_non_imbalance"]

    notional_spread_short = (
            df_in[f"rate_predicted_bess_charge_from_grid"] -
            (
                    (df[f"recent_imbalance_price_long"] + df_in["rate_bess_charge_from_grid_non_imbalance"])
                    / battery_charge_efficiency
            )
    )[df_in[f"imbalance_volume_predicted"] > 0]

    notional_spread_long = -(
            (-df[f"recent_imbalance_price_short"] + df_in["rate_bess_discharge_to_grid_non_imbalance"]) -
            (df_in[f"rate_predicted_bess_discharge_to_grid"])
    )[df_in[f"imbalance_volume_predicted"] < 0]

    df["notional_spread"] = pd.concat([notional_spread_long, notional_spread_short])
    df["prev_sp_notional_spread"] = df["notional_spread"].shift(steps_per_sp).bfill(limit=steps_per_sp-1)

    # Run through each row (where each row represents a time step) and apply the strategy
    for t in df_in.index:

        # Show the user some progress status
        if (t == df_in.index[0]) or (t.date().day == 1 and t.time().hour == 0 and t.time().minute == 0):
            print(f"Simulating {t.date()}...")

        # Set the `soe` column to the value at the start of this time step (the previous value plus the energy
        # transferred in the previous time step)
        soe = last_soe + last_energy_delta - last_bess_losses
        df.loc[t, "soe"] = soe

        system_state = get_system_state(df_in, t, config.niv_cutoff_for_system_state_assumption)

        peak_power = get_peak_power(
            peak_config=config.peak,
            t=t,
            time_step=time_step,
            soe=soe,
            bess_max_power_discharge=df_in.loc[t, "bess_max_power_discharge"],
            microgrid_residual_power=df_in.loc[t, "microgrid_residual_power"],
            system_state=system_state
        )
        if peak_power is not None:
            power = peak_power
        # if config.peak.period and config.peak.period.contains(t):
        #     # The configuration may specify that we ignore the charge/discharge curves and do a full discharge
        #     # for a certain period - probably a DUoS red band
        #
        #     CONFIG_PEAK_DYNAMIC = True
        #     CONFIG_PEAK_PRIORITISE_LOAD = True
        #
        #     bess_max_power_discharge = df_in.loc[t, "bess_max_power_discharge"]
        #
        #     if not CONFIG_PEAK_DYNAMIC:
        #         # We just do a 'dumb' full discharge over the peak until the battery is empty
        #         power = -bess_max_power_discharge
        #     else:
        #         # If the notional battery duration (accounting for grid constraints) is shorter than the peak duration
        #         # then we can get an improvement on the above 'dumb' method by choosing when we discharge into the peak.
        #
        #         peak_end = config.peak.period.period.end_absolute(t)
        #
        #         # This is an approximation because we may be limited by grid constraints and the load and solar levels
        #         # may change throughout the peak.
        #         assumed_time_to_empty_battery = timedelta(hours=(soe / bess_max_power_discharge))
        #
        #         # We want to ensure that we empty the battery completely by the end of the peak period, and there is a
        #         # point into the peak where we must discharge at the max power to ensure that.
        #         latest_time_before_max_discharge = peak_end - assumed_time_to_empty_battery
        #         if t > (latest_time_before_max_discharge - timedelta(seconds=time_step.total_seconds())):
        #             power = -bess_max_power_discharge
        #         else:
        #             # We are early enough in the peak period to have some flexibility about how much we discharge
        #             if system_state == SystemState.LONG:
        #                 if CONFIG_PEAK_PRIORITISE_LOAD:
        #                     # Even though the system is long, discharge to avoid microgrid imports (if any)
        #                     microgrid_residual = df_in.loc[t, "microgrid_residual_power"]
        #                     if microgrid_residual > 0:
        #                         power = -microgrid_residual
        #                     else:
        #                         power = 0
        #                 else:
        #                     # If we are not 'prioritising loads' then hold off on the discharge until the last minute,
        #                     # or until the system is short.
        #                     power = 0
        #             else:
        #                 if CONFIG_PEAK_PRIORITISE_LOAD:
        #                     # Here we want to discharge at the max power we can, whilst ensuring there is enough energy
        #                     # in the battery to service any residual microgrid load at the end of the peak.
        #                     microgrid_residual = df_in.loc[t, "microgrid_residual_power"]
        #
        #                     # Approximate that the residual load will stay the same throughout the peak
        #                     # TODO: we could make some assumption about the residual growing due to less solar later on?
        #                     if microgrid_residual > 0:
        #
        #                         # If we were to discharge at max power, then when would we run out of energy?
        #                         empty_time_without_reserve = t + assumed_time_to_empty_battery
        #
        #                         # This tells us how long and how much energy we need to reserve for servicing load
        #                         reserve_duration = peak_end - empty_time_without_reserve
        #                         reserve_energy = microgrid_residual * (reserve_duration.total_seconds() / 3600)
        #
        #                         # Which, in turn, allows us to calculate the new max discharge rate which would allow us
        #                         # to keep that reserve for the end of the peak
        #                         duration_before_reserve = (peak_end - reserve_duration) - t
        #                         energy_before_reserve = soe - reserve_energy
        #                         power = -energy_before_reserve / (duration_before_reserve.total_seconds() / 3600)
        #
        #                     else:
        #                         power = -bess_max_power_discharge
        #                 else:
        #                     # If we are not 'prioritising loads' then discharge at the max power
        #                     power = -bess_max_power_discharge

        else:
            red_approach_energy, amber_approach_energy = get_peak_approach_energies(
                t=t,
                time_step=time_step,
                soe=soe,
                charge_efficiency=battery_charge_efficiency,
                peak_config=config.peak,
                is_long=system_state == SystemState.LONG
            )

            spread_algo_energy = get_spread_algo_energy(
                notional_spread=df.loc[t, "notional_spread"],
                prev_sp_notional_spread=df.loc[t, "prev_sp_notional_spread"],
                min_spread=config.min_spread,
                short_energy=discharge_energy,
                long_energy=charge_energy,
                system_state=system_state
            )

            if config.microgrid:
                # There is a different niv cutoff threshold for the microgrid algo
                system_state_mg = SystemState.UNKNOWN
                if config.microgrid.imbalance_control:
                    system_state_mg = get_system_state(df_in, t, config.microgrid.imbalance_control.niv_cutoff_for_system_state_assumption)
                microgrid_algo_energy = get_microgrid_algo_energy(
                    config=config.microgrid,
                    microgrid_residual_energy=df_in.loc[t, "microgrid_residual_power"] * time_step_hours,
                    system_state=system_state_mg,
                )
            else:
                microgrid_algo_energy = 0.0

            df.loc[t, "red_approach_distance"] = red_approach_energy
            df.loc[t, "amber_approach_distance"] = amber_approach_energy
            df.loc[t, "spread_algo_energy"] = spread_algo_energy
            df.loc[t, "microgrid_algo_energy"] = microgrid_algo_energy

            if red_approach_energy > 0:
                target_energy_delta = max(red_approach_energy, amber_approach_energy, spread_algo_energy)
            elif amber_approach_energy > 0:
                target_energy_delta = max(amber_approach_energy, spread_algo_energy)
            else:
                target_energy_delta = spread_algo_energy + microgrid_algo_energy

            power = get_power(target_energy_delta, time_step)

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

        df.loc[t, "power"] = power
        df.loc[t, "energy_delta"] = energy_delta
        df.loc[t, "bess_losses"] = bess_losses

        # Save for next iteration...
        last_soe = soe
        last_energy_delta = energy_delta
        last_bess_losses = bess_losses

    if num_skipped_periods > 0:
        time_step_minutes = time_step.total_seconds() / 60
        logging.info(f"Skipped {num_skipped_periods}/{len(df_in)} {time_step_minutes} minute periods (probably due to "
                     f"missing imbalance data)")

    return df[["soe", "energy_delta", "bess_losses", "notional_spread", "red_approach_distance", "amber_approach_distance", "spread_algo_energy", "microgrid_algo_energy"]]


def get_spread_algo_energy(
        notional_spread: float,
        prev_sp_notional_spread: float,
        min_spread: float,
        short_energy: float,
        long_energy: float,
        system_state: SystemState
) -> float:

    if system_state == SystemState.UNKNOWN:
        return 0

    notional_spread_assumed = notional_spread
    if np.isnan(notional_spread_assumed):
        notional_spread_assumed = prev_sp_notional_spread
    if np.isnan(notional_spread_assumed):
        return 0

    if notional_spread_assumed > min_spread:
        if system_state == SystemState.SHORT:
            return -short_energy
        else:
            return long_energy

    return 0
