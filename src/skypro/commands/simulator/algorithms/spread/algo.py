import logging
from datetime import timedelta

import numpy as np
import pandas as pd

from skypro.commands.simulator.algorithms.approach import get_peak_approach_energies
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

    notional_spread_short_ng = (
            df_in[f"rate_predicted_bess_charge_from_grid"] -
            (
                    (df[f"recent_imbalance_price_long"] + df_in["rate_bess_charge_from_grid_non_imbalance"])
                    / battery_charge_efficiency
            )
    )[df_in[f"imbalance_volume_predicted"] > 0]

    notional_spread_long_ng = -(
            (-df[f"recent_imbalance_price_short"] + df_in["rate_bess_discharge_to_grid_non_imbalance"]) -
            (df_in[f"rate_predicted_bess_discharge_to_grid"])
    )[df_in[f"imbalance_volume_predicted"] < 0]

    df["notional_spread_ng"] = pd.concat([notional_spread_long_ng, notional_spread_short_ng])
    # df["notional_spread_mg"] = df["notional_spread_ng"] + 2*(df_in["rate_bess_charge_from_grid_non_imbalance"] + df_in["rate_bess_discharge_to_grid_non_imbalance"])
    #
    df["microgrid_residual_power"] = df_in["load_power"] - df_in["solar_power"]
    # import plotly.express as px
    # df_fig = pd.DataFrame(index=df.index)
    # df_fig["notional_spread_ng"] = df["notional_spread_ng"]
    # df_fig["notional_spread_mg"] = df["notional_spread_mg"]
    # df_fig["microgrid_residual_power"] = df["microgrid_residual_power"]
    # df_fig["rate_predicted_bess_charge_from_grid"] = df_in["rate_predicted_bess_charge_from_grid"]
    # df_fig["rate_predicted_bess_discharge_to_grid"] = -df_in["rate_predicted_bess_discharge_to_grid"]
    # fig = px.line(df_fig, line_shape="hv")
    # fig.update_traces(connectgaps=True)
    # fig.show()

    df["prev_sp_notional_spread_ng"] = df["notional_spread_ng"].shift(steps_per_sp).bfill(limit=steps_per_sp-1)
    # df["prev_sp_notional_spread_mg"] = df["notional_spread_mg"].shift(steps_per_sp).bfill(limit=steps_per_sp-1)

    # Run through each row (where each row represents a time step) and apply the strategy
    for t in df_in.index:

        # Show the user some progress status
        if (t == df_in.index[0]) or (t.date().day == 1 and t.time().hour == 0 and t.time().minute == 0):
            print(f"Simulating {t.date()}...")

        # Set the `soe` column to the value at the start of this time step (the previous value plus the energy
        # transferred in the previous time step)
        soe = last_soe + last_energy_delta - last_bess_losses
        df.loc[t, "soe"] = soe

        if config.peak.period and config.peak.period.contains(t):
            # The configuration may specify that we ignore the charge/discharge curves and do a full discharge
            # for a certain period - probably a DUoS red band
            power = -df_in.loc[t, "bess_max_power_discharge"]

        else:

            imbalance_volume_assumed = df_in.loc[t, "imbalance_volume_predicted"]
            # TODO: optionally only allow this for the first 10m? df_in.loc[t, "time_into_sp"]<timedelta(minutes=10)
            if np.isnan(imbalance_volume_assumed) and \
                    abs(df_in.loc[t, "prev_sp_imbalance_volume_final"]) * 1e3 >= config.volume_cutoff_for_prediction:
                imbalance_volume_assumed = df_in.loc[t, "prev_sp_imbalance_volume_final"]

            df.loc[t, "imbalance_volume_assumed"] = imbalance_volume_assumed

            red_approach_energy, amber_approach_energy = get_peak_approach_energies(
                t=t,
                time_step=time_step,
                soe=soe,
                charge_efficiency=battery_charge_efficiency,
                peak_config=config.peak,
                is_long=imbalance_volume_assumed < 0
            )

            spread_algo_energy_ng = get_spread_algo_energy(
                notional_spread=df.loc[t, "notional_spread_ng"],
                prev_sp_notional_spread=df.loc[t, "prev_sp_notional_spread_ng"],
                min_spread=config.min_spread,
                short_energy=discharge_energy,
                long_energy=charge_energy,
                imbalance_volume_assumed=imbalance_volume_assumed
            )

            # spread_algo_energy_mg = get_spread_algo_energy(
            #     notional_spread=df.loc[t, "notional_spread_mg"],
            #     prev_sp_notional_spread=df.loc[t, "prev_sp_notional_spread_mg"],
            #     min_spread=config.min_spread,
            #     short_energy=discharge_energy,
            #     long_energy=charge_energy,
            #     imbalance_volume_assumed=imbalance_volume_assumed
            # )
            #
            # microgrid_residual_energy = df.loc[t, "microgrid_residual_power"] * time_step_hours
            # if spread_algo_energy_mg > 0:
            #     # The algo wants us to charge the battery from surplus microgrid power - but we can only do that if there
            #     # is indeed a surplus
            #     if microgrid_residual_energy < 0:
            #         # TODO: this should be max of microgrid_residual_power and spread_algo_energy_mg
            #         spread_algo_energy_mg = -microgrid_residual_energy
            #     else:
            #         spread_algo_energy_mg = 0
            # elif spread_algo_energy_mg < 0:
            #     # The algo wants us to discharge the battery to cover the load - but we can only do that if there
            #     # is indeed a residual load
            #     if microgrid_residual_energy > 0:
            #         # TODO: this should be max of microgrid_residual_power and spread_algo_energy_mg
            #         spread_algo_energy_mg = microgrid_residual_energy
            #     else:
            #         spread_algo_energy_mg = 0

            microgrid_algo_energy = 0
            microgrid_residual_energy = df.loc[t, "microgrid_residual_power"] * time_step_hours
            is_short = imbalance_volume_assumed > 0
            if config.microgrid.discharge_into_load_when_short and is_short and microgrid_residual_energy > 0:
                # The system is short (so prices are high) and the microgrid is importing from the grid, so we should
                # try to discharge the battery to cover the load
                microgrid_algo_energy = -microgrid_residual_energy
            elif config.microgrid.charge_from_solar_when_long and not is_short and microgrid_residual_energy < 0:
                # The system is long (so prices are low) and the microgrid is exporting to the grid, so we should
                # try to charge the battery to stop the export
                microgrid_algo_energy = -microgrid_residual_energy


            # TODO: this should not be purely additive
            spread_algo_energy = spread_algo_energy_ng # + spread_algo_energy_mg

            df.loc[t, "red_approach_distance"] = red_approach_energy
            df.loc[t, "amber_approach_distance"] = amber_approach_energy
            df.loc[t, "microgrid_algo_energy"] = microgrid_algo_energy
            df.loc[t, "spread_algo_energy"] = spread_algo_energy

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

    return df[["soe", "energy_delta", "bess_losses", "notional_spread_ng", "red_approach_distance", "amber_approach_distance", "spread_algo_energy", "microgrid_algo_energy"]]


def get_spread_algo_energy(
        notional_spread: float,
        prev_sp_notional_spread: float,
        min_spread: float,
        short_energy: float,
        long_energy: float,
        imbalance_volume_assumed: float
) -> float:

    if np.isnan(imbalance_volume_assumed):
        return 0

    is_currently_short = imbalance_volume_assumed > 0
    notional_spread_assumed = notional_spread
    if np.isnan(notional_spread_assumed):
        notional_spread_assumed = prev_sp_notional_spread
    if np.isnan(notional_spread_assumed):
        return 0

    if notional_spread_assumed > min_spread:
        if is_currently_short:
            return -short_energy
        else:
            return long_energy

    return 0
