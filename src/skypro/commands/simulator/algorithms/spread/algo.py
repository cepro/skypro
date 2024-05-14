import logging
from dataclasses import dataclass
from datetime import timedelta, time, date, datetime
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.express as px
from simt_common.timeutils.days import Days

from skypro.cli_utils.cli_utils import get_user_ack_of_warning_or_exit
from skypro.commands.simulator.cartesian import Curve, Point
from skypro.commands.simulator.config.config import get_relevant_niv_config
import skypro.commands.simulator.config as config


@dataclass
class TimePoint:
    """
    Represents are cartesian point.
    """
    t: time
    y: float



REF_DATE = date(1000, 1, 1)
REF_DATETIME = datetime.combine(REF_DATE, time())


def time_curve_to_curve(time_curve: List[TimePoint]) -> Curve:

    points = []
    for time_point in time_curve:
        dt = datetime.combine(REF_DATE, time_point.t)
        duration = dt - REF_DATETIME
        points.append(Point(
            x=duration.total_seconds(),
            y=time_point.y
        ))

    return Curve(points=points)


red_curve = time_curve_to_curve([
    TimePoint(time(11, 0), 0),
    TimePoint(time(17, 0), 1000)
])
approach_days = Days(name="weekdays", tz_str="UTC")

amber_curve = time_curve_to_curve([
    TimePoint(time(0, 0), 0),
    TimePoint(time(15, 0), 1000)
])


def run_spread_based_algo(
        by_sp: pd.DataFrame,
        import_rates_10m: List,
        import_rates_20m: List,
        import_rates_final: List,
        export_rates_10m: List,
        export_rates_20m: List,
        export_rates_final: List,
        battery_energy_capacity: float,
        battery_charge_efficiency: float,
        battery_nameplate_power: float,
        site_import_limit: float,
        site_export_limit: float,
        niv_chase_periods: List[config.NivPeriod],
        full_discharge_when_export_rate_applies: Optional[str],
) -> pd.DataFrame:
    df = by_sp.copy()

    # These vars keep track of the previous settlement periods values
    last_soe = battery_energy_capacity / 2  # initial SoE is 50%
    last_energy_delta = 0
    last_import_rate_final = None
    last_export_rate_final = None
    last_imbalance_volume_final = None
    num_skipped_periods = 0

    for sp in df.index:
        # Add the individual rates to the dataframe (mostly for user info) and also calculate the summed total
        # import and export rates. This is done for 10mins and 20mins into the SP as well as for the final pricing.
        total_import_rate_10m, total_export_rate_10m = add_rates_to_df(
            df, "10m", import_rates_10m, export_rates_10m, sp
        )
        total_import_rate_20m, total_export_rate_20m = add_rates_to_df(
            df, "20m", import_rates_20m, export_rates_20m, sp
        )
        total_import_rate_final, total_export_rate_final = add_rates_to_df(
            df, "final", import_rates_final, export_rates_final, sp
        )

    df["imbalance_price_20m_long"] = df["imbalance_price_20m"][df["imbalance_volume_20m"] < 0]
    df["imbalance_price_20m_short"] = df["imbalance_price_20m"][df["imbalance_volume_20m"] > 0]
    df["imbalance_price_final_long"] = df["imbalance_price_final"][df["imbalance_volume_final"] < 0]
    df["imbalance_price_final_short"] = df["imbalance_price_final"][df["imbalance_volume_final"] > 0]

    RECENT_SPAN = 4


    # blah = pd.DataFrame(index=df.index)
    # blah["imbalance_price_20m_long"] = df["imbalance_price_20m_long"]
    # blah["dropped_raw"] = df["imbalance_price_20m_long"].rolling(window=RECENT_SPAN, min_periods=1).mean()
    # blah["shifted"] = df["imbalance_price_20m_long"].rolling(window=RECENT_SPAN, min_periods=1).mean().shift(1)



    # blah = pd.DataFrame(index=df.index)
    # blah["imbalance_price_20m_long"] = df["imbalance_price_20m_long"]
    # blah["dropped_raw"] = df["imbalance_price_20m_long"].rolling(window=RECENT_SPAN, min_periods=1).mean()
    # blah["shifted"] = df["imbalance_price_20m_long"].rolling(window=RECENT_SPAN, min_periods=1).mean().shift(1)

    # Calculate the short and long imbalance prices that we have seen historically. We need to shift by 30mins as the
    # data is historical.
    # TODO: if we have a period without short/long samples this ends up just using the latest with a window size of 1, but
    # perhaps we want to keep the average of the window size
    df["recent_imbalance_price_long"] = df["imbalance_price_final_long"].rolling(window=RECENT_SPAN, min_periods=1).mean().ffill().shift(1)
    df["recent_imbalance_price_short"] = df["imbalance_price_final_short"].rolling(window=RECENT_SPAN, min_periods=1).mean().ffill().shift(1)

    df["import_non_imbalance_rates"] = df["rate_import_final"] - df["rate_import_final_imbalance"]
    df["export_non_imbalance_rates"] = df["rate_export_final"] + df["rate_export_final_imbalance"]

    # if is_currently_short:
    #     notional_spread = current_export_rate - recent_long_buy_rate
    # else:
    #     notional_spread = recent_short_sell_rate - current_import_rate
    #
    # df["notional_spread_short"] = (
    #         df["rate_export_final"] - (df["recent_imbalance_price_long"] + df["import_non_imbalance_rates"])
    # )[df["imbalance_volume_final"] > 0]
    #
    # df["notional_spread_long"] = (
    #         df["rate_import_final"] - (df["recent_imbalance_price_short"] + df["export_non_imbalance_rates"])
    # )[df["imbalance_volume_final"] < 0] * -1

    def notional_spread(df, time_str):
        notional_spread_short = (
                df[f"rate_export_{time_str}"] -
                ((df[f"recent_imbalance_price_long"] + df["import_non_imbalance_rates"]) / battery_charge_efficiency)
        )[df[f"imbalance_volume_{time_str}"] > 0]

        notional_spread_long = (
                (df[f"recent_imbalance_price_short"] + df["export_non_imbalance_rates"]) -
                (df[f"rate_import_{time_str}"] / battery_charge_efficiency)
        )[df[f"imbalance_volume_{time_str}"] < 0]

        return pd.concat([notional_spread_long, notional_spread_short])

    df["notional_spread_10m"] = notional_spread(df, "10m")

    df["notional_spread_20m"] = notional_spread(df, "20m")

    df["notional_spread_final"] = notional_spread(df,"final")

    # recent_short_imbalance: float,
    # recent_long_imbalance: float,
    # current_non_imbalance_import_rates: float,
    # current_non_imbalance_export_rates: float,
    # current_imbalance_volume: float,
    # current_imbalance_rate: float,
    # ) -> float:
    #
    px.line(
        df[[
            "imbalance_price_20m_long",
            "imbalance_price_20m_short",
            "imbalance_price_final_long",
            "imbalance_price_final_short",
            "recent_imbalance_price_long",
            "recent_imbalance_price_short",
            "import_non_imbalance_rates",
            "export_non_imbalance_rates",
            "notional_spread_20m",
            "notional_spread_final",
        ]],
        markers=True
    ).show()


    # if is_currently_short:
    #     notional_spread = current_export_rate - recent_long_buy_rate
    # else:
    #     notional_spread = recent_short_sell_rate - current_import_rate
    #
    # df["notional_spread_short"] = (
    #         df["rate_export_final"] - (df["recent_imbalance_price_long"] + df["import_non_imbalance_rates"])
    # )[df["imbalance_volume_final"] > 0]
    #
    # df["notional_spread_long"] = (
    #         df["rate_import_final"] - (df["recent_imbalance_price_short"] + df["export_non_imbalance_rates"])
    # )[df["imbalance_volume_final"] < 0] * -1

    # Run through each row (where each row represents a settlement period) and apply the strategy
    for sp in df.index:

        # Calculate the BESS charge and discharge limits for this SP based on how much solar generation and housing load
        # there is. We need to abide by the overall site import/export limits. And stay within the nameplate inverter
        # capabilities of the BESS
        non_bess_power = df.loc[sp, "load"] - df.loc[sp, "solar"]
        battery_max_power_charge = site_import_limit - non_bess_power
        battery_max_power_discharge = site_export_limit + non_bess_power
        if battery_max_power_charge > battery_nameplate_power:
            battery_max_power_charge = battery_nameplate_power
        if battery_max_power_discharge > battery_nameplate_power:
            battery_max_power_discharge = battery_nameplate_power
        df.loc[sp, "battery_max_power_charge"] = battery_max_power_charge
        df.loc[sp, "battery_max_power_discharge"] = battery_max_power_discharge

        # Show the user some progress status
        if (sp == df.index[0]) or (sp.date().day == 1 and sp.time().hour == 0 and sp.time().minute == 0):
            print(f"Simulating {sp.date()}...")

        # Set the `soe` column to the value at the start of this SP (the previous SP plus the energy transferred in the
        # previous SP)
        if last_energy_delta > 0:
            soe = last_soe + (last_energy_delta * battery_charge_efficiency)  # Apply a charge efficiency
        else:
            soe = last_soe + last_energy_delta
        df.loc[sp, "soe"] = soe

        energy_delta = 0  # the energy transferred in this SP

        # Select the appropriate NIV chasing configuration for this time of day
        # niv_config = get_relevant_niv_config(niv_chase_periods, sp).niv

        imbalance_volume_10m = df.loc[sp, "imbalance_volume_10m"]
        imbalance_volume_20m = df.loc[sp, "imbalance_volume_20m"]
        imbalance_volume_final = df.loc[sp, "imbalance_volume_final"]

        # TODO: improve how unavailable imbalance pricing is actually handled, e.g. we don't need to skip DUoS red band
        #   exports if modo is unavailable at the time.
        if np.isnan(total_import_rate_10m) and np.isnan(total_import_rate_20m):
            num_skipped_periods += 1
            logging.warning(f"Skipping period {sp} due to missing data")
            df.loc[sp, "energy_delta"] = np.NaN
        else:

            # get_target_energy_delta(
            #     recent_short_sell_rate=df.loc[sp, "recent_imbalance_price_short"],
            #     recent_long_buy_rate=df.loc[sp, "recent_imbalance_price_long"],
            #     current_imbalance_volume=,
            #     current_import_rate=,
            #     current_export_rate=,
            # )

            # Each settlement period is split into three chunks of time:
            # 1: 0-10min
            # 2: 10min-20min
            # 3: 20min-30min
            # The actual Go controller is doing better than this as it samples the Modo API every minute or so and
            # adjusts its behaviour in real-time, but limiting this simulation to three samples per SP keeps things
            # tractable.
            # During the first 10mins we can use the previous SP's imbalance data to inform the first 10mins of this SP,
            # During the second 10mins we should have had the first reliable imbalance prices from Modo API, although
            # sometimes it is late and we won't have a price yet. In this case the sim assumes 0 power for this chunk.
            # During the last 10mins we use the latest Modo API data that would have been available at this time.

            power_0m = 0
            # if last_import_rate_final and last_export_rate_final and last_imbalance_volume_final:
            #
            #     if abs(last_imbalance_volume_final) * 1e3 >= niv_config.volume_cutoff_for_prediction:  # MWh to kWh
            #         target_energy_delta_0m = 0
            #         # target_energy_delta_0m = get_target_energy_delta(
            #         #     notional_spread=df.loc[sp, "notional_spread_10m"],
            #         #     current_imbalance_volume=df.loc[sp, "imbalance_volume_10m"]
            #         # )
            #
            #         #     get_target_energy_delta_from_curves(
            #         #     charge_curve=niv_config.charge_curve,
            #         #     discharge_curve=niv_config.discharge_curve,
            #         #     import_rate=last_import_rate_final,
            #         #     export_rate=last_export_rate_final,
            #         #     soe=soe,
            #         #     battery_charge_efficiency=battery_charge_efficiency
            #         # ))
            #         power_0m = get_power(target_energy_delta_0m, timedelta(minutes=30))
            #         power_0m = cap_power(power_0m, battery_max_power_charge, battery_max_power_discharge)

            target_time = datetime.combine(REF_DATE, time(sp.hour, sp.minute, sp.second)) + timedelta(minutes=30)
            now_point = Point(
                x=(target_time - REF_DATETIME).total_seconds(),
                y=soe
            )
            red_approach_distance = red_curve.vertical_distance(now_point)

            amber_approach_distance = amber_curve.vertical_distance(now_point)

            if approach_days.is_on_day(sp) and not np.isnan(red_approach_distance) and red_approach_distance > 0:
                power = get_power(red_approach_distance, timedelta(minutes=30))
                power = cap_power(power, battery_max_power_charge, battery_max_power_discharge)
                power_0m = power
                power_10m = power
                power_20m = power
            elif approach_days.is_on_day(sp) and not np.isnan(amber_approach_distance) and amber_approach_distance > 0:
                if imbalance_volume_final < 0:
                    power = get_power(amber_approach_distance, timedelta(minutes=20))
                    power = cap_power(power, battery_max_power_charge, battery_max_power_discharge)
                    power_10m = power
                    power_20m = power
                else:
                    power_10m = 0
                    power_20m = 0
            else:

                if np.isnan(total_import_rate_10m):
                    # Sometimes the Modo API isn't available at the 10minute mark, so we assume that we do nothing.
                    power_10m = 0
                else:
                    target_energy_delta_10m = get_target_energy_delta(
                        notional_spread=df.loc[sp, "notional_spread_10m"],
                        current_imbalance_volume=df.loc[sp, "imbalance_volume_10m"]
                    )
                    power_10m = get_power(target_energy_delta_10m, timedelta(minutes=20))
                    power_10m = cap_power(power_10m, battery_max_power_charge, battery_max_power_discharge)

                # We then get new imbalance prices 20minutes into the SP, so we can update our behaviour now
                if np.isnan(total_import_rate_20m):
                    # Sometimes the Modo API isn't available at the 20minute mark, so we assume that we do nothing.
                    power_20m = 0
                else:
                    target_energy_delta_20m = get_target_energy_delta(
                        notional_spread=df.loc[sp, "notional_spread_20m"],
                        current_imbalance_volume=df.loc[sp, "imbalance_volume_20m"]
                    )
                    power_20m = get_power(target_energy_delta_20m, timedelta(minutes=10))
                    power_20m = cap_power(power_20m, battery_max_power_charge, battery_max_power_discharge)


            # The configuration may specify that we ignore the charge/discharge curves and do a full discharge
            # for the entire 30 minutes - this is determined by whether a named export rate applies in this SP.
            do_full_discharge = False
            if full_discharge_when_export_rate_applies is not None:
                column_name = f"rate_export_final_{full_discharge_when_export_rate_applies}"
                do_full_discharge = df.loc[sp, column_name] != 0

            if do_full_discharge:
                energy_delta = get_energy(-battery_max_power_discharge, timedelta(minutes=30))
            else:
                energy_delta = (
                        get_energy(power_0m, timedelta(minutes=10)) +
                        get_energy(power_10m, timedelta(minutes=10)) +
                        get_energy(power_20m, timedelta(minutes=10))
                )

        # Cap the SoE at the physical limits of the battery
        if soe + energy_delta > battery_energy_capacity:
            energy_delta = battery_energy_capacity - soe
        elif soe + energy_delta < 0:
            energy_delta = -soe

        df.loc[sp, "energy_delta"] = energy_delta

        # Save for next iteration...
        last_soe = soe
        last_energy_delta = energy_delta
        last_export_rate_final = total_export_rate_final
        last_import_rate_final = total_import_rate_final
        last_imbalance_volume_final = imbalance_volume_final

    if num_skipped_periods > 0:
        get_user_ack_of_warning_or_exit(
            f"Skipped {num_skipped_periods}/{len(by_sp)} periods (probably due to missing imbalance data)"
        )

    return df


def add_rates_to_df(df, tag, import_rates, export_rates, sp) -> Tuple[float, float]:
    """
    Adds each p/kW rate into a column in the dataframe row at `sp`.
    It also calculates the total import and export rates (in p/kW), adds these to the dataframe and returns them.
    """
    total_import_per_kwh = 0.0
    total_export_per_kwh = 0.0
    for rate in import_rates:
        per_kwh = rate.get_per_kwh_rate(sp)
        total_import_per_kwh += per_kwh
        df.loc[sp, f"rate_import_{tag}_{rate.name}"] = per_kwh
    df.loc[sp, f"rate_import_{tag}"] = total_import_per_kwh
    for rate in export_rates:
        per_kwh = rate.get_per_kwh_rate(sp)
        total_export_per_kwh -= per_kwh
        df.loc[sp, f"rate_export_{tag}_{rate.name}"] = per_kwh
    df.loc[sp, f"rate_export_{tag}"] = total_export_per_kwh

    return total_import_per_kwh, total_export_per_kwh


def get_target_energy_delta(
        notional_spread: float,
        current_imbalance_volume: float,
) -> float:
    """
    """
    is_currently_short = current_imbalance_volume > 0

    if notional_spread > -1.0:
        if is_currently_short:
            return -9999
        else:
            return 9999
    else:
        return 0

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
