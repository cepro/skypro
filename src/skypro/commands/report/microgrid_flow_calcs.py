import logging
from datetime import timedelta, datetime
from typing import List, Tuple

import numpy as np
import pandas as pd

from skypro.commands.report.readings import AllReadings
from skypro.common.notice.notice import Notice
from skypro.common.timeutils.timeseries import get_step_size, get_steps_per_hh

from skypro.commands.report.config.config import MicrogridMeters
from skypro.commands.report.warnings import pct_to_notice_level, duration_to_notice_level, missing_data_warnings


def calc_flows(
        time_index: pd.DatetimeIndex,
        step_size: timedelta,
        timezone_str: str,
        readings: AllReadings,
        meter_config: MicrogridMeters
) -> Tuple[pd.DataFrame, List[Notice]]:
    """
    Constructs a dataframe containing the microgrid flows using the given meter and BESS readings. Also returns a list of
    Notices if there are any data quality issues.
    """

    df = pd.DataFrame(index=time_index)
    notices: List[Notice] = []

    # The import/export nomenclature can be confusing for a battery, so prefer the "charge" and "discharge" terminology
    df["bess_discharge"] = readings.bess_meter["energy_imported_active_delta"]
    df["bess_charge"] = readings.bess_meter["energy_exported_active_delta"]
    df["bess_import_cum_reading"] = readings.bess_meter["energy_imported_active_min"]
    df["bess_export_cum_reading"] = readings.bess_meter["energy_exported_active_min"]
    df["soe"] = readings.bess["soe_avg"]
    df["soe"] = df["soe"].ffill(limit=get_steps_per_hh(step_size)-1)

    df["grid_import_cum_reading"] = readings.grid_meter["energy_imported_active_min"]
    df["grid_export_cum_reading"] = readings.grid_meter["energy_exported_active_min"]
    df["grid_import"] = readings.grid_meter["energy_imported_active_delta"]
    df["grid_export"] = readings.grid_meter["energy_exported_active_delta"]

    df, new_notices = synthesise_battery_inverter_if_needed(df, readings.bess["target_power_avg"])
    notices.extend(new_notices)

    df["bess_net"] = df["bess_charge"] - df["bess_discharge"]

    # Taking the net of the EV circuit is probably overkill as the EVs shouldn't be exporting
    df["ev_net"] = readings.ev_meter["energy_imported_active_delta"] - readings.ev_meter["energy_exported_active_delta"]

    df["feeder1_net"] = readings.feeder1_meter["energy_imported_active_delta"] - readings.feeder1_meter["energy_exported_active_delta"]
    df["feeder2_net"] = readings.feeder2_meter["energy_imported_active_delta"] - readings.feeder2_meter["energy_exported_active_delta"]

    # A 'top-down' approach is used to do the calculations: starting with the grid meter data and calculates
    # down towards the feeder level.
    df["grid_net"] = df["grid_import"] - df["grid_export"]
    df["non_bess_net"] = df["grid_net"] - df["bess_net"]
    df["feeders_net"] = df["non_bess_net"] - df["ev_net"]

    # In this context "load" does not include BESS charges
    df["load_not_supplied_by_solar"] = df["non_bess_net"][df["non_bess_net"] > 0]
    df["load_not_supplied_by_solar"] = df["load_not_supplied_by_solar"].fillna(0.0)
    df["solar_not_supplying_load"] = df["non_bess_net"][df["non_bess_net"] < 0] * -1
    df["solar_not_supplying_load"] = df["solar_not_supplying_load"].fillna(0.0)
    df["batt_to_load"] = df[["bess_discharge", "load_not_supplied_by_solar"]].min(axis=1)
    df["batt_to_grid"] = df["bess_discharge"] - df["batt_to_load"]
    df["solar_to_batt"] = df[["bess_charge", "solar_not_supplying_load"]].min(axis=1)
    df["grid_to_batt"] = df["bess_charge"] - df["solar_to_batt"]
    df["solar_to_grid"] = df["solar_not_supplying_load"] - df["solar_to_batt"]
    df["grid_to_load"] = df["load_not_supplied_by_solar"] - df["batt_to_load"]

    # If we have missing meter data then we may be able to approximate it as we do have redundant metering in some cases
    df, new_notices = calculate_missing_net_flows_in_junction(
        df,
        cols_with_direction=[
            ("grid_net", 1),
            ("bess_net", -1),
            ("feeder1_net", -1),
            ("feeder2_net", -1),
            ("ev_net", -1),
        ],
    )
    notices.extend(new_notices)
    notices.extend(missing_data_warnings(df, "Flows metering data for BESS"))

    # We need a half-hourly datetime index for things like emlite meter data
    time_index_hh = pd.date_range(
        start=time_index[0],
        end=time_index[-1],
        freq=timedelta(minutes=30)
    ).tz_convert(timezone_str)

    logging.info("Approximating solar and load...")
    df, new_notices = approximate_solar_and_load(
        df=df,
        plot_meter_readings=readings.plot_meter,
        meter_config=meter_config,
        time_index_hh=time_index_hh
    )
    notices.extend(new_notices)

    df["solar"] = df[["solar_feeder1", "solar_feeder2"]].sum(axis=1)
    df["load"] = df[["plot_load_feeder1", "plot_load_feeder2"]].sum(axis=1)
    df["solar_to_load"] = np.minimum(df["solar"], df["load"])  # requires emlite data

    # These columns are required for the output CSV, and could be calculated, but there's not currently a need for it
    df["solar_to_load_property_level"] = np.nan
    df["solar_to_load_microgrid_level"] = np.nan
    df["bess_losses"] = np.nan
    df["bess_max_charge"] = np.nan
    df["bess_max_discharge"] = np.nan
    df["imbalance_volume_final"] = np.nan
    df["imbalance_volume_predicted"] = np.nan

    return df, notices


def calculate_missing_net_flows_in_junction(
        df: pd.DataFrame,
        cols_with_direction: List[Tuple[str, int]]
) -> Tuple[pd.DataFrame, List[Notice]]:
    """
    Given an electrical junction with N flows in/out, if one of the flows is missing metering data then we can calculate
    it from the others because we have 'redundant metering'. This function calculates any missing flows that it can and
    adds them to the returned dataframe. Also returns any Notices about data quality warnings etc.

    This calculation only works for 'net' flows - i.e. it doesn't work with separate import/export flows.

    `cols_with_direction` is a list of the columns in `df` that make up the electrical junction as well as the
    direction that metered imports/exports are configured. The metering direction is given by either +1 or -1 and only
    needs to be 'internally consistent', i.e. the absolute direction of metered flow doesn't matter as long as all the
    '+1s' are in the same direction and all the '-1s' are in the same direction.
    """

    df = df.copy()
    notices: List[Notice] = []
    cols = [col_with_dir[0] for col_with_dir in cols_with_direction]
    nets_df = df[cols]
    nans_df = nets_df.isna()

    # Loop through each column and make estimates for each in turn
    for col_to_predict, col_to_predict_direction in cols_with_direction:

        # We can only predict rows where there is a single missing value
        rows_to_predict = (nans_df.sum(axis=1) == 1) & (nans_df[col_to_predict])

        if rows_to_predict.sum() > 0:
            # Sum up the energies across all but the column we are predicting for
            total = pd.Series(index=nets_df.index, data=0.0)
            for col_2, direction_2 in cols_with_direction:
                if col_2 != col_to_predict:
                    total = total - nets_df[col_2] * direction_2  # Some metering is done with different directions

            pct_to_fill = (len(rows_to_predict) / len(nets_df)) * 100
            pct_missing = (nans_df[col_to_predict].sum() / len(nans_df)) * 100
            df.loc[rows_to_predict, col_to_predict] = total * col_to_predict_direction
            notices.append(Notice(
                detail=f"{pct_missing:.1f}% of '{col_to_predict}' fields are missing, but {pct_to_fill:.1f}% can be calculated"
                    " using redundant microgrid metering data",
                level=pct_to_notice_level(pct_missing)
            ))

    return df, notices


def approximate_solar_and_load(
        df: pd.DataFrame,
        plot_meter_readings: pd.DataFrame,
        meter_config: MicrogridMeters,
        time_index_hh: pd.DatetimeIndex
) -> Tuple[pd.DataFrame, List[Notice]]:
    """
    Estimates the solar and load values and adds them to the returned dataframe. Also returns any Notices about
    data quality warnings etc.

    At the current microgrids, most of the plot-level load data is available as Emlite meters are in place, except for
    three-phase meters like landlord supplies and EV chargers, which leads to an approx 5% under-estimate of load.
    Most of the plot-level solar data is NOT available as the Emlite meters are not yet communicating/reporting generation.
    Therefore, this function approximates the total solar generation by differencing the feeder-level import/exports with
    the plot-level load data.
    """
    df = df.copy()

    step_size = get_step_size(df.index)
    steps_per_hh = get_steps_per_hh(step_size)

    # Set up a more generic form for feeder config that allows an arbitrary number of feeders:
    feeder_flows_ids = [str(meter_config.feeder_1.feeder_flows_id), str(meter_config.feeder_2.feeder_flows_id)]

    # Filter the plot readings to only include the feeders of interest
    plot_meter_readings = plot_meter_readings[plot_meter_readings["feeder_id"].isin(feeder_flows_ids)]

    load_natures = ["power", "heat"]
    solar_natures = ["solar"]

    # Before we filter the plot readings to include only the time range of interest we first take a record of all the
    # load registers, just in case some registers are completely unavailable during the time range of interest, in which
    # case we wouldn't be able to detect that they are missing
    all_load_registers = plot_meter_readings[plot_meter_readings["nature"].isin(load_natures)]["register_id"].unique()

    # Filter the plot readings to only include the time range of interest
    plot_meter_readings = plot_meter_readings[plot_meter_readings.index.isin(time_index_hh)]

    # Pivot the plot readings so there are columns for each feeder, and a row for each HH
    plot_load_by_register = split_by_register(plot_meter_readings, load_natures, time_index_hh)
    plot_solar_by_register = split_by_register(plot_meter_readings, solar_natures, time_index_hh)

    # Here we ensure that all the load registers get a column, even if they are not present at all in the time frame of
    # interest. We do this so that we can warn the user accurately about the amount of missing data.
    for register in all_load_registers:
        if register not in plot_load_by_register.columns:
            plot_load_by_register[register] = np.nan

    # Estimate missing data
    plot_load_by_register, notices = fill_gaps_in_plot_level_data(plot_load_by_register)

    # Loop for each feeder
    for i, feeder_flows_id in enumerate(feeder_flows_ids):

        # Determine the load registers associated with this feeder
        feeder_load_registers = plot_meter_readings[
            (plot_meter_readings["feeder_id"] == feeder_flows_id) & (plot_meter_readings["nature"].isin(load_natures))
        ]["register_id"].unique()

        plot_loads_in_feeder = plot_load_by_register[feeder_load_registers]

        # The emlite data is half-hourly, so upscale it to 5-minutely resolution by assuming that the load is constant
        # over the half-hour.
        n = i + 1
        df[f"plot_load_feeder{n}"] = plot_loads_in_feeder.sum(axis=1) / steps_per_hh
        df[f"plot_load_feeder{n}"] = df[f"plot_load_feeder{n}"].ffill(limit=steps_per_hh - 1)

        # We can approximate the solar generation by taking the difference of the load and feeder energies:
        df[f"solar_feeder{n}"] = df[f"plot_load_feeder{n}"] - df[f"feeder{n}_net"]
        # This isn't perfect because:
        #  - emlite load data is half-hourly and feeder data is 5-minutely, more granular data is better.
        #  - the feeder data is netted (imports and exports aren't handled separately)
        # These error sources leads to a few side effects:
        #  - Sometimes low levels of solar generation is shown in the middle of the night
        #  - Sometimes solar generation appears negative
        # The overall effect seems to appear like noise around a 'true' solar generation value.

        # Further improve the solar approximation by removing values less than 0 and by zeroing all values where solar
        # meter data is available and indicates that no power was generated
        df[f"solar_feeder{n}"] = np.maximum(df[f"solar_feeder{n}"], 0)
        # There aren't many functional solar meters, but we usually have at least one meter that is reporting values.
        # Sum the metered solar power across all working meters, and if the result is zero then set the total
        # solar approximation to zero also.
        plot_solar_totals = plot_solar_by_register.sum(axis=1, skipna=True)
        plot_solar_totals[plot_solar_by_register.isna().all(
            axis=1)] = np.nan  # If no data is available use NaN rather than 0 in the sum
        indices_to_zero = plot_solar_totals[plot_solar_totals == 0].index
        for period in indices_to_zero:
            # Expand the indices to include each 5 minute period in the half-hour
            indices_to_zero = indices_to_zero.union(pd.date_range(start=period, periods=steps_per_hh, freq=step_size))
        df.loc[indices_to_zero, f"solar_feeder{n}"] = 0.0

    return df, notices


def split_by_register(plot_meter_readings: pd.DataFrame, natures: List[str], time_index_hh: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Takes the plot meter readings (which has a row per reading) and returns a dataframe with a column per register
    and a row for each half-hour.
    """
    if len(plot_meter_readings) == 0:
        return pd.DataFrame(index=time_index_hh)

    df = pd.pivot_table(
        plot_meter_readings[plot_meter_readings["nature"].isin(natures)],
        columns=['register_id'],
        values='kwh',
        index="time",
        aggfunc=dummy_agg
    )
    # Re-indexing ensures any missing half-hours are filled with NaN
    df = df.reindex(time_index_hh)
    return df


def fill_gaps_in_plot_level_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Notice]]:
    """
    Attempts to fill any nans in the plot-level data by approximating based on previous/future values, and returns
    a new dataframe with the approximations, and a list of any Notices regarding data quality.

    See in-line comments for approximation methodologies.
    """

    df = df.copy()
    notices: List[Notice] = []

    num_nans_before = df.isna().sum().sum()
    if num_nans_before > 0:

        # First assume that the value stays the same for 1 hour before or after a gap
        df = df.ffill(limit=2)
        df = df.bfill(limit=2)

        # If there are still gaps, assume that the value is the same as that at the same time n days before/after
        HH_PER_DAY = 48
        for num_days in range(1, 7):
            row_shift = HH_PER_DAY * num_days
            df = df.fillna(df.shift(row_shift))
            df = df.fillna(df.shift(-row_shift))

        # If there are still gaps allow values to be forward/backward filled up to 5 hours
        df = df.ffill(limit=10)
        df = df.bfill(limit=10)

        num_nans_after = df.isna().sum().sum()
        nans_before_pct = (num_nans_before / df.size) * 100
        nans_after_pct = (num_nans_after / df.size) * 100

        notices.append(
            Notice(
                detail=f"{nans_before_pct:.1f}% of the plot-level load data is missing, of that "
                f"{nans_before_pct - nans_after_pct:.1f}% was approximated, the remaining "
                f"{nans_after_pct:.1f}% will be assumed to be 0",
                level=pct_to_notice_level(nans_before_pct),
            )
        )

    return df, notices


def synthesise_battery_inverter_if_needed(df: pd.DataFrame, target_avg_power: pd.Series) -> Tuple[pd.DataFrame, List[Notice]]:
    """
    Attempts to fill any gaps in the `bess_charge` and `bess_discharge` fields of `df` by using the target_avg_power as
    a proxy. Notices are returned if any gaps needed to be filled.

    The `bess_charge` and `bess_discharge` fields are derived from the meter at the battery inverter. If this meter has
    gone offline, and we are missing data then we can use the `target_power_avg` data to approximate as a fallback.
    The `target_power_avg` is reported by the Tesla battery as the power that it's aiming for, so it is normally fairly
    accurate to what was actually delivered.
    """
    df = df.copy()
    df2 = df.copy()

    notices = []

    steps_per_hh = get_steps_per_hh(get_step_size(df.index))

    df2["target_power_avg"] = target_avg_power
    df2["target_power_avg"] = df2["target_power_avg"].ffill(limit=steps_per_hh-1)

    bess_meter_reading_cols = ["bess_charge", "bess_discharge", "bess_import_cum_reading", "bess_export_cum_reading"]
    missing_data_ranges = run_length_encoding(df2[bess_meter_reading_cols].isna().all(axis=1))
    for start, end in missing_data_ranges:

        duration_missing = end - start
        hours_missing = duration_missing.total_seconds() / 3600
        notices.append(Notice(
            detail=f"{hours_missing:.1f}hrs of battery inverter data was synthesised ({start} -> {end})",
            level=duration_to_notice_level(duration_missing),
        ))

        index_before = df2.index.get_loc(start) - 1
        index_after = df2.index.get_loc(end) + 1
        if index_before <= 0 or index_after >= len(df2):
            notices.append(Notice(
                detail=f"{hours_missing:.1f}hrs of battery inverter data could not be synthesised ({start} -> {end}, {index_before} -> {index_after})",
                level=duration_to_notice_level(duration_missing),
            ))
            return df, notices

        hours_per_row = (timedelta(minutes=5) / timedelta(minutes=60))
        df2["estimated_bess_charge"] = -df2[df2["target_power_avg"] < 0]["target_power_avg"] * hours_per_row
        df2["estimated_bess_charge"] = df2["estimated_bess_charge"].fillna(0)
        df2["estimated_bess_discharge"] = df2[df2["target_power_avg"] > 0]["target_power_avg"] * hours_per_row
        df2["estimated_bess_discharge"] = df2["estimated_bess_discharge"].fillna(0)

        df2["estimated_bess_charge_cum_sum"] = df2["estimated_bess_charge"].cumsum()
        df2["estimated_bess_discharge_cum_sum"] = df2["estimated_bess_discharge"].cumsum()

        # Pull the total energy over the range to scale the estimate accurately
        charge_energy = reading_diff(df2["bess_export_cum_reading"], index_before, index_after)
        discharge_energy = reading_diff(df2["bess_import_cum_reading"], index_before, index_after)
        est_charge_energy = reading_diff(df2["estimated_bess_charge_cum_sum"], index_before, index_after)
        est_discharge_energy = reading_diff(df2["estimated_bess_discharge_cum_sum"], index_before, index_after)
        if est_charge_energy == 0:
            charge_scaling = 1.0
        else:
            charge_scaling = charge_energy / est_charge_energy
        if est_discharge_energy == 0:
            discharge_scaling = 1.0
        else:
            discharge_scaling = discharge_energy / est_discharge_energy

        df2["scaled_estimated_bess_charge"] = df2["estimated_bess_charge"] * charge_scaling
        df2["scaled_estimated_bess_discharge"] = df2["estimated_bess_discharge"] * discharge_scaling

        index = pd.date_range(start, end, freq="5min")
        df.loc[index, "bess_charge"] = df2["scaled_estimated_bess_charge"]
        df.loc[index, "bess_discharge"] = df2["scaled_estimated_bess_discharge"]

    return df, notices


def run_length_encoding(series: pd.Series) -> List[Tuple[datetime, datetime]]:
    """
    Returns a List of time ranges for which `series` is true.
    """
    df = pd.DataFrame(index=series.index)
    df["condition"] = series
    df["group"] = (df["condition"] != df["condition"].shift()).cumsum()
    df["time"] = df.index
    true_ranges = df[df["condition"]].groupby("group").agg(start=("time", "first"), end=("time", "last"))

    return list(zip(true_ranges["start"], true_ranges["end"]))


def reading_diff(series: pd.Series, index_start, index_end):
    """
    Returns the difference between the series at the start and end
    """
    return series.iloc[index_end] - series.iloc[index_start]


def dummy_agg(x):
    """
    This is a dummy aggregation function for scenarios where there should not be any aggregation - it just returns the
    value it was given and raises an exception if there is more than one value.
    """
    if len(x) > 1:
        raise ValueError("More than one value in dummy aggregation")
    return x
