import logging
from functools import partial
from typing import Optional, Dict, List, Tuple, Any
from datetime import timedelta

import numpy as np
import pandas as pd


def generate_output_df(
        df: pd.DataFrame,
        int_final_vol_rates_dfs: Dict[str, pd.DataFrame],
        mkt_final_vol_rates_dfs: Dict[str, pd.DataFrame],
        int_live_vol_rates_dfs: Optional[Dict[str, pd.DataFrame]],
        mkt_live_vol_rates_dfs: Optional[Dict[str, pd.DataFrame]],
        mkt_fixed_costs_dfs: Optional[Dict[str, pd.DataFrame]],
        customer_fixed_cost_dfs: Optional[Dict[str, pd.DataFrame]],
        customer_vol_rates_dfs: Optional[Dict[str, pd.DataFrame]],
        load_energy_breakdown_df: Optional[pd.DataFrame],
        aggregate_timebase: Optional[str],
        rate_detail: Optional[str],
        config_entries: List[Tuple[str, Any]]
) -> pd.DataFrame:
    """
    Creates a timeseries of microgrid behaviour suitable for saving to an output CSV file
    """

    output_df = pd.DataFrame(index=df.index)
    output_df["clocktime"] = df.index.tz_convert("Europe/London")

    output_df["agd:load"] = df["load"]
    if load_energy_breakdown_df is not None:
        for col in load_energy_breakdown_df.columns:
            output_df[f"agd:load.{col}"] = load_energy_breakdown_df[col]

    output_df["agd:solar"] = df["solar"]

    output_df["agd:match"] = df["solar_to_load_property_level"]

    output_df["m:upImport"] = df["grid_import"]
    output_df["m:upExport"] = df["grid_export"]

    output_df["m:battSoe"] = df["soe"]
    output_df["m:battCharge"] = df["bess_charge"]
    output_df["m:battDischarge"] = df["bess_discharge"]
    output_df["m:match"] = df["solar_to_load_microgrid_level"]

    output_df["c:battLosses"] = df["bess_losses"]
    output_df["c:limitMaxBattCharge"] = df["bess_max_charge"]
    output_df["c:limitMaxBattDischarge"] = df["bess_max_discharge"]

    # Maps the column names used internally in Skypro to the column names we want in outputs
    output_flow_name_map = {
        "solar_to_grid": "solarToGrid",
        "grid_to_load": "gridToLoad",
        "solar_to_load": "solarToLoad",
        "batt_to_load": "battToLoad",
        "batt_to_grid": "battToGrid",
        "solar_to_batt": "solarToBatt",
        "grid_to_batt": "gridToBatt"
    }

    for internal_flow_name, csv_flow_name in output_flow_name_map.items():
        output_df[f"c:{csv_flow_name}"] = df[internal_flow_name]

    if rate_detail and rate_detail != "all":
        # parse the string like a comma-seperated list of rates to include detail for
        rates_of_interest = rate_detail.split(",")

    # Report on the market and internal volume rates, for both live and final, for each flow. These are identified by
    # column prefixes/suffixes in the CSV column names
    for vol_rates_dfs, csv_col_prefix, csv_col_suffix in [
        (int_final_vol_rates_dfs, "ivRate", "final"),
        (mkt_final_vol_rates_dfs, "mvRate", "final"),
        (int_live_vol_rates_dfs, "ivRate", "live"),
        (mkt_live_vol_rates_dfs, "mvRate", "live"),
    ]:
        if vol_rates_dfs:
            for flow_name, vol_rates_df in vol_rates_dfs.items():
                csv_flow_name = output_flow_name_map[flow_name]
                output_df[f"{csv_col_prefix}:{csv_flow_name}.{csv_col_suffix}"] = vol_rates_df.sum(axis=1, skipna=False)

                # Optionally report the individual rates that make up the total rate for the flow
                if rate_detail and csv_col_suffix == "final":  # Only report final rates at detail for now
                    for detailed_rate_col in vol_rates_df.columns:
                        if rate_detail == "all" or detailed_rate_col in rates_of_interest:
                            output_df[f"{csv_col_prefix}:{csv_flow_name}.{detailed_rate_col}.{csv_col_suffix}"] = vol_rates_df[detailed_rate_col]

                # Avoid the 'highly fragmented' dataframe warning by taking a copy, which de-fragments:
                output_df = output_df.copy()

    output_df = add_peripheral_rates_to_output_df(
        output_df=output_df,
        rate_dfs=mkt_fixed_costs_dfs,
        col_prefix="mfCost",
        rate_detail=rate_detail
    )
    output_df = add_peripheral_rates_to_output_df(
        output_df=output_df,
        rate_dfs=customer_fixed_cost_dfs,
        col_prefix="cfCost",
        rate_detail=rate_detail
    )
    output_df = add_peripheral_rates_to_output_df(
        output_df=output_df,
        rate_dfs=customer_vol_rates_dfs,
        col_prefix="cvRate",
        rate_detail=rate_detail
    )

    output_df["other:imbalanceVolume.final"] = df["imbalance_volume_final"]
    if "imbalance_volume_live" in output_df:
        output_df["other:imbalanceVolume.live"] = df["imbalance_volume_live"]

    if "osam_ncsp" in df.columns:
        output_df["other:osam.ncsp"] = df["osam_ncsp"]

    if aggregate_timebase:
        output_df = aggregate(
            output_df=output_df,
            output_flow_name_map=output_flow_name_map,
            timebase=aggregate_timebase
        )

    # Round any floats because otherwise we can end up with a huge number of decimal places in the CSV which is only
    # going to make the file size bigger than it needs to be
    output_df = output_df.round(decimals=5)

    if "clocktime" in output_df.columns:
        # If this is a timeseries, then set the index to be the UTC time
        output_df.index = output_df["clocktime"].dt.tz_convert("UTC")

    # Add the configuration input to the output file - this is stretching the use of the CSV format a bit, but it means
    # that there is a single output file with full traceability as to all the inputs.
    output_df = with_config_entries(
        df=output_df,
        entries=config_entries
    )

    return output_df


def add_peripheral_rates_to_output_df(
        output_df: pd.DataFrame,
        rate_dfs: Dict[str, pd.DataFrame],
        col_prefix: str,
        rate_detail: str
) -> pd.DataFrame:
    """
    Some rates are not really 'core' to the operation of the reporting itself, but are just passed through into the
    output dataframe to make future analysis of the output CSV easier. This function adds these types of rates into the
    output dataframe. These 'peripheral' rates have their totals calculated and are added to the dataframe with the
    given column prefixes.
    """

    for category, rate_df in rate_dfs.items():
        output_df[f"{col_prefix}:{category}"] = rate_df.sum(axis=1)
        if rate_detail == "all":
            output_df = pd.concat([output_df, rate_df.add_prefix(f"{col_prefix}:{category}.")], axis=1)

    return output_df


def aggregate(output_df: pd.DataFrame, output_flow_name_map: Dict[str, str], timebase: str) -> pd.DataFrame:
    """
    Resamples the output_df into an aggregated timebase, currently supporting timebases of:
    - `30min` which will return a row per half-hour
    - `all` which will return a single row for the entire duration
    """
    sum_dont_skip_nan = partial(pd.Series.sum, skipna=False)

    def first(x):
        return x.iloc[0]

    # Here we need to define how each column in the output_df is aggregated over time. The order of this
    # dictionary defines the order of the output.

    agg_rules = {}

    if timebase == "30min":
        agg_rules["clocktime"] = first
    elif timebase == "all":
        pass  # The time doesn't seem relevant when summarising over all time

    agg_rules = agg_rules | {
        "agd:load": sum_dont_skip_nan,
        "agd:solar": sum_dont_skip_nan,
        "agd:match": sum_dont_skip_nan,

        "m:upImport": sum_dont_skip_nan,
        "m:upExport": sum_dont_skip_nan,
    }

    if timebase == "30min":
        agg_rules = agg_rules | {
            "m:battSoe": first,
            "c:limitMaxBattCharge": sum_dont_skip_nan,  # the limits are in kWh over the period
            "c:limitMaxBattDischarge": sum_dont_skip_nan,
        }
    elif timebase == "all":
        pass  # The battery SoE and limits don't seem relevant when summarising over all time

    agg_rules = agg_rules | {
        "m:battCharge": sum_dont_skip_nan,
        "m:battDischarge": sum_dont_skip_nan,
        "c:battLosses": sum_dont_skip_nan,
        "m:match": sum_dont_skip_nan,
    }

    for _, csv_flow_name in output_flow_name_map.items():
        agg_rules[f"c:{csv_flow_name}"] = sum_dont_skip_nan

    if timebase == "30min":
        agg_rules["other:imbalanceVolume.final"] = "first_ensure_consistent"
    elif timebase == "all":
        pass  # Don't include imbalance volume for full aggregation

    if "other:osam.ncsp" in output_df.columns:
        if timebase == "30min":
            agg_rules["other:osam.ncsp"] = "first_ensure_consistent"
        elif timebase == "all":
            # To summarises the OSAM NCSP across all time, calculate the average, weighted by the applicable volume
            agg_rules["other:osam.ncsp"] = partial(safe_average, weights=output_df["c:gridToBatt"])

    for col in output_df.columns:
        if col.startswith("agd:load."):
            agg_rules[col] = sum_dont_skip_nan

    for col in output_df.columns:
        if col.startswith("mfCost:") or col.startswith("cfCost:"):
            agg_rules[col] = sum_dont_skip_nan
        elif col.startswith("cvRate:"):
            if timebase == "30min":
                agg_rules[col] = "first_ensure_consistent"
            elif timebase == "all":
                agg_rules[col] = partial(safe_average, weights=output_df["agd:load"])

    for col in output_df.columns:
        if (col.startswith("mvRate:") or col.startswith("ivRate:")) and col.endswith(".final"):
            if timebase == "30min":
                agg_rules[col] = "first_ensure_consistent"
            elif timebase == "all":
                flow_name = col.removeprefix("mvRate:").removeprefix("ivRate:").removesuffix(".final")
                # If we are including individual rate detail then we will need to remove the individual rate name, e.g.
                # rate:gridToBatt.duosRed.final -> gridToBatt.duosRed -> gridToBatt
                flow_name_split = flow_name.split(".")
                if len(flow_name_split) > 1:
                    flow_name = flow_name_split[0]

                # To summarises the rates across all time, calculate the average, weighted by the applicable volume
                agg_rules[col] = partial(safe_average, weights=output_df[f"c:{flow_name}"])

        # There doesn't seem to be a sensible way to aggregate *live* rates or volumes to half-hourly, so
        # leave them out.

    logging.info("Ensuring aggregation consistency...")
    # The `first_ensure_consistent` aggregation is actually done in two steps:
    #  1) consistency across each aggregation window is checked
    #  2) the `first` aggregation is applied
    # (Originally a custom aggregation function was used to do both steps but it was very slow)
    first_ensure_equal_cols = []
    for col, rule in agg_rules.items():
        if rule == "first_ensure_consistent":
            first_ensure_equal_cols.append(col)
            agg_rules[col] = first
    period = timedelta(seconds=pd.to_timedelta(output_df.index.freq).total_seconds())
    rows_per_agg_window = int(timedelta(minutes=30) / period)
    ensure_consistent_value_across_aggregation_window(output_df[first_ensure_equal_cols], rows_per_agg_window)

    # Do the actual aggregation
    logging.info("Aggregating...")
    if timebase == "30min":
        output_df = output_df.resample("30min").apply(lambda group: apply_aggregation_functions(group, agg_rules))
    elif timebase == "all":
        output_df = apply_aggregation_functions(output_df, agg_rules)
    else:
        raise ValueError(f"Unknown aggregation timebase: '{timebase}'")

    return output_df


def apply_aggregation_functions(df: pd.DataFrame, agg_rules: Dict) -> pd.DataFrame:
    """
    Applies the aggregation functions defined in `agg_rules` to the `df` across all rows and returns the result.

    In theory, you can use Pandas' built-in `agg` function like: `df.agg(agg_rules).to_frame().transpose()` however,
    this seemed buggy and would not work when doing a weighted average where the weights are all zero (even with the
    `safe_average` function). It also seems to call the custom functions multiple times to run consistency checks, but
    we only want to call them once really as this is quite a slow operation
    """
    result = {}

    for col, func in agg_rules.items():
        if callable(func):  # Custom function
            result[col] = func(df[col])
        else:  # Built-in aggregation like 'sum'
            result[col] = df[col].agg(func)

    result_df = pd.DataFrame(result, index=[0])  # Combine into a DataFrame

    return result_df


def safe_average(a, weights=None):
    """
    Wraps np.average and handles the case where weights sum to zero by returning NaN (np.average throws an exception)
    """

    if weights is not None and np.sum(weights) == 0:
        ret_val = 0.0
    else:
        ret_val = np.average(a, weights=weights)

    return ret_val


def ensure_consistent_value_across_aggregation_window(df: pd.DataFrame, rows_per_agg_window: int):
    """
    This checks that the value in each column doesn't change within the aggregation window.

    For example, if we are aggregating a price from 10-minutely up to 30-minutely, and using the `first` aggregation
    function to extract the first price for each 30minutes, then we want to be sure that the price doesn't change within
    the 30 minute window - otherwise the principle of the aggregation is broken.
    """
    cols = df.columns
    check_df = pd.DataFrame(index=df.index, columns=cols)

    # Get the first value from each aggregation window
    check_df[cols] = df.loc[::rows_per_agg_window, cols]
    check_df = check_df.ffill(limit=rows_per_agg_window - 1)

    assert check_df.equals(df)


def with_config_entries(df: pd.DataFrame, entries: List[Tuple[str, Any]]) -> pd.DataFrame:
    """
    Adds some configuration entries to the output dataframe as extra columns, with just the first row containing the
    given values
    """
    for key, value in entries:
        df[key] = ""
        value_str = str(value)
        df.iloc[0, df.columns.get_loc(key)] = value_str

    return df
