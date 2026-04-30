import logging
from typing import Optional, Callable

import numpy as np
import pandas as pd
import pytz

from skypro.common.config.data_source import ProfileDataSource, ConstantProfileDataSource
from skypro.common.config.data_source_csv import CSVProfileDataSource
from skypro.common.data.utility import get_csv_data_source


def get_profile(
        source: ProfileDataSource,
        time_index: pd.DatetimeIndex,
        file_path_resolver_func: Optional[Callable],
        max_energy_per_interval_kwh: Optional[float] = None,
) -> pd.DataFrame:
    """
    Reads the profile data source and returns a dataframe containing the profile with the given time index.

    If max_energy_per_interval_kwh is set, CSV profiles are filtered: rows with |energy| above the
    threshold are replaced with NaN and linearly interpolated. Default (None) is no filter — raw
    values pass through, so corrupted source data surfaces loudly in downstream sim output rather
    than being silently rewritten.
    """

    if source.csv_profile_data_source:
        df = _get_csv_profile(
            source=source.csv_profile_data_source,
            file_path_resolver_func=file_path_resolver_func,
            max_energy_per_interval_kwh=max_energy_per_interval_kwh,
        )
    elif source.constant_profile_data_source:
        df = _get_constant_profile(
            source=source.constant_profile_data_source,
            time_index=time_index,
        )
    else:
        raise ValueError("Unknown source type")

    return df


def _get_constant_profile(
    source: ConstantProfileDataSource,
    time_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Returns a profile with a constant value
    """
    df = pd.DataFrame(index=time_index)
    df["energy"] = source.value
    return df


def _get_csv_profile(
    source: CSVProfileDataSource,
    file_path_resolver_func: Optional[Callable],
    max_energy_per_interval_kwh: Optional[float] = None,
) -> pd.DataFrame:
    """
    Returns a profile using the given CSV files.

    When max_energy_per_interval_kwh is set, energy values above this absolute threshold are
    treated as anomalous: replaced with NaN, then linearly interpolated (with ffill/bfill at
    edges). Activate it on a per-profile basis from YAML via maxEnergyPerIntervalKwh when
    upstream metering is known to emit corrupt rows. The default (None) is opt-in: raw values
    pass through unchanged unless a threshold is explicitly set.
    """

    df = get_csv_data_source(source, file_path_resolver_func)

    # Defensive filtering: opt-in via max_energy_per_interval_kwh.
    if max_energy_per_interval_kwh is not None and "energy" in df.columns:
        anomalous_mask = df["energy"].abs() > max_energy_per_interval_kwh
        num_anomalous = anomalous_mask.sum()
        if num_anomalous > 0:
            anomalous_rows = df[anomalous_mask]
            max_val = anomalous_rows["energy"].max()
            min_val = anomalous_rows["energy"].min()
            # Surface enough detail for an operator to chase the upstream meter glitch:
            # count, range, and the first/last bad timestamps when available.
            time_col = next(
                (c for c in ("UTCTime", "ClockTime") if c in anomalous_rows.columns),
                None,
            )
            if time_col is not None and len(anomalous_rows) > 0:
                ts_str = (
                    f" Timestamps: {anomalous_rows[time_col].iloc[0]} → "
                    f"{anomalous_rows[time_col].iloc[-1]}."
                )
            else:
                ts_str = ""
            logging.warning(
                f"Dropped {num_anomalous} rows with anomalous energy values "
                f"(>{max_energy_per_interval_kwh} kWh per interval). "
                f"Range: {min_val:.2f} to {max_val:.2f} kWh.{ts_str} "
                f"This typically indicates corrupted meter data."
            )
            # Set anomalous values to NaN (will be interpolated below).
            df.loc[anomalous_mask, "energy"] = np.nan

    # Prefer to use the UTCTime column, but if it's not present then use ClockTime with the Europe/London timezone
    use_clocktime = "UTCTime" not in df.columns or np.all(pd.isnull(df["UTCTime"]))
    if use_clocktime:
        df["ClockTime"] = pd.to_datetime(df["ClockTime"])
        df["ClockTime"] = df["ClockTime"].dt.tz_localize(
            pytz.timezone("Europe/London"),
            ambiguous="NaT",
            nonexistent="NaT"
        )
        num_inc_nan = len(df)
        df = df.dropna(subset=["ClockTime"])
        num_dropped = num_inc_nan - len(df)
        if num_dropped > 0:
            logging.warning(f"Dropped {num_dropped} NaT rows from profile (probably because the UTC time could "
                            f"not be inferred from the ClockTime")
        df["UTCTime"] = df["ClockTime"].dt.tz_convert("UTC")
    else:
        df["UTCTime"] = pd.to_datetime(df["UTCTime"], utc=True)

    df = df.set_index("UTCTime")

    # If we have UTCTime then we don't need the ClockTime column
    if "ClockTime" in df.columns:
        df = df.drop("ClockTime", axis=1)

    # Interpolate any NaN values in energy column (from anomalous data filtering).
    if "energy" in df.columns and df["energy"].isna().any():
        num_nans = df["energy"].isna().sum()
        df["energy"] = df["energy"].interpolate(method="linear")
        # Edge cases (NaN at start/end) — forward/backward fill.
        df["energy"] = df["energy"].ffill().bfill()
        logging.info(f"Interpolated {num_nans} missing energy values in profile")

    return df
