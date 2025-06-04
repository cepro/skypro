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
) -> pd.DataFrame:

    if source.csv_profile_data_source:
        df = _get_csv_profile(
            source=source.csv_profile_data_source,
            file_path_resolver_func=file_path_resolver_func,
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
    df = pd.DataFrame(index=time_index)
    df["energy"] = source.value
    return df


def _get_csv_profile(
    source: CSVProfileDataSource,
    file_path_resolver_func: Optional[Callable],
) -> pd.DataFrame:

    df = get_csv_data_source(source, file_path_resolver_func)

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

    return df
