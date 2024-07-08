import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from simt_common.dataparse.dataparse import read_directory_of_csvs
from simt_common.timeutils.hh_math import floor_hh


class Profiler:
    """
    Profiler handles the scaling of a load or solar energy profile.
    """
    def __init__(
            self,
            scaling_factor: float,
            profile_csv_dir: Optional[str] = None,
            profile_csv: Optional[str] = None,
            energy_cols: Optional[str] = None
    ):
        self._scaling_factor = scaling_factor

        if profile_csv_dir:
            # read in all the profile files in the given directory into a dataframe
            df = read_directory_of_csvs(profile_csv_dir)
        elif profile_csv:
            df = pd.read_csv(profile_csv)
        else:
            raise ValueError("Either a directory containing CSVs or CSV file must be specified")

        df["UTCTime"] = pd.to_datetime(df["UTCTime"], utc=True)
        df = df.set_index("UTCTime")

        # If we have UTCTime then we don't need the ClockTime column
        if "ClockTime" in df.columns:
            df = df.drop("ClockTime", axis=1)

        if energy_cols is None:
            if "energy" not in df.columns:
                raise KeyError("Expected a column called 'energy' in profile")
            self._profile = df["energy"]
        elif energy_cols == "sum-all":
            self._profile = df.sum(axis=1)
        else:
            raise ValueError(f"Unknown energy column option: '{energy_cols}'")

        self._profile = self._profile.sort_index()
        duplicated = self._profile[self._profile.index.duplicated()].index
        if len(duplicated) > 0:
            raise ValueError(f"Duplicate times in profiled data: {duplicated}")

    def get_for(self, times: pd.DatetimeIndex) -> pd.Series:
        """
        Returns the scaled profile for the range of times given
        """

        # We have profile data at half-hour / 30minute granularity, but the requested times may be at a finer resolution
        # so round the times down to the nearest half-hour (floor_hh) to extract the relevant value from the profile
        df_hh = pd.DataFrame(index=pd.Series(times.to_series().apply(lambda t: floor_hh(t)).unique()))
        df_hh["values"] = np.nan

        # Search the profile by offsetting the year by increasing degrees
        for year_offset in range(0, -10, -1):

            # These are the times to search for in the profile
            hh_times_search = df_hh.index.to_series().apply(lambda t: try_offset_year(t, year_offset))
            try:
                # Search the profile
                df_hh["new_finds"] = hh_times_search.apply(lambda t: self._profile.get(t, np.nan)).values

                # Store the found values
                df_hh["values"] = df_hh["values"].fillna(value=df_hh["new_finds"])
            except KeyError:
                # We may not find any of the search times
                pass

            # Stop searching if we have found all the times
            if df_hh["values"].isna().sum() == 0:
                break

        df_hh["values"] = df_hh["values"] * self._scaling_factor

        # Fill any missing profile points at half-hour granularity
        num_nan_1 = df_hh["values"].isna().sum()
        df_hh["values"] = df_hh["values"].ffill(limit=5)
        num_nan_2 = df_hh["values"].isna().sum()
        num_ff = num_nan_1 - num_nan_2
        df_hh["values"] = df_hh["values"].fillna(0)
        num_nan_3 = df_hh["values"].isna().sum()
        num_zerod = num_nan_2 - num_nan_3
        if num_zerod > 0 or num_ff > 0:
            logging.warning(f"{num_nan_1} values in HH profiled data were NaN, {num_ff} have been forward-filled, "
                            f"{num_zerod} have been set to 0.")

        # Up-scale the half-hour granularity to whatever granularity has been requested
        df = pd.DataFrame(index=times)
        resolution_scaling_factor = pd.to_timedelta(times.freq) / timedelta(minutes=30)
        df["values"] = df_hh["values"] * resolution_scaling_factor
        df["values"] = df["values"].ffill()

        return df["values"]


def try_offset_year(t: datetime, year_offset: int) -> datetime:
    """
    Returns the given time offset by the given number of years. This is not always possible - e.g. if the t is Feb 29th
    on a leap year, and the year_offset would lead to a non-leapyear. In which case the original time is returned.
    """
    try:
        new_t = t.replace(year=t.year + year_offset)
    except ValueError:
        new_t = t

    return new_t
