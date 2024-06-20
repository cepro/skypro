import logging
from datetime import datetime, timedelta

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
            profile_csv_dir: str
    ):
        self._scaling_factor = scaling_factor

        # read in all the profile files in the given directory into a dataframe
        df = read_directory_of_csvs(profile_csv_dir)

        # store only the information that we need - which is a pd.Series of the profile, indexed by datetime
        df["UTCTime"] = pd.to_datetime(df["UTCTime"])
        self._profile = df.set_index("UTCTime")["energy"]
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
        for year_offset in range(0, -5, -1):

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
