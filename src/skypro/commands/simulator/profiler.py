import logging
from datetime import datetime
from os import listdir
from os.path import join, isfile

import pandas as pd

from skypro.cli_utils.cli_utils import read_directory_of_csvs


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
        self._profile_searchable = self._profile.index.strftime("%Y-%m-%d %H:%M:%S").str

    def get_for(self, times: pd.Series) -> pd.Series:
        """
        Returns the scaled profile for the range of times given
        """
        values = times.apply(lambda t: self.get_at(t))
        num_nan = values.isna().sum()
        values = values.ffill()
        if num_nan > 0:
            logging.warning(f"Forward filled {num_nan} NaN values in profiled data.")
        return values

    def get_at(self, t: datetime) -> float:
        """
        Returns the scaled profile at the given datetime. This is either found by a direct match on the time
         requested in the raw data or is inferred from other years that are present in the raw data.
        """
        try:
            # First see if there is a value for the actual year requested
            val = self._profile.loc[t]
        except KeyError:
            # If there is no value for the actual year requested, try and find a value for this time in another year
            # that will do the job

            search_str = t.strftime("%m-%d %H:%M:%S")
            matches = self._profile_searchable.contains(search_str)
            if matches.sum() < 1:
                raise KeyError(f"Cannot find profile for YYYY-{search_str} in any year")

            val = self._profile.loc[matches].sort_index().iloc[0]

        return val * self._scaling_factor
