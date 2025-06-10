from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple, TypeVar

import pandas as pd

from skypro.common.timeutils.dayed_period import DayedPeriod

T = TypeVar('T')  # Generic type parameter


class TimeVaryingValue:
    """
    A generic class that manages time-varying values of any type.
    Each value is valid from its start time until the start time of the next value.
    A start_time of None indicates a value that has been active since "the beginning of time".
    The last defined value will be considered active until "the end of time".
    """

    def __init__(self, times_with_values: List[Tuple[Optional[datetime], T]] = None):
        """
        Initialize with a list of (start_time, value) tuples.
        A start_time of None represents "the beginning of time" and should appear first in the list.
        """
        # Make a copy of the input values to avoid modifying the original
        self.times_with_values = list(times_with_values)

        # Sort the values with None/NaT first, then by start_time
        self.times_with_values.sort(key=lambda x: (not (x[0] is None or pd.isnull(x[0])), x[0]))

    def get_value_at(self, t: datetime) -> Optional[T]:
        """
        Returns the value at time t, or None if no value is active.
        """
        applicable_value = None

        for start_time, value in self.times_with_values:
            if start_time is None or pd.isnull(start_time) or start_time <= t:
                applicable_value = value
            else:
                break

        if applicable_value is None:
            raise KeyError(f"Could not find value for time {t}")

        return applicable_value

    def get_all_values(self) -> List[T]:
        """
        Returns a list of all the configured values
        """
        return [time_with_value[1] for time_with_value in self.times_with_values]


@dataclass
class PeriodicValue:
    """
    Holds the float value of a rate, alongside the times of day for which it's active.
    """
    value: float
    periods: List[DayedPeriod]
