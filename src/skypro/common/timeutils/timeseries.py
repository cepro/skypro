from datetime import timedelta

import pandas as pd


def get_step_size(index: pd.DatetimeIndex) -> timedelta:
    """
    Returns the interval between index elements as a native timedelta.
    The pandas equivalent to the timedelta seems to be a bit buggy with time arithmatic, so the native object is
    preferred.
    """
    return timedelta(seconds=pd.to_timedelta(index.freq).total_seconds())


def get_steps_per_hh(step_size: timedelta) -> int:
    """
    Returns the integer number of steps in each half-hour period.
    """

    num_steps = timedelta(minutes=30) / step_size
    int_num_steps = int(num_steps)

    if num_steps != int_num_steps:
        raise ValueError(f"Step size {step_size} does not fit into half hours perfectly.")

    return int_num_steps
