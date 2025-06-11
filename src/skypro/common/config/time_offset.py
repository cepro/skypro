import re
from datetime import timedelta
from typing import Optional


def time_offset_str_to_timedelta(time_offset_str: Optional[str]) -> timedelta:
    """
    Converts a 'time offset string' to a timedelta.
    Currently only strings like "1y", "-1y", "2y" are supported to represent +1, -1 and +2 years respectively.
    """

    if time_offset_str is None:
        return timedelta()

    # Clean and standardize the input
    time_offset_str = time_offset_str.lower().strip()

    # Match patterns like "1y", "2y", "-1y", "-2y"
    match = re.match(r'^(-?)(\d+)y$', time_offset_str)

    if match:
        sign = -1 if match.group(1) else 1
        years = int(match.group(2))
        return timedelta(days=sign * years * 365)

    raise ValueError(f"Unknown time offset string: {time_offset_str}. Expected format 'Xy' or '-Xy' for X years.")
