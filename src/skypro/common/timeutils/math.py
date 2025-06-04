from datetime import datetime


def floor_hh(t: datetime) -> datetime:
    """
    Returns `t` rounded down to the nearest half-hour boundary
    """
    minute = 0
    if t.minute >= 30:
        minute = 30

    return t.replace(minute=minute, second=0, microsecond=0)


def floor_day(t: datetime) -> datetime:
    """
    Returns `t` rounded down to the nearest day boundary.
    `t` must be in the appropriate timezone.
    """
    return t.replace(hour=0, minute=0, second=0, microsecond=0)

