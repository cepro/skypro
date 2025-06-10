from datetime import datetime

import arrow
import pytz


def add_wallclock_days(t: datetime, n: int) -> datetime:
    """
    Adds `n` days to the given time, properly accounting for the timezone.

    The native datetime library treats days as 24hours, and doesn't account for DST boundaries. The Arrow library
    supports days sometimes being 23 or 25 hours and so properly adjusts for DST.
    """
    t_arrow = arrow.get(t)
    t_arrow = t_arrow.shift(days=n)

    t_native = _arrow_to_pydatetime(t_arrow, t.tzinfo)

    return t_native


def _arrow_to_pydatetime(t_arrow, tzinfo) -> datetime:
    """
    Converts an Arrow library datetime to a native python datetime
    """
    return pytz.timezone(str(tzinfo)).localize(datetime(
        t_arrow.year,
        t_arrow.month,
        t_arrow.day,
        t_arrow.hour,
        t_arrow.minute,
        t_arrow.second,
        t_arrow.microsecond,
    ))

