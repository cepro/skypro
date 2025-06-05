from calendar import monthrange
from datetime import date, datetime, timedelta
from typing import Tuple

import pytz


def get_first_and_last_date(month_str: str) -> Tuple[date, date]:
    """
    Returns the first and last date of the given month.
    """
    try:
        start_date = datetime.strptime(month_str, '%Y-%m').date()
    except ValueError:
        raise ValueError("Incorrect month format, should be YYYY-MM")

    num_days = monthrange(start_date.year, start_date.month)[1]
    end_date = date(year=start_date.year, month=start_date.month, day=num_days)

    return start_date, end_date


def get_month_timerange(month_str: str, timezone_str: str) -> Tuple[datetime, datetime]:
    """
    Returns the first instant of the given month, and the first instant of the next month. So the months range
    is start <= t < end.
    Note, this function doesn't use dateutils.relativedelta because it does not seem to honour DST timezones!
    """
    try:
        start = datetime.strptime(month_str, '%Y-%m')
    except ValueError:
        raise ValueError("Incorrect month format, should be YYYY-MM")
    num_days = monthrange(start.year, start.month)[1]
    end = datetime(year=start.year, month=start.month, day=num_days)
    end += timedelta(days=1)

    tz = pytz.timezone(timezone_str)
    start = tz.localize(start)
    end = tz.localize(end)

    return start, end