import pandas as pd
import pytz


def date_and_sp_num_to_utc_datetime(date_series: pd.Series, sp_number_series: pd.Series, sp_timezone_str: str) -> pd.Series:
    """
    Converts the date and GB settlement period number to the tz-aware UTC time.
    """
    tz = pytz.timezone(sp_timezone_str)
    dt = (
        pd.to_datetime(date_series, format="%Y-%m-%d").dt.tz_localize(tz).dt.tz_convert("UTC") +
        pd.to_timedelta((sp_number_series - 1) * 30, unit='minutes')
    )
    return dt
