import functools
import json
import logging

import requests
from datetime import date, datetime, timedelta
from io import StringIO
from typing import Callable

import pandas as pd

from skypro.common.cli_utils.cli_utils import get_user_ack_of_warning_or_exit, read_yaml_file
from skypro.common.data.utility import prepare_data_dir

from skypro.common.timeutils.month_str import get_first_and_last_date
from skypro.commands.pull_elexon_imbalance.utils import daterange, with_retries

ELEXON_API_MAX_RETRIES = 5
ELEXON_API_RETRY_DELAY = timedelta(seconds=1)


def pull_elexon_imbalance(month_str: str, env_file_path: str):
    """
    Pulls a months worth of half-hourly imbalance price and volume data from Elexon and saves it to disk.
    The data is saved in monthly CSV files in the directory defined by the MARKET_DATA_DIR in the environment configuration.
    """

    start_date, end_date = get_first_and_last_date(month_str)

    today = datetime.now().date()
    if end_date > today:
        end_date = today
        get_user_ack_of_warning_or_exit(f"The month has not ended yet, so data will be incomplete after {end_date}")

    env_config = read_yaml_file(env_file_path)
    data_dir = env_config["vars"]["MARKET_DATA_DIR"]

    df = _fetch_multiple_days(
        start=start_date,
        end=end_date,
        fetch_func=_fetch_day,
    )

    df["price"] = df["price"] / 10  # £/MW to p/kW

    prices_file_path = prepare_data_dir(data_dir, "elexon", "imbalance_price", start_date)
    volume_file_path = prepare_data_dir(data_dir, "elexon", "imbalance_volume",  start_date)

    logging.info(f"Saving pricing data to '{prices_file_path}'")
    df[["spUTCTime", "spClockTime", "price"]].to_csv(prices_file_path, index=False)
    logging.info(f"Saving volume data to '{volume_file_path}'")
    df[["spUTCTime", "spClockTime", "volume"]].to_csv(volume_file_path, index=False)


def _fetch_multiple_days(start: date, end: date, fetch_func: Callable) -> pd.DataFrame:
    """
    The elexon API pulls for a single day, so this function calls the elexon API repeatedly for each day and stacks up
    the results.
    """
    df = pd.DataFrame()
    for day in daterange(start, end):

        logging.info(f"Fetching imbalance data for '{str(day)}'...")
        day_df = with_retries(  # The Elexon API can be busy/unreliable at times so use retries to get past temporary failures
            functools.partial(fetch_func, day),
            ELEXON_API_MAX_RETRIES,
            ELEXON_API_RETRY_DELAY
        )

        df = pd.concat([df, day_df])

    # The values come through in the opposite order to what you'd expect
    df = df.sort_values(by=["spUTCTime"], ignore_index=True)

    return df


def _fetch_day(day: date) -> pd.DataFrame:
    """
    Pulls a single days worth of imbalance data from Elexon.
    """

    day_str = day.isoformat()
    # Send a GET request to the API
    response = requests.get(
        url=f"https://data.elexon.co.uk/bmrs/api/v1/balancing/settlement/system-prices/{day_str}",
        params={"format": "json"}
    )
    response.raise_for_status()

    json_data = json.load(StringIO(response.text))
    day_df = pd.DataFrame.from_dict(json_data["data"])

    day_df["startTime"] = pd.to_datetime(day_df["startTime"], utc=True)
    day_df = day_df[["startTime", "systemSellPrice", "netImbalanceVolume"]]
    day_df = day_df.rename(columns={
        "startTime": "spUTCTime",
        "systemSellPrice": "price",
        "netImbalanceVolume": "volume"
    })

    day_df.insert(1, "spClockTime", day_df["spUTCTime"].dt.tz_convert("Europe/London"))

    return day_df
