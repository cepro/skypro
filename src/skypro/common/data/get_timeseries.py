from datetime import datetime
from typing import Optional, Callable, List, Tuple

import pandas as pd

from skypro.common.config.data_source import TimeseriesDataSource, FlowsMarketDataSource, CSVTimeseriesDataSource
from skypro.common.data.utility import get_csv_data_source, drop_extra_rows, sanity_checks
from skypro.common.notice.notice import Notice


def get_timeseries(
        source: TimeseriesDataSource,
        start: Optional[datetime],
        end: Optional[datetime],
        file_path_resolver_func: Optional[Callable],
        db_engine: Optional,
        context: Optional[str],
) -> Tuple[pd.DataFrame, List[Notice]]:
    """
    This reads a data source - either from the flows database or CSV files - and returns a generic time series
    """

    if source.flows_market_data_source:
        df = _get_flows_market_data(
            source=source.flows_market_data_source,
            start=start,
            end=end,
            db_engine=db_engine
        )
    elif source.csv_timeseries_data_source:
        df = _get_csv_timeseries(
            source=source.csv_timeseries_data_source,
            start=start,
            end=end,
            file_path_resolver_func=file_path_resolver_func,
        )
    else:
        raise ValueError("Unknown source type")

    return df, sanity_checks(df, start, end, context)


def _get_flows_market_data(
        source: FlowsMarketDataSource,
        start: datetime,
        end: datetime,
        db_engine
) -> pd.DataFrame:
    """
    Reads data of the given type from the market_data flows table. The returned dataframe will have columns for:
    `time`, `value`, and optionally `predicted_at` (if the data is predictive)
    """

    # TODO: hourly data does not get processed correctly - we need to re-index against the time_index

    # The system is quite flexible to allow interpretation of predictive datasets as non-predictive, and vice-versa, but
    # here there is a sanity check to make sure the user isn't doing something quite wrong:
    if source.is_predictive and source.type in ["elexon-imbalance-price", "elexon-imbalance-volume"]:
        raise ValueError(f"The {source.type} source was configured as predictive, this probably wont work")
    if not source.is_predictive and source.type in ["modo-imbalance-price-forecast", "modo-imbalance-volume-forecast"]:
        raise ValueError(f"The {source.type} source was configured as non-predictive, this probably isn't what you want")

    if source.is_predictive:
        # If the data is 'predictive' then we need to pull not just the latest values, but all the updates that happened
        # along the way.
        query = (
            "SELECT time, created_at, value FROM flows.market_data "
            "JOIN flows.market_data_types on market_data.type = market_data_types.id "
            f"WHERE "
            f"  time >= '{start.isoformat()}' AND "
            f"  time <= '{end.isoformat()}' AND "
            f"  market_data_types.name = '{source.type}' "
            f"order by time asc, created_at asc"
        )
    else:
        # If the data is not 'predictive' then we extract just the latest values for each settlement period via the
        # SELECT DISTINCT ON clause.
        query = (
            " WITH data AS ( "
            "   SELECT time, created_at, value FROM flows.market_data "
            "   JOIN flows.market_data_types on market_data.type = market_data_types.id "
            f"  WHERE "
            f"    time >= '{start.isoformat()}' AND "
            f"    time <= '{end.isoformat()}' AND "
            f"    market_data_types.name = '{source.type}' "
            " )"
            " SELECT DISTINCT ON (time) time, value FROM data "
            " ORDER BY time asc, created_at desc"
        )

    df = pd.read_sql(query, con=db_engine)

    # This should be renaming 'fetch history'? And renamed to predictive in Skypro as required?
    if source.is_predictive:
        df = df.rename(columns={"created_at": "predicted_at"})

    return df


def _get_csv_timeseries(
    source: CSVTimeseriesDataSource,
    start: Optional[datetime],
    end: Optional[datetime],
    file_path_resolver_func: Optional[Callable],
) -> pd.DataFrame:
    """
    Reads timeseries data from the given CSV files.
    """

    df = get_csv_data_source(source, file_path_resolver_func)

    if "utctime" in df.columns:
        df["time"] = pd.to_datetime(df["utctime"], format="ISO8601")
        df = df.drop(columns=["utctime", "clocktime"])
    elif "spUTCTime" in df.columns:
        df["time"] = pd.to_datetime(df["spUTCTime"], format="ISO8601")
        df = df.drop(columns=["spUTCTime", "spClockTime"])

    df = df.rename(columns={
        "price": "value",
        "volume": "value"
    })

    # Remove any data that is outside the time range of interest
    # TODO: only read in CSVs for the months that are required in the first place
    df = drop_extra_rows(df, start, end)

    return df
