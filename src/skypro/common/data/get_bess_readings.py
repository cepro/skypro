from datetime import datetime
from typing import Optional, Callable, Tuple, List

import pandas as pd

from skypro.common.config.data_source import BessReadingDataSource, CSVBessReadingsDataSource
from skypro.common.config.data_source_flows import FlowsBessReadingsDataSource
from skypro.common.data.utility import sanity_checks, get_csv_data_source, drop_extra_rows
from skypro.common.notice.notice import Notice


def get_bess_readings(
        source: BessReadingDataSource,
        start: Optional[datetime],
        end: Optional[datetime],
        file_path_resolver_func: Optional[Callable],
        db_engine: Optional,
        context: Optional[str],
) -> Tuple[pd.DataFrame, List[Notice]]:
    """
    This reads a data source and returns BESS readings in a dataframe- either CSVs from disk or directly from a
    database. Also returns a list of warnings.
    :param source: locates the data in either local files or a remote database
    :param start:
    :param end:
    :param file_path_resolver_func: A function that does any env var substitutions necessary for file paths
    :param db_engine: SQLAlchemy DB engine, as required
    :param context: a string that is added to notices to help the user understand what the data is about
    :return:
    """

    # logging.info(f"Reading data source '{source_str}'...")

    if source.flows_bess_readings_data_source:
        df = _get_flows_bess_readings(
            source=source.flows_bess_readings_data_source,
            start=start,
            end=end,
            db_engine=db_engine
        )
    elif source.csv_bess_readings_data_source:
        df = _get_csv_bess_readings(
            source=source.csv_bess_readings_data_source,
            start=start,
            end=end,
            file_path_resolver_func=file_path_resolver_func,
        )
    else:
        raise ValueError("Unknown source type")

    return df, sanity_checks(df, start, end, context)


def _get_flows_bess_readings(
        source: FlowsBessReadingsDataSource,
        start: datetime,
        end: datetime,
        db_engine
) -> pd.DataFrame:
    """
    Reads bess readings about the given meter from the mg_meter_readings table.
    """

    query = (
        f"SELECT time_b, device_id, soe_avg, target_power_avg "
        "FROM flows.mg_bess_readings_30m WHERE "
        f"time_b >= '{start.isoformat()}' AND "
        f"time_b < '{end.isoformat()}' AND "
        f"device_id = '{source.bess_id}' "
        f"order by time_b"
    )

    df = pd.read_sql(query, con=db_engine)

    df = df.rename(columns={"time_b": "time"})
    df["time"] = pd.to_datetime(df["time"], format="ISO8601")

    return df


def _get_csv_bess_readings(
    source: CSVBessReadingsDataSource,
    start: Optional[datetime],
    end: Optional[datetime],
    file_path_resolver_func: Optional[Callable],
) -> pd.DataFrame:

    df = get_csv_data_source(source, file_path_resolver_func)

    df["time"] = pd.to_datetime(df["utctime"], format="ISO8601")
    df = df.drop(columns=["utctime", "clocktime"])

    # Old CSV files have old naming - bring this up to date with the FlowsDB naming
    df = df.rename(columns={
        "deviceID": "device_id",
        "soeAvg": "soe_avg",
        "targetPowerAvg": "target_power_avg"
    })

    df = df[df["device_id"] == str(source.bess_id)]

    # Remove any data that is outside the time range of interest
    # TODO: only read in CSVs for the months that are required in the first place
    df = drop_extra_rows(df, start, end)

    return df

