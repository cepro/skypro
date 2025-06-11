from datetime import datetime
from typing import Optional, Callable, Tuple, List

import pandas as pd

from skypro.common.config.data_source import MeterReadingDataSource, FlowsMeterReadingsDataSource, \
    CSVMeterReadingsDataSource
from skypro.common.data.utility import get_csv_data_source, drop_extra_rows, sanity_checks, convert_cols_to_str_type
from skypro.common.notice.notice import Notice


def get_meter_readings(
        source: MeterReadingDataSource,
        start: Optional[datetime],
        end: Optional[datetime],
        file_path_resolver_func: Optional[Callable],
        db_engine: Optional,
        context: Optional[str],
) -> Tuple[pd.DataFrame, List[Notice]]:
    """
    This reads a data source - either CSVs from disk or directly from a database - and returns meter readings in a dataframe alongside a list of warnings.
    :param source: locates the data in either local files or a remote database
    :param start: inclusive
    :param end: exclusive
    :param file_path_resolver_func: A function that does any env var substitutions necessary for file paths
    :param db_engine: SQLAlchemy DB engine, as required
    :param context: a string that is added to notices to help the user understand what the data is about
    :return:
    """

    if source.flows_meter_readings_data_source:
        df = _get_flows_meter_readings(
            source=source.flows_meter_readings_data_source,
            start=start,
            end=end,
            db_engine=db_engine
        )
    elif source.csv_meter_readings_data_source:
        df = _get_csv_meter_readings(
            source=source.csv_meter_readings_data_source,
            start=start,
            end=end,
            file_path_resolver_func=file_path_resolver_func,
        )
    else:
        raise ValueError("Unknown source type")

    return df, sanity_checks(df, start, end, context)


def _get_flows_meter_readings(
        source: FlowsMeterReadingsDataSource,
        start: datetime,
        end: datetime,
        db_engine,
) -> pd.DataFrame:
    """
    Pulls readings about the identified meter from the mg_meter_readings table.
    """
    query = (
        f"SELECT time_b, device_id, energy_imported_active_delta, energy_exported_active_delta, "
        "energy_imported_active_min, energy_exported_active_min "
        f"FROM get_meter_readings_5m(start_time => '{start.isoformat()}'::timestamptz, end_time => '{end.isoformat()}'::timestamptz, device_ids => ARRAY['{source.meter_id}'::uuid]) "
        f"order by time_b"
    )
    df = pd.read_sql(query, con=db_engine)

    # SQLAlchemy reads UUIDs as the formal UUID type, but we want them as strings for simplicity
    convert_cols_to_str_type(df, ["device_id"])

    df = df.rename(columns={"time_b": "time"})

    df["time"] = pd.to_datetime(df["time"], format="ISO8601")

    return df


def _get_csv_meter_readings(
    source: CSVMeterReadingsDataSource,
    start: Optional[datetime],
    end: Optional[datetime],
    file_path_resolver_func: Optional[Callable],
) -> pd.DataFrame:
    """
    Pulls readings about the identified meter from the given CSV files.
    """

    df = get_csv_data_source(source, file_path_resolver_func)

    df["time"] = pd.to_datetime(df["utctime"], format="ISO8601")
    df = df.drop(columns=["utctime", "clocktime"])

    # Old CSV files have old naming - bring this up to date with the FlowsDB naming
    df = df.rename(columns={
        "deviceID": "device_id",
        "energyImportedActiveDelta": "energy_imported_active_delta",
        "energyExportedActiveDelta": "energy_exported_active_delta",
        "energyImportedActiveMin": "energy_imported_active_min",
        "energyExportedActiveMin": "energy_exported_active_min"
    })

    df = df[df["device_id"] == str(source.meter_id)]

    # Remove any data that is outside the time range of interest
    # TODO: only read in CSVs for the months that are required in the first place
    df = drop_extra_rows(df, start, end)

    return df

