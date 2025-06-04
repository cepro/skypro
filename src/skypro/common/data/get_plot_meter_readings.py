from datetime import datetime
from typing import Optional, Callable, List, Tuple

import pandas as pd

from skypro.common.config.data_source import PlotMeterReadingDataSource
from skypro.common.config.data_source_csv import CSVPlotMeterReadingsDataSource
from skypro.common.config.data_source_flows import FlowsPlotMeterReadingsDataSource
from skypro.common.data.utility import get_csv_data_source, drop_extra_rows, sanity_checks
from skypro.common.notice.notice import Notice


def get_plot_meter_readings(
        source: PlotMeterReadingDataSource,
        start: Optional[datetime],
        end: Optional[datetime],
        file_path_resolver_func: Optional[Callable],
        db_engine: Optional,
        context: Optional[str],
) -> Tuple[pd.DataFrame, List[Notice]]:
    """
    This reads a data source and returns plot-level meter readings in a dataframe- either CSVs from disk or directly
    from a database
    :param source: locates the data in either local files or a remote database
    :param start:
    :param end:
    :param file_path_resolver_func: A function that does any env var substitutions necessary for file paths
    :param db_engine: SQLAlchemy DB engine, as required
    :param context: a string that is added to notices to help the user understand what the data is about
    :return:
    """

    # logging.info(f"Reading data source '{source_str}'...")

    if source.flows_plot_meter_readings_data_source:
        df = _get_flows_plot_meter_readings(
            source=source.flows_plot_meter_readings_data_source,
            start=start,
            end=end,
            db_engine=db_engine
        )
    elif source.csv_plot_meter_readings_data_source:
        df = _get_csv_plot_meter_readings(
            source=source.csv_plot_meter_readings_data_source,
            start=start,
            end=end,
            file_path_resolver_func=file_path_resolver_func,
        )
    else:
        raise ValueError("Unknown source type")

    return df, sanity_checks(df, start, end, context)


def _get_flows_plot_meter_readings(
        source: FlowsPlotMeterReadingsDataSource,
        start: datetime,
        end: datetime,
        db_engine
) -> pd.DataFrame:
    """
    Reads Emlite plot meter readings that are on the given feeders.
    """

    feeder_id_list_str = ', '.join(f"'{str(u)}'::uuid" for u in source.feeder_ids)

    query = (
        "SELECT "
        "rih.timestamp as timestamp, "
        "fr.id as feeder_id, "
        "mr.register_id as register_id, "
        "mr2.nature as nature, "
        "rih.kwh AS kwh "
        "FROM flows.register_interval_hh rih "
        "JOIN flows.meter_registers mr ON mr.register_id = rih.register_id "
        "JOIN flows.meter_registers mr2 on mr.register_id = mr2.register_id "
        "JOIN flows.service_head_meter shm on shm.meter = mr2.meter_id "
        "JOIN flows.service_head_registry shr on shr.id = shm.service_head "
        "JOIN flows.feeder_registry fr on fr.id = shr.feeder "
        f"WHERE rih.timestamp >= '{start.isoformat()}' "
        f"AND rih.timestamp < '{end.isoformat()}' "
        f"AND fr.id = ANY(ARRAY[{feeder_id_list_str}]) "
        f"order by rih.timestamp, fr.id, mr.register_id"
    )
    # TODO: we may want a more rigourous check for meters that are  missing ALL data for the enitre month?

    df = pd.read_sql(query, con=db_engine)

    df = df.rename(columns={"timestamp": "time"})
    df["time"] = pd.to_datetime(df["time"], format="ISO8601")

    return df


def _get_csv_plot_meter_readings(
    source: CSVPlotMeterReadingsDataSource,
    start: Optional[datetime],
    end: Optional[datetime],
    file_path_resolver_func: Optional[Callable],
) -> pd.DataFrame:

    df = get_csv_data_source(source, file_path_resolver_func)

    df["time"] = pd.to_datetime(df["utctime"], format="ISO8601")
    df = df.drop(columns=["utctime", "clocktime"])

    # Old CSV files have old naming - bring this up to date with the FlowsDB naming
    df = df.rename(columns={
        "feederID": "feeder_id",
        "registerID": "register_id",
        "nature": "nature",
        "energyImportedActiveDelta": "kwh",
    })

    df = df[df["feeder_id"].isin([str(id) for id in source.feeder_ids])]

    # Remove any data that is outside the time range of interest
    # TODO: only read in CSVs for the months that are required in the first place
    df = drop_extra_rows(df, start, end)

    return df

