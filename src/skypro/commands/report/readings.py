import logging
from dataclasses import dataclass
from typing import Tuple, List

import pandas as pd

from skypro.common.config.data_source import MeterReadingDataSource
from skypro.common.data.get_bess_readings import get_bess_readings
from skypro.common.data.get_meter_readings import get_meter_readings
from skypro.common.data.get_plot_meter_readings import get_plot_meter_readings
from skypro.common.notice.notice import Notice


@dataclass
class AllReadings:
    """
    This structure holds the various readings that are required for reporting.
    """
    bess: pd.DataFrame
    bess_meter: pd.DataFrame
    ev_meter: pd.DataFrame
    feeder1_meter: pd.DataFrame
    feeder2_meter: pd.DataFrame
    grid_meter: pd.DataFrame
    plot_meter: pd.DataFrame


def get_readings(config, time_index, flows_db_url, file_path_resolver_func) -> Tuple[AllReadings, List[Notice]]:
    """
    Pulls in the various meter and BESS readings (these will be resolved either from DB or local CSV files)
    """
    notices: List[Notice] = []

    # This just wraps `get_meter_readings` for convenience
    def _get_meter_readings_wrapped(source: MeterReadingDataSource, context: str) -> pd.DataFrame:
        logging.info(f"Loading {context}...")
        data_df, new_notices_2 = get_meter_readings(
            source=source,
            start=time_index[0],
            end=time_index[-1],
            file_path_resolver_func=file_path_resolver_func,
            db_engine=flows_db_url,
            context=context
        )
        notices.extend(new_notices_2)

        data_df = data_df.set_index("time")
        return data_df

    mg_meter_config = config.reporting.metering.microgrid_meters  # For convenience
    bess_meter_readings = _get_meter_readings_wrapped(mg_meter_config.bess_inverter.data_source, "bess meter readings")
    grid_meter_readings = _get_meter_readings_wrapped(mg_meter_config.main_incomer.data_source, "grid meter readings")
    feeder1_meter_readings = _get_meter_readings_wrapped(mg_meter_config.feeder_1.data_source, "feeder1 meter readings")
    feeder2_meter_readings = _get_meter_readings_wrapped(mg_meter_config.feeder_2.data_source, "feeder2 meter readings")
    ev_meter_readings = _get_meter_readings_wrapped(mg_meter_config.ev_charger.data_source, "ev charger meter readings")

    logging.info("Loading plot meter readings...")
    plot_meter_readings, new_notices = get_plot_meter_readings(
        source=config.reporting.metering.plot_meters.data_source,
        start=time_index[0],
        end=time_index[-1],
        file_path_resolver_func=file_path_resolver_func,
        db_engine=flows_db_url,
        context="plot meter readings"
    )
    notices.extend(new_notices)
    plot_meter_readings = plot_meter_readings.set_index("time")
    logging.info("Loading bess readings...")
    bess_readings, new_notices = get_bess_readings(
        source=config.reporting.bess.data_source,
        start=time_index[0],
        end=time_index[-1],
        file_path_resolver_func=file_path_resolver_func,
        db_engine=flows_db_url,
        context="bess readings"
    )
    notices.extend(new_notices)
    bess_readings = bess_readings.set_index("time")

    all_readings = AllReadings(
        bess=bess_readings,
        bess_meter=bess_meter_readings,
        ev_meter=ev_meter_readings,
        feeder1_meter=feeder1_meter_readings,
        feeder2_meter=feeder2_meter_readings,
        grid_meter=grid_meter_readings,
        plot_meter=plot_meter_readings,
    )
    return all_readings, notices
