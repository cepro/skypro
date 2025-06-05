from typing import Dict, Optional
from uuid import UUID

import yaml
from marshmallow_dataclass import dataclass
from skypro.common.config.bill_match import BillMatchLineItem
from skypro.common.config.path_field import PathField, PathType
from skypro.common.config.rates_dataclasses import Rates
from skypro.common.config.utility import field_with_opts
from skypro.common.config.data_source import MeterReadingDataSource, PlotMeterReadingDataSource, BessReadingDataSource

"""
This file contains configuration schema that is used for reporting
"""


@dataclass
class MicrogridMeter:
    data_source: MeterReadingDataSource = field_with_opts(key="dataSource")


@dataclass
class MicrogridFeederMeter:
    data_source: MeterReadingDataSource = field_with_opts(key="dataSource")
    feeder_id: UUID = field_with_opts(key="feederId")


@dataclass
class MicrogridMeters:
    bess_inverter: MicrogridMeter = field_with_opts(key="bessInverter")
    main_incomer: MicrogridMeter = field_with_opts(key="mainIncomer")
    ev_charger: MicrogridMeter = field_with_opts(key="evCharger")
    feeder_1: MicrogridFeederMeter = field_with_opts(key="feeder1")
    feeder_2: MicrogridFeederMeter = field_with_opts(key="feeder2")


@dataclass
class PlotMeters:
    data_source: PlotMeterReadingDataSource = field_with_opts(key="dataSource")


@dataclass
class Metering:
    plot_meters: Optional[PlotMeters] = field_with_opts(key="plotMeters")
    microgrid_meters: MicrogridMeters = field_with_opts(key="microgridMeters")


@dataclass
class Bess:
    energy_capacity: float = field_with_opts(key="energyCapacity")
    data_source: BessReadingDataSource = field_with_opts(key="dataSource")


@dataclass
class GridConnection:
    import_capacity: float = field_with_opts(key="importCapacity")
    export_capacity: float = field_with_opts(key="exportCapacity")


@dataclass
class SpreadScenario:
    import_duration: float = field_with_opts(key="importDuration")
    export_duration: float = field_with_opts(key="exportDuration")
    highlight: bool


@dataclass
class BillMatchImportOrExport:
    line_items: Dict[str, BillMatchLineItem] = field_with_opts(key="lineItems")


@dataclass
class BillMatch:
    import_direction: Optional[BillMatchImportOrExport] = field_with_opts(key="import")
    export_direction: Optional[BillMatchImportOrExport] = field_with_opts(key="export")


@dataclass
class Reporting:
    metering: Metering
    bess: Bess
    grid_connection: GridConnection = field_with_opts(key="gridConnection")
    spread_scenarios: Optional[Dict[str, SpreadScenario]] = field_with_opts(key="spreadScenarios")
    bill_match: BillMatch = field_with_opts(key="billMatch")
    profiles_save_dir: PathType = field_with_opts(key="profilesSaveDir")
    rates: Rates


@dataclass
class Config:
    reporting: Reporting


def parse_config(file_path: str, env_vars: dict) -> Config:

    # Read in the main config file
    with open(file_path) as config_file:
        # Here we parse the config file as YAML, which is a superset of JSON so allows us to parse JSON files as well
        config_dict = yaml.safe_load(config_file)

        # Set up the variables that are substituted into file paths
        PathField.vars_for_substitution = env_vars

        config = Config.Schema().load(config_dict)

    return config
