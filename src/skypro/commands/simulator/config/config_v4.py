from dataclasses import field
from datetime import datetime
from typing import Dict

from marshmallow_dataclass import dataclass

from simt_common.jsonconfig.utility import name_in_json

from skypro.commands.simulator.config.config_common import Site, Strategy, ImbalanceDataSource, Rates


"""
This file contains configuration schema specific to V4
"""


@dataclass
class Timeframe:
    start: datetime
    end: datetime


class Output:
    csv_30min: str = name_in_json("csv30min")
    csv_summary: str = name_in_json("csvSummary")


@dataclass
class SimulationV4:
    output: Output
    timeframe: Timeframe
    site: Site
    strategy: Strategy
    imbalance_data_source: ImbalanceDataSource = name_in_json("imbalanceDataSource")
    rates: Rates


@dataclass
class ConfigV4:
    config_format_version: str = field(metadata={"data_key": "configFormatVersion"})
    sandbox: dict  # a space for the user to define YAML anchors which is not parsed/used by the program
    cases: Dict[str, SimulationV4]
