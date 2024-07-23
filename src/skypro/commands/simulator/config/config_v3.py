from dataclasses import field
from datetime import datetime

from marshmallow_dataclass import dataclass

from simt_common.jsonconfig.utility import name_in_json

from skypro.commands.simulator.config.config_common import Strategy, ImbalanceDataSource, Rates, Site

"""
This file contains configuration schema specific to V3
"""


@dataclass
class SimulationV3:
    start: datetime
    end: datetime
    site: Site
    strategy: Strategy
    imbalance_data_source: ImbalanceDataSource = name_in_json("imbalanceDataSource")
    rates: Rates


@dataclass
class ConfigV3:
    config_format_version: str = field(metadata={"data_key": "configFormatVersion"})
    simulation: SimulationV3
