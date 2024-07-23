from dataclasses import field
from datetime import datetime
from typing import Dict, Optional

from marshmallow_dataclass import dataclass

from simt_common.jsonconfig.utility import name_in_json

from skypro.commands.simulator.config.config_common import Site, Strategy, ImbalanceDataSource, Rates, PathType

"""
This file contains configuration schema specific to V4
"""


@dataclass
class TimeFrame:
    start: datetime
    end: datetime


@dataclass
class OutputSummary:
    csv: PathType


@dataclass
class OutputSimulation:
    csv: PathType
    aggregate: Optional[str]
    rate_detail: Optional[str] = name_in_json("rateDetail")


@dataclass
class OutputLoad:
    csv: PathType
    aggregate: Optional[str]


@dataclass
class Output:
    summary: Optional[OutputSummary]
    simulation: Optional[OutputSimulation]
    load: Optional[OutputLoad]


@dataclass
class SimulationV4:
    output: Output
    timeframe: TimeFrame = name_in_json("timeFrame")
    site: Site
    strategy: Strategy
    imbalance_data_source: ImbalanceDataSource = name_in_json("imbalanceDataSource")
    rates: Rates

    @property
    def start(self) -> datetime:
        return self.timeframe.start

    @property
    def end(self) -> datetime:
        return self.timeframe.end


@dataclass
class ConfigV4:
    config_format_version: str = field(metadata={"data_key": "configFormatVersion"})
    sandbox: Optional[dict]  # a space for the user to define YAML anchors which is not parsed/used by the program
    cases: Dict[str, SimulationV4]
