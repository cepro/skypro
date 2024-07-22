from datetime import datetime
from typing import List
from packaging.version import Version

import yaml

from skypro.commands.simulator.config.config_common import NivPeriod
from skypro.commands.simulator.config.config_v3 import ConfigV3
from skypro.commands.simulator.config.config_v4 import ConfigV4

"""
This module handles parsing of the JSON or YAML configuration file for the Simulation script.
Marshmallow (and marshmallow-dataclass) is used to validate and parse the JSON into the classes defined below.
"""


def parse_config(file_path: str) -> ConfigV3 | ConfigV4:
    # Read in the main config file
    with open(file_path) as config_file:
        # Here we parse the config file as YAML, which is a superset of JSON so allows us to parse JSON files as well
        config_dict = yaml.safe_load(config_file)

        if "configFormatVersion" not in config_dict:
            raise ValueError("Missing configFormatVersion from configuration file.")

        version = Version(config_dict["configFormatVersion"])

        if version.major == 3:
            config = ConfigV3.Schema().load(config_dict)
        elif version.major == 4:
            config = ConfigV4.Schema().load(config_dict)
        else:
            raise ValueError(f"Unknown config version: {config_dict['configFormatVersion']}")

    return config


def get_relevant_niv_config(niv_periods: List[NivPeriod], t: datetime) -> NivPeriod:
    """
    Returns the first NivPeriod instance that corresponds with the given time.
    """
    for niv_period in niv_periods:
        if niv_period.period.contains(t):
            return niv_period
    raise ValueError(f"No niv chase configuration matches the time '{t}'")