from datetime import datetime
from typing import List
from packaging.version import Version

import yaml
from simt_common.cli_utils.cliutils import substitute_vars

from skypro.commands.simulator.config.config_common import NivPeriod, PathField
from skypro.commands.simulator.config.config_v3 import ConfigV3
from skypro.commands.simulator.config.config_v4 import ConfigV4, SimulationCaseV4

"""
This module handles parsing of the JSON or YAML configuration file for the Simulation script.
Marshmallow (and marshmallow-dataclass) is used to validate and parse the JSON into the classes defined below.
"""


def parse_config(file_path: str, env_vars: dict) -> ConfigV3 | ConfigV4:
    # Read in the main config file
    with open(file_path) as config_file:
        # Here we parse the config file as YAML, which is a superset of JSON so allows us to parse JSON files as well
        config_dict = yaml.safe_load(config_file)

        if "configFormatVersion" not in config_dict:
            raise ValueError("Missing configFormatVersion from configuration file.")

        version = Version(config_dict["configFormatVersion"])

        # Set up the variables that are substituted into file paths
        PathField.vars_for_substitution = env_vars
        if version.major == 4 and "variables" in config_dict:
            # In config v4 there may be variables defined at the file level as well as env vars
            file_vars = config_dict["variables"]
            # Allow the file-level variables to contain env level variables, which we resolve here:
            for name, value in file_vars.items():
                file_vars[name] = substitute_vars(value, env_vars)
            PathField.vars_for_substitution = env_vars | file_vars

        # Parse the configs
        if version.major == 3:
            config = ConfigV3.Schema().load(config_dict)
        elif version.major == 4:
            config = ConfigV4.Schema().load(config_dict)
        else:
            raise ValueError(f"Unknown config version: {config_dict['configFormatVersion']}")

        if version.major == 4:
            # There is also a special variable `$CASE_NAME` which should resolve to the name of the case, which can't
            # be handled with the above mechanism... manually go through a substitute that here... this isn't a
            # particularly elegant mechanism. A better way may be to somehow integrate it into the PathField class, or
            # to just do all the substitutions here but in a generic way with 'deep reflection' of the config structure
            # looking for `PathField` types.
            sim_config: SimulationCaseV4
            for sim_name, sim_config in config.simulations.items():
                case_name_dict = {"_SIM_NAME": sim_name}
                if sim_config.output:
                    if sim_config.output.simulation:
                        sim_config.output.simulation.csv = substitute_vars(sim_config.output.simulation.csv, case_name_dict)
                    if sim_config.output.summary:
                        sim_config.output.summary.csv = substitute_vars(sim_config.output.summary.csv, case_name_dict)
                    if sim_config.output.load:
                        sim_config.output.load.csv = substitute_vars(sim_config.output.load.csv, case_name_dict)

    return config


def get_relevant_niv_config(niv_periods: List[NivPeriod], t: datetime) -> NivPeriod:
    """
    Returns the first NivPeriod instance that corresponds with the given time.
    """
    for niv_period in niv_periods:
        if niv_period.period.contains(t):
            return niv_period
    raise ValueError(f"No niv chase configuration matches the time '{t}'")
