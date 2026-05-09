import copy
import os
from packaging.version import Version

import yaml
from skypro.common.cli_utils.cli_utils import substitute_vars
from skypro.common.config.path_field import PathField

from skypro.commands.simulator.config.config import Config, SimulationCase

"""
This module handles parsing of the YAML configuration file for the Simulation script.
Marshmallow (and marshmallow-dataclass) is used to validate and parse the YAML into the classes defined below.
"""


def parse_config(file_path: str, env_vars: dict) -> Config:
    # Read in the main config file
    with open(file_path) as config_file:
        # Here we parse the config file as YAML, which is a superset of JSON so allows us to parse JSON files as well
        config_dict = yaml.safe_load(config_file)

        if "configFormatVersion" not in config_dict:
            raise ValueError("Missing configFormatVersion from configuration file.")

        version = Version(config_dict["configFormatVersion"])

        # Drop scenarios annotated `enabled: false` before strict schema validation.
        # Lets externally-managed scenarios (e.g. Monte-Carlo runs whose outputs are
        # produced outside skypro) sit alongside live ones in the same simulate.yaml.
        sims = config_dict.get("simulations") or {}
        for name in list(sims):
            cfg = sims[name]
            if isinstance(cfg, dict) and cfg.get("enabled") is False:
                del sims[name]

        # Set up the variables that are substituted into file paths
        PathField.vars_for_substitution = env_vars
        if version.major == 4 and "variables" in config_dict:
            # In config v4 there may be variables defined at the file level as well as env vars
            file_vars = config_dict["variables"]
            # Allow the file-level variables to contain env level variables, which we resolve here:
            for name, value in file_vars.items():
                file_vars[name] = substitute_vars(value, env_vars)
            PathField.vars_for_substitution = env_vars | file_vars

        config = Config.Schema().load(config_dict)

        # Fan out any sim with a `finals: {name: Rates, ...}` block into one expanded sim per variant,
        # sharing dispatch and live rates but settling against its own final rates. The expanded
        # sim_name is `<orig>.<variant>`. Each variant's CSV paths are de-clashed: paths containing
        # `$_SIM_NAME` get the variant suffix via the substitution loop below; hardcoded paths get
        # `.<variant>` inserted before the extension.
        config.simulations = _expand_multi_final_simulations(config.simulations)

        if version.major == 4:
            # There is also a special variable `$CASE_NAME` which should resolve to the name of the case, which can't
            # be handled with the above mechanism... manually go through a substitute that here... this isn't a
            # particularly elegant mechanism. A better way may be to somehow integrate it into the PathField class, or
            # to just do all the substitutions here but in a generic way with 'deep reflection' of the config structure
            # looking for `PathField` types.
            sim_config: SimulationCase
            for sim_name, sim_config in config.simulations.items():
                case_name_dict = {"_SIM_NAME": sim_name}
                if sim_config.output:
                    if sim_config.output.simulation:
                        sim_config.output.simulation.csv = substitute_vars(sim_config.output.simulation.csv, case_name_dict)
                    if sim_config.output.summary:
                        sim_config.output.summary.csv = substitute_vars(sim_config.output.summary.csv, case_name_dict)

    return config


def _expand_multi_final_simulations(simulations):
    """Replace each multi-`finals` simulation with N single-`final` simulations.

    Sim name becomes `<orig>.<variant>`; CSV paths get a `.<variant>` suffix when the original
    path doesn't already use `$_SIM_NAME` (which the later substitution loop will rewrite). Order
    is preserved: variants land in the position of their original sim, in declaration order.
    """
    expanded = {}
    for sim_name, sim_config in simulations.items():
        finals = sim_config.rates.finals
        if finals is None:
            expanded[sim_name] = sim_config
            continue
        for variant_name, variant_final in finals.items():
            expanded_name = f"{sim_name}.{variant_name}"
            expanded_sim = copy.deepcopy(sim_config)
            expanded_sim.rates.final = variant_final
            expanded_sim.rates.finals = None
            if expanded_sim.output is not None:
                if expanded_sim.output.simulation is not None:
                    expanded_sim.output.simulation.csv = _add_variant_suffix(
                        expanded_sim.output.simulation.csv, variant_name
                    )
                if expanded_sim.output.summary is not None:
                    expanded_sim.output.summary.csv = _add_variant_suffix(
                        expanded_sim.output.summary.csv, variant_name
                    )
            expanded[expanded_name] = expanded_sim
    return expanded


def _add_variant_suffix(csv_path: str, variant_name: str) -> str:
    """Insert `.<variant>` before the extension when the path doesn't already use `$_SIM_NAME`.

    Paths using `$_SIM_NAME` get the variant suffix naturally because the sim_name is now
    `<orig>.<variant>` and the substitution loop runs after fan-out — leave those untouched.
    Hardcoded paths must be de-clashed manually so two variants don't overwrite the same file.
    """
    if "$_SIM_NAME" in csv_path:
        return csv_path
    base, ext = os.path.splitext(csv_path)
    return f"{base}.{variant_name}{ext}"
