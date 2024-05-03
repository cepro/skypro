from dataclasses import field
from datetime import datetime
from typing import List, Optional

import numpy as np
from marshmallow_dataclass import dataclass

from simt_common.jsonconfig.utility import name_in_json, enforce_one_option
from simt_common.jsonconfig.dayed_period import DayedPeriodType
from skypro.commands.simulator.config.curve import (CurveType)

"""
This module handles parsing of the JSON configuration file for the Simulation script.
Marshmallow (and marshmallow-dataclass) is used to validate and parse the JSON into the classes defined below.
"""


MAJOR_VERSION = 2
MINOR_VERSION = 0
PATCH_VERSION = 0


@dataclass
class GridConnection:
    import_limit: float = name_in_json("importLimit")
    export_limit: float = name_in_json("exportLimit")


@dataclass
class Site:
    grid_connection: GridConnection = field(metadata={"data_key": "gridConnection"})


@dataclass
class SolarProfile:
    profile_dir: str = name_in_json("profileDir")
    profiled_size_kwp: float = name_in_json("profiledSizeKwp")
    scaled_size_kwp: float = name_in_json("scaledSizeKwp")


@dataclass
class Solar:
    constant: Optional[float] = field(default=np.NaN)
    profile: Optional[SolarProfile] = field(default=None)

    def __post_init__(self):
        enforce_one_option([self.constant, self.profile], "'constant' or 'profile' solar")


@dataclass
class LoadProfile:
    profile_dir: str = name_in_json("profileDir")
    profiled_num_plots: float = name_in_json("profiledNumPlots")
    scaled_num_plots: float = name_in_json("scaledNumPlots")


@dataclass
class Load:
    constant: Optional[float] = field(default=np.NaN)
    profile: Optional[LoadProfile] = field(default=None)

    def __post_init__(self):
        enforce_one_option([self.constant, self.profile], "'constant' or 'profile' load")


@dataclass
class Bess:
    energy_capacity: float = name_in_json("energyCapacity")
    nameplate_power: float = name_in_json("nameplatePower")
    charge_efficiency: float = name_in_json("chargeEfficiency")


@dataclass
class Site:
    grid_connection: GridConnection = name_in_json("gridConnection")
    solar: Solar
    load: Load
    bess: Bess


@dataclass
class Niv:
    """
    The configuration to do NIV chasing.
    """
    charge_curve: CurveType = name_in_json("chargeCurve")
    discharge_curve: CurveType = name_in_json("dischargeCurve")
    curve_shift_long: float = name_in_json("curveShiftLong")
    curve_shift_short: float = name_in_json("curveShiftShort")
    volume_cutoff_for_prediction: float = field(metadata={"data_key": "volumeCutoffForPrediction", "allow_nan": True})


@dataclass
class NivPeriod:
    """
    Represents a NIV chasing configuration for a particular period of time.
    """
    period: DayedPeriodType
    niv: Niv


@dataclass
class ImbalanceDataSource:
    price_dir: str = name_in_json("priceDir")
    volume_dir: str = name_in_json("volumeDir")


@dataclass
class Rates:
    """
    Note that this class just holds the paths to the rates/supply point configuration files. The actual parsing of the
    contents of these files is done in the common.config.rates module.
    """
    supply_points_config_file: str = name_in_json("supplyPointsConfigFile")
    rates_config_files: List[str] = name_in_json("ratesConfigFiles")


@dataclass
class PriceCurveAlgo:
    do_full_discharge_when_export_rate_applies: str = name_in_json("doFullDischargeWhenExportRateApplies")
    niv_chase_periods: List[NivPeriod] = name_in_json("nivChasePeriods")


@dataclass
class Strategy:
    price_curve_algo: PriceCurveAlgo = name_in_json("priceCurveAlgo")

    def __post_init__(self):
        # There is currently only one option - but this is here as a placeholder for when other algos are available
        enforce_one_option([self.price_curve_algo], "'priceCurveAlgo'")

@dataclass
class Simulation:
    start: datetime
    end: datetime
    site: Site
    strategy: Strategy
    imbalance_data_source: ImbalanceDataSource = name_in_json("imbalanceDataSource")
    rates: Rates


@dataclass
class Config:
    config_format_version: str = field(metadata={"data_key": "configFormatVersion"})
    simulation: Simulation

    def __post_init__(self):
        version_numbers = self.config_format_version.split(".")
        if len(version_numbers) != 3:
            raise ValueError("Config format version number must be in the semver format MAJOR.MINOR.PATCH")

        major = version_numbers[0]
        # minor = version_numbers[1]
        # patch = version_numbers[2]

        if major != str(MAJOR_VERSION):
            raise ValueError(f"Config format major version number must be {MAJOR_VERSION}.")


def parse_config(file_path: str) -> Config:
    # Read in the main config file
    with open(file_path) as config_file:
        config_str = config_file.read()
        config = Config.Schema().loads(config_str)

    return config


def get_relevant_niv_config(niv_periods: List[NivPeriod], t: datetime) -> NivPeriod:
    """
    Returns the first NivPeriod instance that corresponds with the given time.
    """
    for niv_period in niv_periods:
        if niv_period.period.contains(t):
            return niv_period
    raise ValueError(f"No niv chase configuration matches the time '{t}'")