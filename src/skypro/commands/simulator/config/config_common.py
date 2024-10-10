import os
from dataclasses import field
from datetime import timedelta
from typing import List, Optional, Annotated

import numpy as np
from marshmallow import fields
from marshmallow_dataclass import dataclass
from simt_common.cli_utils.cliutils import substitute_vars

from simt_common.jsonconfig.utility import name_in_json, enforce_one_option
from simt_common.jsonconfig.dayed_period import DayedPeriodType
from skypro.commands.simulator.config.curve import (CurveType)


"""
This file contains configuration schema that is used for both V3 and V4 config
"""


class PathField(fields.Field):
    """
    This Marshmallow field type is used to deserialize file paths. It expands the local user tilde symbol and also
    substitutes variables (in the $VAR_NAME format).
    The variables must be first set on the `vars_for_substitution` class variable before deserializing.
    """

    vars_for_substitution = {}  # class variable defines any variables for substitution into the paths

    def _serialize(self, value, attr, obj, **kwargs):
        raise ValueError("Serialization not yet defined")

    def _deserialize(self, value, attr, data, **kwargs):
        # Expand any `~/` syntax and $ENV_VARS that are used
        return os.path.expanduser(substitute_vars(value, PathField.vars_for_substitution))


# The marshmallow_dataclass library doesn't use the PathField directly, but instead needs a Type defining:
PathType = Annotated[str, PathField]


@dataclass
class GridConnection:
    import_limit: float = name_in_json("importLimit")
    export_limit: float = name_in_json("exportLimit")


@dataclass
class Profile:
    # Tag is an optional name to assign to the profile. The advantage of this over the name being a dict key is that
    # arrays preserve order and the order of the load profiles may become important down the line.
    tag: Optional[str] = name_in_json("tag")

    profile_dir: Optional[PathType] = name_in_json("profileDir")
    profile_csv: Optional[PathType] = name_in_json("profileCsv")

    energy_cols: Optional[str] = name_in_json("energyCols")

    # These are deprecated - ClockTime will be automatically used as Europe/London if UTCTime can't be found
    parse_clock_time: Optional[bool] = name_in_json("parseClockTime")
    clock_time_zone: Optional[str] = name_in_json("clockTimeZone")

    scaling_factor: Optional[float] = name_in_json("scalingFactor")
    profiled_num_plots: Optional[float] = name_in_json("profiledNumPlots")
    scaled_num_plots: Optional[float] = name_in_json("scaledNumPlots")
    profiled_size_kwp: Optional[float] = name_in_json("profiledSizeKwp")
    scaled_size_kwp: Optional[float] = name_in_json("scaledSizeKwp")

    def __post_init__(self):
        enforce_one_option([self.profile_dir, self.profile_csv], "'profileDir' or 'profileCsv'")

        # There are three ways of setting the scaling factor: by 'kwp' fields; by 'num plot' fields; or by
        # explicitly setting the 'scalingFactor'. This is partly to support older configurations.
        if self.scaling_factor is None:
            if (self.profiled_num_plots is not None) and (self.scaled_num_plots is not None):
                self.scaling_factor = self.scaled_num_plots / self.profiled_num_plots

            if (self.profiled_size_kwp is not None) and (self.scaled_size_kwp is not None):
                if self.scaling_factor is not None:
                    raise ValueError("Profile can be scaled by either 'num plots' or 'kwp', but not both.")
                self.scaling_factor = self.scaled_size_kwp / self.profiled_size_kwp

            if self.scaling_factor is None:
                self.scaling_factor = 1


@dataclass
class Solar:
    constant: Optional[float] = field(default=np.nan)
    profile: Optional[Profile] = field(default=None)

    def __post_init__(self):
        enforce_one_option([self.constant, self.profile], "'constant' or 'profile' solar")


@dataclass
class Load:
    constant: Optional[float] = field(default=np.nan)
    profile: Optional[Profile] = field(default=None)
    profiles: Optional[List[Profile]] = field(default=None)

    def __post_init__(self):
        enforce_one_option(
            [self.constant, self.profile, self.profiles],
            "'constant', 'profile' or 'profiles' load"
        )


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
    price_dir: PathType = name_in_json("priceDir")
    volume_dir: PathType = name_in_json("volumeDir")


@dataclass
class RatesFiles:
    bess_charge_from_solar: List[PathType] = name_in_json("solarToBatt")
    bess_charge_from_grid: List[PathType] = name_in_json("gridToBatt")
    bess_discharge_to_grid: List[PathType] = name_in_json("battToGrid")
    bess_discharge_to_load: List[PathType] = name_in_json("battToLoad")
    solar_to_grid: List[PathType] = name_in_json("solarToGrid")
    solar_to_load: List[PathType] = name_in_json("solarToLoad")
    load_from_grid: List[PathType] = name_in_json("gridToLoad")


@dataclass
class Rates:
    """
    Note that this class just holds the paths to the rates/supply point configuration files. The actual parsing of the
    contents of these files is done in the common.config.rates module.
    """
    supply_points_config_file: PathType = name_in_json("supplyPointsConfigFile")
    imbalance_data_source: ImbalanceDataSource = name_in_json("imbalanceDataSource")
    files: RatesFiles


@dataclass
class Approach:
    to_soe: float = name_in_json("toSoe")
    assumed_charge_power: float = name_in_json("assumedChargePower")
    encourage_charge_duration_factor: float = name_in_json("encourageChargeDurationFactor")
    force_charge_duration_factor: float = name_in_json("forceChargeDurationFactor")
    charge_cushion: timedelta = field(metadata={"precision": "minutes", "data_key": "chargeCushionMins"})


@dataclass
class PeakDynamic:
    prioritise_residual_load: bool = name_in_json("prioritiseResidualLoad")


@dataclass
class Peak:
    period: DayedPeriodType = name_in_json("period")
    approach: Approach = name_in_json("approach")
    dynamic: Optional[PeakDynamic] = name_in_json("dynamic")


@dataclass
class MicrogridLocalControl:
    import_avoidance: bool = name_in_json("importAvoidance")
    export_avoidance: bool = name_in_json("exportAvoidance")


@dataclass
class MicrogridImbalanceControl:
    discharge_into_load_when_short: bool = name_in_json("dischargeIntoLoadWhenShort")
    charge_from_solar_when_long: bool = name_in_json("chargeFromSolarWhenLong")
    niv_cutoff_for_system_state_assumption: float = field(metadata={"data_key": "nivCutoffForSystemStateAssumption", "allow_nan": True})


@dataclass
class Microgrid:
    local_control: Optional[MicrogridLocalControl] = name_in_json("localControl")
    imbalance_control: Optional[MicrogridImbalanceControl] = name_in_json("imbalanceControl")


@dataclass
class PriceCurveAlgo:
    microgrid: Optional[Microgrid] = name_in_json("microgrid")
    peak: Peak = name_in_json("peak")
    niv_chase_periods: List[NivPeriod] = name_in_json("nivChasePeriods")


@dataclass
class SpreadAlgoFixedAction:
    charge_power: float = name_in_json("chargePower")
    discharge_power: float = name_in_json("dischargePower")


@dataclass
class SpreadAlgo:
    min_spread: float = name_in_json("minSpread")
    recent_pricing_span: int = name_in_json("recentPricingSpan")
    niv_cutoff_for_system_state_assumption: float = field(metadata={"data_key": "nivCutoffForSystemStateAssumption", "allow_nan": True})
    fixed_action: SpreadAlgoFixedAction = name_in_json("fixedAction")
    microgrid: Optional[Microgrid] = name_in_json("microgrid")
    peak: Peak = name_in_json("peak")


@dataclass
class OptimiserBlocks:
    """Defines how the whole simulation period is split into smaller duration optimisations that are stacked on top of
    each other, and any settings that are applied to each of those smaller duration optimisations."""
    max_avg_cycles_per_day: float = name_in_json("maxAvgCyclesPerDay")
    max_optimal_tolerance: Optional[float] = field(metadata={"data_key": "maxOptimalTolerance"}, default=0.02)
    max_computation_secs: Optional[int] = field(metadata={"data_key": "maxComputationSecs"}, default=10)
    duration_days: Optional[int] = field(metadata={"data_key": "durationDays"}, default=3)
    used_duration_days: Optional[int] = field(metadata={"data_key": "usedDurationDays"}, default=5)

    def __post_init__(self):
        if self.duration_days <= 0 or self.used_duration_days <= 0:
            raise ValueError("both usedDurationDays and durationDays must be greater than 0.")
        if self.used_duration_days > self.duration_days:
            raise ValueError("usedDurationDays must not be larger than durationDays.")


@dataclass
class Optimiser:
    blocks: OptimiserBlocks


@dataclass
class Strategy:
    price_curve_algo: Optional[PriceCurveAlgo] = name_in_json("priceCurveAlgo")
    spread_algo: Optional[SpreadAlgo] = name_in_json("spreadAlgo")
    optimiser: Optional[Optimiser] = name_in_json("perfectHindsightOptimiser")

    def __post_init__(self):
        enforce_one_option([self.price_curve_algo, self.spread_algo, self.optimiser], "'priceCurveAlgo', 'spreadAlgo', 'perfectHindsightOptimiser'")
