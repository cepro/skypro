from dataclasses import field
from datetime import timedelta
from typing import List, Optional

from marshmallow_dataclass import dataclass
from simt_common.data.config import DataSource
from simt_common.rates.parse_config.dayed_period import DayedPeriodType

from skypro.commands.simulator.config.curve import (CurveType)
from skypro.commands.simulator.config.path_field import PathType
from skypro.commands.simulator.config.utility import name_in_json, enforce_one_option

"""
This file contains configuration schema that is used for both V3 and V4 config
"""


@dataclass
class GridConnection:
    import_limit: float = name_in_json("importLimit")
    export_limit: float = name_in_json("exportLimit")


@dataclass
class Profile:
    # Tag is an optional name to assign to the profile. The advantage of this over the name being a dict key is that
    # arrays preserve order and the order of the load profiles may become important down the line.
    tag: Optional[str] = name_in_json("tag")

    source: DataSource

    energy_cols: Optional[str] = name_in_json("energyCols")

    scaling_factor: Optional[float] = name_in_json("scalingFactor")
    profiled_num_plots: Optional[float] = name_in_json("profiledNumPlots")
    scaled_num_plots: Optional[float] = name_in_json("scaledNumPlots")
    profiled_size_kwp: Optional[float] = name_in_json("profiledSizeKwp")
    scaled_size_kwp: Optional[float] = name_in_json("scaledSizeKwp")

    def __post_init__(self):
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
class SolarOrLoad:
    profiles: List[Profile] = field(default=None)


@dataclass
class Bess:
    energy_capacity: float = name_in_json("energyCapacity")
    nameplate_power: float = name_in_json("nameplatePower")
    charge_efficiency: float = name_in_json("chargeEfficiency")


@dataclass
class Site:
    grid_connection: GridConnection = name_in_json("gridConnection")
    solar: SolarOrLoad
    load: SolarOrLoad
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
class Approach:
    to_soe: float = name_in_json("toSoe")
    encourage_to_soe: Optional[float] = name_in_json("encourageToSoe")
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
    duration_days: Optional[int] = field(metadata={"data_key": "durationDays"}, default=5)
    used_duration_days: Optional[int] = field(metadata={"data_key": "usedDurationDays"}, default=3)
    do_active_export_constraint_management: Optional[bool] = field(metadata={"data_key": "doActiveExportConstraintManagement"}, default=False)
    do_active_import_constraint_management: Optional[bool] = field(metadata={"data_key": "doActiveImportConstraintManagement"}, default=False)

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
