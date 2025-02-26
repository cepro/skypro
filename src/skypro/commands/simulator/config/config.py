from dataclasses import field
from datetime import timedelta, datetime
from typing import List, Optional, Dict

from marshmallow_dataclass import dataclass
from simt_common.config.data_source import ProfileDataSource
from simt_common.config.dayed_period import DayedPeriodType
from simt_common.config.path_field import PathType
from simt_common.config.rates_dataclasses import Rates
from simt_common.config.utility import field_with_opts, enforce_one_option

from skypro.commands.simulator.config.curve import (CurveType)

"""
This file contains configuration schema that is used for both V3 and V4 config
"""


@dataclass
class GridConnection:
    import_limit: float = field_with_opts(key="importLimit")
    export_limit: float = field_with_opts(key="exportLimit")


@dataclass
class Profile:
    # Tag is an optional name to assign to the profile. The advantage of this over the name being a dict key is that
    # arrays preserve order and the order of the load profiles may become important down the line.
    tag: Optional[str] = field_with_opts(key="tag")

    source: ProfileDataSource

    energy_cols: Optional[str] = field_with_opts(key="energyCols")

    scaling_factor: Optional[float] = field_with_opts(key="scalingFactor")
    profiled_num_plots: Optional[float] = field_with_opts(key="profiledNumPlots")
    scaled_num_plots: Optional[float] = field_with_opts(key="scaledNumPlots")
    profiled_size_kwp: Optional[float] = field_with_opts(key="profiledSizeKwp")
    scaled_size_kwp: Optional[float] = field_with_opts(key="scaledSizeKwp")

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
    profile: Optional[Profile]
    profiles: Optional[List[Profile]] = field(default=None)

    def __post_init__(self):
        enforce_one_option([self.profiles, self.profile], "'profile', 'profiles")


@dataclass
class Bess:
    energy_capacity: float = field_with_opts(key="energyCapacity")
    nameplate_power: float = field_with_opts(key="nameplatePower")
    charge_efficiency: float = field_with_opts(key="chargeEfficiency")


@dataclass
class Site:
    grid_connection: GridConnection = field_with_opts(key="gridConnection")
    solar: SolarOrLoad
    load: SolarOrLoad
    bess: Bess


@dataclass
class RemoteSite:
    allow_flow_to_site: bool = field_with_opts(key="allowFlowToSite")
    solar: SolarOrLoad


@dataclass
class Niv:
    """
    The configuration to do NIV chasing.
    """
    charge_curve: CurveType = field_with_opts(key="chargeCurve")
    discharge_curve: CurveType = field_with_opts(key="dischargeCurve")
    curve_shift_long: float = field_with_opts(key="curveShiftLong")
    curve_shift_short: float = field_with_opts(key="curveShiftShort")
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
    to_soe: float = field_with_opts(key="toSoe")
    encourage_to_soe: Optional[float] = field_with_opts(key="encourageToSoe")
    assumed_charge_power: float = field_with_opts(key="assumedChargePower")
    encourage_charge_duration_factor: float = field_with_opts(key="encourageChargeDurationFactor")
    force_charge_duration_factor: float = field_with_opts(key="forceChargeDurationFactor")
    charge_cushion: timedelta = field(metadata={"precision": "minutes", "data_key": "chargeCushionMins"})


@dataclass
class PeakDynamic:
    prioritise_residual_load: bool = field_with_opts(key="prioritiseResidualLoad")


@dataclass
class Peak:
    period: DayedPeriodType = field_with_opts(key="period")
    approach: Approach = field_with_opts(key="approach")
    dynamic: Optional[PeakDynamic] = field_with_opts(key="dynamic")


@dataclass
class MicrogridLocalControl:
    import_avoidance: bool = field_with_opts(key="importAvoidance")
    export_avoidance: bool = field_with_opts(key="exportAvoidance")


@dataclass
class MicrogridImbalanceControl:
    discharge_into_load_when_short: bool = field_with_opts(key="dischargeIntoLoadWhenShort")
    charge_from_solar_when_long: bool = field_with_opts(key="chargeFromSolarWhenLong")
    niv_cutoff_for_system_state_assumption: float = field(metadata={"data_key": "nivCutoffForSystemStateAssumption", "allow_nan": True})


@dataclass
class Microgrid:
    local_control: Optional[MicrogridLocalControl] = field_with_opts(key="localControl")
    imbalance_control: Optional[MicrogridImbalanceControl] = field_with_opts(key="imbalanceControl")


@dataclass
class PriceCurveAlgo:
    microgrid: Optional[Microgrid] = field_with_opts(key="microgrid")
    peak: Peak = field_with_opts(key="peak")
    niv_chase_periods: List[NivPeriod] = field_with_opts(key="nivChasePeriods")


@dataclass
class SpreadAlgoFixedAction:
    charge_power: float = field_with_opts(key="chargePower")
    discharge_power: float = field_with_opts(key="dischargePower")


@dataclass
class SpreadAlgo:
    min_spread: float = field_with_opts(key="minSpread")
    recent_pricing_span: int = field_with_opts(key="recentPricingSpan")
    niv_cutoff_for_system_state_assumption: float = field(metadata={"data_key": "nivCutoffForSystemStateAssumption", "allow_nan": True})
    fixed_action: SpreadAlgoFixedAction = field_with_opts(key="fixedAction")
    microgrid: Optional[Microgrid] = field_with_opts(key="microgrid")
    peak: Peak = field_with_opts(key="peak")


@dataclass
class OptimiserBlocks:
    """Defines how the whole simulation period is split into smaller duration optimisations that are stacked on top of
    each other, and any settings that are applied to each of those smaller duration optimisations."""
    max_avg_cycles_per_day: float = field_with_opts(key="maxAvgCyclesPerDay")
    max_optimal_tolerance: Optional[float] = field_with_opts(key="maxOptimalTolerance", default=0.02)
    max_computation_secs: Optional[int] = field_with_opts(key="maxComputationSecs", default=10)
    duration_days: Optional[int] = field_with_opts(key="durationDays", default=5)
    used_duration_days: Optional[int] = field_with_opts(key="usedDurationDays", default=3)
    do_active_export_constraint_management: Optional[bool] = field_with_opts(key="doActiveExportConstraintManagement", default=False)
    do_active_import_constraint_management: Optional[bool] = field_with_opts(key="doActiveImportConstraintManagement", default=False)

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
    price_curve_algo: Optional[PriceCurveAlgo] = field_with_opts(key="priceCurveAlgo")
    spread_algo: Optional[SpreadAlgo] = field_with_opts(key="spreadAlgo")
    optimiser: Optional[Optimiser] = field_with_opts(key="perfectHindsightOptimiser")

    def __post_init__(self):
        enforce_one_option([self.price_curve_algo, self.spread_algo, self.optimiser], "'priceCurveAlgo', 'spreadAlgo', 'perfectHindsightOptimiser'")


@dataclass
class TimeFrame:
    start: datetime
    end: datetime


@dataclass
class OutputSummary:
    csv: PathType
    rate_detail: Optional[str] = field_with_opts(key="rateDetail")


@dataclass
class OutputSimulation:
    csv: PathType
    aggregate: Optional[str]
    rate_detail: Optional[str] = field_with_opts(key="rateDetail")


@dataclass
class SimOutput:
    summary: Optional[OutputSummary]
    simulation: Optional[OutputSimulation]


@dataclass
class AllSimulationsOutput:
    summary: Optional[OutputSummary]


@dataclass
class AllSimulations:
    output: Optional[AllSimulationsOutput]


@dataclass
class AllRates:
    live: Rates
    final: Rates


@dataclass
class SimulationCaseV4:
    output: Optional[SimOutput]
    timeframe: TimeFrame = field_with_opts(key="timeFrame")
    site: Site
    remote_site: Optional[RemoteSite] = field_with_opts(key="remoteSite")
    strategy: Strategy
    rates: AllRates

    @property
    def start(self) -> datetime:
        return self.timeframe.start

    @property
    def end(self) -> datetime:
        return self.timeframe.end


@dataclass
class ConfigV4:
    config_format_version: str = field_with_opts(key="configFormatVersion")
    sandbox: Optional[dict]  # a space for the user to define YAML anchors, which is not parsed/used by the program
    variables: Optional[dict]  # a space for the user to define file-level variables that are substituted into paths, which is not otherwise parsed/used by the program
    all_sims: Optional[AllSimulations] = field_with_opts(key="allSimulations")
    simulations: Dict[str, SimulationCaseV4]
