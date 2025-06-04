from dataclasses import field
from datetime import timedelta, datetime
from typing import List, Optional, Dict

from marshmallow_dataclass import dataclass
from skypro.common.config.data_source import ProfileDataSource
from skypro.common.config.dayed_period import DayedPeriodType
from skypro.common.config.path_field import PathType
from skypro.common.config.rates_dataclasses import Rates
from skypro.common.config.utility import field_with_opts, enforce_one_option

from skypro.commands.simulator.config.curve import (CurveType)

"""
This file contains configuration schema, implemented with Marshmallow Dataclass. The higher-level configuration structures are defined towards
the end of the file, and the lower-level structures towards the top.
"""


@dataclass
class Profile:
    """
    Represents a solar or load profile, which can be scaled.
    """

    # Tag is an optional name to assign to the profile. The advantage of this over the name being a dict key is that
    # arrays preserve order and the order of the load profiles may become important down the line.
    tag: Optional[str] = field_with_opts(key="tag")

    source: ProfileDataSource

    energy_cols: Optional[str] = field_with_opts(key="energyCols")

    # There are various ways to configure the scaling of the profile (e.g. number of plots, kwp of solar capacity, etc.).
    # But in the end it all boils down to a scaling factor.
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
    """
    Holds either solar or load profiles. Allows either a single `profile` or an array of `profiles` to be defined.
    """
    profile: Optional[Profile]
    profiles: Optional[List[Profile]] = field(default=None)

    def __post_init__(self):
        enforce_one_option([self.profiles, self.profile], "'profile', 'profiles")


@dataclass
class GridConnectionVariations:
    """
    Allows a grid connection size to be varied over the course of a simulation. The variations are defined by solar or load profiles.
    This is useful for simulating a grid connection that is shared by solar or load which is 'external' to the microgrid we want to simulate.
    So, although the solar or load is external to our microgrid, it does affect the size of the grid connection from the point of view of our microgrid.
    """
    load: SolarOrLoad
    generation: SolarOrLoad


@dataclass
class GridConnection:
    import_limit: float = field_with_opts(key="importLimit")
    export_limit: float = field_with_opts(key="exportLimit")
    variations: Optional[GridConnectionVariations] = field_with_opts(key="variations")


@dataclass
class Bess:
    energy_capacity: float = field_with_opts(key="energyCapacity")
    nameplate_power: float = field_with_opts(key="nameplatePower")
    charge_efficiency: float = field_with_opts(key="chargeEfficiency")


@dataclass
class Site:
    """
    Holds information about the physical site we want to simulate.
    """
    grid_connection: GridConnection = field_with_opts(key="gridConnection")
    solar: SolarOrLoad
    load: SolarOrLoad
    bess: Bess


@dataclass
class Niv:
    """
    Configuration to do NIV chasing using the price curve algorithm.
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
    This allows us to have different NIV chasing configurations for different times of day, or for weekends vs weekdays, etc.
    """
    period: DayedPeriodType
    niv: Niv


@dataclass
class Approach:
    """
    Configures how the battery algorithm should behave as we approach a peak period. We normally want to ensure a certain SoE ahead
    of a peak period, and the approach configuration tries to get us to a target SoE in a cost-effective way.

    The peak approach algorithm effectively draws a line on a chart of SoE vs Time (see below). As we approach the `peak_start` time,
    we may find that our SoE is below the line - in which case the approach algorithm kicks in and charges the battery. If our SoE is
    high enough that it sits above the line at a given time, then the approach algorithm doesn't need to do anything.

    SoE
    |
    |                                                                       *Target SoE
    |                                                                 *      |
    |                                                        *               |
    |                                               *                        |
    |                                     *                                  |
    |                            *                                           |
    |                    *                                                   |
    |            *                                                           |
    --------*----------------------------------------------------------------|-----------------> time
            approach_start                                                   peak_start

    The approach algorithm actually defines two lines:
    - A 'force charge' line which defines an absolute minimum SoE that we must get to ahead of the peak. If we are below the 'force charge line' then
      the battery will be charged even if the system is short (chances are, that even if prices are high now, they are only going to get higher as we
      reach the peak, so this is often a good strategy).
    - An 'encourage charge' line which defines a softer target SoE. If we are below the 'encourage charge line' then we will only charge if the system
      is long (and prices are relatively low).

    The approach_start time on the illustration above is derived from the `assumed_charge_power` and the `encourage/force_charge_duration_factor`,
    """
    to_soe: float = field_with_opts(key="toSoe")  # The SoE that we MUST reach ahead of the peak.
    encourage_to_soe: Optional[float] = field_with_opts(key="encourageToSoe")  # The SoE that we would LIKE to reach ahead of the peak.
    assumed_charge_power: float = field_with_opts(key="assumedChargePower")  # The actual charge power varies depending on site solar/load and grid constraints, but this is the configured estimate of how fast we can charge.
    encourage_charge_duration_factor: float = field_with_opts(key="encourageChargeDurationFactor")
    force_charge_duration_factor: float = field_with_opts(key="forceChargeDurationFactor")

    # charge_cushion shifts the approach so that we reach the target SoE before the actual start of the peak - this is useful because prices can get very high as
    # we near the peak, so it's often better to have already charged to the target SoE an hour or so before.
    charge_cushion: timedelta = field(metadata={"precision": "minutes", "data_key": "chargeCushionMins"})


@dataclass
class PeakDynamic:
    prioritise_residual_load: bool = field_with_opts(key="prioritiseResidualLoad")


@dataclass
class Peak:
    """
    Configures the time of a peak period to discharge into (usually a strong price signal from DUoS red bands)
    """
    period: DayedPeriodType = field_with_opts(key="period")
    approach: Approach = field_with_opts(key="approach")
    dynamic: Optional[PeakDynamic] = field_with_opts(key="dynamic")


@dataclass
class MicrogridLocalControl:
    """
    Configures control of the microgrid that doesn't need any external price signals:
    - import avoidance which prevents the microgrid from importing into load
    - export avoidance which prevents the microgrid from exporting solar onto the national grid
    """
    import_avoidance: bool = field_with_opts(key="importAvoidance")
    export_avoidance: bool = field_with_opts(key="exportAvoidance")


@dataclass
class MicrogridImbalanceControl:
    """
    Configures control of the microgrid, based on the imbalance state of the national grid.
    """
    discharge_into_load_when_short: bool = field_with_opts(key="dischargeIntoLoadWhenShort")
    charge_from_solar_when_long: bool = field_with_opts(key="chargeFromSolarWhenLong")

    # If the NIV in the previous SP was higher than this, then we will assume that the grid will stay long or short for the following period.
    niv_cutoff_for_system_state_assumption: float = field(metadata={"data_key": "nivCutoffForSystemStateAssumption", "allow_nan": True})


@dataclass
class Microgrid:
    """
    Configures control of the microgrid load and solar generation.
    """
    local_control: Optional[MicrogridLocalControl] = field_with_opts(key="localControl")
    imbalance_control: Optional[MicrogridImbalanceControl] = field_with_opts(key="imbalanceControl")


@dataclass
class PriceCurveAlgo:
    """
    Configures the price curve algorithm.
    """
    microgrid: Optional[Microgrid] = field_with_opts(key="microgrid")
    peak: Peak = field_with_opts(key="peak")
    niv_chase_periods: List[NivPeriod] = field_with_opts(key="nivChasePeriods")


@dataclass
class OptimiserBlocks:
    """
    Defines how an LP optimisation simulation is split into smaller duration optimisations that are stacked on top of
    each other, and any settings that are applied to each of those smaller duration optimisations.
    """
    max_avg_cycles_per_day: float = field_with_opts(key="maxAvgCyclesPerDay")

    # These settings allow us to reduce the effectiveness of a perfect hindsight optimisation - with the aim of making it more realistic.
    no_optional_charging_in_lowest_priced_quantile: Optional[float] = field_with_opts(key="noOptionalChargingInLowestPricedQuantile", default=None)
    no_optional_actions_in_first_ten_mins_except_for_period: Optional[DayedPeriodType] = field_with_opts(key="noOptionalActionsInFirstTenMinsExceptForPeriod", default=None)

    # These settings control the optimisation process itself, the defaults are likely fine.
    max_optimal_tolerance: Optional[float] = field_with_opts(key="maxOptimalTolerance", default=0.02)
    max_computation_secs: Optional[int] = field_with_opts(key="maxComputationSecs", default=10)

    # How long each optimisation block should be for, and how much of the optimisation block to actually use (the end of each optimisation block is discarded
    # because the behaviour is not reliable at the end - see simulator/algorithms/lp/optimiser.py.
    duration_days: Optional[int] = field_with_opts(key="durationDays", default=5)
    used_duration_days: Optional[int] = field_with_opts(key="usedDurationDays", default=3)

    # Allows the battery to facilitate solar or load capacity that would normally exceed the grid constraints.
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
    """
    Configures which optimisation strategy to use for a simulation - at the moment either the price curve algo or the perfect hindsight optimiser.
    """
    price_curve_algo: Optional[PriceCurveAlgo] = field_with_opts(key="priceCurveAlgo")
    optimiser: Optional[Optimiser] = field_with_opts(key="perfectHindsightOptimiser")

    def __post_init__(self):
        enforce_one_option([self.price_curve_algo, self.optimiser], "'priceCurveAlgo', 'perfectHindsightOptimiser'")


@dataclass
class TimeFrame:
    """
    Configures when the simulation should run for.
    """
    start: datetime
    end: datetime


@dataclass
class OutputSummary:
    """
    Configures an output file for the simulation summary.
    """
    csv: PathType
    rate_detail: Optional[str] = field_with_opts(key="rateDetail")  # if 'all', then full details of individual rates will be output.


@dataclass
class OutputSimulation:
    """
    Configures an output file for the detailed simulation timeseries.
    """
    csv: PathType
    aggregate: Optional[str]  # If "30min" then the output CSV will have a row per half-hour, if missing then the CSV will have a row per simulation time step.
    rate_detail: Optional[str] = field_with_opts(key="rateDetail")  # if 'all', then full details of individual rates will be output.


@dataclass
class SimOutput:
    summary: Optional[OutputSummary]
    simulation: Optional[OutputSimulation]


@dataclass
class AllSimulationsOutput:
    """
    Configures an output file to summarise the results of all simulations. This is only generated if the `--sim all` option is used. It's useful if your YAML configuration
    file defines multiple simulations, and you want to run them all and get an overview of all the results.
    """
    summary: Optional[OutputSummary]


@dataclass
class AllSimulations:
    output: Optional[AllSimulationsOutput]


@dataclass
class AllRates:
    """
    Configures two sets of rates:
    - live: these are the rates which the algorithm thinks are going to happen at the time it is making decisions
    - final: these are the rates which are used after the algorithm has run, and determine the actual costs/revenues of the simulation run.

    It is sometimes useful to have two sets of rates, for example, we might want to use Modo imbalance price predictions as 'live' and the
    actual Elexon prices as 'final'.
    """
    live: Rates
    final: Rates


@dataclass
class SimulationCase:
    """
    Configures an individual simulation. Each YAML configuration may define multiple simulations.
    """
    output: Optional[SimOutput]
    timeframe: TimeFrame = field_with_opts(key="timeFrame")
    site: Site
    strategy: Strategy
    rates: AllRates

    @property
    def start(self) -> datetime:
        return self.timeframe.start

    @property
    def end(self) -> datetime:
        return self.timeframe.end


@dataclass
class Config:
    """
    The highest level configuration class for simulations.
    """
    config_format_version: str = field_with_opts(key="configFormatVersion")
    sandbox: Optional[dict]  # A space for the user to define YAML anchors, which is not parsed/used by the program
    variables: Optional[dict]  # A space for the user to define file-level variables that are substituted into paths, which is not otherwise parsed/used by the program
    all_sims: Optional[AllSimulations] = field_with_opts(key="allSimulations")  # We may want to summarise the outputs of all the simulations.
    simulations: Dict[str, SimulationCase]  # Multiple simulations may be defined, each must have a unique name.
