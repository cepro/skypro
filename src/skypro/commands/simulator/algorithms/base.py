"""Base class for control strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class StrategyContext:
    """
    Context passed to extension strategies.

    Contains the data and configuration needed to run a control strategy.
    """
    df: pd.DataFrame  # Input dataframe with solar, load, time_into_sp, etc.
    bess_energy_capacity: float  # kWh
    bess_nameplate_power: float  # kW
    bess_charge_efficiency: float  # 0-1
    license_file: Optional[str] = None  # Path to license file for premium strategies


class ControlStrategy(ABC):
    """
    Abstract base class for battery control strategies.

    All strategies (built-in and extensions) should implement this interface.
    The `run()` method must return a DataFrame with at minimum these columns:
    - soe: State of energy at start of each timestep (kWh)
    - energy_delta: Energy transferred in/out of battery (kWh, positive=charge)
    - bess_losses: Energy lost due to charge inefficiency (kWh)
    """

    @abstractmethod
    def __init__(self, context: StrategyContext, config: dict):
        """
        Initialize the strategy.

        Args:
            context: StrategyContext containing input data and battery config
            config: Strategy-specific configuration from YAML
        """
        pass

    @abstractmethod
    def run(self) -> pd.DataFrame:
        """
        Execute the control strategy.

        Returns:
            DataFrame with columns: soe, energy_delta, bess_losses
            May include additional debug columns.
        """
        pass
