import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

import pandas as pd

from skypro.common.rates.rates import VolRate, Rate, FixedRate

"""
This file has functions to convert Rate instances into time-series dataframes.
"""


@dataclass
class VolRatesForEnergyFlows:
    """Holds the volume-based (p/kWh) rates for each energy flow in a microgrid"""
    solar_to_batt: List[VolRate] = field(default_factory=list)
    grid_to_batt: List[VolRate] = field(default_factory=list)
    batt_to_load: List[VolRate] = field(default_factory=list)
    batt_to_grid: List[VolRate] = field(default_factory=list)
    solar_to_grid: List[VolRate] = field(default_factory=list)
    solar_to_load: List[VolRate] = field(default_factory=list)
    grid_to_load: List[VolRate] = field(default_factory=list)

    def get_all_sets_named(self) -> List[Tuple[str, List[VolRate]]]:
        """
        Returns all the sets of rates, in a tuple with the set name
        """
        return [
            ("solar_to_batt", self.solar_to_batt),
            ("grid_to_batt", self.grid_to_batt),
            ("batt_to_load", self.batt_to_load),
            ("batt_to_grid", self.batt_to_grid),
            ("solar_to_grid", self.solar_to_grid),
            ("solar_to_load", self.solar_to_load),
            ("grid_to_load", self.grid_to_load),
        ]


def get_vol_rates_dfs(time_index: pd.DatetimeIndex, all_rates: VolRatesForEnergyFlows, log: bool = True) -> (Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]):
    """
    Returns the market and internal volume-based rates (p/kWh) as dataframes for each flow.
    The first dictionary contains the "market rates" - i.e. those that we are actually charged or paid by our suppliers.
    The second dictionary contains the "internal rates" - i.e. those that we use for internal accounting/decision-making
    and includes the opportunity costs.
    For example, the "solar_to_batt" market rate may be zero because no money changes hands with a third
    party when we charge from on-site solar; however, the internal rate is likely non-zero because we account for the
    opportunity cost of not exporting that solar power to the grid.

    Both dictionaries are keyed by the flow name, and contain dataframes.

    Each dataframe has a column for each individual rate (e.g. DUoS red/amber/green, supplier fees, imbalance, etc)
    """
    mkt_rates_dfs = {}
    cache = {}

    for rate_set_name, rate_set in all_rates.get_all_sets_named():
        if log:
            logging.info(f"Calculating rates for {rate_set_name}...")
        set_df = pd.DataFrame(index=time_index)
        for rate in rate_set:
            rate_id = id(rate)
            if rate_id not in cache:
                cache[rate_id] = rate.get_per_kwh_rate_series(time_index)
            per_kwh = cache[rate_id]
            if rate.name in set_df.columns:
                raise ValueError(f"Rate '{rate.name}' is defined twice for '{rate_set_name}' flow.")
            set_df[rate.name] = per_kwh

        mkt_rates_dfs[rate_set_name] = set_df

    int_rates_dfs = {}

    # These flows rates are the same for both internal and market
    for flow_name in ["grid_to_batt", "batt_to_grid"]:
        int_rates_dfs[flow_name] = mkt_rates_dfs[flow_name]

    # We call these internal rates are zero, because they are not effected by the bess control algorithm:
    for flow_name in ["solar_to_grid", "grid_to_load"]:
        int_rates_dfs[flow_name] = pd.DataFrame(index=time_index)

    # The cost of the bess strategy includes the 'opportunity cost' associated with what would have happened to the
    # energy if the BESS didn't act.
    # If we hadn't charged the bess from solar then we would have been paid to export it to the grid
    int_rates_dfs["solar_to_batt"] = pd.concat([
        mkt_rates_dfs["solar_to_batt"].add_prefix("mkt_"),
        -mkt_rates_dfs["solar_to_grid"].add_prefix("int_")
    ], axis=1)

    # If we hadn't supplied the loads from the BESS then we would have paid to import from the grid
    int_rates_dfs["batt_to_load"] = pd.concat([
        mkt_rates_dfs["batt_to_load"].add_prefix("mkt_"),
        -mkt_rates_dfs["grid_to_load"].add_prefix("int_")
    ], axis=1)

    # We value the solar to load flow internally at the rate which we would have paid to import from the grid
    int_rates_dfs["solar_to_load"] = pd.concat([
        mkt_rates_dfs["solar_to_load"].add_prefix("mkt_"),
        -mkt_rates_dfs["grid_to_load"].add_prefix("int_")
    ], axis=1)

    return mkt_rates_dfs, int_rates_dfs


def get_rates_dfs_by_type(
        time_index: pd.DatetimeIndex,
        rates_by_category: Dict[str, List[Rate]],
        allow_vol_rates: bool
) -> (Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]):
    """Returns two dictionaries of dataframes:
       - The first has dataframes containing any fixed costs in pence, keyed by category
       - The second has dataframes containing any volumetric rates in p/kWh, keyed by category
    """

    fixed_costs_dfs = {}
    vol_rates_dfs = {}

    for category, rates in rates_by_category.items():
        fixed_costs_dfs[category] = pd.DataFrame(index=time_index)
        vol_rates_dfs[category] = pd.DataFrame(index=time_index)

        for rate in rates:
            # Fixed costs and volume-based rates go into different columns
            if isinstance(rate, FixedRate):
                fixed_costs_dfs[category][rate.name] = rate.get_cost_series(time_index)
            elif isinstance(rate, VolRate):
                if not allow_vol_rates:
                    raise ValueError("Volumetric rate found but not allowed")
                vol_rates_dfs[category][rate.name] = rate.get_per_kwh_rate_series(time_index)

    return fixed_costs_dfs, vol_rates_dfs
