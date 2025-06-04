from typing import Dict, List

import pandas as pd

from skypro.common.rates.rates import Rate, FixedRate, VolRate


def get_rates_dfs_by_type(
        time_index: pd.DatetimeIndex,
        rates_by_category: Dict[str, List[Rate]],
        allow_vol_rates: bool
) -> (Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]):
    """Returns two dictionary dataframes:
       - The first has dataframes containing any fixed costs in pence, keyed by category
       - The second has dataframes containing any volumetric rates in p/kWh, keyed by category

    Some rates are not really 'core' to the operation of the simulations/reporting itself, but are just passed through into the
    output to make future analysis of the output CSV easier, or for bill matching. These rates could be called 'peripheral'.
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
