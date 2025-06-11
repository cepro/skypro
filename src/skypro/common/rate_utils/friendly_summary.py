from datetime import datetime
from typing import List, Dict, Any

import pandas as pd

from skypro.common.rates.rates import Rate, FlatVolRate, PeriodicFlatVolRate, ShapedVolRate, MultiplierVolRate, OSAMFlatVolRate, RegularFixedRate
from skypro.common.rates.time_varying_value import PeriodicValue


def get_friendly_rates_summary(rates: List[Rate], t: datetime) -> pd.DataFrame:
    """
    Returns a nicely formatted summary dataframe of the given rates (which can then be printed with tabulate).
    The values of the rates can change over time, so `t` is used to specify the time at which the values are used for presentation.
    """
    df = pd.DataFrame([_format_rate_info(rate, t) for rate in rates])
    df = df.sort_values(by=["Type", "Name"])

    return df.drop(["Type"], axis=1)


def _format_rate_info(rate: Rate, t: datetime) -> Dict[str, Any]:
    """Extract relevant information from a Rate object for tabulation."""
    rate_info = {
        "Name": rate.name,
        "Type": type(rate).__name__,
        "Value": "",
        "Periods": "",
        "Supply Point": ""
    }

    if isinstance(rate, OSAMFlatVolRate):
        rate_info["Value"] = f"{rate.time_varying_value.get_value_at(t):.3f} p/kWh OSAM"
        rate_info["Supply Point"] = rate.supply_point.name

    elif isinstance(rate, FlatVolRate):
        rate_info["Value"] = f"{rate.time_varying_value.get_value_at(t):.3f} p/kWh"
        rate_info["Supply Point"] = rate.supply_point.name

    elif isinstance(rate, PeriodicFlatVolRate):
        periodic_value: PeriodicValue = rate.time_varying_value.get_value_at(t)
        rate_info["Value"] = f"{periodic_value.value:.3f} p/kWh"
        periods = periodic_value.periods
        period_strs = [f"{period.days} {period.period.start} -> {period.period.end}" for period in periods]
        rate_info["Periods"] = "[" + ",\n".join(period_strs) + "]"
        rate_info["Supply Point"] = rate.supply_point.name

    elif isinstance(rate, ShapedVolRate):
        # For shaped rates, show range
        rate_info["Value"] = f"{rate.pricing.min():.2f} -> {rate.pricing.max():.2f} p/kWh"
        rate_info["Supply Point"] = rate.supply_point.name

    elif isinstance(rate, MultiplierVolRate):
        factor = rate.time_varying_factor.get_value_at(t)
        rates_to_multiple_str = ",\n".join([r.name for r in rate.rates_to_multiply])
        rate_info["Value"] = f"{factor * 100:.1f}% of [{rates_to_multiple_str}]"

    elif isinstance(rate, RegularFixedRate):
        rate_info["Value"] = f"{rate.daily_costs.get_value_at(t):.3f} p/day"

    return rate_info
