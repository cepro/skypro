from abc import ABC
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import pandas as pd

from skypro.common.rates.supply_point import SupplyPoint
from skypro.common.rates.time_varying_value import TimeVaryingValue, PeriodicValue
from skypro.common.timeutils.timeseries import get_step_size


class Rate(ABC):
    """
    Abstract base class for all rates
    """
    def __init__(self, name: str):
        self.name = name


class VolRate(Rate):
    """
    Abstract base class for all p/kWh rates (i.e. the cost is related to the volume of energy)
    """

    def get_per_kwh_rate(self, t: datetime) -> float:
        raise NotImplementedError()

    def get_per_kwh_rate_series(self, time_index: pd.DatetimeIndex) -> pd.Series:
        raise NotImplementedError()

    def get_cost(self, t: datetime, energy: float) -> float:
        raise NotImplementedError()


class FixedRate(Rate):
    """
    Abstract base class for all rates that are not effected by the volume of energy
    """

    def get_cost_series(self, index: pd.DatetimeIndex) -> pd.Series:
        raise NotImplementedError()


class FlatVolRate(VolRate):
    """
    Represents a p/kW charge at a flat rate.
    The flat rate can be for 'all time' or it can vary occasionally over time (for example year by year).
    """
    def __init__(self, name: str, values: List[Tuple[Optional[datetime], float]], supply_point: SupplyPoint):
        super().__init__(name)
        self.supply_point = supply_point
        self.time_varying_value = TimeVaryingValue(values)

    def get_per_kwh_rate(self, t: datetime) -> float:
        rate_value = self.time_varying_value.get_value_at(t)
        return rate_value * self.supply_point.line_loss_factor

    def get_per_kwh_rate_series(self, time_index: pd.DatetimeIndex) -> pd.Series:

        rate_series = pd.Series(index=time_index)

        for start_time, _ in self.time_varying_value.times_with_values:

            per_kwh_rate = FlatVolRate.get_per_kwh_rate(self, start_time)  # subclasses may override this method, but we want this class' version
            if start_time is None or pd.isnull(start_time):
                rate_series[:] = per_kwh_rate
            else:
                rate_series.loc[rate_series.index >= start_time] = per_kwh_rate

        return rate_series

    def get_cost(self, t: datetime, energy: float) -> float:
        return self.get_per_kwh_rate(t) * energy

    def __str__(self) -> str:
        return f"{self.name}: {self.time_varying_value.get_all_values()} p/kW, {self.supply_point.name}"


class PeriodicFlatVolRate(VolRate):
    """
    Represents a flat rate, e.g. 10p/kWh that is only charged at certain times of day - e.g. this can be used for DUoS
    bands that apply only at certain times of day and is different for weekdays and weekends.
    """
    def __init__(
            self,
            name: str,
            periodic_values: List[Tuple[Optional[datetime], PeriodicValue]],
            supply_point: SupplyPoint,
    ):
        super().__init__(name)
        self.supply_point = supply_point
        self.time_varying_value = TimeVaryingValue(periodic_values)

    def get_per_kwh_rate(self, t: datetime) -> float:

        periodic_value = self.time_varying_value.get_value_at(t)

        for period in periodic_value.periods:
            if period.contains(t):
                return periodic_value.value * self.supply_point.line_loss_factor

        return 0.0

    def get_per_kwh_rate_series(self, time_index: pd.DatetimeIndex) -> pd.Series:

        values = time_index.to_series().apply(lambda t: self.get_per_kwh_rate(t))
        values.index = time_index

        return values

    def get_cost(self, t: datetime, energy: float) -> float:
        return self.get_per_kwh_rate(t) * energy

    def __str__(self) -> str:
        return f"{self.name}: {self.time_varying_value.get_all_values()}, {self.supply_point.name}"


class ShapedVolRate(VolRate):
    """
    Represents a rate that varies over time (normally half-hourly) as given in a pricing pd.Series.
    E.g. used for imbalance pricing.
    """
    def __init__(self, name: str, pricing: pd.Series, supply_point: SupplyPoint):
        """The pricing series must have a DatetimeIndex (usually half-hourly) and have a price value in units of
        `p/kW`."""
        super().__init__(name)
        self.pricing = pricing
        self.supply_point = supply_point

    def get_per_kwh_rate(self, t: datetime) -> float:
        try:
            price = self.pricing.loc[t]
        except KeyError:
            # Give a more helpful error message:
            raise KeyError(f"Pricing data not available for the {self.name} charge for the period {t}")

        return price * self.supply_point.line_loss_factor

    def get_per_kwh_rate_series(self, time_index: pd.DatetimeIndex) -> pd.Series:
        return self.pricing.loc[time_index] * self.supply_point.line_loss_factor

    def get_cost(self, t: datetime, energy: float) -> float:
        return self.get_per_kwh_rate(t) * energy

    def __str__(self) -> str:
        return f"{self.name}: variable, {self.supply_point.name}"


class MultiplierVolRate(VolRate):
    """
    Represents a percentage fee that is applied to the revenue of other volume rates.
    E.g. Statkraft takes 5% of all export earnings (both Imbalance and DUoS).
    """
    def __init__(self, name: str, mode: str, factors: List[Tuple[Optional[datetime], float]]):
        super().__init__(name)
        self.rates_to_multiply = None

        if mode != "all-in-this-direction":
            raise ValueError("The multiplier rate currently only supports 'all-in-this-direction'")

        self.mode = mode

        self.time_varying_factor = TimeVaryingValue(factors)

    def __str__(self) -> str:
        names = [rate.name for rate in self.rates_to_multiply]
        return f"{self.name}: {'|'.join([str(v*100) for v in self.time_varying_factor.get_all_values()])}% of: {names}"

    def set_all_rates_in_set(self, rates: List[Rate]):
        """
        Sets all the rates in this set, some of which will then be select to be part of the multiplication. You can
        provide a list that contains self, and it will automatically be excluded.
        """
        self.rates_to_multiply = []
        if self.mode == "all-in-this-direction":
            for rate in rates:
                if not isinstance(rate, MultiplierVolRate):
                    self.rates_to_multiply.append(rate)

    def get_per_kwh_rate(self, t: datetime) -> float:

        factor = self.time_varying_factor.get_value_at(t)

        rates_to_multiply_per_kwh_rate = 0.0
        for rate_to_multiply in self.rates_to_multiply:
            rates_to_multiply_per_kwh_rate += rate_to_multiply.get_per_kwh_rate(t)

        per_kwh_rate = rates_to_multiply_per_kwh_rate * factor
        return per_kwh_rate

    def get_per_kwh_rate_series(self, time_index: pd.DatetimeIndex) -> pd.Series:

        values = time_index.to_series().apply(lambda t: self.get_per_kwh_rate(t))
        values.index = time_index

        return values

    def get_cost(self, t: datetime, energy: float) -> float:
        return self.get_per_kwh_rate(t) * energy


class OSAMFlatVolRate(FlatVolRate):
    """
    Represents a p/kWh rate that is subject to OSAM exemption, e.g. under P395
    """
    def __init__(self, name: str,  rates: List[Tuple[Optional[datetime], float]], supply_point: SupplyPoint):
        super().__init__(name, rates, supply_point)
        self.ncsp = pd.Series()

    def __str__(self) -> str:
        return f"{self.name}: OSAM {self.time_varying_value.get_all_values()} p/kWh"

    def add_ncsp(self, ncsp: pd.Series):
        """
        Appends the given NCSP to the known set. You can call this method once with all the NCSPs if you know them
        all at once (e.g. if reporting over a historical dataset), or you can call it multiple times to append new data
        as it becomes available (e.g. if simulating and generating data as you go).
        """

        if len(self.ncsp) == 0:
            self.ncsp = ncsp.copy()
        else:
            self.ncsp = pd.concat([self.ncsp, ncsp])
            # It's possible that we are passed newer data that overlaps with existing data, so drop the duplicates,
            # keeping the latest
            self.ncsp = self.ncsp[~self.ncsp.index.duplicated(keep='last')]

    def get_per_kwh_rate(self, t: datetime) -> float:
        """
        Returns the p/kWh rate, adjusted based on the non-chargeable proportion
        """
        base_rate = super().get_per_kwh_rate(t)
        return base_rate * (1.0 - self.ncsp.loc[t])

    def get_per_kwh_rate_series(self, time_index: pd.DatetimeIndex) -> pd.Series:

        values = time_index.to_series().apply(lambda t: self.get_per_kwh_rate(t))
        values.index = time_index
        return values

    def get_per_kwh_base_rate_series(self, time_index: pd.DatetimeIndex) -> pd.Series:
        """
        Returns the p/kWh 'base rate'- i.e. the rate without OSAM adjustment
        """
        return super().get_per_kwh_rate_series(time_index)


class RegularFixedRate(FixedRate):
    """
    Represents a fixed p/day charge.
    """
    def __init__(self, name: str, daily_costs: List[Tuple[Optional[datetime], float]]):
        super().__init__(name)
        self.daily_costs = TimeVaryingValue(daily_costs)

    def __str__(self) -> str:
        return f"{self.name}: {'|'.join([str(v) for v in self.daily_costs.get_all_values()])} p/day"

    def get_cost_series(self, index: pd.DatetimeIndex) -> pd.Series:
        """
        Returns the pence cost split evenly over the given time index as a series
        """
        step_size = get_step_size(index)
        scaling_factor = step_size / timedelta(hours=24)  # TODO: this does not account for DST boundary days which don't have 24 hours, but it's likely close enough

        cost_series = pd.Series(index=index)

        for start_time, daily_cost in self.daily_costs.times_with_values:

            interval_cost = daily_cost * scaling_factor

            if start_time is None or pd.isnull(start_time):
                cost_series[:] = interval_cost
            else:
                cost_series.loc[cost_series.index >= start_time] = interval_cost

        return cost_series
