from datetime import datetime, timedelta
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import pytz

from skypro.commands.simulator.algorithms.system_state import SystemState
from skypro.commands.simulator.cartesian import Curve, Point
from skypro.commands.simulator.config.config import Peak

TIMEZONE_STR = "Europe/London"
REF_DATETIME = pytz.timezone(TIMEZONE_STR).localize(datetime(year=2000, month=1, day=1))


def get_peak_power(
        peak_config: Optional[Peak],
        t: datetime,
        time_step: timedelta | pd.Timedelta,
        soe: float,
        bess_max_power_discharge: float,
        microgrid_residual_power: float,
        system_state: SystemState
) -> Optional[float]:
    """
    Returns the power to deliver during the peak period, or None if the peak is not active/relevant at this time.
    """
    # There is a strange bug with pd.TimeDelta where it doesn't behave correctly when differencing - convert to
    # a `timedelta` type
    time_step = timedelta(seconds=time_step.total_seconds())

    if not peak_config:
        return None

    if not peak_config.period.contains(t):
        return None

    if not peak_config.dynamic:
        # Just do a 'dumb' full discharge over the peak until the battery is empty
        return -bess_max_power_discharge

    # If the notional battery duration (accounting for grid constraints) is shorter than the peak duration
    # then we can get an improvement on the above 'dumb' method by choosing when we discharge into the peak.
    peak_end = peak_config.period.period.end_absolute(t)

    # This is an approximation because we may be limited by grid constraints and the load and solar levels
    # may change throughout the peak.
    assumed_time_to_empty_battery = timedelta(hours=(soe / bess_max_power_discharge))

    # We want to ensure that we empty the battery completely by the end of the peak period, and there is a
    # point into the peak where we must discharge at the max power to ensure that.
    latest_time_before_max_discharge = peak_end - assumed_time_to_empty_battery
    if t > (latest_time_before_max_discharge - time_step):
        return -bess_max_power_discharge

    # We are early enough in the peak period to have some flexibility about how much we discharge
    if system_state == SystemState.LONG:
        if not peak_config.dynamic.prioritise_residual_load:
            # If we are not 'prioritising loads' then hold off on the discharge until the last minute,
            # or until the system is short.
            return 0.0

        # Even though the system is long (and prices relatively low), discharge to avoid microgrid imports (if any)
        if microgrid_residual_power > 0:
            return -microgrid_residual_power
        else:
            return 0.0

    else: # TODO: this 'else' currently includes SystemState.UNKNOWN, we may want to adjust the behaviour around that

        if not peak_config.dynamic.prioritise_residual_load:
            # If we are not 'prioritising loads' then discharge at the max power
            return -bess_max_power_discharge

        # Here we want to discharge at the max power we can, whilst ensuring there is enough energy
        # in the battery to service any residual microgrid load at the end of the peak.
        # Approximate that the residual load will stay the same throughout the peak
        # TODO: we could make some assumption about the residual growing due to less solar later on?
        if microgrid_residual_power <= 0:
            # There is no residual load (solar may be supplying load?) so just discharge at max power
            return -bess_max_power_discharge

        # If we were to discharge at max power, then when would we run out of energy?
        empty_time_without_reserve = t + assumed_time_to_empty_battery

        # This tells us how long and how much energy we need to reserve for servicing the residual load
        reserve_duration = peak_end - empty_time_without_reserve
        reserve_energy = microgrid_residual_power * (reserve_duration.total_seconds() / 3600)

        # Which, in turn, allows us to calculate the new max discharge rate which would allow us
        # to keep that reserve for the end of the peak
        duration_before_reserve = (peak_end - reserve_duration) - t
        if duration_before_reserve.total_seconds() <= 0:
            return -bess_max_power_discharge
        energy_before_reserve = soe - reserve_energy
        return -energy_before_reserve / (duration_before_reserve.total_seconds() / 3600)


def get_peak_approach_energies(
        t: datetime,
        time_step: timedelta,
        soe: float,
        charge_efficiency: float,
        peak_config: Peak,
        is_long: bool,
) -> Tuple[float, float]:
    """
    Returns the charge energy required due to the "force" and "encourage" peak approach configuration
    :param t: the time now
    :param time_step: the size of the simulation time step
    :param soe: the current battery soe
    :param charge_efficiency:
    :param peak_config:
    :param is_long: indicates if the system is long or short - the encourage curve is only used when the system is long
    :return:
    """
    # TODO: this approach won't work if the approach curve crosses over a midnight boundary

    if not peak_config.period or peak_config.approach.to_soe == 0:
        return 0.0, 0.0

    t = t.astimezone(pytz.timezone(TIMEZONE_STR))

    if not peak_config.period.days.is_on_day(t):
        return 0.0, 0.0

    peak_start = peak_config.period.period.start_absolute(t.date())
    peak_end = peak_config.period.period.end_absolute(t.date())

    reference_point = _datetime_point(
        t=t + time_step,
        y=soe
    )

    force_curve = _get_approach_curve(
        peak_start=peak_start,
        peak_end=peak_end,
        to_soe=peak_config.approach.to_soe,
        charge_efficiency=charge_efficiency,
        assumed_charge_power=peak_config.approach.assumed_charge_power,
        charge_cushion=peak_config.approach.charge_cushion,
        charge_duration_factor=peak_config.approach.force_charge_duration_factor
    )
    force_energy = force_curve.vertical_distance(reference_point)
    if np.isnan(force_energy) or force_energy < 0:
        force_energy = 0

    if is_long:
        encourage_curve = _get_approach_curve(
            peak_start=peak_start,
            peak_end=peak_end,
            to_soe=peak_config.approach.to_soe,
            charge_efficiency=charge_efficiency,
            assumed_charge_power=peak_config.approach.assumed_charge_power,
            charge_cushion=peak_config.approach.charge_cushion,
            charge_duration_factor=peak_config.approach.encourage_charge_duration_factor
        )
        encourage_energy = encourage_curve.vertical_distance(reference_point)
        if np.isnan(encourage_energy) or encourage_energy < 0:
            encourage_energy = 0
    else:
        encourage_energy = 0

    return force_energy, encourage_energy


def _get_approach_curve(
        peak_start: datetime,
        peak_end: datetime,
        to_soe: float,
        charge_efficiency: float,
        assumed_charge_power: float,
        charge_cushion: timedelta,
        charge_duration_factor: float
) -> Curve:
    """
    Returns a curve representing the boundary of the peak approach
    """
    # how long is the approach
    approach_duration = timedelta(
        hours=((to_soe / assumed_charge_power) / charge_efficiency) * charge_duration_factor
    )

    approach_curve = Curve(points=[
        _datetime_point(t=peak_start - approach_duration - charge_cushion, y=0),
        _datetime_point(t=peak_start - charge_cushion, y=to_soe),
        _datetime_point(t=peak_end, y=to_soe),
    ])

    return approach_curve


def _datetime_point(t: datetime, y: float) -> Point:
    """
    Returns a Point object that encodes a time of day.
    This uses a reference datetime to convert a time into a float number of seconds, so may not work over midnight
    boundaries.
    """
    duration = t - REF_DATETIME
    return Point(
        x=duration.total_seconds(),
        y=y
    )
