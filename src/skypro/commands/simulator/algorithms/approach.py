from datetime import datetime, timedelta
from typing import Tuple

import numpy as np
import pytz
from skypro.commands.simulator.cartesian import Curve, Point
from skypro.commands.simulator.config.config import Peak

TIMEZONE_STR = "Europe/London"
REF_DATETIME = pytz.timezone(TIMEZONE_STR).localize(datetime(year=2000, month=1, day=1))


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

    if not peak_config.period:
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
