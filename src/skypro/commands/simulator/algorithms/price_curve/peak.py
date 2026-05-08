from datetime import datetime, timedelta
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import pytz

from skypro.commands.simulator.algorithms.price_curve.system_state import SystemState
from skypro.commands.simulator.cartesian import Curve, Point
from skypro.commands.simulator.config import Peak

TIMEZONE = pytz.timezone("Europe/London")
REF_DATETIME = TIMEZONE.localize(datetime(year=2000, month=1, day=1))


def _find_active_peak(peaks: List[Peak], t: datetime) -> Optional[Peak]:
    """Return the peak whose period contains `t`, or None.

    Raises ValueError if more than one peak's period claims `t` (overlapping peaks are
    a configuration error — discharge logic for a single timestep can only be driven by one peak).
    """
    matches = [p for p in peaks if p.period.contains(t)]
    if len(matches) > 1:
        raise ValueError(
            f"Multiple peaks active at {t}: {[str(p.period) for p in matches]}. "
            "Peak periods must not overlap."
        )
    return matches[0] if matches else None


def _find_next_peak(peaks: List[Peak], t: datetime) -> Optional[Peak]:
    """Return the next peak that starts strictly after `t` on the same calendar day (in `t`'s tz)
    and applies to that day-class. None if no peak qualifies.

    The approach machinery (force/encourage charging up to a target SoE before the peak) only makes
    sense for an upcoming peak; once the peak starts the dispatch logic in `get_peak_power` takes over.
    """
    upcoming = []
    for p in peaks:
        if not p.period.days.is_on_day(t):
            continue
        start = p.period.period.start_absolute(t.date())
        if start > t:
            upcoming.append((start, p))
    if not upcoming:
        return None
    upcoming.sort(key=lambda pair: pair[0])
    return upcoming[0][1]


def get_peak_power(
        peaks: List[Peak],
        t: datetime,
        time_step: timedelta | pd.Timedelta,
        soe: float,
        bess_max_power_discharge: float,
        microgrid_residual_power: float,
        system_state: SystemState
) -> Optional[float]:
    """
    Returns the power to deliver during a peak period, or None if no peak is active at this time.
    """
    # There is a strange bug with pd.TimeDelta where it doesn't behave correctly when differencing - convert to
    # a `timedelta` type
    time_step = timedelta(seconds=time_step.total_seconds())

    peak_config = _find_active_peak(peaks, t)
    if peak_config is None:
        return None

    if not peak_config.dynamic:
        # Just do a 'dumb' full discharge over the peak until the battery is empty
        return -bess_max_power_discharge

    # If the notional battery duration (accounting for grid constraints) is shorter than the peak duration
    # then we can get an improvement on the above 'dumb' method by choosing when we discharge into the peak.
    peak_end = peak_config.period.period.end_absolute(t)

    # `min_end_of_peak_soe` reserves SoE for post-peak niv-chase. It's subtracted
    # from `soe` for the time-to-empty calc, which is what creates slack — without
    # it, time_to_empty often consumes the full peak window so HOLD-on-LONG never
    # gets a chance to fire.
    dischargeable_soe = max(0.0, soe - peak_config.dynamic.min_end_of_peak_soe)

    # This is an approximation because we may be limited by grid constraints which depend on the load and solar levels
    # which, in turn, may change throughout the peak.
    if bess_max_power_discharge <= 0:
        assumed_time_to_empty_battery = timedelta(minutes=0)
    else:
        assumed_time_to_empty_battery = timedelta(hours=(dischargeable_soe / bess_max_power_discharge))

    # We want to ensure that we empty the battery completely by the end of the peak period, and there is a
    # point into the peak where we must discharge at the max power to ensure that.
    latest_time_before_max_discharge = peak_end - assumed_time_to_empty_battery
    if t > (latest_time_before_max_discharge - time_step):
        return -bess_max_power_discharge

    # We are early enough in the peak period to have some flexibility about how much we discharge
    if system_state == SystemState.LONG or system_state == SystemState.UNKNOWN:
        if not peak_config.dynamic.prioritise_residual_load:
            # If we are not 'prioritising loads' then hold off on the discharge until the last minute,
            # or until the system is short.
            return 0.0

        # Even though the system is long (and prices relatively low), discharge to avoid microgrid imports (if any)
        if microgrid_residual_power > 0:
            return -microgrid_residual_power
        else:
            return 0.0

    else:

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

        if reserve_energy > dischargeable_soe:
            # The assumptions around how much we needed to reserve were wrong, and so we are going to run out of energy.
            # Just do our best to service the residual load at this point:
            return -microgrid_residual_power

        # Knowing the reserve_energy allows us to calculate the new max discharge rate which would allow us
        # to keep that reserve for the end of the peak
        duration_before_reserve = (peak_end - reserve_duration) - t
        if duration_before_reserve.total_seconds() <= 0:
            # TODO: this should return microgrid_residual_power?!
            return -bess_max_power_discharge
        energy_before_reserve = dischargeable_soe - reserve_energy
        return -energy_before_reserve / (duration_before_reserve.total_seconds() / 3600)


def get_peak_approach_energies(
        t: datetime,
        time_step: timedelta,
        soe: float,
        charge_efficiency: float,
        peaks: List[Peak],
        is_long: bool,
) -> Tuple[float, float]:
    """
    Returns the charge energy required due to the "force" and "encourage" peak approach configuration.

    The 'encourage' peak approach will try to get the battery to a target SoE by charging whenever the system is long
    AND the current SoE is below a threshold which is defined by the timings in the configuration.

    The 'force' peak approach will get the battery to a target SoE by charging, even if the system is short,
    AND the current SoE is below a threshold which is defined by the timings in the configuration.

    With multiple peaks configured, the next-upcoming peak today is the one whose approach drives charging.
    Once that peak starts, dispatch in `get_peak_power` takes over; immediately after it ends, the following
    peak (if any later today) becomes the next-upcoming and its approach kicks in.

    FOr a more detailed description of this mechanism, see the docstring on the `Approach` configuration class in simulator/config/config.py

    :param t: the time now
    :param time_step: the size of the simulation time step
    :param soe: the current battery soe
    :param charge_efficiency:
    :param peaks: list of peak configurations (single-peak schemas pass a one-element list)
    :param is_long: indicates if the system is long or short - the encourage curve is only used when the system is long
    :return:
    """
    # TODO: this approach won't work if the approach curve crosses over a midnight boundary

    if not peaks:
        return 0.0, 0.0

    t = t.astimezone(TIMEZONE)

    peak_config = _find_next_peak(peaks, t)
    if peak_config is None:
        return 0.0, 0.0

    if peak_config.approach.to_soe == 0 and peak_config.approach.encourage_to_soe == 0:
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
        if peak_config.approach.encourage_to_soe:
            encourage_to_soe = peak_config.approach.encourage_to_soe
        else:
            encourage_to_soe = peak_config.approach.to_soe
        encourage_curve = _get_approach_curve(
            peak_start=peak_start,
            peak_end=peak_end,
            to_soe=encourage_to_soe,
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
