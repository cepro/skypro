from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import List

import numpy as np
from simt_common.timeutils.days import Days

from skypro.commands.simulator.cartesian import Curve, Point


@dataclass
class TimePoint:
    """
    Represents are cartesian point.
    """
    t: time
    y: float



REF_DATE = date(1000, 1, 1)
REF_DATETIME = datetime.combine(REF_DATE, time())


def time_curve_to_curve(time_curve: List[TimePoint]) -> Curve:

    points = []
    for time_point in time_curve:
        dt = datetime.combine(REF_DATE, time_point.t)
        duration = dt - REF_DATETIME
        points.append(Point(
            x=duration.total_seconds(),
            y=time_point.y
        ))

    return Curve(points=points)


# TODO: deal with DST properly
red_curve = time_curve_to_curve([
    TimePoint(time(10, 0), 0),
    TimePoint(time(15, 30), 1000),  # The soe is currently allowed to get 'behind' the curve, so allow an extra 30 mins to 'catch up with the curve'
    TimePoint(time(16, 0), 1000)
])
approach_days = Days(name="weekdays", tz_str="UTC")

amber_curve = time_curve_to_curve([
    TimePoint(time(0, 0), 0),
    TimePoint(time(15, 0), 1000),
    TimePoint(time(16, 0), 1000)
])


def get_red_approach_energy(t, soe) -> float:

    if not approach_days.is_on_day(t):
        return 0

    # TODO: use time_step rather than hard-coding 10mins - but the addition didn't seem to work?!
    target_time = datetime.combine(REF_DATE, time(t.hour, t.minute, t.second)) + timedelta(minutes=10)
    now_point = Point(
        x=(target_time - REF_DATETIME).total_seconds(),
        y=soe
    )

    red_approach_distance = red_curve.vertical_distance(now_point)
    if np.isnan(red_approach_distance) or red_approach_distance < 0:
        return 0

    return red_approach_distance


def get_amber_approach_energy(t, soe, imbalance_volume_assumed) -> float:

    if imbalance_volume_assumed > 0 or not approach_days.is_on_day(t):
        return 0

    # TODO: use time_step rather than hard-coding 10mins - but the addition didn't seem to work?!
    target_time = datetime.combine(REF_DATE, time(t.hour, t.minute, t.second)) + timedelta(minutes=10)
    now_point = Point(
        x=(target_time - REF_DATETIME).total_seconds(),
        y=soe
    )

    amber_approach_distance = amber_curve.vertical_distance(now_point)
    if np.isnan(amber_approach_distance) or amber_approach_distance < 0:
        return 0

    return amber_approach_distance