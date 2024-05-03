from dataclasses import dataclass
from itertools import tee
from typing import List

import numpy as np


@dataclass
class Point:
    """
    Represents are cartesian point.
    """
    x: float
    y: float


class Curve:
    """
    Represents a 'curve' made up of multiple cartesian points, each joined using linear interpolation.
    """
    def __init__(self, points: List[Point]):
        self._points = points

    def __str__(self):
        ret_str = "["
        for p in self._points:
            ret_str += f"({p.x:.1f}, {p.y:.1f}) "
        ret_str += "]"

        return ret_str

    def __repr__(self):
        return self.__str__()

    def vertical_distance(self, p: Point) -> float:
        """
        Returns the vertical (y-axis) distance from the given point to the Curve, a positive number indicating that the
        point is below the curve, and vice-versa.
        NaN is returned if the distance could not be calculated, this can happen if the given point is not within the
        horizontal span of the curve.
        """
        # Loop over each pair of points in the curve
        for p1, p2 in pairwise(self._points):
            # Check if the given point is 'within the vertical band' of the two current points
            if p1.x < p.x < p2.x:
                # Use linear interpolation to find where the curve is at the given point
                curve_y = np.interp(p.x, [p1.x, p2.x], [p1.y, p2.y])
                distance = curve_y - p.y
                return float(distance)

        return np.NaN


# TODO: should be able to remove this if we specify Python 3.10
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)
