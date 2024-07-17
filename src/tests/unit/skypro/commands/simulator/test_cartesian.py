import unittest
from dataclasses import dataclass

from skypro.commands.simulator.cartesian import Curve, Point


class TestCurve(unittest.TestCase):

    def test_vertical_distance(self):

        @dataclass
        class Case:
            msg: str
            curve: Curve
            point: Point
            expected_distance: float

        cases = [
            Case(
                msg="below",
                curve=Curve(points=[
                    Point(0, 0),
                    Point(10, 10)
                ]),
                point=Point(5, 0),
                expected_distance=5.0,
            ),
            Case(
                msg="above",
                curve=Curve(points=[
                    Point(0, 0),
                    Point(10, 10)
                ]),
                point=Point(5, 10),
                expected_distance=-5.0,
            )
        ]

        for case in cases:
            with self.subTest(case.msg):
                distance = case.curve.vertical_distance(case.point)
                self.assertEqual(distance, case.expected_distance)

