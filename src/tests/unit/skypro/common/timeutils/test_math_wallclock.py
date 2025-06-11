import unittest
from dataclasses import dataclass
from datetime import datetime

import pytz

from skypro.common.timeutils.math_wallclock import add_wallclock_days


class TestAddWallclockDays(unittest.TestCase):

    def test_add_wallclock_days(self):

        @dataclass
        class SubCase:
            msg: str
            t: datetime
            n_days: int
            expected_t: datetime

        def make_datetime(datetime_str, tz_str):
            t = datetime.fromisoformat(datetime_str)
            tz = pytz.timezone(tz_str)
            t = tz.localize(t)
            return t

        sub_cases = [
            SubCase(
                msg="GMT +1d -> GMT",
                t=make_datetime("2024-02-03T11:00:00", "Europe/London"),
                n_days=1,
                expected_t=datetime.fromisoformat("2024-02-04T11:00:00+00:00"),
            ),
            SubCase(
                msg="GMT -5d -> GMT",
                t=make_datetime("2024-02-03T23:00:00", "Europe/London"),
                n_days=-5,
                expected_t=datetime.fromisoformat("2024-01-29T23:00:00+00:00"),
            ),
            SubCase(
                msg="BST +3d -> BST",
                t=make_datetime("2024-06-03T12:00:00", "Europe/London"),
                n_days=3,
                expected_t=datetime.fromisoformat("2024-06-06T12:00:00+01:00"),
            ),
            SubCase(
                msg="BST -1d -> BST",
                t=make_datetime("2024-06-03T00:00:00", "Europe/London"),
                n_days=-1,
                expected_t=datetime.fromisoformat("2024-06-02T00:00:00+01:00"),
            ),
            SubCase(
                msg="GMT +30d -> BST",
                t=make_datetime("2024-03-15T08:00:00", "Europe/London"),
                n_days=30,
                expected_t=datetime.fromisoformat("2024-04-14T08:00:00+01:00"),
            ),
            SubCase(
                msg="BST -30d -> GMT",
                t=make_datetime("2024-04-14T12:00:00", "Europe/London"),
                n_days=-30,
                expected_t=datetime.fromisoformat("2024-03-15T12:00:00+00:00"),
            ),
            SubCase(
                msg="BST +5d -> GMT",
                t=make_datetime("2024-10-26T12:00:00", "Europe/London"),
                n_days=5,
                expected_t=datetime.fromisoformat("2024-10-31T12:00:00+00:00"),
            ),
            SubCase(
                msg="GMT -5d -> BST",
                t=make_datetime("2024-10-31T20:00:00", "Europe/London"),
                n_days=-5,
                expected_t=datetime.fromisoformat("2024-10-26T20:00:00+01:00"),
            ),
        ]

        for sub_case in sub_cases:
            with self.subTest(msg=sub_case.msg):
                result = add_wallclock_days(sub_case.t, sub_case.n_days)
                self.assertEqual(result, sub_case.expected_t)


