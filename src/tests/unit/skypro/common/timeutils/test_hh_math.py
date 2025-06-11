import unittest
from dataclasses import dataclass
from datetime import datetime

from skypro.common.timeutils.math import floor_hh


class TestSettlementPeriods(unittest.TestCase):

    def test_floor_hh(self):

        @dataclass
        class SubCase:
            msg: str
            t: datetime
            expected_t: datetime

        sub_cases = [
            SubCase(
                msg="already on hh boundary",
                t=datetime.fromisoformat("2024-02-03T11:00:00+00:00"),
                expected_t=datetime.fromisoformat("2024-02-03T11:00:00+00:00"),
            ),
            SubCase(
                msg="just into HH",
                t=datetime.fromisoformat("2024-02-03T11:01:02+00:00"),
                expected_t=datetime.fromisoformat("2024-02-03T11:00:00+00:00"),
            ),
            SubCase(
                msg="well into HH",
                t=datetime.fromisoformat("2024-02-03T11:29:02+00:00"),
                expected_t=datetime.fromisoformat("2024-02-03T11:00:00+00:00"),
            ),
            SubCase(
                msg="In BST",
                t=datetime.fromisoformat("2024-06-03T11:29:02+01:00"),
                expected_t=datetime.fromisoformat("2024-06-03T11:00:00+01:00"),
            )
        ]

        for sub_case in sub_cases:
            with self.subTest(msg=sub_case.msg):
                result = floor_hh(sub_case.t)
                self.assertEqual(result, sub_case.expected_t)


