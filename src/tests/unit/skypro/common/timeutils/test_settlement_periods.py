import unittest
from datetime import date, datetime

import pandas as pd

from skypro.common.timeutils.settlement_periods import date_and_sp_num_to_utc_datetime


class TestSettlementPeriods(unittest.TestCase):

    def test_date_and_sp_num_to_utc_datetime(self):

        date_series = pd.Series([
            date(2024, 1, 1),
            date(2024, 1, 1),
            date(2024, 1, 1),
            date(2024, 5, 4),  # in DST/BST
            date(2024, 5, 4),  # in DST/BST
        ])

        sp_number_series = pd.Series([
            1,
            2,
            48,
            20,
            1
        ])

        expected_datetime_series = pd.Series([
            datetime.fromisoformat("2024-01-01T00:00:00+00:00"),
            datetime.fromisoformat("2024-01-01T00:30:00+00:00"),
            datetime.fromisoformat("2024-01-01T23:30:00+00:00"),
            datetime.fromisoformat("2024-05-04T08:30:00+00:00"),
            datetime.fromisoformat("2024-05-03T23:00:00+00:00"),
        ])

        datetime_series = date_and_sp_num_to_utc_datetime(
            date_series=date_series,
            sp_number_series=sp_number_series,
            sp_timezone_str="Europe/London"
        )
        pd.testing.assert_series_equal(datetime_series, expected_datetime_series)

