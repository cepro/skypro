import unittest
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import pandas as pd

from skypro.commands.report.flow_calcs import calculate_missing_net_flows_in_junction


class TestJunction(unittest.TestCase):

    def test_calculate_missing_net_flows_in_junction(self):

        @dataclass
        class SubCase:
            msg: str
            df: pd.DataFrame
            cols_with_direction: List[Tuple[str, int]]
            expected_df: pd.DataFrame
            expected_num_notices: int

        sub_cases = [
            SubCase(
                msg="Nothing to calculate",
                df=pd.DataFrame({
                    "A": [1, 2, 3],
                    "B": [4, 5, 6],
                    "C": [7, 8, 9]
                }),
                cols_with_direction=[
                    ("A", 1),
                    ("B", 1),
                    ("C", 1),
                ],
                expected_df=pd.DataFrame({
                    "A": [1, 2, 3],
                    "B": [4, 5, 6],
                    "C": [7, 8, 9]
                }),
                expected_num_notices=0
            ),
            SubCase(
                msg="A is missing",
                df=pd.DataFrame({
                    "A": [np.nan, np.nan, np.nan],
                    "B": [2, 1, -1],
                    "C": [3, 1, 1]
                }),
                cols_with_direction=[
                    ("A", 1),
                    ("B", -1),
                    ("C", 1),
                ],
                expected_df=pd.DataFrame({
                    "A": [-1, 0, -2],
                    "B": [2, 1, -1],
                    "C": [3, 1, 1]
                }),
                expected_num_notices=1
            ),
            SubCase(
                msg="B/C is missing",
                df=pd.DataFrame({
                    "A": [0, 1, 2],
                    "B": [np.nan, np.nan, -1],
                    "C": [3, 1, np.nan]
                }),
                cols_with_direction=[
                    ("A", 1),
                    ("B", -1),
                    ("C", 1),
                ],
                expected_df=pd.DataFrame({
                    "A": [0, 1, 2],
                    "B": [3, 2, -1],
                    "C": [3, 1, -3]
                }),
                expected_num_notices=2
            ),
        ]

        for sub_case in sub_cases:
            with (self.subTest(sub_case.msg)):
                df, notices = calculate_missing_net_flows_in_junction(
                    df=sub_case.df,
                    cols_with_direction=sub_case.cols_with_direction,
                )
                self.assertEqual(
                    np.isclose(df, sub_case.expected_df, rtol=0.01).all().all(),
                    True
                )
                print(notices)
                self.assertEqual(len(notices), sub_case.expected_num_notices)

