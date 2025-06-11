import unittest
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import pytz

from skypro.common.rate_utils.osam import calculate_osam_ncsp


class TestCalculateOsamNcsp(unittest.TestCase):

    def test_calculate_osam_ncsp(self):

        @dataclass
        class SubCase:
            msg: str
            df: pd.DataFrame
            index_to_calc_for: pd.DatetimeIndex
            expected_nscp: pd.Series

        data_index = pd.date_range(
            start=gb_time("2024-09-27T00:00:00"),
            end=gb_time("2024-09-27T23:30:00"),
            freq='30min'
        )
        index_to_calc_for = pd.date_range(
            start=gb_time("2024-09-28T00:00:00"),
            end=gb_time("2024-09-28T23:30:00"),
            freq='30min'
        )

        def initialise_df(index: pd.DatetimeIndex):
            df = pd.DataFrame(index=index)
            for col in ["imp_bp", "exp_bp", "imp_stor", "exp_stor", "imp_gen", "exp_gen"]:
                df[col] = 0.0
            return df

        grid_trade_all_df = initialise_df(data_index)
        solar_shift_all_df = initialise_df(data_index)
        grid_trade_some_df = initialise_df(data_index)

        # 5kWh from Grid -> Stor and back out again
        grid_trade_all_df.loc[gb_time("2024-09-27T01:00:00"), "imp_bp"] = 5.0
        grid_trade_all_df.loc[gb_time("2024-09-27T01:00:00"), "imp_stor"] = 5.0
        grid_trade_all_df.loc[gb_time("2024-09-27T07:00:00"), "exp_bp"] = 5.0
        grid_trade_all_df.loc[gb_time("2024-09-27T07:00:00"), "exp_stor"] = 5.0

        # 5kWh from Grid -> Stor, 1kWh back out grid, and 4kWh to load
        grid_trade_some_df.loc[gb_time("2024-09-27T01:00:00"), "imp_bp"] = 5.0
        grid_trade_some_df.loc[gb_time("2024-09-27T01:00:00"), "imp_stor"] = 5.0
        grid_trade_some_df.loc[gb_time("2024-09-27T07:00:00"), "exp_bp"] = 1.0
        grid_trade_some_df.loc[gb_time("2024-09-27T07:00:00"), "exp_stor"] = 5.0

        # 5kWh from Gen -> Stor and then Stor -> Other
        solar_shift_all_df.loc[gb_time("2024-09-27T12:00:00"), "exp_gen"] = 5.0
        solar_shift_all_df.loc[gb_time("2024-09-27T12:00:00"), "imp_stor"] = 5.0
        solar_shift_all_df.loc[gb_time("2024-09-27T19:00:00"), "exp_stor"] = 5.0

        # Do an equal mix of solar shifting and grid trading
        solar_shift_and_grid_trade_df = grid_trade_all_df.copy()
        solar_shift_and_grid_trade_df.loc[gb_time("2024-09-27T12:00:00"), "exp_gen"] = 5.0
        solar_shift_and_grid_trade_df.loc[gb_time("2024-09-27T12:00:00"), "imp_stor"] = 5.0
        solar_shift_and_grid_trade_df.loc[gb_time("2024-09-27T19:00:00"), "exp_stor"] = 5.0

        data_index_2 = pd.date_range(
            start=gb_time("2024-10-15T00:00:00"),
            end=gb_time("2024-10-30T23:30:00"),
            freq='30min'
        )
        index_to_calc_for_2 = pd.date_range(
            start=gb_time("2024-10-17T04:00:00"),
            end=gb_time("2024-10-31T23:30:00"),
            freq='30min'
        )
        long_df = initialise_df(data_index_2)

        for day in data_index_2.day.unique():
            # This is the same as grid_trade_some_df, but just repeated over many days
            long_df.loc[gb_time(f"2024-10-{day}T01:00:00"), "imp_bp"] = 5.0
            long_df.loc[gb_time(f"2024-10-{day}T01:00:00"), "imp_stor"] = 5.0
            long_df.loc[gb_time(f"2024-10-{day}T07:00:00"), "exp_bp"] = 1.0
            long_df.loc[gb_time(f"2024-10-{day}T07:00:00"), "exp_stor"] = 5.0

        sub_cases = [
            SubCase(
                msg="Only grid trading everything gives NSCP 1.0",
                df=grid_trade_all_df,
                index_to_calc_for=index_to_calc_for,
                expected_nscp=pd.Series(index=index_to_calc_for, data=1.0)
            ),
            SubCase(
                msg="Grid trading 1/5th gives NSCP 0.2",
                df=grid_trade_some_df,
                index_to_calc_for=index_to_calc_for,
                expected_nscp=pd.Series(index=index_to_calc_for, data=0.2)
            ),
            SubCase(
                msg="Only solar shifting gives NSCP 0.0 (but this would later be applied to a zero grid -> stor flow)",
                df=solar_shift_all_df,
                index_to_calc_for=index_to_calc_for,
                expected_nscp=pd.Series(index=index_to_calc_for, data=0.0)
            ),
            SubCase(
                # It's frustrating that this is the case, as it's not optimal/fair settlement for us
                msg="Equal mix of grid trading and solar storage gives an NSCP of 0.5",
                df=solar_shift_and_grid_trade_df,
                index_to_calc_for=index_to_calc_for,
                expected_nscp=pd.Series(index=index_to_calc_for, data=0.5)
            ),
            SubCase(
                # It's frustrating that this is the case, as it's not optimal/fair settlement for us
                msg="Long time range over clock change",
                df=long_df,
                index_to_calc_for=index_to_calc_for_2,
                expected_nscp=pd.Series(index=index_to_calc_for_2, data=0.2)
            ),
        ]

        for sub_case in sub_cases:
            with self.subTest(msg=sub_case.msg):
                result, _ = calculate_osam_ncsp(
                    df=sub_case.df,
                    index_to_calc_for=sub_case.index_to_calc_for,
                    imp_bp_col="imp_bp",
                    exp_bp_col="exp_bp",
                    imp_stor_col="imp_stor",
                    exp_stor_col="exp_stor",
                    imp_gen_col="imp_gen",
                    exp_gen_col="exp_gen",
                )
                self.assertEqual(result.equals(sub_case.expected_nscp), True)


def gb_time(s) -> datetime:
    return pytz.timezone("Europe/London").localize(datetime.fromisoformat(s))
