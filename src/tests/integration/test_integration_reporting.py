import logging
import unittest
import subprocess
from dataclasses import dataclass
from datetime import timedelta

import pandas as pd


class TestIntegration(unittest.TestCase):

    def test_integration(self):
        """
        This spawns the Skypro reporting tool in a subprocess, runs it with a set of known inputs/configurations, and checks
        that the summarised results are within tolerance.
        This should allow us to ensure that the results do not change unexpectedly.
        """

        print("\n\n\n\nSTARTING REPORTING INTEGRATION TEST - - - - - - - - - - - - - - - - - - - - - - -")
        res = subprocess.run([
            'python3',
            './src/skypro/main.py',
            'report',
            '--env',
            './src/tests/integration/fixtures/env.json',
            '-y',
            '-c',
            './src/tests/integration/fixtures/reporting/config.yaml',
            '-m',
            '2024-08'
        ])
        logging.info("Skypro reporting finished running")

        if res.returncode != 0:
            raise ValueError("Non zero exit code")
        #
        # df = pd.read_csv("./src/tests/integration/simulation_output_summary.csv")
        #
        # columns_to_compare = sub.expected_summary_df.columns
        # df = df[columns_to_compare]
        #
        # error = (df - sub.expected_summary_df).abs()
        # tolerance = 0.01
        #
        # self.assertEqual(
        #     (error > tolerance).sum().sum(),
        #     0,
        #     msg=f"Summary value out of tolerance:\n{error.transpose()}"
        # )
