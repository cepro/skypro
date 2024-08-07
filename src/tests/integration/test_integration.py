import logging
import unittest
import subprocess

import pandas as pd


class TestIntegration(unittest.TestCase):

    def test_integration(self):
        """
        This spawns the Skypro cli tool in a subprocess, runs it with a set of known inputs/configurations, and checks
        that the summarised results are within tolerance.
        This should allow us to ensure that the results do not change unexpectedly, although of course if the algorithm
        genuinely changes then we should expect the results to change and update them appropriately here.
        """

        logging.info("Starting integration test...")
        res = subprocess.run([
            'python3',
            './src/skypro/main.py',
            'simulate',
            '--env',
            './src/tests/integration/fixtures/env.json',
            '-y',
            '--config',
            './src/tests/integration/fixtures/config.json',
            '--output-summary',
            './src/tests/integration/output_summary.csv',
        ])
        logging.info("Skypro finished running")

        if res.returncode != 0:
            raise ValueError("Non zero exit code")

        df = pd.read_csv("./src/tests/integration/output_summary.csv")

        # The avg rate columns are a simple calculation from the other two columns, so don't bother testing these
        df = df.drop(columns=["int_avg_rate", "ext_avg_rate"])
        df = df.set_index("flow")

        expected_df = pd.DataFrame.from_dict({
            "flow": ["gridToBatt", "battToGrid", "solarToGrid", "gridToLoad", "solarToBatt", "battToLoad", "solarToLoad"],
            "volume": [29774.99, 22557.51, 872.40, 29364.75, 309.14, 3339.95, 5022.52],
            "int_cost": [2421.65, -3956.13, -72.35, 3180.37, 18.97, -728.92, -535.79],
            "ext_cost": [2421.65, -3956.13, -72.35, 3180.37, 0.0, 0.0, 0.0]
        })
        expected_df = expected_df.set_index("flow")

        self.assertEqual(df.columns.to_list(), expected_df.columns.to_list())

        error = (df - expected_df).abs()
        tolerance = 0.1

        self.assertEqual(
            (error > tolerance).sum().sum(),
            0,
            msg=f"Summary value out of tolerance:\n{error.transpose()}"
        )
