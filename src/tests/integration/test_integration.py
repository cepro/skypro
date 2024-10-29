import logging
import unittest
import subprocess
from dataclasses import dataclass

import pandas as pd


class TestIntegration(unittest.TestCase):

    def test_integration(self):
        """
        This spawns the Skypro cli tool in a subprocess, runs it with a set of known inputs/configurations, and checks
        that the summarised results are within tolerance.
        This should allow us to ensure that the results do not change unexpectedly, although of course if the algorithm
        genuinely changes then we should expect the results to change and update them appropriately here.
        """

        @dataclass
        class SubTest:
            msg: str
            sim_name: str
            expected_summary_df: pd.DataFrame

        subtests = [
            SubTest(
                msg="integrationTestPriceCurve",
                sim_name="integrationTestPriceCurve",
                expected_summary_df=pd.DataFrame.from_dict({
                    "c:solarToGrid": [872.40],
                    "c:gridToLoad": [29364.75],
                    "c:solarToLoad": [5022.52],
                    "c:battToLoad": [3339.95],
                    "c:battToGrid": [22557.51],
                    "c:solarToBatt": [309.14],
                    "c:gridToBatt": [29774.99],

                    "irate:gridToBatt.final": [8.1332],
                    "irate:battToGrid.final": [-17.5380],
                    "irate:solarToGrid.final": [-8.2932],
                    "irate:gridToLoad.final": [10.8306],
                    "irate:solarToBatt.final": [6.1364],
                    "irate:battToLoad.final": [-21.8243],
                    "irate:solarToLoad.final": [-10.6678],

                    "rate:gridToBatt.final": [8.1332],
                    "rate:battToGrid.final": [-17.5380],
                    "rate:solarToGrid.final": [-8.2932],
                    "rate:gridToLoad.final": [10.8306],
                    "rate:solarToBatt.final": [0.0],
                    "rate:battToLoad.final": [0.0],
                    "rate:solarToLoad.final": [0.0],
                })
            ),
            SubTest(
                msg="integrationTestPerfectHindsightLP",
                sim_name="integrationTestPerfectHindsightLP",
                expected_summary_df=pd.DataFrame.from_dict({
                    "c:solarToGrid": [277.51],
                    "c:gridToLoad": [18951.33],
                    "c:solarToLoad": [5022.52],
                    "c:battToLoad": [13753.37],
                    "c:battToGrid": [31305.08],
                    "c:solarToBatt": [904.03],
                    "c:gridToBatt": [51352.98],

                    "irate:gridToBatt.final": [7.4687],
                    "irate:battToGrid.final": [-16.4390],
                    "irate:solarToGrid.final": [-10.8284],
                    "irate:gridToLoad.final": [9.1808],
                    "irate:solarToBatt.final": [6.7763],
                    "irate:battToLoad.final": [-15.7736],
                    "irate:solarToLoad.final": [-10.6678],

                    "rate:gridToBatt.final": [7.4687],
                    "rate:battToGrid.final": [-16.4390],
                    "rate:solarToGrid.final": [-10.8284],
                    "rate:gridToLoad.final": [9.1808],
                    "rate:solarToBatt.final": [0.0],
                    "rate:battToLoad.final": [0.0],
                    "rate:solarToLoad.final": [0.0],
                })
            ),
        ]

        for sub in subtests:
            with self.subTest(sub.msg):
                logging.info(f"Starting integration test '{sub.msg}'...")
                res = subprocess.run([
                    'python3',
                    './src/skypro/main.py',
                    'simulate',
                    '--env',
                    './src/tests/integration/fixtures/env.json',
                    '-y',
                    '--config',
                    './src/tests/integration/fixtures/config.yaml',
                    '--sim',
                    sub.sim_name,
                ])
                logging.info("Skypro finished running")

                if res.returncode != 0:
                    raise ValueError("Non zero exit code")

                df = pd.read_csv("./src/tests/integration/output_summary.csv")

                columns_to_compare = sub.expected_summary_df.columns
                df = df[columns_to_compare]

                error = (df - sub.expected_summary_df).abs()
                tolerance = 0.01

                self.assertEqual(
                    (error > tolerance).sum().sum(),
                    0,
                    msg=f"Summary value out of tolerance:\n{error.transpose()}"
                )
