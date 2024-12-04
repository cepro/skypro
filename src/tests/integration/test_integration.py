import logging
import unittest
import subprocess
from dataclasses import dataclass
from datetime import timedelta

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

        # The simulation end time is 23:40:00, which leads to a strange number of minutes:
        simulation_duration = timedelta(minutes=43180)
        num_days_simulated = simulation_duration / timedelta(days=1)

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

                    "ivRate:gridToBatt.final": [8.1332],
                    "ivRate:battToGrid.final": [-17.5380],
                    "ivRate:solarToGrid.final": [0.0],
                    "ivRate:gridToLoad.final": [0.0],
                    "ivRate:solarToBatt.final": [6.1364],
                    "ivRate:battToLoad.final": [-21.8243],
                    "ivRate:solarToLoad.final": [0.0],

                    "mvRate:gridToBatt.final": [8.1332],
                    "mvRate:battToGrid.final": [-17.5380],
                    "mvRate:solarToGrid.final": [-8.2932],
                    "mvRate:gridToLoad.final": [10.8306],
                    "mvRate:solarToBatt.final": [0.0],
                    "mvRate:battToLoad.final": [0.0],
                    "mvRate:solarToLoad.final": [0.0],

                    "cvRate:domestic": [-21.0],

                    # the fixed charges are applied to a number of days
                    "cfCost:standingCharge": [-2000 * num_days_simulated],
                    "mfCost:meterManagementFee": [1250 * num_days_simulated],
                    "mfCost:supplierFee": [300 * num_days_simulated],
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

                    "ivRate:gridToBatt.final": [7.4687],
                    "ivRate:battToGrid.final": [-16.4390],
                    "ivRate:solarToGrid.final": [0.0],
                    "ivRate:gridToLoad.final": [0.0],
                    "ivRate:solarToBatt.final": [6.7763],
                    "ivRate:battToLoad.final": [-15.7736],
                    "ivRate:solarToLoad.final": [0.0],

                    "mvRate:gridToBatt.final": [7.4687],
                    "mvRate:battToGrid.final": [-16.4390],
                    "mvRate:solarToGrid.final": [-10.8284],
                    "mvRate:gridToLoad.final": [9.1808],
                    "mvRate:solarToBatt.final": [0.0],
                    "mvRate:battToLoad.final": [0.0],
                    "mvRate:solarToLoad.final": [0.0],
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
