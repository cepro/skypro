import logging
import unittest
import subprocess
from dataclasses import dataclass

import pandas as pd


class TestIntegration(unittest.TestCase):

    def test_integration(self):
        """
        This spawns the Skypro reporting tool in a subprocess, runs it with a set of known inputs/configurations, and checks
        that the summarised results are within tolerance.
        This allows us to ensure that the results do not change unexpectedly whilst developing features.
        """

        @dataclass
        class SubTest:
            msg: str
            month: str
            expected_summary_df: pd.DataFrame

        subtests = [
            SubTest(
                msg="report202408",
                month="2024-08",
                expected_summary_df=pd.DataFrame.from_dict({
                    "agd:load": [9290.688],
                    "agd:solar": [21969.16780],
                    "m:upImport": [31385.62743],
                    "m:upExport": [38970.21875],
                    "m:battCharge": [30922.89062],
                    "m:battDischarge": [26568.10938],
                    "c:solarToGrid": [12384.01863],
                    "c:gridToLoad": [4571.26804],
                    "c:solarToLoad": [4860.90924],
                    "c:battToLoad": [456.48816],
                    "c:battToGrid": [26111.62122],
                    "c:solarToBatt": [4583.11013],
                    "c:gridToBatt": [26339.78050],
                    "other:osam.ncsp": [0.96923],
                    "mfCost:import": [48050.00000],
                    "mfCost:export": [0.00000],
                    "cfCost:all": [-62000.00000],
                    "cvRate:all": [-21.00000],
                    "ivRate:gridToBatt.final": [6.00367],
                    "ivRate:battToGrid.final": [-16.24747],
                    "ivRate:solarToGrid.final": [0.00000],
                    "ivRate:gridToLoad.final": [0.00000],
                    "ivRate:solarToBatt.final": [3.88745],
                    "ivRate:battToLoad.final": [-19.21144],
                    "ivRate:solarToLoad.final": [-10.17220],
                    "mvRate:solarToBatt.final": [0.00000],
                    "mvRate:gridToBatt.final": [6.00367],
                    "mvRate:battToLoad.final": [0.00000],
                    "mvRate:battToGrid.final": [-16.24747],
                    "mvRate:solarToGrid.final": [-5.39167],
                    "mvRate:solarToLoad.final": [0.00000],
                    "mvRate:gridToLoad.final": [9.74784],
                })
            ),
        ]

        for sub in subtests:
            with self.subTest(sub.msg):
                print(f"\n\n\n\nSTARTING REPORTING INTEGRATION TEST '{sub.msg}' - - - - - - - - - - - - - - - - - - - - - - -")
                res = subprocess.run([
                    'python3',
                    './src/skypro/main.py',
                    'report',
                    '--env',
                    './src/tests/integration/fixtures/env.json',
                    '-y',
                    '-c',
                    './src/tests/integration/fixtures/reporting/config.yaml',
                    '-s',
                    './src/tests/integration/reporting_output_summary.csv',
                    '-m',
                    '2024-08'
                ])
                logging.info("Skypro reporting finished running")

                if res.returncode != 0:
                    raise ValueError("Non zero exit code")

                df = pd.read_csv("./src/tests/integration/reporting_output_summary.csv")

                columns_to_compare = sub.expected_summary_df.columns
                df = df[columns_to_compare]

                error = (df - sub.expected_summary_df).abs()
                tolerance = 0.01

                self.assertEqual(
                    (error > tolerance).sum().sum(),
                    0,
                    msg=f"Summary value out of tolerance:\n{error.transpose()}"
                )
