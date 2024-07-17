import logging
import unittest
import subprocess

import pandas as pd


class TestIntegration(unittest.TestCase):

    def test_integration(self):

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

        expected_df = pd.DataFrame(index=[0], data={
            "bess_charge_from_grid": 29774.994364,
            "bess_charge_from_grid_cost": 2421.649273,
            "bess_charge_from_solar": 309.138352,
            "bess_charge_from_solar_cost": 18.969905,
            "bess_discharge_to_grid": 22557.510469,
            "bess_discharge_to_grid_cost": -3956.128884,
            "bess_discharge_to_load": 3339.945498,
            "bess_discharge_to_load_cost": -728.922904,
            "bess_losses": 4512.619907,
            "solar_to_load": 5022.524864,
            "solar_to_load_cost": -535.794159,
            "solar_to_grid": 872.404967,
            "solar_to_grid_cost": -72.345166,
            "load_from_grid": 29364.752639,
            "load_from_grid_cost": 3180.368026,
        })

        self.assertEqual(df.columns.to_list(), expected_df.columns.to_list())

        error = (df - expected_df).abs()
        tolerance = 0.1

        self.assertEqual(
            (error > tolerance).sum().sum(),
            0,
            msg=f"Summary value out of tolerance:\n{error.transpose()}"
        )
