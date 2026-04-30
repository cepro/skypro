import unittest
import tempfile
import os

from skypro.common.data.get_profile import _get_csv_profile
from skypro.common.config.data_source_csv import CSVProfileDataSource


class TestGetProfileDefensiveFiltering(unittest.TestCase):
    """Anomaly-filter is opt-in via max_energy_per_interval_kwh. Without it, raw values pass
    through unchanged so corrupt source data surfaces loudly downstream."""

    def test_filter_off_by_default(self):
        """Without max_energy_per_interval_kwh, raw values pass through (even huge ones)."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("UTCTime,ClockTime,energy\n")
            f.write("2025-01-01 00:00:00+00:00,2025-01-01 00:00:00+00:00,10.0\n")
            # Massive value — but no threshold set, so it stays.
            f.write("2025-01-01 00:30:00+00:00,2025-01-01 00:30:00+00:00,1000000.0\n")
            f.write("2025-01-01 01:00:00+00:00,2025-01-01 01:00:00+00:00,12.0\n")
            temp_path = f.name

        try:
            source = CSVProfileDataSource(dir=None, file=temp_path)
            df = _get_csv_profile(source, None)  # threshold defaults to None
            energies = df['energy'].tolist()
            self.assertAlmostEqual(energies[0], 10.0)
            self.assertAlmostEqual(energies[1], 1000000.0)  # untouched
            self.assertAlmostEqual(energies[2], 12.0)
        finally:
            os.unlink(temp_path)

    def test_anomalous_values_are_filtered_and_interpolated_when_configured(self):
        """With threshold set, values above it are NaN'd then linearly interpolated."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("UTCTime,ClockTime,energy\n")
            f.write("2025-01-01 00:00:00+00:00,2025-01-01 00:00:00+00:00,10.0\n")
            f.write("2025-01-01 00:30:00+00:00,2025-01-01 00:30:00+00:00,12.0\n")
            f.write("2025-01-01 01:00:00+00:00,2025-01-01 01:00:00+00:00,1000000.0\n")
            f.write("2025-01-01 01:30:00+00:00,2025-01-01 01:30:00+00:00,14.0\n")
            f.write("2025-01-01 02:00:00+00:00,2025-01-01 02:00:00+00:00,16.0\n")
            temp_path = f.name

        try:
            source = CSVProfileDataSource(dir=None, file=temp_path)
            df = _get_csv_profile(source, None, max_energy_per_interval_kwh=500)
            self.assertEqual(len(df), 5)
            energies = df['energy'].tolist()
            self.assertAlmostEqual(energies[0], 10.0)
            self.assertAlmostEqual(energies[1], 12.0)
            self.assertAlmostEqual(energies[2], 13.0)  # interpolated mid-gap
            self.assertAlmostEqual(energies[3], 14.0)
            self.assertAlmostEqual(energies[4], 16.0)
            self.assertTrue(all(e <= 500 for e in energies))
        finally:
            os.unlink(temp_path)

    def test_normal_values_unchanged_when_threshold_set(self):
        """If threshold is set but no values exceed it, the data passes through untouched."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("UTCTime,ClockTime,energy\n")
            f.write("2025-01-01 00:00:00+00:00,2025-01-01 00:00:00+00:00,10.0\n")
            f.write("2025-01-01 00:30:00+00:00,2025-01-01 00:30:00+00:00,20.0\n")
            f.write("2025-01-01 01:00:00+00:00,2025-01-01 01:00:00+00:00,30.0\n")
            temp_path = f.name

        try:
            source = CSVProfileDataSource(dir=None, file=temp_path)
            df = _get_csv_profile(source, None, max_energy_per_interval_kwh=500)
            energies = df['energy'].tolist()
            self.assertAlmostEqual(energies[0], 10.0)
            self.assertAlmostEqual(energies[1], 20.0)
            self.assertAlmostEqual(energies[2], 30.0)
        finally:
            os.unlink(temp_path)

    def test_edge_anomaly_uses_ffill_bfill(self):
        """Anomaly at start of series gets back-filled (interpolation can't help at edges)."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("UTCTime,ClockTime,energy\n")
            f.write("2025-01-01 00:00:00+00:00,2025-01-01 00:00:00+00:00,999999.0\n")
            f.write("2025-01-01 00:30:00+00:00,2025-01-01 00:30:00+00:00,10.0\n")
            f.write("2025-01-01 01:00:00+00:00,2025-01-01 01:00:00+00:00,15.0\n")
            temp_path = f.name

        try:
            source = CSVProfileDataSource(dir=None, file=temp_path)
            df = _get_csv_profile(source, None, max_energy_per_interval_kwh=500)
            energies = df['energy'].tolist()
            self.assertAlmostEqual(energies[0], 10.0)  # back-filled
            self.assertAlmostEqual(energies[1], 10.0)
            self.assertAlmostEqual(energies[2], 15.0)
        finally:
            os.unlink(temp_path)

    def test_negative_anomalous_values_filtered(self):
        """Filter uses abs(), so large-magnitude negative values are also caught."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("UTCTime,ClockTime,energy\n")
            f.write("2025-01-01 00:00:00+00:00,2025-01-01 00:00:00+00:00,10.0\n")
            f.write("2025-01-01 00:30:00+00:00,2025-01-01 00:30:00+00:00,-1000000.0\n")
            f.write("2025-01-01 01:00:00+00:00,2025-01-01 01:00:00+00:00,20.0\n")
            temp_path = f.name

        try:
            source = CSVProfileDataSource(dir=None, file=temp_path)
            df = _get_csv_profile(source, None, max_energy_per_interval_kwh=500)
            energies = df['energy'].tolist()
            self.assertAlmostEqual(energies[0], 10.0)
            self.assertAlmostEqual(energies[1], 15.0)  # interpolated
            self.assertAlmostEqual(energies[2], 20.0)
        finally:
            os.unlink(temp_path)


if __name__ == '__main__':
    unittest.main()
