"""
Tests for the FlowFiles polymorphic schema added in skypro v2.x.

Each flow on a `RatesFiles` block can be expressed as either:
  - a legacy list of paths: `gridToBatt: [a.json, b.json]`
  - a dict carrying an override: `gridToBatt: { rates: [...], imbalanceDataSourceOverride: ... }`

Both shapes deserialise to a `FlowFiles` instance.
"""
import unittest

from skypro.common.config.path_field import PathField
from skypro.common.config.rates_dataclasses import RatesFiles


def _make_rates_files_dict(grid_to_batt):
    """Helper: minimal RatesFiles input with all flows present."""
    return {
        "solarToBatt": [],
        "gridToBatt": grid_to_batt,
        "battToGrid": [],
        "battToLoad": [],
        "solarToGrid": [],
        "solarToLoad": [],
        "gridToLoad": [],
    }


class TestFlowFilesShapes(unittest.TestCase):

    def setUp(self):
        # PathField uses class-level state for env var substitution; ensure deterministic.
        PathField.vars_for_substitution = {}

    def test_legacy_list_shape(self):
        rates_files = RatesFiles.Schema().load(_make_rates_files_dict(
            grid_to_batt=["/tmp/a.json", "/tmp/b.json"]
        ))
        self.assertEqual(rates_files.grid_to_batt.rates, ["/tmp/a.json", "/tmp/b.json"])
        self.assertIsNone(rates_files.grid_to_batt.imbalance_data_source_override)

    def test_dict_shape_without_override(self):
        rates_files = RatesFiles.Schema().load(_make_rates_files_dict(
            grid_to_batt={"rates": ["/tmp/a.json"]}
        ))
        self.assertEqual(rates_files.grid_to_batt.rates, ["/tmp/a.json"])
        self.assertIsNone(rates_files.grid_to_batt.imbalance_data_source_override)

    def test_dict_shape_with_override(self):
        rates_files = RatesFiles.Schema().load(_make_rates_files_dict(
            grid_to_batt={
                "rates": ["/tmp/a.json"],
                "imbalanceDataSourceOverride": {
                    "price": {"csvTimeseries": {"dir": "/tmp/price"}},
                    "volume": {"csvTimeseries": {"dir": "/tmp/volume"}},
                },
            }
        ))
        override = rates_files.grid_to_batt.imbalance_data_source_override
        self.assertIsNotNone(override)
        self.assertEqual(override.price.csv_timeseries_data_source.dir, "/tmp/price")
        self.assertEqual(override.volume.csv_timeseries_data_source.dir, "/tmp/volume")

    def test_other_flows_default_to_no_override(self):
        rates_files = RatesFiles.Schema().load(_make_rates_files_dict(
            grid_to_batt={
                "rates": ["/tmp/a.json"],
                "imbalanceDataSourceOverride": {
                    "price": {"csvTimeseries": {"dir": "/tmp/price"}},
                    "volume": {"csvTimeseries": {"dir": "/tmp/volume"}},
                },
            }
        ))
        # Sibling flows declared as legacy lists must still parse and have no override.
        self.assertIsNone(rates_files.grid_to_load.imbalance_data_source_override)
        self.assertEqual(rates_files.grid_to_load.rates, [])


if __name__ == "__main__":
    unittest.main()
