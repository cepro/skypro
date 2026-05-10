"""
Unit tests for the YAML parser helpers added in the multi-and-per-flow-rates work:

  - `_add_variant_suffix`: collision-free CSV path naming for fanned-out variants
  - `_expand_multi_final_simulations`: fan-out from `rates.finals` dict into
    one SimulationCase per variant

End-to-end behaviour (`enabled:false` drop, schema validation) is exercised
via the existing integration test in `test_integration_simulator.py`.
"""
import unittest
from types import SimpleNamespace

from skypro.commands.simulator.config.parse_config import (
    _add_variant_suffix,
    _expand_multi_final_simulations,
)


def _make_sim(*, finals=None, final="legacy_final", csv="$_SIM_NAME.csv"):
    """Build a SimulationCase-shaped object with just enough structure for the
    fan-out helper. We use SimpleNamespace because `_expand_multi_final_simulations`
    only accesses `.rates.{final,finals}` and `.output.summary.csv`.
    """
    rates = SimpleNamespace(final=final, finals=finals)
    summary = SimpleNamespace(csv=csv)
    output = SimpleNamespace(summary=summary, simulation=None)
    return SimpleNamespace(rates=rates, output=output)


class TestAddVariantSuffix(unittest.TestCase):

    def test_inserts_before_extension(self):
        self.assertEqual(_add_variant_suffix("/tmp/out.csv", "v1"), "/tmp/out.v1.csv")

    def test_skipped_when_sim_name_token_present(self):
        # $_SIM_NAME paths get the variant suffix naturally via the substitution
        # loop after fan-out (sim name is now `<orig>.<variant>`); leaving these
        # untouched here avoids double-suffixing.
        self.assertEqual(_add_variant_suffix("$_SIM_NAME.csv", "v1"), "$_SIM_NAME.csv")
        self.assertEqual(
            _add_variant_suffix("outputs/$_SIM_NAME.summary.csv", "v1"),
            "outputs/$_SIM_NAME.summary.csv",
        )

    def test_works_without_extension(self):
        self.assertEqual(_add_variant_suffix("/tmp/out", "v1"), "/tmp/out.v1")


class TestExpandMultiFinalSimulations(unittest.TestCase):

    def test_legacy_single_final_unchanged(self):
        sim = _make_sim(final="legacy_final", finals=None)
        result = _expand_multi_final_simulations({"legacy": sim})
        self.assertEqual(list(result.keys()), ["legacy"])
        self.assertIs(result["legacy"], sim)

    def test_finals_block_expands_to_n_sims(self):
        sim = _make_sim(final=None, finals={"plain": "rates_a", "alt": "rates_b"})
        result = _expand_multi_final_simulations({"hmce": sim})
        self.assertEqual(set(result.keys()), {"hmce.plain", "hmce.alt"})
        self.assertNotIn("hmce", result)

    def test_each_variant_has_resolved_final_and_no_finals(self):
        sim = _make_sim(final=None, finals={"a": "rates_a", "b": "rates_b"})
        result = _expand_multi_final_simulations({"hmce": sim})
        self.assertEqual(result["hmce.a"].rates.final, "rates_a")
        self.assertIsNone(result["hmce.a"].rates.finals)
        self.assertEqual(result["hmce.b"].rates.final, "rates_b")
        self.assertIsNone(result["hmce.b"].rates.finals)

    def test_sim_name_token_paths_pass_through(self):
        # The substitution loop downstream substitutes $_SIM_NAME with `<orig>.<variant>`
        # so we don't add the suffix here.
        sim = _make_sim(final=None, finals={"a": "_", "b": "_"}, csv="$_SIM_NAME.csv")
        result = _expand_multi_final_simulations({"sim": sim})
        self.assertEqual(result["sim.a"].output.summary.csv, "$_SIM_NAME.csv")
        self.assertEqual(result["sim.b"].output.summary.csv, "$_SIM_NAME.csv")

    def test_hardcoded_csv_path_gets_variant_suffix(self):
        sim = _make_sim(final=None, finals={"a": "_", "b": "_"}, csv="/tmp/out.csv")
        result = _expand_multi_final_simulations({"clash": sim})
        self.assertEqual(result["clash.a"].output.summary.csv, "/tmp/out.a.csv")
        self.assertEqual(result["clash.b"].output.summary.csv, "/tmp/out.b.csv")

    def test_variants_are_deep_copied(self):
        # Mutating one variant's output must not bleed into a sibling's.
        sim = _make_sim(final=None, finals={"a": "_", "b": "_"})
        result = _expand_multi_final_simulations({"sim": sim})
        result["sim.a"].output.summary.csv = "MUTATED"
        self.assertNotEqual(result["sim.b"].output.summary.csv, "MUTATED")

    def test_order_preserved(self):
        # Mixed legacy + multi-final preserves declaration order; each variant lands
        # in the position of its parent sim, in finals-declaration order.
        first = _make_sim(final="x", finals=None)
        second = _make_sim(final=None, finals={"alpha": "_", "beta": "_"})
        third = _make_sim(final="y", finals=None)
        result = _expand_multi_final_simulations({
            "first": first, "second": second, "third": third,
        })
        self.assertEqual(
            list(result.keys()),
            ["first", "second.alpha", "second.beta", "third"],
        )


if __name__ == "__main__":
    unittest.main()
