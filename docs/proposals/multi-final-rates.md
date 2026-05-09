# Skypro core change — support multiple `ratesFinal` per simulation

**Audience:** Skypro engineer (Python core)
**Author:** Damon Rand
**Status:** Brief — design + implementation TBD

## Background — why this matters

Today skypro's simulation is configured with a single pair of rates:

```yaml
rates:
  live:  { ... }     # what the optimiser sees, drives dispatch
  final: { ... }     # what gets settled, drives reported margin
```

In several real workflows we want to evaluate the **same dispatch
decision against multiple alternative settlement rate structures**.
Re-running the optimiser is unnecessary in those cases — the dispatch
is locked once `ratesLive` is fixed; only the cost columns change.

### Concrete example — Axle / flex on/off

A common pattern: optimise against base imbalance prices (`ratesLive =
imbalance`), then settle the same dispatch under two final structures
in parallel:

1. `final.no_flex = imbalance` — what the operator earns without any
   flex aggregator (counterfactual).
2. `final.with_flex = imbalance + axle_premium × bess_share` — what
   the operator actually banks once Axle / a Virtual Trading Party
   takes their cut.

Today this requires running skypro twice. The optimiser runs both
times even though it's the same dispatch question. We want one
optimiser run, two parallel settlement column-sets in the output CSV.

### Other use cases

- **Tariff alternatives**: Trio vs flat vs Octopus Tracker for the same
  dispatch. Useful for proposals to housing developers / customers.
- **OSAM on/off** as a sensitivity (subject to the OSAM-NCSP caveat
  below — it's dispatch-driven, not rate-driven, so OSAM-NCSP is a
  shared input across variants).
- **Per-flow settlement structures** that exist today but require
  separate sims: e.g. site MPAN settles at imbalance, BESS MPAN
  settles at imbalance+flex (the BSCP550 pattern at HMCE — though
  see the per-flow-imbalance-source-override.md brief for a more
  targeted fix to that specific case).

## What to build

Extend the YAML schema so `rates.final` can be **either** a single
`Rates` object (backward-compatible) **or** a dict of named variants:

```yaml
rates:
  live: { ... }
  final:
    no_flex:    { ... }
    with_flex:  { ... }
```

Output CSV columns get a variant suffix: `mvRate:battToGrid.final[no_flex]`,
`mvRate:battToGrid.final[with_flex]`, etc. Single-variant case (current
schema) stays as `mvRate:battToGrid.final` — no breaking change.

## Existing entry points worth knowing about

(Verified via read of skypro core. Treat as starting points; full
design is the implementer's call.)

- **Schema**: `AllRates` dataclass at
  `src/skypro/commands/simulator/config/config.py:378`. Make
  `final` polymorphic.
- **Re-rating loop**: `_process_final_rates()` at
  `src/skypro/commands/simulator/main.py:328`. The function
  is pure (input → output) given a fixed dispatch. Loop it N times
  once the optimiser is done.
- **Output column generation**: `generate_output_df()` at
  `src/skypro/common/microgrid_analysis/output.py:98`.
  Currently iterates over `(int_final, mkt_final, int_live, mkt_live)`
  rate dicts. Extend to iterate per variant.
- **OSAM/P395 NCSP**: `calculate_osam_ncsp()` at
  `src/skypro/common/rate_utils/osam.py:15`. NCSP is
  **dispatch-dependent but rate-variant-independent** — compute once
  after the optimiser, reuse across all variants.
- **`skypro report`** reuses `generate_output_df()`. Same change
  benefits report-side rate switching for free.

## Rough scope

**~3–4 hours** as a focused PR. Not a week-long architectural project.

Roughly:
1. Schema + parser tweak (singular vs dict `final`).
2. Hoist the OSAM NCSP calc above the re-rating loop.
3. Loop `_process_final_rates()` per variant; collect per-variant
   rate dataframes.
4. Extend column naming in `generate_output_df()` to emit
   `*.final[<variant>]` when multiple variants are configured.
5. Backward-compat regression test on the single-variant path.

## Known risks / red flags

1. **Column naming ambiguity.** Don't keep a rollup
   `mvRate:battToGrid.final` alongside `mvRate:battToGrid.final[v1]`
   and `[v2]` — downstream consumers (skypro-fresh dashboards,
   axle_reconcile, ad-hoc analysis scripts) would have to guess
   whether the rollup is a sum, an average, or stale. Cleaner: drop
   the rollup, require variant names in brackets when multiple are
   configured.
2. **OSAM rate-instance state mutation.** `OSAMFlatVolRate.add_ncsp()`
   mutates the rate instance in place. If the same OSAM rate object
   appears in multiple final variants, the second variant's add_ncsp
   could double-apply. Either deep-copy rate instances per variant
   in the loop, or refactor NCSP to be a runtime parameter rather
   than mutated state.
3. **CSV row-width explosion.** With N variants, you get roughly
   N×2 extra columns per flow per HH. For 12-month detail dumps
   with 3 variants, expect ~3× the file size. Consider an opt-in
   summary-only output mode for variant-heavy runs, or document
   the cost.

## Out of scope for this change

- Changes to the optimiser or algorithm layer — none required.
- Changes to YAML rate parsing primitives — already list-aware.
- OSAM math itself — unchanged.
- Per-flow `imbalanceDataSource` override — separate brief at
  `per-flow-imbalance-source-override.md`. Both changes are
  independent and can ship in either order.
- skypro-fresh dashboard surfaces for the new variant columns
  (consumer-side work, separate).

## Validation hint

Run `hmce.202604` Axle-aware scenarios with `final = { with_axle,
without_axle }`. The `with_axle` column-set should exactly equal
the current single-`final` output. The `without_axle` column-set
should equal what the existing Impr0 family scenarios
(`hmce.202604.*.imb-imb.*.bscp550.imb-imbflex` etc.) produce today
when run as standalone sims with `ratesLive ≠ ratesFinal`. After
this change ships, those Impr0 sims become redundant — one
multi-`ratesFinal` sim replaces the pair.
