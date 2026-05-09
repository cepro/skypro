# Skypro core change — per-flow `imbalanceDataSource` override

**Audience:** Skypro engineer (Python core)
**Author:** Damon Rand
**Status:** ✅ IMPLEMENTED in v2.2.0 (commit `ca77995` on `feature/multi-and-per-flow-rates`).
The implementation matches this brief — schema lives inside `RatesFiles`
as a new `FlowFiles` dataclass with optional `imbalanceDataSourceOverride`.
Legacy list shape continues to parse. See `CHANGELOG.md` v2.2.0 for the
final YAML reference.

## Background — why this matters

Today skypro's `Rates` config sets `imbalanceDataSource` once at the
rates-block level. Every flow that consumes imbalance (via the
`imbalance` rate type, optionally with a `multiplierRate` like
Statkraft × imbalance) reads from that single source:

```yaml
rates:
  final:
    imbalanceDataSource: <one source per rates block>
    files:
      gridToBatt:  [ ... ]   # all see the same imbalance source
      gridToLoad:  [ ... ]
      battToGrid:  [ ... ]
      solarToGrid: [ ... ]
```

This works for single-MPAN sites where the entire boundary settles
against one imbalance signal. It breaks for **two-MPAN BSCP550 sites**
where the BESS sits on its own MPAN and the residential supply is on
a different one — the two MPANs have structurally different
imbalance treatments and shouldn't share a source.

### Concrete problem — HMCE BSCP550 + Axle leak

At HMCE Apr-26 the BSCP550 metering split puts the BESS on its own
MPAN. Under that arrangement:

- **BESS MPAN** participates in Axle's flex programme → its imbalance
  signal gets the Axle premium stacked on top (via
  `imbalance_price_with_axle/`).
- **Site MPAN** (residential supply) does NOT participate in Axle —
  it should see plain imbalance (`imbalance_price/`).

Because skypro has only one `imbalanceDataSource` per rates block,
the current `simulate.yaml` configures **all flows** in a BSCP550
rates anchor to read from `imbSrc_elexon_axle` (Axle-stacked). The
result: `gridToLoad` and `solarToGrid` (site-MPAN flows) absorb the
Axle premium they shouldn't see.

Quantified bias: **~£500–700/mo over-attribution** on HMCE Apr-26
BSCP550 scenarios (Axle window HHs × site-load and site-export volumes
× BESS-share-scaled Axle premium). The simulator's `margin` column on
those scenarios overstates HMCE's actual revenue by this amount, and
the optimiser sees biased prices on site flows so dispatch decisions
are slightly off too.

We worked around this in
`projects/mgfl/hazelmead/202604/tuning_history.md` (2026-05-09 entry)
by publishing the biased numbers with an explicit caveat. This change
closes the gap.

## What to build

Allow individual flows in a rates block to override the block-level
`imbalanceDataSource`. Either as a per-flow override or as a property
of the `imbalance` rate type. Sketch (exact schema TBD):

```yaml
rates:
  final:
    imbalanceDataSource: *imbSrc_elexon_plain     # default for site flows
    files:
      gridToBatt:                                  # BESS MPAN
        imbalanceDataSourceOverride: *imbSrc_elexon_axle
        rates:
          - dno_fees_southwest_import.json
          - supply_fees_unify_import.json
      battToGrid:                                  # BESS MPAN
        imbalanceDataSourceOverride: *imbSrc_elexon_axle
        rates:
          - dno_fees_southwest_export.json
          - supply_fees_statkraft_export.json
      gridToLoad:                                  # Site MPAN — default
        rates:
          - dno_fees_southwest_import.json
          - supply_fees_unify_import.json
          - final_consumption_levies.yaml
      solarToGrid:                                 # Site MPAN — default
        rates:
          - dno_fees_southwest_export.json
          - supply_fees_statkraft_export.json
```

Backward-compat: if no per-flow override is set, behaviour matches
today (block-level source applies to all flows). The current YAML
shape (`gridToBatt: [<file>, <file>]`) should stay valid; the new
override-capable shape is a per-flow opt-in.

## Existing entry points worth knowing about

(Treat as starting points; full design is the implementer's call.)

- **Schema**: `Rates.files` parsing in
  `src/skypro/commands/simulator/config/config.py`. The `files` field
  currently accepts `Dict[str, List[str]]`; needs to accept a richer
  per-flow shape that carries an optional override.
- **Imbalance rate construction**: wherever the `imbalance` rate type
  is built from rate files — that's the consumption point that
  currently reads the block-level `imbalanceDataSource`. The override
  should plug in there.
- **`generate_output_df()`** in `src/skypro/common/microgrid_analysis/output.py`
  — should already cope as long as the rate engine produces the right
  per-flow rate dataframes. Output column naming unchanged.

## Rough scope

**~4–6 hours** as a focused PR. Slightly bigger than the
multi-`ratesFinal` change because it touches per-flow rate
construction, not just an outer loop.

Roughly:
1. Schema change: new per-flow override field (config.py).
2. Plumb the override through to the `imbalance` rate constructor.
3. Backward-compat: existing list-shape continues to work.
4. Unit test: a rates block with mixed per-flow imbalance sources
   produces the expected per-flow rate dataframes.
5. Integration test: re-run a BSCP550 HMCE scenario before/after,
   confirm Axle premium only lands on `battToGrid` / `gridToBatt`.

## Known risks / red flags

1. **Multiplier rates referencing the wrong source.** Files like
   `supply_fees_statkraft_export.json` reference imbalance via
   `multiplierRate`. With per-flow override, the multiplier needs to
   resolve to the per-flow source, not the block-level one. Verify
   the multiplier-resolve path picks up overrides correctly.
2. **OSAM/P395 NCSP coupling.** OSAM's NCSP factor is dispatch-driven
   and shared across flows. The per-flow override doesn't change
   NCSP — it just changes which imbalance signal each flow's rate
   stack consumes. Sanity-check that NCSP application still resolves
   per-flow, not per-imbalance-source.
3. **YAML readability degrades** with the richer per-flow shape.
   Consider keeping the simple list shape as the default and only
   requiring the dict shape when an override is set. Don't force every
   site to migrate.

## Out of scope for this change

- Multi-`ratesFinal` per simulation (separate brief at
  `multi-final-rates.md`). Both changes are independent and can
  ship in either order.
- Changes to the optimiser, algorithm layer, or output column naming.
- Restructuring how `imbSrc_*` anchors are defined in YAML (the
  override pattern reuses existing imbalance source anchors).
- skypro-fresh dashboard surfaces (the change is invisible to the
  consumer side as long as output column names stay stable).

## Validation hint

**HMCE 202604 is the validation case.** After this change ships:

1. Update `projects/mgfl/hazelmead/202604/simulate.yaml` BSCP550
   rate anchors (`ratesFinal_bscp550`, `ratesLive_impr12`,
   `ratesLive_impr2`) to override imbalance source per flow:
   - `gridToBatt`, `battToGrid` → `imbSrc_*_axle`
   - `gridToLoad`, `solarToGrid` → `imbSrc_*_plain`
2. Re-run `./tools/rebuild projects/mgfl/hazelmead/202604/`.
3. Compare margin columns for the HEADLINE scenario
   `hmce.202604.1609kWh-210kWp.imb-imbflex.cc-twopeaks.bscp550`
   against the 2026-05-09 entry in
   `projects/mgfl/hazelmead/tuning_history.md` — should drop by
   ~£500–700/mo, removing the over-attribution to gridToLoad +
   solarToGrid.
4. Once 202604 validates, roll the same per-flow wiring out to other
   HMCE timeframes (202601, 202602, etc.).

NCESF doesn't need migration — its solarToGrid uses a flat PPA rate
that doesn't reference imbalance, and `gridToLoad` = 0 (no onsite
load), so there's nothing to leak Axle onto.
