# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.2.0] - 2026-05-10

### Added

- **`enabled: false` annotation on simulations**: Scenarios marked
  `enabled: false` in `simulate.yaml` are now dropped before strict
  schema validation, letting externally-managed scenarios (e.g.
  Monte-Carlo runs whose outputs are produced outside skypro) coexist
  with live scenarios in the same yaml. Removes the need for callers
  (e.g. the rebuilder in skypro-service) to write a temp staged yaml
  to scrub them out.

- **Per-flow `imbalanceDataSourceOverride` on `RatesFiles`**: Each flow
  in a rates block can now override the block-level `imbalanceDataSource`
  via a new dict shape:

  ```yaml
  rates:
    final:
      imbalanceDataSource: *imbSrc_default     # block-level default
      files:
        gridToBatt:                            # legacy list shape — inherits default
          - dno_import.json
          - supply_import.json
        solarToGrid:                           # new dict shape with override
          rates:
            - dno_export.json
            - supply_statkraft_export.json
          imbalanceDataSourceOverride: *imbSrc_plain
  ```

  Closes the structural Axle-premium leak on two-MPAN BSCP550 sites
  where the BESS and site MPANs settle against different imbalance
  signals. Backward-compatible — legacy list shape on flows continues
  to parse.

- **Multi-`finals` per simulation**: Declare N settlement variants in
  one simulation. Mutually exclusive with the legacy `final: <Rates>`
  field (same precedent as `peak`/`peaks` in priceCurveAlgo):

  ```yaml
  rates:
    live:  *ratesLive_basecase
    finals:
      fullfcl:      *ratesFinal_fullfcl
      trio_imbflex: *ratesFinal_trio_imbflex
  ```

  Fanned out at parse time into one expanded `SimulationCase` per
  variant (sim name `<orig>.<variant>`). CSV paths get an automatic
  variant suffix when they don't use `$_SIM_NAME`. The optimiser runs
  N times — accepted trade-off for YAML ergonomics; the
  single-dispatch / multi-settlement-column variant is deferred (see
  `docs/proposals/multi-final-rates.md`).

### Changed

- `RatesFiles` flow fields are now `FlowFilesType` (a `FlowFiles`
  dataclass) rather than `List[PathType]`. Schema parsing accepts
  either list or dict shape transparently. **No migration needed for
  existing YAML configs.** Internal callers that previously accessed
  e.g. `rates_files.grid_to_batt[0]` need to use
  `rates_files.grid_to_batt.rates[0]`. Only consumer affected was
  `parse_vol_rates_files_for_all_energy_flows`, which has been
  updated.

- `parse_vol_rates_files_for_all_energy_flows` accepts an optional
  `flow_imbalance_pricings: Dict[str, pd.Series]` keyword argument
  for per-flow override pricings. Cache key now
  `(file_list_str, id(pricing))` so flows with the same files but
  different overrides don't share rate instances.

### Compatibility

- Legacy single-`final` configs unchanged. Verified by integration
  tests (`integrationTestPriceCurve`, `integrationTestPriceCurveMultiPeak`,
  `integrationTestPerfectHindsightLP`) — bit-identical LP output
  within tolerance `0.01`.
- Legacy list-shape rate files continue to parse and behave
  identically.
- `ratesDB` source rejects per-flow overrides with a clear error
  (override only supported with the YAML `files` source).

## [2.0.5] - 2026-05-07

### Fixed

- `skypro simulate` now skips constructing a SQLAlchemy engine for the
  flows database when all imbalance data sources are `csvTimeseries`.
  Previously the engine was created eagerly and unconditionally, so a
  placeholder or unparseable `env_config["flows"]["dbUrl"]` raised
  before any simulation logic ran — blocking standalone CSV-only
  configuration bundles. When at least one source is `flowsMarketData`
  the engine is still constructed, with a clearer error if `dbUrl` is
  missing.

## [2.0.0] - Unreleased

### Breaking Changes

- **Environment configuration**: Added required `flux` section to env.json for database connections
- The `skypro report` command now requires a `flux` database URL in the environment configuration

### Added

- Support for separate `flows` and `flux` database schemas
- Configurable schema names via env.json (`schema` field in `flows` and `flux` sections)
- Schema defaults: `flows.schema = "flows"`, `flux.schema = "flux"`

### Changed

- Meter readings function (`get_meter_readings_5m`) now queries from `flux` schema
- BESS readings now query from `flux.mg_bess_readings_30m`
- Market data (imbalance prices) now query from `flux.market_data`
- Plot meter readings continue to query from `flows` schema tables

### Migration Guide

Update your env.json to include the `flux` section:

```json
{
  "flows": { "dbUrl": "postgres://..." },
  "flux": { "dbUrl": "postgres://...", "schema": "flux" },
  "rates": { "dbUrl": "postgres://..." }
}
```

For databases where all data is in the `flows` schema (legacy setup), set `flux.schema = "flows"`:

```json
{
  "flux": { "dbUrl": "postgres://...", "schema": "flows" }
}
```

## [1.2.0] - 2024

### Added

- Example configurations for simulations (self-contained examples with data)
- LP optimiser with constraint management example
- Multiple tagged load profiles example
- File-based rates example
- Peak configuration can now be disabled in price curve algorithm

### Fixed

- Database-based simulations now work correctly
