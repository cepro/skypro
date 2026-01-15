# Skypro Codebase

Python CLI for microgrid simulation and reporting. Published to pypi.org.

## Project Structure

```
src/skypro/
├── main.py              # CLI entry point
├── commands/
│   ├── simulator/       # `skypro simulate` command
│   └── report/          # `skypro report` command
├── common/
│   └── rates/           # Rate modelling (volumetric, fixed, market, internal)
└── reporting_webapp/    # DEPRECATED Streamlit app
```

## CLI Commands

### `skypro simulate`
```bash
skypro simulate -c <config.yaml> -o ./output.csv --plot
skypro simulate --help
```

Projects microgrid behaviour over historical data with configurable control strategies:
- `perfectHindsightOptimiser` - LP-based optimal hindsight
- `priceCurveAlgo` - Real-time price curve following

### `skypro report`
```bash
skypro report -c <config.yaml> -m 2025-04 --plot
skypro report --help
```

Analyzes real metering data, generates supplier invoice estimates, reports data inconsistencies as Notices.

## Seven Energy Flows

```
solar_to_batt    solar_to_load    solar_to_grid
grid_to_batt     grid_to_load
batt_to_load     batt_to_grid
```

Each flow has associated market rates (actual £) and internal rates (notional value for optimization).

## Configuration

### Environment (`~/.simt/env.json`)

Since v2.0.0, the environment file requires a `flux` section for the flux database:

```json
{
  "vars": { "SKYPRO_DIR": "/path/to/skyprospector.com" },
  "flows": { "dbUrl": "postgres://..." },
  "flux": { "dbUrl": "postgres://...", "schema": "flux" },
  "rates": { "dbUrl": "postgres://..." }
}
```

**Schema configuration:**
- `flows.schema` - defaults to "flows" (plot meter tables)
- `flux.schema` - defaults to "flux" (meter readings function, BESS readings, market data)

For legacy databases where everything is in `flows` schema, set `flux.schema = "flows"`.

### Simulation Config (YAML)
See `src/tests/integration/fixtures/simulation/config.yaml` for annotated example.

Key sections:
- `timeFrame` - start/end dates
- `site.gridConnection` - import/export limits (kW)
- `site.bess` - energyCapacity (kWh), nameplatePower (kW), chargeEfficiency
- `site.solar/load` - profile sources (CSV dirs, constants, scaling)
- `rates` - JSON/YAML files for each of the 7 flows
- `strategy` - control algorithm config
- `output` - CSV paths, aggregation, detail level

## Development

**IMPORTANT:** Always ask the user before pushing commits to the `main` branch on GitHub.

### Install
```bash
pip install --upgrade skypro
```

### Run Tests
```bash
PYTHONPATH=src python -m unittest discover --start-directory src
```

### Publish to PyPI

**Credentials:** Store PyPI API token in `~/.simt/pypi.token` (one line, token only).

Generate token at: https://pypi.org/manage/account/token/

```bash
# 1. Update version in pyproject.toml
# 2. Build
python -m build

# 3. Publish (reads token from ~/.simt/pypi.token)
TWINE_USERNAME=__token__ TWINE_PASSWORD=$(cat ~/.simt/pypi.token) python -m twine upload dist/*

# Or use uv:
uv publish dist/* --token $(cat ~/.simt/pypi.token)
```

**Test PyPI** (for feature branches):
```bash
# Token from https://test.pypi.org/manage/account/token/
# Store in ~/.simt/testpypi.token
TWINE_USERNAME=__token__ TWINE_PASSWORD=$(cat ~/.simt/testpypi.token) python -m twine upload --repository testpypi dist/*
```

## Key Concepts

### Rates
- **Volumetric** (p/kWh) - DUoS, supplier fees, levies
- **Fixed** (p/day, p/kVA/day) - standing charges
- **Market** - actual cashflows with suppliers
- **Internal** - opportunity cost for optimization

### OSAM (P395)
On-site Allocation Methodology for calculating final demand levies. Runs in parallel with Skypro's own methodology; discrepancies reported as Notices.

### Control Strategies
- **Perfect Hindsight LP** - Optimal solution knowing future prices
- **Price Curve Algorithm** - Real-time with NIV chase, peak shaving, load following

## Dependencies

pandas, plotly, pulp, pendulum, sqlalchemy, psycopg2-binary, marshmallow, pyyaml

## Merge Log

| Date | PR | Branch | Summary |
|------|-----|--------|---------|
| 2026-01-12 | - | main | License changed from AGPL-3.0 to MIT |
| 2026-01-09 | #59 | bugfix/nan-threshold-aggregation | Fix NaN propagation in rate averages and cost totals (v2.0.1) |
| 2026-01-09 | #58 | feature/flux-schema-support | Flux/flows schema separation for DB queries (v2.0.0) |
| 2025-08-12 | #50 | example-configs | Self-contained example configs, peak config toggle |
| 2025-08-07 | #49 | load-discrepancy-include-ev | Include EV load in totals, load energy breakdown in CSV |
| 2025-08-07 | #48 | remove-streamlit-app | Revert streamlit removal, add deprecated warning |
| 2025-08-05 | #47 | remove-streamlit-app | Remove streamlit webapp (later reverted) |
| 2025-08-05 | #46 | customer-rates-from-db | Customer rates from database (v1.1.0) |
| 2025-06-11 | #44 | open-source-preparation | Merge simt-common into repo, anonymise profiles, add tests |
| 2025-06-04 | #42 | open-source-review | Code cleanup, drop spread algo, improve comments |
| 2025-05-15 | #41 | rates-db | Rates database support, site-specific rates (v0.19.0) |
