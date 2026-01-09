# Skypro Codebase

Python CLI for microgrid simulation and reporting. Published to test.pypi.org.

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

### Install
```bash
pip install --upgrade --extra-index-url https://test.pypi.org/simple/ skypro
```

### Run Tests
```bash
PYTHONPATH=src python -m unittest discover --start-directory src
```

### Publish
1. Update version in `pyproject.toml`
2. `poetry build`
3. `poetry publish -r test-pypi`

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
