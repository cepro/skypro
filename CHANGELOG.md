# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
