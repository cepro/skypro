# Skypro simulator

Simulations model a microgrids behaviour, battery control algorithms, and the resulting costs/revenues.

Simulations are run against historical/pre-defined load and solar profiles, national SSP/imbalance pricing, and site-specific rate information like DUoS and Supplier fees.
A simulation can be run over any length of time, but simulating a full year captures seasonality well.
The simulations time span is split into ten minute chunks - at the start of each ten minutes the control algorithm is given any information that would reasonably be available to it at the time, and then makes a decision about battery control.

When the simulation has finished some summary statistics are given (similar to the reporting tool), this includes:
- Supplier invoice estimates, i.e. the costs associated with imports and revenues associated with exports.
- Summaries of energy flows in the microgrid, and their associated prices and costs.
- Solar statistics, including self-use.
- BESS statistics, including cycling and roundtrip efficiency.


## Usage

See `skypro simulate -h` for help with command line options.

## Configuration

See the integration tests for an example configuration YAML file with inline comments: `src/tests/integration/fixtures/config.yaml`

Also, see the `src/skypro/commands/simulator/config/config.py` file for the configuration definition in code with comments.