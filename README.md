# Skypro

Skypro is a collection of simulation and reporting tools for microgrids.

## Simulation CLI tool
The Skypro simulator can project microgrid behaviours, costs and revenues.
This tool is run on the command line using `skypro simulate`.
See `src/skypro/commands/simulator/README.md` for more information.

## Reporting CLI tool
The Skypro reporting tool collates data to documents and analyse real-world performance of microgrids.
This tool is run on the command line using `skypro report`.
See `src/skypro/commands/report/README.md` for more information.

## Reporting web app
The Skypro reporting web app makes reporting results accessible to non-cli users.
This is run using Streamlit.
See `src/skypro/reporting_webapp/README.md` for more information.

## Rates and energy flows
Information about the costs and revenues associated with using power are fundamental to the codebase and a high-level understanding of how they are modelled is important for interpreting results.
See `src/skypro/common/rates/README.md` for a background on how rates and energy flows are modelled in the codebase.

## Secrets
env file


## Installation

To install from test.pypi:
- `pip3 install --upgrade --extra-index-url https://test.pypi.org/simple/ skypro`


## Development

### Running tests
To run the tests: `PYTHONPATH=src python -m unittest discover  --start-directory src`

Some of the tests akin to integration tests which run a whole simulation and check the results.
To run an individual integration test directly: `simulate --env src/tests/integration/fixtures/env.json --config src/tests/integration/fixtures/config.yaml -y --sim integrationTestPerfectHindsightLP`

### Publishing
To publish the repository to test.pypi:
1. Make your code changes
2. Update the semver version number in `pyproject.toml`
3. Commit to git and push
4. Run `poetry build` and observe the new version number
5. Run `poetry publish -r test-pypi` to publish

If this is your first time publishing to test.pypi then you will also need to do the following steps before publishing:
1. `poetry config repositories.test-pypi https://test.pypi.org/legacy/`
2. `poetry config pypi-token.test-pypi  pypi-YYYYYYYY` using an API token from your account on the test.pypi website.
