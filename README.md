# skypro-cli
Scenario analysis tool for microgrids

## Installation

To install from test.pypi:
- `pip3 install --upgrade --extra-index-url https://test.pypi.org/simple/ skypro`

## Usage
See `skypro -h` for help.

To run a BESS simulation and plot the results: `skypro simulate -c <config-file> -o ./output.csv --plot`


## Running tests
To run the unit tests: `PYTHONPATH=src python -m unittest discover  --start-directory src`

To run integration test directly: `simulate --env /Users/marcuswood/Desktop/all/repos/skypro-cli/src/tests/integration/fixtures/env.json --config /Users/marcuswood/Desktop/all/repos/skypro-cli/src/tests/integration/fixtures/config.yaml -y --sim integrationTestPerfectHindsightLP`

### Publishing to test pypi
To publish the repository to test.pypi:
1. Make your code changes
2. Update the semver version number in `pyproject.toml`
3. Commit to git and push
4. Run `poetry build` and observe the new version number
5. Run `poetry publish -r test-pypi` to publish

If this is your first time publishing to test.pypi then you will also need to do the following steps before publishing:
1. `poetry config repositories.test-pypi https://test.pypi.org/legacy/`
2. `poetry config pypi-token.test-pypi  pypi-YYYYYYYY` using an API token from your account on the test.pypi website.
