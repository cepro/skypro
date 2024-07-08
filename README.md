# skypro-cli
Scenario analysis tool for microgrids

## Installation

To install from test.pypi:
- `pip3 install --upgrade --extra-index-url https://test.pypi.org/simple/ skypro`

## Usage
See `skypro -h` for help.

To run a BESS simulation and plot the results: `skypro simulate -c <config-file> -o ./output.csv --plot`


### Scenario configuration file

A JSON configuration file defines the simulation scenario and is in the format shown in the example below:
```json
{
  "configFormatVersion": "2.0.0",
  "simulation": {
    "start": "2024-03-01T00:00:00+00:00", -- when to begin the simulation 
    "end": "2024-04-01T00:00:00+00:00", -- when to end the simulation
    "site": {
      "gridConnection": {
        "importLimit": 220,
        "exportLimit": 475
      },
       "solar": { -- defines either a scaled profile, or a consant solar output
         "profile": {
           "profileDir": "~/simt_data/site/solar_profile/",
           "profiledSizeKwp": 4.4,
           "scaledSizeKwp": 210
         },
         "constant": 0
      },
      "load": { -- defines either a scaled profile, or a constant domestic load
        "profile": {
           "profileDir": "~/simt_data/site/load_profile/",
           "profiledNumPlots": 10,
           "scaledNumPlots": 54
         },
        "constant": 20
      },
      "bess": {
        "energyCapacity": 1280,
        "nameplatePower": 565,
        "chargeEfficiency": 0.85
      }
    },
    "strategy": {
      "priceCurveAlgo": { -- this is currently the only supported algorithm
        "doFullDischargeWhenExportRateApplies": "duosRed",
        "nivChasePeriods": [
          {
            "period": {  -- the days and times at which the following "niv chasing" configuration applies
              "days": "all:Europe/London",
              "start": "00:00:00:Europe/London",
              "end": "23:59:59:Europe/London"
            },
            "niv": {
              "chargeCurve": [ -- defines the Price vs SoE curve that defines when the battery should charge 
                {"x": -Infinity, "y": 1280},
                {"x": 4, "y": 1280},
                {"x": 6, "y": 1000},
                {"x": 12, "y": 0}
              ],
              "dischargeCurve": [ -- defines the Price vs SoE curve that defines when the battery should discharge
                {"x": 8, "y": 1280},
                {"x": 18, "y": 1000},
                {"x": 25, "y": 1000},
                {"x": 35, "y": 0},
                {"x": Infinity, "y": 0}
              ],
              "curveShiftLong": 6,  -- bias towards charging when then the imbalance volume is negative
              "curveShiftShort": 2,  -- bias towards discharging when then the imbalance volume is positive
              "volumeCutoffForPrediction": 150000  -- when the previous imbalance volume is larger than this number, then the imbalance price from the previous SP is used for the first 10 minutes of the next SP 
            }
          }
        ]
      }
    },
    "imbalanceDataSource": {  -- location of imbalance pricing and volume data
       "priceDir": "~/simt_data/modo/imbalance_price/",
       "volumeDir": "~/simt_data/modo/imbalance_volume/"
    },
    "rates": {  -- location of rates JSON configuraiton files
      "supplyPointsConfigFile": "./config/rates/supply_points_a.json",
      "ratesConfigFiles": [
        "./config/rates/dno_fees_wpd.json",
        "./config/rates/supply_fees_a.json"
      ]
    }
  }
}
```

## Running tests
To run the unit tests: `python -m unittest discover  --start-directory src`

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
