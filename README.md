# skypro-cli
Scenario analysis tool for microgrids

## Installation

To install from test.pypi:
- `pip3 install --extra-index-url https://test.pypi.org/simple/ skypro`

## Usage
See `skypro -h` for help.

To run a BESS simulation and plot the results: `skypro simulate -c <scenario-configuration> -o ./output.csv --plot`


### Scenario configuration file

A JSON configuration file defines the simulation scenario and is in the format shown in the example below:
```JSON
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
              "chargeCurve": [
                {
                  "x": -Infinity,
                  "y": 1280
                },
                {
                  "x": 4,
                  "y": 1280
                },
                {
                  "x": 6,
                  "y": 1000
                },
                {
                  "x": 12,
                  "y": 0
                }
              ],
              "dischargeCurve": [
                {
                  "x": 8,
                  "y": 1280
                },
                {
                  "x": 18,
                  "y": 1000
                },
                {
                  "x": 25,
                  "y": 1000
                },
                {
                  "x": 35,
                  "y": 0
                },
                {
                  "x": Infinity,
                  "y": 0
                }
              ],
              "curveShiftLong": 6,
              "curveShiftShort": 2,
              "volumeCutoffForPrediction": Infinity
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
