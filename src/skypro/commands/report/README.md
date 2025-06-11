# Skypro reporting

The reporting CLI tool collates historical metering actuals and rate/pricing information about a microgrid, and then analyses the data to produce a report which gives:
- Supplier invoice estimates, i.e. the costs associated with imports and revenues associated with exports.
- Summaries of energy flows in the microgrid, and their associated prices and costs.
- Solar statistics, including self-use.
- BESS statistics, including cycling and roundtrip efficiency.

The tool also reports on data inconsistencies by presenting 'Notices' to the user, which are graded according to how serious the issue is.
Some data inconsistencies are always going to be present because, for example, meters are not 100% accurate.
However, other data inconsistencies may be transient, for example, some metering data may be temporarily missing and so the reporting tool may need to make approximations.  


## Usage

See `skypro report -h` for help with command line options.

## Configuration

TODO: see integration tests for an example configuration YAML file with inline comments.

Also, see the `skypro/commands/report/config/config.py` file for the configuration definition in code with comments.