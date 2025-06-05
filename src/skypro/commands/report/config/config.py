from typing import Dict, Optional
from uuid import UUID

import yaml
from marshmallow_dataclass import dataclass
from skypro.common.config.bill_match import BillMatchLineItem
from skypro.common.config.path_field import PathField, PathType
from skypro.common.config.rates_dataclasses import Rates
from skypro.common.config.utility import field_with_opts
from skypro.common.config.data_source import MeterReadingDataSource, PlotMeterReadingDataSource, BessReadingDataSource

"""
This file contains configuration schema that is used for reporting actuals on microgrid costs/revenues and battery performance.
The higher-level configuration structures are defined towards the end of the file, and the lower-level structures towards the top.
"""


@dataclass
class MicrogridMeter:
    """
    Configures the data source for a microgrid-level meter (e.g. an Acuvim meter)
    """
    data_source: MeterReadingDataSource = field_with_opts(key="dataSource")


@dataclass
class MicrogridFeederMeter:
    """
    Configures the data source for a microgrid-level feeder meter (e.g. an Acuvim meter), alongside the
    ID that the feeder is assigned in the Flows database.
    """
    data_source: MeterReadingDataSource = field_with_opts(key="dataSource")
    feeder_flows_id: UUID = field_with_opts(key="feederId")


@dataclass
class MicrogridMeters:
    """
    Holds configuration for all the microgrid-level meters on a site
    """
    bess_inverter: MicrogridMeter = field_with_opts(key="bessInverter")
    main_incomer: MicrogridMeter = field_with_opts(key="mainIncomer")
    ev_charger: MicrogridMeter = field_with_opts(key="evCharger")
    feeder_1: MicrogridFeederMeter = field_with_opts(key="feeder1")
    feeder_2: MicrogridFeederMeter = field_with_opts(key="feeder2")


@dataclass
class PlotMeters:
    """
    Configures the data source for plot-level meters (e.g. Emlite meters)
    """
    data_source: PlotMeterReadingDataSource = field_with_opts(key="dataSource")


@dataclass
class Metering:
    """
    Holds configuration for both plot-level and microgrid-level meters
    """
    plot_meters: Optional[PlotMeters] = field_with_opts(key="plotMeters")
    microgrid_meters: MicrogridMeters = field_with_opts(key="microgridMeters")


@dataclass
class Bess:
    """
    Configures the size of a BESS in a microgrid, as well as the data source for BESS readings (which
    hold things like state of energy kWh etc)
    """
    energy_capacity: float = field_with_opts(key="energyCapacity")
    data_source: BessReadingDataSource = field_with_opts(key="dataSource")


@dataclass
class GridConnection:
    """
    Configures the size of the sites grid connection in kVA (this is used for calculating things like
    the £/kVA/day costs).
    """
    import_capacity: float = field_with_opts(key="importCapacity")
    export_capacity: float = field_with_opts(key="exportCapacity")


@dataclass
class BillMatchImportOrExport:
    """
    Configures a set of 'line items' that appear on a Suppliers invoice. Suppliers each have their own
    way of formatting their invoices, and this allows us to express our costs/revenues in the same way as the
    Suppliers, so we can easily compare. For example, some may have a line for "non commodity", and "duos", etc.
    """
    line_items: Dict[str, BillMatchLineItem] = field_with_opts(key="lineItems")


@dataclass
class BillMatch:
    """
    Optionally configures the import and export Supplier invoice formats, see above.
    """
    import_direction: Optional[BillMatchImportOrExport] = field_with_opts(key="import")
    export_direction: Optional[BillMatchImportOrExport] = field_with_opts(key="export")


@dataclass
class Reporting:
    """
    Holds all configuration required to configure reporting of actual costs/revenues of a microgrid.
    """
    metering: Metering
    bess: Bess
    grid_connection: GridConnection = field_with_opts(key="gridConnection")
    bill_match: BillMatch = field_with_opts(key="billMatch")
    profiles_save_dir: PathType = field_with_opts(key="profilesSaveDir")  # Optionally save out the load and solar profiles to disk (useful for running simulations with actual solar and load profiles).
    rates: Rates


@dataclass
class Config:
    reporting: Reporting


def parse_config(file_path: str, env_vars: dict) -> Config:
    """
    Reads in the given file, and returns the parsed Config instance.
    Variables may be specified to substitute into any file paths that are specified in the config - e.g. some paths
    appear like "$SKYPRO_DIR/my_sims/blah.yaml" and the $SKYPRO_DIR will be substituted if there is a corresponding
    entry in `env_vars`.
    """

    # Read in the main config file
    with open(file_path) as config_file:
        # Here we parse the config file as YAML, which is a superset of JSON so allows us to parse JSON files as well
        config_dict = yaml.safe_load(config_file)

        # Set up the variables that are substituted into file paths
        PathField.vars_for_substitution = env_vars

        config = Config.Schema().load(config_dict)

    return config
