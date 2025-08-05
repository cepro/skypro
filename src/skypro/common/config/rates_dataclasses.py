from typing import List, Optional, Dict

from marshmallow_dataclass import dataclass

from skypro.common.config.data_source import ImbalanceDataSource
from skypro.common.config.path_field import PathType
from skypro.common.config.utility import field_with_opts, enforce_one_option


@dataclass
class SiteSpecifier:
    """
    Configures the region (e.g. 'south_west') and LV/HV banding for a site. This allows locational and
    banding-specific rates to be pulled from a rates database automatically. For example, if we know
    that a site is in the south_west region on the LV3 band then we can pull all the DUoS rates, and
    capacity charges etc.
    """
    region: str
    bands: List[str]


@dataclass
class CustomerRatesDB:
    """
    Configures rates for customers (i.e. domestic homes) to be pulled from a database
    """
    import_bundles: List[str] = field_with_opts(key="importBundles")  # Names of any import rate bundles to use for the customer load
    export_bundles: List[str] = field_with_opts(key="exportBundles")  # Names of any export rate bundles to use for the customer export


@dataclass
class RatesDB:
    """
    Configures rates to be pulled from a database.
    """
    supply_points_name: str = field_with_opts(key="supplyPoints")  # The name of the supply point as defined in the database
    site_specific: SiteSpecifier = field_with_opts(key="siteSpecific")  # This will resolve location and banding-specific rates.
    import_bundles: List[str] = field_with_opts(key="importBundles")  # Names of any import rate bundles to use in addition to the site specific ones (e.g. Supplier arrangements)
    export_bundles: List[str] = field_with_opts(key="exportBundles")  # Names of any export rate bundles to use in addition to the site specific ones (e.g. Supplier arrangements).
    future_offset_str: Optional[str] = field_with_opts(key="futureOffset")  # For simulations, it can be useful to bring the rates forwards in time, for example we might want to use the 2025 rates for a simulation run over 2024
    customer: Optional[CustomerRatesDB]  # Optionally define rates for customers - these are only really used for reporting purposes as this doesn't affect control algorithms


@dataclass
class RatesFiles:
    """
    Configures rates to be pulled from YAML files, with a list of files for each microgrid flow.
    Each rate definition file may define one or more rates.
    """
    solar_to_batt: List[PathType] = field_with_opts(key="solarToBatt")
    grid_to_batt: List[PathType] = field_with_opts(key="gridToBatt")
    batt_to_grid: List[PathType] = field_with_opts(key="battToGrid")
    batt_to_load: List[PathType] = field_with_opts(key="battToLoad")
    solar_to_grid: List[PathType] = field_with_opts(key="solarToGrid")
    solar_to_load: List[PathType] = field_with_opts(key="solarToLoad")
    grid_to_load: List[PathType] = field_with_opts(key="gridToLoad")


@dataclass
class ExperimentalRates:
    """
    The "market fixed rates" and "customer load rates" are in this experimental configuration
    block as a beta feature.
    Market fixed rates are £/day or £/kVA/day rates charged by third parties (e.g. suppliers/DNOs)
    Customer rates are the p/kWh and standing charge that we charge to our microgrid customers.
    """
    mkt_fixed_files: Dict[str, List[PathType]] = field_with_opts(key="marketFixedCostFiles")
    customer_load_files: Dict[str, List[PathType]] = field_with_opts(key="customerLoadFiles")


@dataclass
class Rates:
    """
    Configures the rates that a microgrid is exposed to. These can be configured to come from either local YAML files or
    to come from a rates database.
    """
    imbalance_data_source: ImbalanceDataSource = field_with_opts(key="imbalanceDataSource")  # Some rates are based on the imbalance price, so this configures the data source for imbalance data.
    files: Optional[RatesFiles]  # Used if the rates are coming from local YAML files
    rates_db: Optional[RatesDB] = field_with_opts(key="ratesDB")  # Used if the rates are coming from a database

    # TODO: the following elements are only relevant if RatesFiles is configured, so they should be moved into the `RatesFiles` object. But this would be a breaking change for existing configurations.
    experimental: Optional[ExperimentalRates]
    supply_points_config_file: Optional[PathType] = field_with_opts(key="supplyPointsConfigFile")  # This is

    def __post_init__(self):
        enforce_one_option([self.files, self.rates_db], "'files' or 'ratesDB'")

        if self.rates_db is None and self.supply_points_config_file is None:
            raise ValueError("If using rates 'files' than you must specify the 'supplyPointsConfigFile'")

        if self.rates_db is not None and self.supply_points_config_file is not None:
            raise ValueError("If using 'ratesDB' than you must not specify the 'supplyPointsConfigFile'")
