from typing import List, Optional, Dict

from marshmallow_dataclass import dataclass

from skypro.common.config.data_source import ImbalanceDataSource
from skypro.common.config.path_field import PathType
from skypro.common.config.utility import field_with_opts, enforce_one_option


@dataclass
class RatesFiles:
    solar_to_batt: List[PathType] = field_with_opts(key="solarToBatt")
    grid_to_batt: List[PathType] = field_with_opts(key="gridToBatt")
    batt_to_grid: List[PathType] = field_with_opts(key="battToGrid")
    batt_to_load: List[PathType] = field_with_opts(key="battToLoad")
    solar_to_grid: List[PathType] = field_with_opts(key="solarToGrid")
    solar_to_load: List[PathType] = field_with_opts(key="solarToLoad")
    grid_to_load: List[PathType] = field_with_opts(key="gridToLoad")


@dataclass
class ExperimentalRates:
    mkt_fixed_files: Dict[str, List[PathType]] = field_with_opts(key="marketFixedCostFiles")
    customer_load_files: Dict[str, List[PathType]] = field_with_opts(key="customerLoadFiles")


@dataclass
class SiteSpecifier:
    region: str
    bands: List[str]


@dataclass
class RatesDB:
    supply_points_name: str = field_with_opts(key="supplyPoints")
    site_specific: SiteSpecifier = field_with_opts(key="siteSpecific")
    import_bundles: List[str] = field_with_opts(key="importBundles")
    export_bundles: List[str] = field_with_opts(key="exportBundles")
    future_offset_str: Optional[str] = field_with_opts(key="futureOffset")


@dataclass
class Rates:
    """
    Note that this class just holds the paths to the rates/supply point configuration files. The actual parsing of the
    contents of these files is done in the common.config.rates module.
    """
    supply_points_config_file: Optional[PathType] = field_with_opts(key="supplyPointsConfigFile")  # This is only specified if using RatesFiles, otherwise the supply points are pulled from the database
    imbalance_data_source: ImbalanceDataSource = field_with_opts(key="imbalanceDataSource")
    files: Optional[RatesFiles]
    rates_db: Optional[RatesDB] = field_with_opts(key="ratesDB")
    experimental: Optional[ExperimentalRates]

    def __post_init__(self):
        enforce_one_option([self.files, self.rates_db], "'files' or 'ratesDB'")

        if self.rates_db is None and self.supply_points_config_file is None:
            raise ValueError("If using rates 'files' than you must specify the 'supplyPointsConfigFile'")

        if self.rates_db is not None and self.supply_points_config_file is not None:
            raise ValueError("If using 'ratesDB' than you must not specify the 'supplyPointsConfigFile'")
