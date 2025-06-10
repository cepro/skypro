from typing import Optional, List
from uuid import UUID

from marshmallow_dataclass import dataclass

from skypro.common.config.utility import enforce_one_option, field_with_opts


@dataclass
class CSVDataSource:
    """
    Base class for any data sources that come from CSV files. The data could be in a single `file` or in a
    directory (`dir`) of CSV files.
    """
    dir: Optional[str]
    file: Optional[str]

    def __post_init__(self):
        enforce_one_option([self.dir, self.file], "'dir' or 'file'")


@dataclass
class CSVMeterReadingsDataSource(CSVDataSource):
    """
    Configures microgrid-level meter readings that are read from CSV files
    """
    meter_id: UUID = field_with_opts(key="meterId")


@dataclass
class CSVBessReadingsDataSource(CSVDataSource):
    """
    Configures BESS readings that are read from CSV files
    """
    bess_id: UUID = field_with_opts(key="bessId")


@dataclass
class CSVPlotMeterReadingsDataSource(CSVDataSource):
    """
    Configures plot-level meter readings that are read from CSV files
    """
    feeder_ids: List[UUID] = field_with_opts(key="feederIds")


@dataclass
class CSVProfileDataSource(CSVDataSource):
    """
    Configures a load or solar profile that is read from CSV files.
    """
    pass  # There are no extra fields beyond the CSVDataSource base class.


@dataclass
class CSVTimeseriesDataSource(CSVDataSource):
    """
    Configures a generic timeseries that is read from CSV files (for example imbalance price or volume).

    A timeseries can be marked as 'predictive' - in which case there will be multiple predictions for the
    same point in time. For example, Modo predicts the imbalance price for a single settlement period
    many times, with the later predictions having increasing accuracy.
    """
    is_predictive: Optional[bool] = field_with_opts(key="isPredictive", default=False)
