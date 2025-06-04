from typing import Optional, List
from uuid import UUID

from marshmallow_dataclass import dataclass

from skypro.common.config.utility import enforce_one_option, field_with_opts


@dataclass
class CSVDataSource:
    dir: Optional[str]
    file: Optional[str]

    # energy_cols: Optional[str] = name_in_json("energyCols")

    def __post_init__(self):
        enforce_one_option([self.dir, self.file], "'dir' or 'file'")


@dataclass
class CSVMeterReadingsDataSource(CSVDataSource):
    meter_id: UUID = field_with_opts(key="meterId")


@dataclass
class CSVBessReadingsDataSource(CSVDataSource):
    bess_id: UUID = field_with_opts(key="bessId")


@dataclass
class CSVPlotMeterReadingsDataSource(CSVDataSource):
    feeder_ids: List[UUID] = field_with_opts(key="feederIds")


@dataclass
class CSVProfileDataSource(CSVDataSource):
    pass


@dataclass
class CSVTimeseriesDataSource(CSVDataSource):
    # Optionally mark the data as 'predictive' which means there may be multiple predictions for a given value
    is_predictive: Optional[bool] = field_with_opts(key="isPredictive", default=False)
