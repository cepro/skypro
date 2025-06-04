from typing import Optional, List
from uuid import UUID

from marshmallow_dataclass import dataclass

from skypro.common.config.utility import field_with_opts


@dataclass
class FlowsMarketDataSource:
    type: str

    # Optionally mark the data as 'predictive' which means there may be multiple predictions for a given value
    is_predictive: Optional[bool] = field_with_opts(key="isPredictive", default=False)


@dataclass
class FlowsMeterReadingsDataSource:
    meter_id: UUID = field_with_opts(key="meterId")


@dataclass
class FlowsBessReadingsDataSource:
    bess_id: UUID = field_with_opts(key="bessId")


@dataclass
class FlowsPlotMeterReadingsDataSource:
    feeder_ids: List[UUID] = field_with_opts(key="feederIds")  # Pulls data only for meters that sit on these feeders
