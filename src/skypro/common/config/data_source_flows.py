from typing import Optional, List
from uuid import UUID

from marshmallow_dataclass import dataclass

from skypro.common.config.utility import field_with_opts


@dataclass
class FlowsMeterReadingsDataSource:
    """
    Configures microgrid-level meter readings that are read from the Flows database.
    """
    meter_id: UUID = field_with_opts(key="meterId")


@dataclass
class FlowsBessReadingsDataSource:
    """
    Configures BESS readings that are read from the Flows database.
    """
    bess_id: UUID = field_with_opts(key="bessId")


@dataclass
class FlowsPlotMeterReadingsDataSource:
    """
    Configures plot-level meter readings that are read from the Flows database.
    """
    feeder_ids: List[UUID] = field_with_opts(key="feederIds")  # Pulls data only for meters that sit on these feeders


@dataclass
class FlowsMarketDataSource:
    """
    Configures a timeseries that is read from the Flows market_data table (for example imbalance price or volume).

    A timeseries can be marked as 'predictive' - in which case there will be multiple predictions for the
    same point in time. For example, Modo predicts the imbalance price for a single settlement period
    many times, with the later predictions having increasing accuracy.
    """
    type: str  # Each type of market data has an associated string, e.g. `modo-imbalance-price-forecast` or `epex-day-ahead-half-hourly-price`.

    is_predictive: Optional[bool] = field_with_opts(key="isPredictive", default=False)

