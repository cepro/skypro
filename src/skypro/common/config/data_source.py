from typing import Optional

from marshmallow_dataclass import dataclass

from skypro.common.config.data_source_csv import CSVPlotMeterReadingsDataSource, CSVMeterReadingsDataSource, \
    CSVTimeseriesDataSource, CSVBessReadingsDataSource, CSVProfileDataSource
from skypro.common.config.data_source_flows import FlowsPlotMeterReadingsDataSource, FlowsMeterReadingsDataSource, \
    FlowsMarketDataSource, FlowsBessReadingsDataSource
from skypro.common.config.utility import enforce_one_option, field_with_opts


@dataclass
class BessReadingDataSource:
    """
    A source of 'bess readings' data - e.g. SoE, target power, etc. Mostly used for reporting on actuals. The data can come either from CSV or DB.
    """
    flows_bess_readings_data_source: Optional[FlowsBessReadingsDataSource] = field_with_opts(key="flowsBessReadings")
    csv_bess_readings_data_source: Optional[CSVBessReadingsDataSource] = field_with_opts(key="csvBessReadings")

    def __post_init__(self):
        enforce_one_option(
            [self.flows_bess_readings_data_source, self.csv_bess_readings_data_source],
            "'flowsBessReadings', 'csvBessReadings'"
        )


@dataclass
class PlotMeterReadingDataSource:
    """
    A source of data that provides Emlite plot-level meter readings, either from CSV or DB
    """
    flows_plot_meter_readings_data_source: Optional[FlowsPlotMeterReadingsDataSource] = field_with_opts(key="flowsPlotMeterReadings")
    csv_plot_meter_readings_data_source: Optional[CSVPlotMeterReadingsDataSource] = field_with_opts(key="csvPlotMeterReadings")

    def __post_init__(self):
        enforce_one_option(
            [self.flows_plot_meter_readings_data_source, self.csv_plot_meter_readings_data_source],
            "'flowsPlotMeterReadings', 'csvPlotMeterReadings'"
        )


@dataclass
class MeterReadingDataSource:
    """
    A source of data that provides microgrid (Acuvim) meter readings, either from CSV or DB
    """
    flows_meter_readings_data_source: Optional[FlowsMeterReadingsDataSource] = field_with_opts(key="flowsMeterReadings")
    csv_meter_readings_data_source: Optional[CSVMeterReadingsDataSource] = field_with_opts(key="csvMeterReadings")

    def __post_init__(self):
        enforce_one_option(
            [self.flows_meter_readings_data_source, self.csv_meter_readings_data_source],
            "'flowsMeterReadings', 'csvMeterReadings'"
        )


@dataclass
class TimeseriesDataSource:
    """
    A source of data that provides generic timeseries data, either from CSV or DB
    """
    flows_market_data_source: Optional[FlowsMarketDataSource] = field_with_opts(key="flowsMarketData")
    csv_timeseries_data_source: Optional[CSVTimeseriesDataSource] = field_with_opts(key="csvTimeseries")

    def __post_init__(self):
        enforce_one_option([
            self.flows_market_data_source,
            self.csv_timeseries_data_source,
        ],
            "'flowsMarketData', 'csvTimeseries'"
        )


@dataclass
class ConstantProfileDataSource:
    """
    Represents a constant flat value - usually used to set Solar generation or Load to zero.
    """
    value: float


@dataclass
class ProfileDataSource:
    """
    A source of data that provides a profile for either Solar or Load etc. In the future this may support configuring a
    profile stored in a database, but at the moment only a CSV source is supported.
    """
    csv_profile_data_source: Optional[CSVProfileDataSource] = field_with_opts(key="csvProfile")
    constant_profile_data_source: Optional[ConstantProfileDataSource] = field_with_opts(key="constant")

    def __post_init__(self):
        enforce_one_option([self.csv_profile_data_source, self.constant_profile_data_source], "'csvProfile', 'constant'")


@dataclass
class ImbalanceDataSource:
    """
    Configures the system imbalance price and volume data sources.
    """
    price: TimeseriesDataSource = field_with_opts(key="price")
    volume: TimeseriesDataSource = field_with_opts(key="volume")
