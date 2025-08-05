from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, List, Callable, cast, Tuple

import pandas as pd
from skypro.common.config.rates_parse_yaml import parse_supply_points, parse_vol_rates_files_for_all_energy_flows, parse_rate_files
from skypro.common.config.rates_parse_db import get_rates_from_db
from skypro.common.data.get_timeseries import get_timeseries
from skypro.common.notice.notice import Notice
from skypro.common.rate_utils.to_dfs import VolRatesForEnergyFlows
from skypro.common.rates.rates import FixedRate, VolRate
from skypro.common.timeutils.timeseries import get_steps_per_hh, get_step_size

from skypro.commands.report.config.config import Config
from skypro.commands.report.warnings import missing_data_warnings


@dataclass
class ParsedRates:
    """
    This is just a container to hold the various rate objects
    """
    mkt_vol: VolRatesForEnergyFlows = field(default_factory=VolRatesForEnergyFlows)   # Volume-based (p/kWh) market rates for each energy flow, as predicted in real-time
    mkt_fix: Dict[str, List[FixedRate]] = field(default_factory=dict)  # Fixed p/day rates associated with market/suppliers, keyed by a string which can be used to categorise
    customer_vol: Dict[str, List[VolRate]] = field(default_factory=dict)  # Volume rates charged to customers, keyed by a string which can be used to categorise
    customer_fix: Dict[str, List[FixedRate]] = field(default_factory=dict)  # Fixed rates charged to customers, keyed by a string which can be used to categorise


def get_rates_from_config(
        time_index: pd.DatetimeIndex,
        config: Config,
        file_path_resolver_func: Callable,
        flows_db_engine,
        rates_db_engine,
) -> Tuple[ParsedRates, List[Notice]]:
    """
    This reads the rates configuration block and returns the ParsedRates, and a list of Notices if there are issues with data quality.
    The rates may be configured to be read from YAML files or from a database.
    """

    notices: List[Notice] = []

    # Read in Elexon imbalance price, which is sometimes required by Rate objects
    elexon, new_notices = get_timeseries(
        source=config.reporting.rates.imbalance_data_source.price,
        start=time_index[0],
        end=time_index[-1],
        file_path_resolver_func=file_path_resolver_func,
        db_engine=flows_db_engine,
        context="elexon imbalance data"
    )
    notices.extend(new_notices)

    elexon.index = elexon["time"]
    elexon = elexon.sort_index()

    imbalance_pricing = pd.DataFrame(index=time_index)
    imbalance_pricing["imbalance_price"] = elexon["value"]
    # We are working with a 5 minutely dataframe, but pricing changes every 30mins, so copy pricing for every 5 minute:
    imbalance_pricing = imbalance_pricing.ffill(limit=get_steps_per_hh(get_step_size(time_index))-1)  # limit forward fill to 25 minutes

    # Sanity check for missing data
    notices.extend(missing_data_warnings(imbalance_pricing, "Elexon imbalance data"))

    # Rates can either be read from the "rates database" or from local YAML files
    db_config = config.reporting.rates.rates_db
    if db_config is not None:
        db_rates = get_rates_from_db(
            supply_points_name=db_config.supply_points_name,
            site_region=db_config.site_specific.region,
            site_bands=db_config.site_specific.bands,
            import_bundle_names=db_config.import_bundles,
            export_bundle_names=db_config.export_bundles,
            db_engine=rates_db_engine,
            imbalance_pricing=imbalance_pricing["imbalance_price"],
            import_grid_capacity=config.reporting.grid_connection.import_capacity,
            export_grid_capacity=config.reporting.grid_connection.export_capacity,
            future_offset=timedelta(seconds=0),
            customer_import_bundle_names=db_config.customer.import_bundles if db_config.customer is not None else [],
            customer_export_bundle_names=db_config.customer.export_bundles if db_config.customer is not None else [],
        )

        parsed_rates = ParsedRates(
            mkt_vol=db_rates.mkt_vol_by_flow,
            mkt_fix={
                "import": db_rates.mkt_fix_import,
                "export": db_rates.mkt_fix_export,
            },
            customer_vol={
                "import": db_rates.customer_vol_import,
                "export": db_rates.customer_vol_export,
            },
            customer_fix={
                "import": db_rates.customer_fix_import,
                "export": db_rates.customer_fix_export,
            },
        )
    else:  # Read rates from local YAML files...
        # Parse the supply points config file:
        supply_points = parse_supply_points(
            supply_points_config_file=config.reporting.rates.supply_points_config_file
        )

        parsed_rates = ParsedRates()
        parsed_rates.mkt_vol = parse_vol_rates_files_for_all_energy_flows(
            rates_files=config.reporting.rates.files,
            supply_points=supply_points,
            imbalance_pricing=imbalance_pricing["imbalance_price"],
            file_path_resolver_func=file_path_resolver_func,
        )

        # The rates files can include an 'experimental' block which contains beta features like fixed market rates and customer rates.
        exp_config = config.reporting.rates.experimental
        if exp_config:
            if exp_config.mkt_fixed_files:
                # Read in fixed rates just to output them in the CSV
                for category_str, files in exp_config.mkt_fixed_files.items():
                    rates = parse_rate_files(
                        files=files,
                        supply_points=supply_points,
                        imbalance_pricing=None,
                        file_path_resolver_func=file_path_resolver_func
                    )
                    for rate in rates:
                        if not isinstance(rate, FixedRate):
                            raise ValueError(f"Only fixed rates can be specified in the fixedMarketFiles, got: '{rate.name}'")
                    parsed_rates.mkt_fix[category_str] = cast(List[FixedRate], rates)

            if exp_config.customer_load_files:
                for category_str, files in exp_config.customer_load_files.items():
                    rates = parse_rate_files(
                        files=files,
                        supply_points=supply_points,
                        imbalance_pricing=None,
                        file_path_resolver_func=file_path_resolver_func
                    )
                    parsed_rates.customer_fix[category_str] = []
                    parsed_rates.customer_vol[category_str] = []
                    for rate in rates:
                        if isinstance(rate, FixedRate):
                            parsed_rates.customer_fix[category_str].append(rate)
                        elif isinstance(rate, VolRate):
                            parsed_rates.customer_vol[category_str].append(rate)
                        else:
                            raise ValueError(f"Unknown rate type: {rate}")

    return parsed_rates, notices
