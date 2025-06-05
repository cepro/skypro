from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, List, Callable, cast, Tuple

import pandas as pd
from skypro.common.config.rates_parse_yaml import parse_supply_points, parse_vol_rates_files_for_all_energy_flows, parse_rate_files
from skypro.common.config.rates_parse_db import get_rates_from_db
from skypro.common.data.get_timeseries import get_timeseries
from skypro.common.notice.notice import Notice
from skypro.common.rates.microgrid import VolRatesForEnergyFlows
from skypro.common.rates.rates import FixedRate, Rate
from skypro.common.timeutils.timeseries import get_steps_per_hh, get_step_size

from skypro.commands.report.config.config import Config
from skypro.commands.report.warnings import missing_data_warnings


@dataclass
class ParsedRates:
    """
    This is just a container to hold the various rate objects
    """
    mkt_vol: VolRatesForEnergyFlows = field(default_factory=VolRatesForEnergyFlows)   # Volume-based (p/kWh) market rates for each energy flow, as predicted in real-time
    mkt_fix: Dict[str, List[FixedRate]] = field(default_factory=dict)  # Fixed p/day rates associated with market/suppliers, keyed by user-specified string which can be used to categorise
    customer: Dict[str, List[Rate]] = field(default_factory=dict)  # Volume and fixed rates charged to customers, keyed by user-specified string which can be used to categorise


def get_rates_from_config(
        time_index: pd.DatetimeIndex,
        config: Config,
        file_path_resolver_func: Callable,
        flows_db_engine,
        rates_db_engine,
) -> Tuple[ParsedRates, List[Notice]]:
    """
    This reads the rates files defined in the given rates configuration block and returns the ParsedRates,
    and a dataframe containing live and final imbalance data.
    """

    notices: List[Notice] = []

    # Read in Elexon imbalance price
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
    if config.reporting.rates.rates_db is not None:
        mkt_vol, fixed_import, fixed_export = get_rates_from_db(
            supply_points_name=config.reporting.rates.rates_db.supply_points_name,
            site_region=config.reporting.rates.rates_db.site_specific.region,
            site_bands=config.reporting.rates.rates_db.site_specific.bands,
            import_bundle_names=config.reporting.rates.rates_db.import_bundles,
            export_bundle_names=config.reporting.rates.rates_db.export_bundles,
            db_engine=rates_db_engine,
            imbalance_pricing=imbalance_pricing["imbalance_price"],
            import_grid_capacity=config.reporting.grid_connection.import_capacity,
            export_grid_capacity=config.reporting.grid_connection.export_capacity,
            future_offset=timedelta(seconds=0)
        )

        parsed_rates = ParsedRates(
            mkt_vol=mkt_vol,
            mkt_fix={
                "import": fixed_import,
                "export": fixed_export
            },
            customer={}  # TODO: read customer rates
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
                    parsed_rates.customer[category_str] = parse_rate_files(
                        files=files,
                        supply_points=supply_points,
                        imbalance_pricing=None,
                        file_path_resolver_func=file_path_resolver_func
                    )

    return parsed_rates, notices
