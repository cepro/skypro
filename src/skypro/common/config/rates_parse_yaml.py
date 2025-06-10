import yaml
import logging
import os
from typing import Dict, List, Optional, Callable, cast

import pandas as pd

from skypro.common.config.rates_dataclasses import RatesFiles
from skypro.common.rates.microgrid import VolRatesForEnergyFlows
from skypro.common.config.dayed_period import DayedPeriodField
from skypro.common.rates.rates import Rate, ShapedVolRate, FlatVolRate, PeriodicFlatVolRate, MultiplierVolRate, \
    OSAMFlatVolRate, RegularFixedRate, VolRate, PeriodicValue
from skypro.common.rates.supply_point import SupplyPoint

"""
This file handles parsing of rates from YAML/JSON configurations 
"""


def parse_supply_points(supply_points_config_file: str) -> Dict[str, SupplyPoint]:
    """
    Reads the supply point configuration file and returns dictionary of associated SupplyPoint objects, keyed
    by name.
    """

    supply_points_config_file = os.path.expanduser(supply_points_config_file)

    with open(supply_points_config_file) as config_data:
        supply_points_config = yaml.safe_load(config_data)["supplyPoints"]

    supplyPoints = {}

    # First create the supply points
    for name, config in supply_points_config.items():
        supplyPoints[name] = SupplyPoint(
            name=name,
            line_loss_factor=config["lossFactor"]
        )

    return supplyPoints


def parse_vol_rates_files_for_all_energy_flows(
        rates_files: RatesFiles,
        supply_points: Dict[str, SupplyPoint],
        imbalance_pricing: pd.Series,
        file_path_resolver_func: Callable,
) -> VolRatesForEnergyFlows:
    """
    Reads the rates files for each flow (JSON or YAML) and returns only the volume-based rates objects for each energy
    flow. Fixed charges like £/day are not returned.
    """

    # This is a rudimentary caching mechanism to spot if two flows have identical files and re-use the same rate
    # instances in that case.
    flows = {
        "solar_to_batt": {
            "files": rates_files.solar_to_batt
        },
        "grid_to_batt": {
            "files": rates_files.grid_to_batt
        },
        "batt_to_load": {
            "files": rates_files.batt_to_load
        },
        "batt_to_grid": {
            "files": rates_files.batt_to_grid
        },
        "solar_to_grid": {
            "files": rates_files.solar_to_grid
        },
        "solar_to_load": {
            "files": rates_files.solar_to_load
        },
        "grid_to_load": {
            "files": rates_files.grid_to_load
        },
    }

    cached: Dict[str, List[VolRate]] = {}
    for flow_name, flow_info in flows.items():
        files_str = str(flow_info["files"])
        if files_str not in cached:
            rates = parse_rate_files(
                files=flow_info["files"],
                supply_points=supply_points,
                imbalance_pricing=imbalance_pricing,
                file_path_resolver_func=file_path_resolver_func,
            )
            # check that the rates are all volume-based, and not fixed rates
            for rate in rates:
                if not isinstance(rate, VolRate):
                    raise ValueError(f"Flow '{flow_name}' specifies a non-volume based rate: '{rate.name}'")

            cached[files_str] = cast(List[VolRate], rates)

    def pull_from_cache(name: str) -> List[VolRate]:
        """
        Convenience function to pull the rates associated with the given flow name from the cache.
        This function captures the `cached` variable.
        """
        return cached[str(flows[name]["files"])]

    all_rates = VolRatesForEnergyFlows(
        solar_to_batt=pull_from_cache("solar_to_batt"),
        grid_to_batt=pull_from_cache("grid_to_batt"),
        batt_to_load=pull_from_cache("batt_to_load"),
        solar_to_grid=pull_from_cache("solar_to_grid"),
        solar_to_load=pull_from_cache("solar_to_load"),
        grid_to_load=pull_from_cache("grid_to_load"),
        batt_to_grid=pull_from_cache("batt_to_grid"),
    )

    # This runs through all the rates in each set and if there is a multiplier rate present then it will be
    # informed of all the other rates in the set, so that it knows what rates to multiply.
    for _, rates in all_rates.get_all_sets_named():
        for rate in rates:
            if isinstance(rate, MultiplierVolRate):
                rate.set_all_rates_in_set(rates)

    # OSAM is a special kind of rate that can only ever be on grid imports to the battery.
    # This runs through all the rates in each set and checks that only the `grid_to_batt` flow has OSAM rates.
    for flow_name, rates in all_rates.get_all_sets_named():
        if flow_name == "grid_to_batt":
            continue
        for rate in rates:
            if isinstance(rate, OSAMFlatVolRate):
                raise ValueError(f"There are OSAM rates configured for the '{flow_name}' flow. OSAM rates can only be applied to the 'grid_to_batt' flow.")

    return all_rates


def parse_rate_files(
        files: List[str],
        supply_points: Dict[str, SupplyPoint],
        imbalance_pricing: Optional[pd.Series],
        file_path_resolver_func: Callable,
) -> List[Rate]:
    """
    Reads the list of rates files and returns a list of Rate objects
    """
    rates = []
    for file in files:
        rates.extend(_parse_rate_file(file, supply_points, imbalance_pricing, file_path_resolver_func))

    return rates


def _parse_rate_file(
        file: str,
        supply_points: Dict[str, SupplyPoint],
        imbalance_pricing: Optional[pd.Series],
        file_path_resolver_func: Callable
) -> List[Rate]:
    """
    Reads the single rates file and returns a list of Rate objects
    """
    resolved_file = file_path_resolver_func(file)

    with open(resolved_file) as file_data:
        return _parse_rates(
            rates_config=yaml.safe_load(file_data),
            supply_points=supply_points,
            imbalance_pricing=imbalance_pricing * 1 if imbalance_pricing is not None else None,
        )


def _parse_rates(
        rates_config: Dict,
        supply_points: Dict[str, SupplyPoint],
        imbalance_pricing: Optional[pd.Series],
) -> List[Rate]:
    """
    Parses a configuration specifying a set of rates and returns lists of associated rate objects. The imbalance pricing
    is used if there are any imbalance rates in the configuration.
    """
    rates = []
    keys_used = []  # keep track of which keys dict keys we used, so we can alert if some were missed (not supported)

    # Create imbalance rates
    if "imbalance" in rates_config:
        keys_used.append("imbalance")
        rates.append(_parse_imbalance_rate(
            config=rates_config["imbalance"],
            supply_points=supply_points,
            imbalance_pricing=imbalance_pricing
        ))

    # TODO: This should be renamed "powerFlat"?
    # Create fixed p/kW rates
    if "powerFixed" in rates_config:
        keys_used.append("powerFixed")
        for name, config in rates_config["powerFixed"].items():
            rates.append(FlatVolRate(
                name=name,
                values=[(None, config["rate"])],
                supply_point=supply_points[config["supplyPoint"]],
            ))

    # Create "PeriodicFlat" rates
    if "periodicFlat" in rates_config:
        keys_used.append("periodicFlat")
        for name, config in rates_config["periodicFlat"].items():
            rates.append(_parse_periodic_flat_rate(
                config=config,
                name=name,
                supply_points=supply_points
            ))

    # Create DUoS rates, which have become the same thing as "periodicFlat" rates, just under a different name
    if "duos" in rates_config:
        keys_used.append("duos")
        for name, config in rates_config["duos"].items():
            rates.append(_parse_periodic_flat_rate(
                config=config,
                name=name,
                supply_points=supply_points
            ))

    if "multiplierRate" in rates_config:
        keys_used.append("multiplierRate")
        # multiplierRate is a special type of fee that is applied on top of other rates.
        rate = MultiplierVolRate(
            name=rates_config["multiplierRate"]["name"],
            factors=[(None, rates_config["multiplierRate"]["factor"])],
            mode=rates_config["multiplierRate"]["ratesToMultiply"]
        )
        rates.append(rate)

    if "osam" in rates_config:
        keys_used.append("osam")
        # OSAM is a special type of fee that is applied based on the on-site allocation methodology (P395)
        for name, config in rates_config["osam"].items():
            rates.append(OSAMFlatVolRate(
                name=name,
                rates=[(None, config["rate"])],
                supply_point=supply_points[config["supplyPoint"]],
            ))

    if "regularFixed" in rates_config:
        keys_used.append("regularFixed")
        for name, config in rates_config["regularFixed"].items():
            rates.append(RegularFixedRate(
                name=name,
                daily_costs=[(None, config["daily"])]
            ))

    mismatch = set(rates_config.keys()) ^ set(keys_used)
    if len(mismatch) > 0:
        raise ValueError(f"Rate type unknown: {mismatch}")

    return rates


def _parse_imbalance_rate(
        config: Dict,
        supply_points: Dict[str, SupplyPoint],
        imbalance_pricing: Optional[pd.Series] = None
) -> ShapedVolRate:
    """
    Parses the configuration of an imbalance rate into a rate object
    """
    supplyPointName = config["supplyPoint"]
    # The supply point should always be NSP when using imbalance pricing
    if supplyPointName != "nsp" and supplyPointName != "NSP":
        raise ValueError(f"Imbalance rate supply point should probably be nsp, but was '{supplyPointName}'")

    multiplier = 1
    name = "imbalance"
    if config["isExport"]:
        multiplier = -1
        name = "gimbalance"

    return ShapedVolRate(
        name=name,
        pricing=imbalance_pricing * multiplier,
        supply_point=supply_points[supplyPointName],
    )


def _parse_periodic_flat_rate(
        config: Dict,
        name: str,
        supply_points: Dict[str, SupplyPoint],
) -> PeriodicFlatVolRate:
    """
    Parses the configuration of a 'periodic flat' rate into a rate object
    """
    supplyPointName = config["supplyPoint"]

    periods = []
    for period_config in config["periods"]:
        period = DayedPeriodField().deserialize(period_config)
        periods.append(period)

    return PeriodicFlatVolRate(
        name=name,
        periodic_values=[(None, PeriodicValue(value=config["rate"], periods=periods))],
        supply_point=supply_points[supplyPointName],
    )
