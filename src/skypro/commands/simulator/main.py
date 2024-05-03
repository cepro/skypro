import logging
from typing import Optional

import numpy as np

from simt_common.jsonconfig.rates import parse_rates, parse_supply_points, collate_import_and_export_rate_configurations

from skypro.commands.simulator.algorithms.price_curve import run_price_curve_imbalance_algorithm
from skypro.commands.simulator.config import parse_config
from skypro.commands.simulator.output import save_output
from skypro.commands.simulator.parse_imbalance_data import read_imbalance_data
from skypro.commands.simulator.profiler import Profiler
from skypro.commands.simulator.results import explore_results


def simulate(config_file_path: str, do_plots: bool, output_file_path: Optional[str] = None):

    logging.info("Simulator - - - - - - - - - - - - -")

    # Parse the main config file
    logging.info(f"Using config file: {config_file_path}")
    config = parse_config(config_file_path)

    # Parse the supply points config file:
    supply_points = parse_supply_points(
        supply_points_config_file=config.simulation.rates.supply_points_config_file
    )

    # Read all the rates config files and sort into import and export rate configurations. The actual
    # import/export configurations are actually parsed into Rates objects later.
    rates_import_config, rates_export_config = collate_import_and_export_rate_configurations(
        rates_config_files=config.simulation.rates.rates_config_files
    )

    logging.info("Reading pricing files...")

    # Imbalance pricing/volume data can come from either Modo or Elexon, Modo is 'predictive' and it's predictions
    # change over the course of the SP, whereas Elexon publishes a single figure for each SP in hindsight.
    by_sp = read_imbalance_data(
        price_dir=config.simulation.imbalance_data_source.price_dir,
        volume_dir=config.simulation.imbalance_data_source.volume_dir,
    )
    if config.simulation.start < by_sp.index[0]:
        raise ValueError(f"Simulation start time is outside of imbalance data range")
    if config.simulation.end > by_sp.index[-1]:
        raise ValueError(f"Simulation end time is outside of imbalance data range")

    # Remove any extra settlement periods that are in the data, but are outside the start and end times
    by_sp = by_sp[by_sp.index >= config.simulation.start]
    by_sp = by_sp[by_sp.index <= config.simulation.end]

    import_rates_10m = parse_rates(
        rates_config=rates_import_config,
        supply_points=supply_points,
        imbalance_pricing=by_sp["imbalance_price_10m"],
    )
    import_rates_20m = parse_rates(
        rates_config=rates_import_config,
        supply_points=supply_points,
        imbalance_pricing=by_sp["imbalance_price_20m"],
    )
    import_rates_final = parse_rates(
        rates_config=rates_import_config,
        supply_points=supply_points,
        imbalance_pricing=by_sp["imbalance_price_final"],
    )
    export_rates_10m = parse_rates(
        rates_config=rates_export_config,
        supply_points=supply_points,
        imbalance_pricing=by_sp["imbalance_price_10m"]*-1,  # imbalance price is inverted for exports
    )
    export_rates_20m = parse_rates(
        rates_config=rates_export_config,
        supply_points=supply_points,
        imbalance_pricing=by_sp["imbalance_price_20m"]*-1,  # imbalance price is inverted for exports
    )
    export_rates_final = parse_rates(
        rates_config=rates_export_config,
        supply_points=supply_points,
        imbalance_pricing=by_sp["imbalance_price_final"]*-1,  # imbalance price is inverted for exports
    )

    # Log the parsed rates for user information
    for rate in import_rates_final:
        logging.info(f"Import rate {rate}")
    for rate in export_rates_final:
        logging.info(f"Export rate {rate}")

    logging.info("Generating solar profile...")
    solar_config = config.simulation.site.solar
    if solar_config.profile is not None:
        solarProfiler = Profiler(
            profile_csv_dir=solar_config.profile.profile_dir,
            scaling_factor=(solar_config.profile.scaled_size_kwp / solar_config.profile.profiled_size_kwp)
        )
        by_sp["solar"] = solarProfiler.get_for(by_sp.index.to_series()) * 2  # Multiply to convert kWh over HH to kW
    elif not np.isnan(solar_config.constant):
        by_sp["solar"] = solar_config.constant
    else:
        raise ValueError("Solar configuration must be either 'profile' or 'constant'")

    # Create domestic load data
    logging.info("Generating domestic load profile...")
    load_config = config.simulation.site.load
    if load_config.profile is not None:
        loadProfiler = Profiler(
            profile_csv_dir=load_config.profile.profile_dir,
            scaling_factor=(load_config.profile.scaled_num_plots / load_config.profile.profiled_num_plots)
        )
        by_sp["load"] = loadProfiler.get_for(by_sp.index.to_series()) * 2  # Multiply to convert kWh over HH to kW
    elif not np.isnan(load_config.constant):
        by_sp["load"] = load_config.constant
    else:
        raise ValueError("Load configuration must be either 'profile' or 'constant'")

    if config.simulation.strategy.price_curve_algo:
        df = run_price_curve_imbalance_algorithm(
            by_sp,
            import_rates_10m=import_rates_10m,
            import_rates_20m=import_rates_20m,
            import_rates_final=import_rates_final,
            export_rates_10m=export_rates_10m,
            export_rates_20m=export_rates_20m,
            export_rates_final=export_rates_final,
            battery_energy_capacity=config.simulation.site.bess.energy_capacity,
            battery_charge_efficiency=config.simulation.site.bess.charge_efficiency,
            battery_nameplate_power=config.simulation.site.bess.nameplate_power,
            site_import_limit=config.simulation.site.grid_connection.import_limit,
            site_export_limit=config.simulation.site.grid_connection.export_limit,
            niv_chase_periods=config.simulation.strategy.price_curve_algo.niv_chase_periods,
            full_discharge_when_export_rate_applies
            =config.simulation.strategy.price_curve_algo.do_full_discharge_when_export_rate_applies,
        )
    else:
        raise ValueError("Unknown algorithm chosen")

    if output_file_path:
        save_output(df, config, output_file_path)

    explore_results(
        df=df,
        do_plots=do_plots,
        battery_energy_capacity=config.simulation.site.bess.energy_capacity,
        battery_nameplate_power=config.simulation.site.bess.nameplate_power,
        site_import_limit=config.simulation.site.grid_connection.import_limit,
        site_export_limit=config.simulation.site.grid_connection.export_limit,
        import_rates=import_rates_final,
        export_rates=export_rates_final,
    )
