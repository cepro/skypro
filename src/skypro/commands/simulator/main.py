import logging
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd

from simt_common.jsonconfig.rates import parse_supply_points, process_rates_for_all_energy_flows
from simt_common.rates.microgrid import get_rates_dfs

from skypro.cli_utils.cli_utils import substitute_vars, read_json_file
from skypro.commands.simulator.algorithms.price_curve import run_price_curve_imbalance_algorithm
from skypro.commands.simulator.algorithms.spread.algo_2 import run_spread_based_algo_2
from skypro.commands.simulator.config import parse_config
from skypro.commands.simulator.output import save_output
from skypro.commands.simulator.parse_imbalance_data import read_imbalance_data
from skypro.commands.simulator.profiler import Profiler
from skypro.commands.simulator.results import explore_results


STEP_SIZE = timedelta(minutes=10)
STEPS_PER_SP = int(timedelta(minutes=30) / STEP_SIZE)
assert((timedelta(minutes=30) / STEP_SIZE) == STEPS_PER_SP)  # Check that we have an exact number of steps per SP


def simulate(config_file_path: str, env_file_path: str, do_plots: bool, output_file_path: Optional[str] = None):

    logging.info("Simulator - - - - - - - - - - - - -")

    # Parse the main config file
    logging.info(f"Using config file: {config_file_path}")
    config = parse_config(config_file_path)

    env_vars = read_json_file(env_file_path)["vars"]

    # Parse the supply points config file:
    supply_points = parse_supply_points(
        supply_points_config_file=substitute_vars(config.simulation.rates.supply_points_config_file, env_vars)
    )

    # Run the simulation at 10 minute granularity
    time_index = pd.date_range(config.simulation.start, config.simulation.end, freq=STEP_SIZE)
    time_index = time_index.tz_convert("UTC")

    # Imbalance pricing/volume data can come from either Modo or Elexon, Modo is 'predictive' and it's predictions
    # change over the course of the SP, whereas Elexon publishes a single figure for each SP in hindsight.
    df = read_imbalance_data(
        time_index=time_index,
        price_dir=substitute_vars(config.simulation.imbalance_data_source.price_dir, env_vars),
        volume_dir=substitute_vars(config.simulation.imbalance_data_source.volume_dir, env_vars),
    )

    predicted_rates = process_rates_for_all_energy_flows(
        config=config.simulation.rates.files,
        env_vars=env_vars,
        supply_points=supply_points,
        imbalance_pricing=df["imbalance_price_predicted"]
    )

    final_rates = process_rates_for_all_energy_flows(
        config=config.simulation.rates.files,
        env_vars=env_vars,
        supply_points=supply_points,
        imbalance_pricing=df["imbalance_price_final"]
    )

    # Log the parsed rates for user information
    for name, rate_set in predicted_rates.get_all_sets_named():
        for rate in rate_set:
            logging.info(f"Flow: {name}, Rate: {rate}")

    logging.info("Calculating predicted rates...")
    predicted_rates_dfs = get_rates_dfs(time_index, predicted_rates)
    logging.info("Calculating final rates...")
    final_rates_dfs = get_rates_dfs(time_index, final_rates)

    # Add the total rate of each energy flow to the dataframe
    for set_name, rates_df in predicted_rates_dfs.items():
        df[f"rate_predicted_{set_name}"] = rates_df.sum(axis=1, skipna=False)
    for set_name, rates_df in final_rates_dfs.items():
        df[f"rate_final_{set_name}"] = rates_df.sum(axis=1, skipna=False)

    logging.info("Generating solar profile...")
    solar_config = config.simulation.site.solar
    if solar_config.profile is not None:
        solarProfiler = Profiler(
            profile_csv_dir=substitute_vars(solar_config.profile.profile_dir, env_vars),
            scaling_factor=(solar_config.profile.scaled_size_kwp / solar_config.profile.profiled_size_kwp)
        )
        solar_energy = solarProfiler.get_for(time_index)
        df["solar_power"] = solar_energy / (STEP_SIZE.total_seconds()/3600)
    elif not np.isnan(solar_config.constant):
        df["solar_power"] = solar_config.constant
    else:
        raise ValueError("Solar configuration must be either 'profile' or 'constant'")

    # Create domestic load data
    logging.info("Generating domestic load profile...")
    load_config = config.simulation.site.load
    if load_config.profile is not None:
        loadProfiler = Profiler(
            profile_csv_dir=substitute_vars(load_config.profile.profile_dir, env_vars),
            scaling_factor=(load_config.profile.scaled_num_plots / load_config.profile.profiled_num_plots)
        )
        load_energy = loadProfiler.get_for(time_index)
        df["load_power"] = load_energy / (STEP_SIZE.total_seconds()/3600)
    elif not np.isnan(load_config.constant):
        df["load_power"] = load_config.constant
    else:
        raise ValueError("Load configuration must be either 'profile' or 'constant'")

    # Calculate the BESS charge and discharge limits based on how much solar generation and housing load
    # there is. We need to abide by the overall site import/export limits. And stay within the nameplate inverter
    # capabilities of the BESS
    non_bess_power = df["load_power"] - df["solar_power"]
    df["bess_max_power_charge"] = ((config.simulation.site.grid_connection.import_limit - non_bess_power).
                                    clip(upper=config.simulation.site.bess.nameplate_power))
    df["bess_max_power_discharge"] = ((config.simulation.site.grid_connection.export_limit + non_bess_power).
                                       clip(upper=config.simulation.site.bess.nameplate_power))

    # The algo sometimes needs the previous SP's final rates. The algo processes each step as a row, so make the
    # previous SPs values available in each row/step.
    cols_to_shift = [
        "imbalance_volume_final",
        "imbalance_price_final",
        "rate_final_bess_charge_from_solar",
        "rate_final_bess_charge_from_grid",
        "rate_final_bess_discharge_to_load",
        "rate_final_bess_discharge_to_grid",
        "rate_final_solar_to_grid",
        "rate_final_load_from_grid",
    ]
    for col in cols_to_shift:
        df[f"prev_sp_{col}"] = df[col].shift(STEPS_PER_SP)

    # Only share the columns that are relevant with the algo
    cols_to_share_with_algo = [
        "load_power",
        "solar_power",
        "bess_max_power_charge",
        "bess_max_power_discharge",
        "imbalance_volume_predicted",
        "rate_predicted_bess_charge_from_solar",
        "rate_predicted_bess_charge_from_grid",
        "rate_predicted_bess_discharge_to_load",
        "rate_predicted_bess_discharge_to_grid",
        "rate_predicted_solar_to_grid",
        "rate_predicted_load_from_grid",
        "prev_sp_imbalance_price_final",
        "prev_sp_imbalance_volume_final",
        "prev_sp_rate_final_bess_charge_from_solar",
        "prev_sp_rate_final_bess_charge_from_grid",
        "prev_sp_rate_final_bess_discharge_to_load",
        "prev_sp_rate_final_bess_discharge_to_grid",
        "prev_sp_rate_final_solar_to_grid",
        "prev_sp_rate_final_load_from_grid",
    ]
    if config.simulation.strategy.price_curve_algo:
        df_algo = run_price_curve_imbalance_algorithm(
            df_in=df[cols_to_share_with_algo],
            battery_energy_capacity=config.simulation.site.bess.energy_capacity,
            battery_charge_efficiency=config.simulation.site.bess.charge_efficiency,
            niv_chase_periods=config.simulation.strategy.price_curve_algo.niv_chase_periods,
            full_discharge_period=config.simulation.strategy.price_curve_algo.full_discharge_period,
        )
    elif config.simulation.strategy.spread_algo:
        # TODO: make this more elegant and check the numbers - particularly not certain on the discharge side, although
        #       the numbers are small
        df["rate_bess_charge_from_grid_non_imbalance"] = df["rate_final_bess_charge_from_grid"] - df["imbalance_price_final"]
        df["rate_bess_discharge_to_grid_non_imbalance"] = df["rate_final_bess_discharge_to_grid"] + df["imbalance_price_final"]

        cols_to_share_with_algo.extend(["rate_bess_charge_from_grid_non_imbalance", "rate_bess_discharge_to_grid_non_imbalance"])
        df_algo = run_spread_based_algo_2(
            df_in=df[cols_to_share_with_algo],
            battery_energy_capacity=config.simulation.site.bess.energy_capacity,
            battery_charge_efficiency=config.simulation.site.bess.charge_efficiency,
            full_discharge_period=config.simulation.strategy.spread_algo.full_discharge_period,
        )

    else:
        raise ValueError("Unknown algorithm chosen")

    df = pd.concat([df, df_algo], axis=1)

    if output_file_path:
        save_output(df, config, output_file_path)

    explore_results(
        df=df,
        final_rates_dfs=final_rates_dfs,
        do_plots=do_plots,
        battery_energy_capacity=config.simulation.site.bess.energy_capacity,
        battery_nameplate_power=config.simulation.site.bess.nameplate_power,
        site_import_limit=config.simulation.site.grid_connection.import_limit,
        site_export_limit=config.simulation.site.grid_connection.export_limit,
    )
