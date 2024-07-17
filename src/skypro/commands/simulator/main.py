import logging
import os
from datetime import timedelta
from functools import reduce
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytz

from simt_common.jsonconfig.rates import parse_supply_points, process_rates_for_all_energy_flows
from simt_common.rates.microgrid import get_rates_dfs
from simt_common.timeutils.hh_math import floor_hh

from skypro.cli_utils.cli_utils import substitute_vars, read_json_file
from skypro.commands.simulator.algorithms.price_curve.algo import run_price_curve_imbalance_algo
from skypro.commands.simulator.algorithms.spread.algo import run_spread_based_algo
from skypro.commands.simulator.config import parse_config
from skypro.commands.simulator.config.config import Solar, Load
from skypro.commands.simulator.output import save_output
from skypro.commands.simulator.parse_imbalance_data import read_imbalance_data
from skypro.commands.simulator.profiler import Profiler
from skypro.commands.simulator.results import explore_results


STEP_SIZE = timedelta(minutes=10)
STEPS_PER_SP = int(timedelta(minutes=30) / STEP_SIZE)
assert((timedelta(minutes=30) / STEP_SIZE) == STEPS_PER_SP)  # Check that we have an exact number of steps per SP


def simulate(
        config_file_path: str,
        env_file_path: str,
        do_plots: bool,
        output_file_path: Optional[str] = None,
        output_summary_file_path: Optional[str] = None,
        output_aggregate: Optional[str] = None,
        output_rate_detail: Optional[bool] = False,
):

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
    time_index = pd.date_range(
        start=config.simulation.start.astimezone(pytz.UTC),
        end=config.simulation.end.astimezone(pytz.UTC),
        freq=STEP_SIZE
    )
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
    df["solar_power"] = generate_profile(
        time_index=time_index,
        config=config.simulation.site.solar,
        env_vars=env_vars,
        do_plots=do_plots,
        context_hint="Solar"
    )

    # Create load data
    logging.info("Generating load profile...")
    df["load_power"] = generate_profile(
        time_index=time_index,
        config=config.simulation.site.load,
        env_vars=env_vars,
        do_plots=do_plots,
        context_hint="Load"
    )

    # Calculate the BESS charge and discharge limits based on how much solar generation and housing load
    # there is. We need to abide by the overall site import/export limits. And stay within the nameplate inverter
    # capabilities of the BESS
    df["microgrid_residual_power"] = df["load_power"] - df["solar_power"]
    df["bess_max_power_charge"] = ((config.simulation.site.grid_connection.import_limit - df["microgrid_residual_power"]).
                                    clip(upper=config.simulation.site.bess.nameplate_power))
    df["bess_max_power_discharge"] = ((config.simulation.site.grid_connection.export_limit + df["microgrid_residual_power"]).
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

    # Calculate settlement period timings
    df["sp"] = df.index.to_series().apply(lambda t: floor_hh(t))
    df["time_into_sp"] = df.index.to_series() - df["sp"]
    df["time_left_of_sp"] = timedelta(minutes=30) - df["time_into_sp"]

    # Only share the columns that are relevant with the algo
    cols_to_share_with_algo = [
        "sp",
        "time_into_sp",
        "time_left_of_sp",
        "load_power",
        "solar_power",
        "microgrid_residual_power",
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
        df_algo = run_price_curve_imbalance_algo(
            df_in=df[cols_to_share_with_algo],
            battery_energy_capacity=config.simulation.site.bess.energy_capacity,
            battery_charge_efficiency=config.simulation.site.bess.charge_efficiency,
            config=config.simulation.strategy.price_curve_algo
        )
    elif config.simulation.strategy.spread_algo:
        # TODO: there should be a more elegant way of doing this - but at the moment the spread based algo needs to know
        #  the "non imbalance" rates, seperated from the imbalance rates. These are calculated here, because if the algo
        #  did these calculations itself then we would need to share 'final' rates with it which is best avoided as it
        #  shouldn't know about final rates.

        df["rate_bess_charge_from_grid_non_imbalance"] = df["rate_final_bess_charge_from_grid"] - df["imbalance_price_final"]
        df["rate_bess_discharge_to_grid_non_imbalance"] = df["rate_final_bess_discharge_to_grid"] + df["imbalance_price_final"]
        cols_to_share_with_algo.extend(["rate_bess_charge_from_grid_non_imbalance", "rate_bess_discharge_to_grid_non_imbalance"])

        df_algo = run_spread_based_algo(
            df_in=df[cols_to_share_with_algo],
            battery_energy_capacity=config.simulation.site.bess.energy_capacity,
            battery_charge_efficiency=config.simulation.site.bess.charge_efficiency,
            config=config.simulation.strategy.spread_algo,
        )
    else:
        raise ValueError("Unknown algorithm chosen")

    df = pd.concat([df, df_algo], axis=1)

    df = calculate_microgrid_flows(df)

    if output_file_path:
        save_output(
            df=df,
            final_rates_dfs=final_rates_dfs,
            config=config,
            output_file_path=output_file_path,
            aggregate=output_aggregate,
            rate_detail=output_rate_detail,
        )

    explore_results(
        df=df,
        final_rates_dfs=final_rates_dfs,
        do_plots=do_plots,
        battery_energy_capacity=config.simulation.site.bess.energy_capacity,
        battery_nameplate_power=config.simulation.site.bess.nameplate_power,
        site_import_limit=config.simulation.site.grid_connection.import_limit,
        site_export_limit=config.simulation.site.grid_connection.export_limit,
        summary_csv_path=output_summary_file_path
    )


def calculate_microgrid_flows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the individual flows of energy around the microgrid and adds them to the dataframe
    """
    df = df.copy()
    time_step_hours = (STEP_SIZE.total_seconds() / 3600)

    df["bess_discharge"] = -df["energy_delta"][df["energy_delta"] < 0]
    df["bess_discharge"] = df["bess_discharge"].fillna(0)
    df["bess_charge"] = df["energy_delta"][df["energy_delta"] > 0]
    df["bess_charge"] = df["bess_charge"].fillna(0)

    df["bess_max_charge"] = df["bess_max_power_charge"] * time_step_hours
    df["bess_max_discharge"] = df["bess_max_power_discharge"] * time_step_hours

    # Calculate load and solar energies from the power
    df["solar"] = df["solar_power"] * time_step_hours
    df["load"] = df["load_power"] * time_step_hours
    df["solar_to_load"] = df[["solar", "load"]].min(axis=1)
    df["load_not_supplied_by_solar"] = df["load"] - df["solar_to_load"]
    df["solar_not_supplying_load"] = df["solar"] - df["solar_to_load"]

    df["bess_discharge_to_load"] = df[["bess_discharge", "load_not_supplied_by_solar"]].min(axis=1)
    df["bess_discharge_to_grid"] = df["bess_discharge"] - df["bess_discharge_to_load"]

    df["bess_charge_from_solar"] = df[["bess_charge", "solar_not_supplying_load"]].min(axis=1)
    df["bess_charge_from_grid"] = df["bess_charge"] - df["bess_charge_from_solar"]

    df["load_from_grid"] = df["load_not_supplied_by_solar"] - df["bess_discharge_to_load"]
    df["solar_to_grid"] = df["solar_not_supplying_load"] - df["bess_charge_from_solar"]

    return df


def generate_profile(
        time_index: pd.DatetimeIndex,
        config: Solar | Load,
        env_vars,
        do_plots: bool,
        context_hint: str
) -> pd.Series:

    if config.profile:
        profile_configs = [config.profile]
    elif config.profiles:
        profile_configs = config.profiles
    elif not np.isnan(config.constant):
        return pd.Series(index=time_index, data=config.constant)
    else:
        raise ValueError("Configuration must have either 'profile', 'profiles' or 'constant'")

    all_power_series = []
    all_power_names = []
    for profile_config in profile_configs:
        if profile_config.profile_csv:
            name = profile_config.profile_csv
        else:
            name = profile_config.profile_dir
        name = os.path.basename(os.path.normpath(name))

        logging.info(f"Generating load profile for {name}...")
        profiler = Profiler(
            scaling_factor=profile_config.scaling_factor,
            profile_csv=substitute_vars(profile_config.profile_csv, env_vars),
            profile_csv_dir=substitute_vars(profile_config.profile_dir, env_vars),
            energy_cols=profile_config.energy_cols,
            parse_clock_time=profile_config.parse_clock_time,
            clock_time_zone=profile_config.clock_time_zone,
        )
        energy = profiler.get_for(time_index)
        power = energy / (STEP_SIZE.total_seconds() / 3600)

        all_power_series.append(power)
        all_power_names.append(name)

    total_power = reduce(lambda x, y: x.add(y, fill_value=0), all_power_series)
    if do_plots:
        fig = go.Figure()
        for i, series in enumerate(all_power_series):
            fig.add_trace(
                go.Scatter(
                    x=series.index,
                    y=series,
                    mode='lines',
                    name=f"power-{i}-{all_power_names[i]}"
                )
            )
        fig.add_trace(go.Scatter(x=total_power.index, y=total_power, name="total-power"))
        fig.update_layout(
            title=f"{context_hint} Profile(s)",
            yaxis_title="Power (kW)"
        )
        fig.show()

    return total_power
