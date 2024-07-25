import logging
import os
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytz

from simt_common.jsonconfig.rates import parse_supply_points, process_rates_for_all_energy_flows
from simt_common.rates.microgrid import get_rates_dfs
from simt_common.timeutils.hh_math import floor_hh

from skypro.cli_utils.cli_utils import read_json_file, set_auto_accept_cli_warnings
from skypro.commands.simulator.algorithms.price_curve.algo import run_price_curve_imbalance_algo
from skypro.commands.simulator.algorithms.spread.algo import run_spread_based_algo
from skypro.commands.simulator.config import parse_config, Solar, Load, ConfigV3, ConfigV4
from skypro.commands.simulator.config.config_v3 import SimulationCaseV3
from skypro.commands.simulator.config.config_v4 import SimulationCaseV4, OutputLoad, OutputSimulation, OutputSummary
from skypro.commands.simulator.output import save_simulation_output
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
        skip_cli_warnings: bool,
        chosen_sim_name: Optional[str] = None,
        v3_output_file_path: Optional[str] = None,
        v3_output_summary_file_path: Optional[str] = None,
        v3_output_aggregate: Optional[str] = None,
        v3_output_rate_detail: Optional[bool] = False,
):

    logging.info("Simulator - - - - - - - - - - - - -")

    set_auto_accept_cli_warnings(skip_cli_warnings)

    logging.info(f"Using env var file: {os.path.expanduser(env_file_path)}")
    env_vars = read_json_file(env_file_path)["vars"]

    # Parse the main config file
    logging.info(f"Using config file: {config_file_path}")
    config: ConfigV3 | ConfigV4 = parse_config(config_file_path, env_vars)

    if isinstance(config, ConfigV3):
        simulations = {"v3sim": config.simulation}
        if chosen_sim_name:
            raise ValueError("The --sim option is not compatible with a v3 config")
    elif isinstance(config, ConfigV4):
        if v3_output_file_path:
            raise ValueError("The --output option is not compatible with a v4 config")
        if v3_output_summary_file_path:
            raise ValueError("The --output-summary option is not compatible with a v4 config")
        if v3_output_aggregate:
            raise ValueError("The --aggregate option is not compatible with a v4 config")
        if v3_output_rate_detail:
            raise ValueError("The --rate-detail option is not compatible with a v4 config")

        if not chosen_sim_name:
            raise ValueError("You must specify the --sim to run when using a V4 configuration file.")
        if chosen_sim_name == "all":
            simulations = config.simulations
        elif chosen_sim_name in config.simulations.keys():
            simulations = {chosen_sim_name: config.simulations[chosen_sim_name]}
        else:
            raise KeyError(f"Simulation case '{chosen_sim_name}' is not defined in the configuration.")
    else:
        raise AssertionError("Configuration type unknown")

    summary_df = pd.DataFrame()

    sim_config: SimulationCaseV3 | SimulationCaseV4
    for sim_name, sim_config in simulations.items():

        print("\n\n")
        logging.info(f"Running simulation '{sim_name}'...")

        sim_summary_df = run_one_simulation(
            sim_config=sim_config,
            do_plots=do_plots,
            v3_output_aggregate=v3_output_aggregate,
            v3_output_file_path=v3_output_file_path,
            v3_output_rate_detail=v3_output_rate_detail,
            v3_output_summary_file_path=v3_output_summary_file_path
        )

        # Maintain a dataframe containing the summaries of each simulation
        sim_summary_df.index = pd.Series([sim_name])
        summary_df = pd.concat([summary_df, sim_summary_df], axis=0)

    if isinstance(config, ConfigV4):
        if chosen_sim_name == "all" and config.all_sims_output and config.all_sims_output.summary:
            # Optionally write a CSV file containing the summaries of all the simulations
            summary_df.to_csv(config.all_sims_output.summary.csv, index_label="simulation")


def run_one_simulation(
        sim_config: SimulationCaseV3 | SimulationCaseV4,
        do_plots: bool,
        v3_output_aggregate,
        v3_output_file_path,
        v3_output_rate_detail,
        v3_output_summary_file_path
) -> pd.DataFrame:

    # Parse the supply points config file:
    supply_points = parse_supply_points(
        supply_points_config_file=sim_config.rates.supply_points_config_file
    )
    # Run the simulation at 10 minute granularity
    time_index = pd.date_range(
        start=sim_config.start.astimezone(pytz.UTC),
        end=sim_config.end.astimezone(pytz.UTC),
        freq=STEP_SIZE
    )
    time_index = time_index.tz_convert("UTC")
    # Imbalance pricing/volume data can come from either Modo or Elexon, Modo is 'predictive' and it's predictions
    # change over the course of the SP, whereas Elexon publishes a single figure for each SP in hindsight.
    df = read_imbalance_data(
        time_index=time_index,
        price_dir=sim_config.imbalance_data_source.price_dir,
        volume_dir=sim_config.imbalance_data_source.volume_dir,
    )
    predicted_rates = process_rates_for_all_energy_flows(
        bess_charge_from_solar=sim_config.rates.files.bess_charge_from_solar,
        bess_charge_from_grid=sim_config.rates.files.bess_charge_from_grid,
        bess_discharge_to_load=sim_config.rates.files.bess_discharge_to_load,
        bess_discharge_to_grid=sim_config.rates.files.bess_discharge_to_grid,
        solar_to_grid=sim_config.rates.files.solar_to_grid,
        load_from_grid=sim_config.rates.files.load_from_grid,
        supply_points=supply_points,
        imbalance_pricing=df["imbalance_price_predicted"]
    )
    final_rates = process_rates_for_all_energy_flows(
        bess_charge_from_solar=sim_config.rates.files.bess_charge_from_solar,
        bess_charge_from_grid=sim_config.rates.files.bess_charge_from_grid,
        bess_discharge_to_load=sim_config.rates.files.bess_discharge_to_load,
        bess_discharge_to_grid=sim_config.rates.files.bess_discharge_to_grid,
        solar_to_grid=sim_config.rates.files.solar_to_grid,
        load_from_grid=sim_config.rates.files.load_from_grid,
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
    df["solar_power"] = process_profiles(
        time_index=time_index,
        config=sim_config.site.solar,
        do_plots=do_plots,
        context_hint="Solar",
        output_config=None
    )
    load_output_config: Optional[OutputLoad]
    if isinstance(sim_config, SimulationCaseV3):
        load_output_config = None
    else:
        load_output_config = sim_config.output.load if sim_config.output else None
    # Create load data
    logging.info("Generating load profile...")
    df["load_power"] = process_profiles(
        time_index=time_index,
        config=sim_config.site.load,
        do_plots=do_plots,
        context_hint="Load",
        output_config=load_output_config
    )
    # Calculate the BESS charge and discharge limits based on how much solar generation and housing load
    # there is. We need to abide by the overall site import/export limits. And stay within the nameplate inverter
    # capabilities of the BESS
    df["microgrid_residual_power"] = df["load_power"] - df["solar_power"]
    df["bess_max_power_charge"] = ((sim_config.site.grid_connection.import_limit - df["microgrid_residual_power"]).
                                   clip(upper=sim_config.site.bess.nameplate_power))
    df["bess_max_power_discharge"] = ((sim_config.site.grid_connection.export_limit + df["microgrid_residual_power"]).
                                      clip(upper=sim_config.site.bess.nameplate_power))
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
    if sim_config.strategy.price_curve_algo:
        df_algo = run_price_curve_imbalance_algo(
            df_in=df[cols_to_share_with_algo],
            battery_energy_capacity=sim_config.site.bess.energy_capacity,
            battery_charge_efficiency=sim_config.site.bess.charge_efficiency,
            config=sim_config.strategy.price_curve_algo
        )
    elif sim_config.strategy.spread_algo:
        # TODO: there should be a more elegant way of doing this - but at the moment the spread based algo needs to know
        #  the "non imbalance" rates, seperated from the imbalance rates. These are calculated here, because if the algo
        #  did these calculations itself then we would need to share 'final' rates with it which is best avoided as it
        #  shouldn't know about final rates.

        df["rate_bess_charge_from_grid_non_imbalance"] = df["rate_final_bess_charge_from_grid"] - df[
            "imbalance_price_final"]
        df["rate_bess_discharge_to_grid_non_imbalance"] = df["rate_final_bess_discharge_to_grid"] + df[
            "imbalance_price_final"]
        cols_to_share_with_algo.extend(
            ["rate_bess_charge_from_grid_non_imbalance", "rate_bess_discharge_to_grid_non_imbalance"])

        df_algo = run_spread_based_algo(
            df_in=df[cols_to_share_with_algo],
            battery_energy_capacity=sim_config.site.bess.energy_capacity,
            battery_charge_efficiency=sim_config.site.bess.charge_efficiency,
            config=sim_config.strategy.spread_algo,
        )
    else:
        raise ValueError("Unknown algorithm chosen")
    df = pd.concat([df, df_algo], axis=1)
    df = calculate_microgrid_flows(df)
    simulation_output_config: Optional[OutputSimulation] = None
    if isinstance(sim_config, SimulationCaseV3):
        if v3_output_file_path:
            simulation_output_config = OutputSimulation(
                csv=v3_output_file_path,
                aggregate=v3_output_aggregate,
                rate_detail=v3_output_rate_detail
            )
    else:
        simulation_output_config = sim_config.output.simulation if sim_config.output else None
    if simulation_output_config:
        save_simulation_output(
            df=df,
            final_rates_dfs=final_rates_dfs,
            sim_config=sim_config,
            output_config=simulation_output_config
        )
    summary_output_config: Optional[OutputSummary] = None
    if isinstance(sim_config, SimulationCaseV3):
        if v3_output_summary_file_path:
            summary_output_config = OutputSummary(
                csv=v3_output_summary_file_path
            )
    else:
        summary_output_config = sim_config.output.summary if sim_config.output else None
    sim_summary_df = explore_results(
        df=df,
        final_rates_dfs=final_rates_dfs,
        do_plots=do_plots,
        battery_energy_capacity=sim_config.site.bess.energy_capacity,
        battery_nameplate_power=sim_config.site.bess.nameplate_power,
        site_import_limit=sim_config.site.grid_connection.import_limit,
        site_export_limit=sim_config.site.grid_connection.export_limit,
        summary_output_config=summary_output_config
    )

    return sim_summary_df


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


def process_profiles(
        time_index: pd.DatetimeIndex,
        config: Solar | Load,
        do_plots: bool,
        context_hint: str,
        output_config: Optional[OutputLoad]
) -> pd.Series:
    """
    Reads the specified profile configuration and returns the total power in a pd.Series.
    This function also optionally plots the profiles and exports a CSV of the profiles broken down by category.
    """

    if config.profile:
        profile_configs = [config.profile]
    elif config.profiles:
        profile_configs = config.profiles
    elif not np.isnan(config.constant):
        return pd.Series(index=time_index, data=config.constant)
    else:
        raise ValueError("Configuration must have either 'profile', 'profiles' or 'constant'")

    energy_df = pd.DataFrame(index=time_index)
    power_df = pd.DataFrame(index=time_index)

    for i, profile_config in enumerate(profile_configs):
        if profile_config.tag:
            name = profile_config.tag
        else:
            if profile_config.profile_csv:
                name = profile_config.profile_csv
            else:
                name = profile_config.profile_dir
            name = os.path.basename(os.path.normpath(name))
            name = f"profile-{i}-{name}"

        logging.info(f"Generating load profile for {name}...")
        profiler = Profiler(
            scaling_factor=profile_config.scaling_factor,
            profile_csv=profile_config.profile_csv,
            profile_csv_dir=profile_config.profile_dir,
            energy_cols=profile_config.energy_cols,
            parse_clock_time=profile_config.parse_clock_time,
            clock_time_zone=profile_config.clock_time_zone,
        )
        energy_df[name] = profiler.get_for(time_index)
        power_df[name] = energy_df[name] / (STEP_SIZE.total_seconds() / 3600)

    total_power = power_df.sum(axis=1)  # TODO: check

    if do_plots:
        fig = go.Figure()
        for name in power_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=power_df.index,
                    y=power_df[name],
                    mode='lines',
                    name=name
                )
            )
        fig.add_trace(go.Scatter(x=total_power.index, y=total_power, name="total-power"))
        fig.update_layout(
            title=f"{context_hint} Profile(s)",
            yaxis_title="Power (kW)"
        )
        fig.show()

    if output_config:
        # Optionally export a CSV of the load profiles

        if output_config.aggregate:
            # Optionally aggregate to 30mins to reduce the size of the CSV
            if output_config.aggregate == "30min":
                # Energy profiles are in kWh should all be summed to aggregate
                aggregate_energy_df = energy_df.resample("30min").sum()
            else:
                raise ValueError(f"Unknown aggregate option: '{output_config.aggregate}'")
        else:
            aggregate_energy_df = energy_df

        aggregate_energy_df.to_csv(output_config.csv, index_label="UTCTime")

    return total_power
