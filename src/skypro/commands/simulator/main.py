import importlib.metadata
import logging
import os
from dataclasses import dataclass, field
from datetime import timedelta
from functools import partial
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytz

from simt_common.jsonconfig.rates import parse_supply_points, parse_rates_files_for_all_energy_flows, parse_rate_files
from simt_common.microgrid.output import generate_output_df
from simt_common.rates.microgrid import get_vol_rates_dfs, VolRatesForEnergyFlows
from simt_common.rates.osam import calculate_osam_ncsp
from simt_common.rates.rates import FixedRate, Rate, OSAMFlatVolRate
from simt_common.timeutils.math import floor_hh

from skypro.cli_utils.cli_utils import read_json_file, set_auto_accept_cli_warnings
from skypro.commands.simulator.algorithms.lp.optimiser import Optimiser
from skypro.commands.simulator.algorithms.price_curve.algo import PriceCurveAlgo
from skypro.commands.simulator.config import parse_config, Solar, Load, ConfigV4
from skypro.commands.simulator.config.config_v4 import SimulationCaseV4, AllRates
from skypro.commands.simulator.config.path_field import resolve_file_path
from skypro.commands.simulator.microgrid import calculate_microgrid_flows
from skypro.commands.simulator.parse_imbalance_data import read_imbalance_data, normalise_final_imbalance_data, \
    normalise_live_imbalance_data
from skypro.commands.simulator.profiler import Profiler
from skypro.commands.simulator.results import explore_results


STEP_SIZE = timedelta(minutes=10)
STEP_SIZE_HRS = STEP_SIZE.total_seconds() / 3600
STEPS_PER_SP = int(timedelta(minutes=30) / STEP_SIZE)
assert ((timedelta(minutes=30) / STEP_SIZE) == STEPS_PER_SP)  # Check that we have an exact number of steps per SP


@dataclass
class ParsedRates:
    """
    This is just a container to hold the various rates
    """
    # TODO: Rename live_vol -> live_sup_vol? and fixed_market -> sup_fix?
    live_vol: VolRatesForEnergyFlows = field(default_factory=VolRatesForEnergyFlows)   # Volume-based (p/kWh) rates for each energy flow, as predicted in real-time
    final_vol: VolRatesForEnergyFlows = field(default_factory=VolRatesForEnergyFlows)  # Volume-based (p/kWh) rates for each energy flow
    fixed_market: List[FixedRate] = field(default_factory=list)      # Fixed p/day rates associated with suppliers
    customer: List[Rate] = field(default_factory=list)               # Volume and fixed rates charged to customers


def simulate(
        config_file_path: str,
        env_file_path: str,
        do_plots: bool,
        skip_cli_warnings: bool,
        chosen_sim_name: Optional[str] = None,
):

    logging.info("Simulator - - - - - - - - - - - - -")

    set_auto_accept_cli_warnings(skip_cli_warnings)

    logging.info(f"Using env var file: {os.path.expanduser(env_file_path)}")
    env_vars = read_json_file(env_file_path)["vars"]

    # Parse the main config file
    logging.info(f"Using config file: {config_file_path}")
    config: ConfigV4 = parse_config(config_file_path, env_vars)

    if not chosen_sim_name:
        raise ValueError("You must specify the --sim to run.")
    if chosen_sim_name == "all":
        simulations = config.simulations
    elif chosen_sim_name in config.simulations.keys():
        simulations = {chosen_sim_name: config.simulations[chosen_sim_name]}
    else:
        raise KeyError(f"Simulation case '{chosen_sim_name}' is not defined in the configuration.")

    if config.all_sims and config.all_sims.output and config.all_sims.output.summary and config.all_sims.output.summary.rate_detail:
        raise ValueError(
            "The 'rateDetail' option is invalid for allSimulations - please specify the rateDetail option within"
            " each simulations' summary output configuration."
        )

    summary_df = pd.DataFrame()

    for sim_name, sim_config in simulations.items():

        print("\n\n")
        logging.info(f"Running simulation '{sim_name}' from {sim_config.start} to {sim_config.end}...")

        sim_summary_df = run_one_simulation(
            sim_config=sim_config,
            do_plots=do_plots,
            env_vars=env_vars
        )
        # Maintain a dataframe containing the summaries of each simulation
        sim_summary_df.insert(0, "sim_name", sim_name)
        summary_df = pd.concat([summary_df, sim_summary_df], axis=0)

    if chosen_sim_name == "all" and config.all_sims and config.all_sims.output and config.all_sims.output.summary:
        # Optionally write a CSV file containing the summaries of all the simulations
        summary_df.to_csv(config.all_sims.output.summary.csv, index=False)


def run_one_simulation(
        sim_config: SimulationCaseV4,
        do_plots: bool,
        env_vars: Dict,
) -> pd.DataFrame:
    """
    Runs a single simulation as defined by the configuration and returns a dataframe containing a summary of the results
    """

    time_index_start = sim_config.start.astimezone(pytz.UTC)
    time_index_end = sim_config.end.astimezone(pytz.UTC) - STEP_SIZE
    if time_index_end <= time_index_start:
        raise ValueError("Simulation end time is before the start time")

    # The simulation runs at 10 minute granularity, create a time index for that
    time_index = pd.date_range(
        start=sim_config.start.astimezone(pytz.UTC),
        end=sim_config.end.astimezone(pytz.UTC) - STEP_SIZE,
        freq=STEP_SIZE
    )
    time_index = time_index.tz_convert(pytz.timezone("Europe/London"))

    # Extract the rates objects from the config files
    rates, imbalance_df = get_rates_from_config(time_index, sim_config.rates, env_vars)

    # Log the parsed rates for user information
    for name, rate_set in rates.live_vol.get_all_sets_named():
        for rate in rate_set:
            logging.info(f"Flow: {name}, Rate: {rate}")

    df = imbalance_df[["imbalance_volume_live", "imbalance_volume_final"]].copy()

    # Process solar profiles
    solar_energy_breakdown_df, total_solar_power = process_profiles(
        time_index=time_index,
        config=sim_config.site.solar,
        do_plots=do_plots,
        context_hint="Solar"
    )
    df["solar"] = solar_energy_breakdown_df.sum(axis=1)
    df["solar_power"] = total_solar_power

    # Process load profiles
    load_energy_breakdown_df, total_load_power = process_profiles(
        time_index=time_index,
        config=sim_config.site.load,
        do_plots=do_plots,
        context_hint="Load"
    )
    df["load"] = load_energy_breakdown_df.sum(axis=1)
    df["load_power"] = total_load_power

    # Calculate the BESS charge and discharge limits based on how much solar generation and housing load
    # there is. We need to abide by the overall site import/export limits. And stay within the nameplate inverter
    # capabilities of the BESS
    df["microgrid_residual_power"] = df["load_power"] - df["solar_power"]
    df["bess_max_power_charge"] = ((sim_config.site.grid_connection.import_limit - df["microgrid_residual_power"]).
                                   clip(upper=sim_config.site.bess.nameplate_power))
    df["bess_max_power_discharge"] = ((sim_config.site.grid_connection.export_limit + df["microgrid_residual_power"]).
                                      clip(upper=sim_config.site.bess.nameplate_power))
    # Also store the energy equivalent of the powers
    df["bess_max_charge"] = df["bess_max_power_charge"] * STEP_SIZE_HRS
    df["bess_max_discharge"] = df["bess_max_power_discharge"] * STEP_SIZE_HRS

    # Calculate settlement period timings
    df["sp"] = df.index.to_series().apply(lambda t: floor_hh(t))
    df["time_into_sp"] = df.index.to_series() - df["sp"]
    df["time_left_of_sp"] = timedelta(minutes=30) - df["time_into_sp"]

    # Only share the columns that are relevant with the algo
    cols_to_share_with_algo = [
        "sp",
        "time_into_sp",
        "time_left_of_sp",
        "solar",
        "solar_power",
        "load",
        "load_power",
        "microgrid_residual_power",
        "bess_max_power_charge",
        "bess_max_power_discharge",
        "imbalance_volume_live",
    ]

    # Run the configured algo
    if sim_config.strategy.price_curve_algo:
        algo = PriceCurveAlgo(
            algo_config=sim_config.strategy.price_curve_algo,
            bess_config=sim_config.site.bess,
            live_vol_rates=rates.live_vol,
            df=df[cols_to_share_with_algo]
        )
        df_algo = algo.run()
    elif sim_config.strategy.optimiser:

        cols_to_share_with_algo.extend([
            "bess_max_charge",
            "bess_max_discharge",
        ])
        opt = Optimiser(
            algo_config=sim_config.strategy.optimiser,
            bess_config=sim_config.site.bess,
            final_vol_rates=rates.final_vol,
            df=df[cols_to_share_with_algo],
        )
        df_algo = opt.run()
    else:
        raise ValueError("Unknown algorithm chosen")

    check_algo_result_consistency(
        df_algo=df_algo,
        df_in=df,
        battery_charge_efficiency=sim_config.site.bess.charge_efficiency,
    )

    # Add the results of the algo into the main dataframe
    df = pd.concat([df, df_algo], axis=1)

    df = calculate_microgrid_flows(df)

    # The algorithm has used the 'live' rates that were available at the simulated time, now we ascertain the 'final'
    # rates for use in reporting.
    df["osam_ncsp"], osam_df = calculate_osam_ncsp(
        df=df,
        index_to_calc_for=df.index,
        imp_bp_col="grid_import",
        exp_bp_col="grid_export",
        imp_stor_col="bess_charge",
        exp_stor_col="bess_discharge",
        imp_gen_col=None,
        exp_gen_col="solar",
    )
    # Inform any OSAM rate objects about the NCSP
    osam_rates = []
    for rate in rates.final_vol.bess_charge_from_grid:
        if isinstance(rate, OSAMFlatVolRate):
            rate.add_ncsp(df["osam_ncsp"])
            osam_rates.append(rate)

    # Next we can calculate the individual p/kWh rates that apply for today
    final_ext_vol_rates_dfs, final_int_vol_rates_dfs = get_vol_rates_dfs(df.index, rates.final_vol)

    # Then we sum up the individual rates to create a total for each flow
    for set_name, vol_rates_df in final_ext_vol_rates_dfs.items():
        df[f"vol_rate_final_{set_name}"] = vol_rates_df.sum(axis=1, skipna=False)
    for set_name, vol_rates_df in final_int_vol_rates_dfs.items():
        df[f"int_vol_rate_final_{set_name}"] = vol_rates_df.sum(axis=1, skipna=False)

    # Generate an output file if configured to do so
    simulation_output_config = sim_config.output.simulation if sim_config.output else None
    if simulation_output_config and simulation_output_config.csv:
        logging.info(f"Generating output to {simulation_output_config.csv}...")
        generate_output_df(
            df=df,
            int_final_vol_rates_dfs=final_int_vol_rates_dfs,
            ext_final_vol_rates_dfs=final_ext_vol_rates_dfs,
            int_live_vol_rates_dfs=None,  # These 'live' rates aren't available in the output CSV at the moment as they are
            ext_live_vol_rates_dfs=None,  # calculated by the price curve algo internally and not returned
            fixed_market_rates=rates.fixed_market,
            customer_rates=rates.customer,
            load_energy_breakdown_df=load_energy_breakdown_df,
            aggregate_timebase=simulation_output_config.aggregate,
            rate_detail=simulation_output_config.rate_detail,
            config_entries=[
                ("skypro.version", importlib.metadata.version('skypro')),
                ("start", sim_config.start.isoformat()),
                ("end", sim_config.end.isoformat()),
                ("site.gridConnection.importLimit", sim_config.site.grid_connection.import_limit),
                ("site.gridConnection.exportLimit", sim_config.site.grid_connection.export_limit),
                ("site.solar.constant", sim_config.site.solar.constant),
                ("site.solar.profile", sim_config.site.solar.profile),
                ("site.load.constant", sim_config.site.load.constant),
                ("site.load.profile", sim_config.site.load.profile),
                ("site.bess.energyCapacity", sim_config.site.bess.energy_capacity),
                ("site.bess.nameplatePower", sim_config.site.bess.nameplate_power),
                ("site.bess.chargeEfficiency", sim_config.site.bess.charge_efficiency),
                ("strategy.priceCurveAlgo", sim_config.strategy.price_curve_algo),
                ("rates", sim_config.rates),
            ]
        ).to_csv(
            simulation_output_config.csv,
            index_label="utctime"
        )

    save_summary = sim_config.output and sim_config.output.summary and sim_config.output.summary.csv
    if save_summary:
        logging.info(f"Generating summary to {sim_config.output.summary.csv}...")
    else:
        logging.info("Generating summary...")

    # The summary dataframe is just an output dataframe with aggregate_timebase set to 'all'
    sim_summary_df = generate_output_df(
        df=df,
        int_final_vol_rates_dfs=final_int_vol_rates_dfs,
        ext_final_vol_rates_dfs=final_ext_vol_rates_dfs,
        int_live_vol_rates_dfs=None,  # These 'live' rates aren't available in the output CSV at the moment as they are
        ext_live_vol_rates_dfs=None,  # calculated by the price curve algo internally and not returned
        fixed_market_rates=rates.fixed_market,
        customer_rates=rates.customer,
        load_energy_breakdown_df=load_energy_breakdown_df,
        aggregate_timebase="all",
        rate_detail=sim_config.output.summary.rate_detail if (sim_config.output and sim_config.output.summary) else None,
        config_entries=[],
    )

    if save_summary:
        sim_summary_df.to_csv(sim_config.output.summary.csv, index=False)

    explore_results(
        df=df,
        final_ext_vol_rates_dfs=final_ext_vol_rates_dfs,
        final_int_vol_rates_dfs=final_int_vol_rates_dfs,
        do_plots=do_plots,
        battery_energy_capacity=sim_config.site.bess.energy_capacity,
        battery_nameplate_power=sim_config.site.bess.nameplate_power,
        site_import_limit=sim_config.site.grid_connection.import_limit,
        site_export_limit=sim_config.site.grid_connection.export_limit,
        osam_rates=osam_rates,
        osam_df=osam_df,
    )

    return sim_summary_df


def get_rates_from_config(
        time_index: pd.DatetimeIndex,
        rates_config: AllRates,
        env_vars: Dict
) -> Tuple[ParsedRates, pd.DataFrame]:
    """
    This reads the rates files defined in the given rates configuration block and returns the ParsedRates,
    and a dataframe containing live and final imbalance data.
    """
    final_supply_points = parse_supply_points(
        supply_points_config_file=rates_config.final.supply_points_config_file
    )
    live_supply_points = parse_supply_points(
        supply_points_config_file=rates_config.live.supply_points_config_file
    )

    final_price_df, final_volume_df = read_imbalance_data(
        start=time_index[0],
        end=time_index[-1],
        price_dir=rates_config.final.imbalance_data_source.price_dir,
        volume_dir=rates_config.final.imbalance_data_source.volume_dir,
    )
    live_price_df, live_volume_df = read_imbalance_data(
        start=time_index[0],
        end=time_index[-1],
        price_dir=rates_config.live.imbalance_data_source.price_dir,
        volume_dir=rates_config.live.imbalance_data_source.volume_dir,
    )

    final_df = normalise_final_imbalance_data(time_index, final_price_df, final_volume_df)
    live_df = normalise_live_imbalance_data(time_index, live_price_df, live_volume_df)
    df = pd.concat([final_df, live_df], axis=1)

    parsed_rates = ParsedRates()

    file_path_resolver_func = partial(resolve_file_path, env_vars=env_vars)

    parsed_rates.final_vol = parse_rates_files_for_all_energy_flows(
        bess_charge_from_solar=rates_config.final.files.bess_charge_from_solar,
        bess_charge_from_grid=rates_config.final.files.bess_charge_from_grid,
        bess_discharge_to_load=rates_config.final.files.bess_discharge_to_load,
        bess_discharge_to_grid=rates_config.final.files.bess_discharge_to_grid,
        solar_to_grid=rates_config.final.files.solar_to_grid,
        solar_to_load=rates_config.final.files.solar_to_load,
        load_from_grid=rates_config.final.files.load_from_grid,
        supply_points=final_supply_points,
        imbalance_pricing=df["imbalance_price_final"],
        file_path_resolver_func=file_path_resolver_func
    )
    parsed_rates.live_vol = parse_rates_files_for_all_energy_flows(
        bess_charge_from_solar=rates_config.live.files.bess_charge_from_solar,
        bess_charge_from_grid=rates_config.live.files.bess_charge_from_grid,
        bess_discharge_to_load=rates_config.live.files.bess_discharge_to_load,
        bess_discharge_to_grid=rates_config.live.files.bess_discharge_to_grid,
        solar_to_grid=rates_config.live.files.solar_to_grid,
        solar_to_load=rates_config.live.files.solar_to_load,
        load_from_grid=rates_config.live.files.load_from_grid,
        supply_points=live_supply_points,
        imbalance_pricing=df["imbalance_price_live"],
        file_path_resolver_func=file_path_resolver_func
    )

    if rates_config.final.experimental:
        if rates_config.final.experimental.fixed_market_files:
            # Read in fixed rates just to output them in the CSV
            parsed_rates.fixed_market = parse_rate_files(
                files=rates_config.final.experimental.fixed_market_files,
                supply_points=final_supply_points,
                imbalance_pricing=None,
                file_path_resolver_func=file_path_resolver_func,
            )

        if rates_config.final.experimental.customer_load_files:
            parsed_rates.customer = parse_rate_files(
                files=rates_config.final.experimental.customer_load_files,
                supply_points=final_supply_points,
                imbalance_pricing=None,
                file_path_resolver_func=file_path_resolver_func,
            )

    # TODO: we need to be sure that there are no fixed rates in live + final, and only fixed rates in fixed_market_rates

    return parsed_rates, df


def check_algo_result_consistency(df_algo: pd.DataFrame, df_in: pd.DataFrame, battery_charge_efficiency: float):
    """
    Does various checks to ensure that the algorithm results are viable. The algos generate their results in
    different ways, so we want to check that they are all following basic rules here.
    These could be written as unit tests, but they run quickly so there's no harm in running them over every
    result set that is generated.
    """

    # TODO: rename energy_delta in whole project

    tolerance = 0.01

    # Calculate the energy delta from the soe and check that it matches the energy delta given
    soe_diff = df_algo["soe"].diff().shift(-1)
    soe_diff.iloc[-1] = 0.0
    energy_delta_check = soe_diff.copy()
    charges = energy_delta_check[energy_delta_check > 0] / battery_charge_efficiency
    discharges = energy_delta_check[energy_delta_check < 0]
    energy_delta_check.loc[charges.index] = charges
    energy_delta_check.iloc[-1] = df_algo["energy_delta"].iloc[-1]  # There isn't a valid soe diff on the last row
    if (df_algo["energy_delta"] - energy_delta_check).abs().max() > tolerance:
        raise AssertionError("Algorithm solution has inconsistent energy delta")

    # Check the bess losses are expected given the SoE
    bess_losses = charges * (1 - battery_charge_efficiency)
    bess_losses_check = pd.Series(index=df_algo.index, data=0.0)
    bess_losses_check.loc[bess_losses.index] = bess_losses
    bess_losses_error = bess_losses_check - df_algo["bess_losses"]
    bess_losses_error = bess_losses_error.iloc[:-1]  # There isn't a valid check for the last row
    if bess_losses_error.abs().max() > tolerance:
        raise AssertionError("Algorithm solution has inconsistent bess losses")

    # Check that the max charge/discharges are not breached
    if (charges.abs() > (df_in["bess_max_charge"].loc[charges.index] + tolerance)).sum() > 0:
        raise AssertionError("Algorithm solution charges at too high a rate")
    if (discharges.abs() > (df_in["bess_max_discharge"].loc[discharges.index] + tolerance)).sum() > 0:
        raise AssertionError("Algorithm solution discharges at too high a rate")


def process_profiles(
        time_index: pd.DatetimeIndex,
        config: Solar | Load,
        do_plots: bool,
        context_hint: str
) -> (pd.DataFrame, pd.Series):
    """
    Reads the specified profile configuration and returns a dataframe of the individual profiled energies, as well as
    the summed total power in a pd.Series.
    This function also optionally plots the profiles and exports a CSV of the profiles broken down by category.
    """

    if config.profile:
        profile_configs = [config.profile]
    elif isinstance(config, Load) and config.profiles:
        profile_configs = config.profiles
    elif not np.isnan(config.constant):
        energy = pd.Series(index=time_index, data=config.constant)
        return energy.to_frame(), energy_to_power(energy)
    else:
        raise ValueError("Configuration must have either 'profile', 'profiles' or 'constant'")

    energy_df = pd.DataFrame(index=time_index)
    power_df = pd.DataFrame(index=time_index)

    for i, profile_config in enumerate(profile_configs):
        if profile_config.tag:
            tag = profile_config.tag
        else:
            tag = "untagged"

        logging.info(f"Generating {context_hint} profile for '{tag}'...")
        profiler = Profiler(
            scaling_factor=profile_config.scaling_factor,
            profile_csv=profile_config.profile_csv,
            profile_csv_dir=profile_config.profile_dir,
            energy_cols=profile_config.energy_cols,
        )
        energy = profiler.get_for(time_index)
        power = energy_to_power(energy)

        # There may be multiple profiles under the same tag - in which case the profiles are added together under the
        # tag name.
        if tag in energy_df.columns:
            energy_df[tag] = energy_df[tag] + energy
            power_df[tag] = power_df[tag] + power
        else:
            energy_df[tag] = energy
            power_df[tag] = power

    total_power = power_df.sum(axis=1)
    if do_plots:
        fig = go.Figure()
        for tag in power_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=power_df.index,
                    y=power_df[tag],
                    mode='lines',
                    name=tag
                )
            )
        fig.add_trace(go.Scatter(x=total_power.index, y=total_power, name="total-power"))
        fig.update_layout(
            title=f"{context_hint} Profile(s)",
            yaxis_title="Power (kW)"
        )
        fig.show()

    return energy_df, total_power


def energy_to_power(energy: pd.Series) -> pd.Series:
    return energy / (STEP_SIZE.total_seconds() / 3600)
