import logging
import os
from dataclasses import dataclass
from datetime import timedelta, datetime
from typing import Optional, Callable, List, Dict

import numpy as np
import pandas as pd
from skypro.common.config.data_source import MeterReadingDataSource
from skypro.common.data.get_bess_readings import get_bess_readings
from skypro.common.data.get_meter_readings import get_meter_readings
from skypro.common.data.get_plot_meter_readings import get_plot_meter_readings
from skypro.common.microgrid_analysis.breakdown import breakdown_microgrid_flows, MicrogridBreakdown
from skypro.common.microgrid_analysis.daily_gains import plot_daily_gains
from skypro.common.microgrid_analysis.output import generate_output_df
from skypro.common.notice.notice import Notice
from skypro.common.rates.bill_match import bill_match
from skypro.common.rates.osam import calculate_osam_ncsp
from skypro.common.rates.peripheral import get_rates_dfs_by_type
from skypro.common.rates.rates import OSAMFlatVolRate
from skypro.common.rates.rates_friendly_summary import get_friendly_rates_summary
from skypro.common.timeutils.timeseries import get_steps_per_hh
from tabulate import tabulate

from skypro.common.cli_utils.cli_utils import substitute_vars, set_auto_accept_cli_warnings
from skypro.common.rates.microgrid import get_vol_rates_dfs

from skypro.common.cli_utils.cli_utils import read_yaml_file
from skypro.common.data.utility import prepare_data_dir
from skypro.commands.report.time import get_month_timerange
from skypro.commands.report.config.config import parse_config, Config
from skypro.commands.report.microgrid_flow_calcs import calculate_missing_net_flows_in_junction, calc_flows, \
    synthesise_battery_inverter_if_needed, approximate_solar_and_load
from skypro.commands.report.plots import plot_load_and_solar
from skypro.commands.report.rates import get_rates_from_config
from skypro.commands.report.warnings import missing_data_warnings, energy_discrepancy_warnings

TIMEZONE_STR = "Europe/London"


@dataclass
class Report:
    """
    Holds the results of a reporting run
    """
    df: pd.DataFrame  # a time-series with detailed info

    # The various rates and costs separated by flow
    mkt_vol_rates_dfs: Dict[str, pd.DataFrame]
    int_vol_rates_dfs: Dict[str, pd.DataFrame]
    mkt_fixed_cost_dfs: Dict[str, pd.DataFrame]
    customer_fixed_cost_dfs: Dict[str, pd.DataFrame]
    customer_vol_rates_dfs: Dict[str, pd.DataFrame]

    breakdown: MicrogridBreakdown  # Some key results which are shared with Skypro

    osam_rates: List[OSAMFlatVolRate]
    osam_df: pd.DataFrame

    num_days: float  # how many days have been reported over

    total_bess_discharged: float
    total_bess_charged: float

    roundtrip_efficiency: float
    total_cycles: float

    notices: List[Notice]  # Any warnings that the user should be aware of


def report_cli(
        config_file_path: str,
        month_str: str,
        env_file_path: str,
        do_plots: bool,
        do_save_profiles: bool,
        output_file_path: Optional[str] = None
):
    """
    Analyses how the battery has performed and produces a set of statistics and plots.
    """
    logging.info("Reporting - - - - - - - - - - - - -")
    logging.info(f"Using config file: {config_file_path}")

    set_auto_accept_cli_warnings(False)

    env_config = read_yaml_file(env_file_path)
    env_vars = env_config["vars"]

    def file_path_resolver_func(file: str) -> str:
        """
        Substitutes env vars and resolves `~` in file paths. This captures the env_vars variable.
        """
        return os.path.expanduser(substitute_vars(file, env_vars))

    # Read in the main config file
    config = parse_config(config_file_path, env_vars=env_vars)

    step_size = timedelta(minutes=5)
    start, end = get_month_timerange(month_str, TIMEZONE_STR)
    end = end - step_size

    result = report(
        config=config,
        flows_db_url=env_config["flows"]["dbUrl"],
        rates_db_url=env_config["rates"]["dbUrl"],
        start=start,
        end=end,
        step_size=step_size,
        file_path_resolver_func=file_path_resolver_func,
    )

    if do_plots:
        plot_load_and_solar(result.df)

    if output_file_path:
        generate_output_df(
            df=result.df,
            int_final_vol_rates_dfs=result.int_vol_rates_dfs,
            mkt_final_vol_rates_dfs=result.mkt_vol_rates_dfs,
            int_live_vol_rates_dfs=None,
            mkt_live_vol_rates_dfs=None,
            mkt_fixed_costs_dfs=result.mkt_fixed_cost_dfs,
            customer_fixed_cost_dfs=result.customer_fixed_cost_dfs,
            customer_vol_rates_dfs=result.customer_vol_rates_dfs,
            load_energy_breakdown_df=None,
            aggregate_timebase="30min",
            rate_detail=None,
            config_entries=[],
        ).to_csv(
            output_file_path,
            index_label="utctime"
        )

    if do_save_profiles:
        save_profiles(
            df=result.df,
            save_dir=config.reporting.profiles_save_dir,
            start=start
        )

    if do_plots:
        plot_daily_gains(result.breakdown.int_vol_costs_dfs)

    log_report_summary(config, result)


def report(
    config: Config,
    flows_db_url: str,
    rates_db_url: str,
    start: datetime,
    end: datetime,
    step_size: timedelta,
    file_path_resolver_func: Callable,
) -> Report:
    """
    Pulls the data and runs the actual reporting calculations and returns the various results.
    """

    logging.info(f"Reporting for period {start} -> {end}")

    notices: List[Notice] = []

    # 5 minutely data is best as the BESS strategy changes within the half-hour
    time_index = pd.date_range(start, end, freq=step_size).tz_convert(TIMEZONE_STR)

    # we also need a half-hourly datetime index for things like emlite meter data
    time_index_hh = pd.date_range(start, end, freq=timedelta(minutes=30)).tz_convert(TIMEZONE_STR)

    rates, new_notices = get_rates_from_config(
        time_index=time_index,
        config=config,
        file_path_resolver_func=file_path_resolver_func,
        flows_db_engine=flows_db_url,
        rates_db_engine=rates_db_url,
    )
    notices.extend(new_notices)

    print(f"\nIMPORT RATES (at {time_index[0]}, grid to battery)")
    print(tabulate(
        tabular_data=get_friendly_rates_summary(rates.mkt_vol.grid_to_batt, time_index[0]),
        headers="keys",
        tablefmt="presto",
        showindex=False
    ))

    print(f"\nEXPORT RATES (at {time_index[0]}, battery to grid)")
    print(tabulate(
        tabular_data=get_friendly_rates_summary(rates.mkt_vol.batt_to_grid, time_index[0]),
        headers="keys",
        tablefmt="presto",
        showindex=False
    ))

    print(f"\nFIXED RATES (at {time_index[0]})")
    print(tabulate(
        tabular_data=get_friendly_rates_summary(rates.mkt_fix["import"] + rates.mkt_fix["export"], time_index[0]),
        headers="keys",
        tablefmt="presto",
        showindex=False
    ))
    print("")

    def _get_meter_readings(source: MeterReadingDataSource, context: str) -> pd.DataFrame:
        # This just wraps `get_meter_readings` for convenience
        logging.info(f"Loading {context}...")
        data_df, new_notices = get_meter_readings(
            source=source,
            start=time_index[0],
            end=time_index[-1],
            file_path_resolver_func=file_path_resolver_func,
            db_engine=flows_db_url,
            context=context
        )
        notices.extend(new_notices)

        data_df = data_df.set_index("time")
        return data_df

    mg_meter_config = config.reporting.metering.microgrid_meters  # For convenience
    bess_meter_readings = _get_meter_readings(mg_meter_config.bess_inverter.data_source, "bess meter readings")
    grid_meter_readings = _get_meter_readings(mg_meter_config.main_incomer.data_source, "grid meter readings")
    feeder1_meter_readings = _get_meter_readings(mg_meter_config.feeder_1.data_source, "feeder1 meter readings")
    feeder2_meter_readings = _get_meter_readings(mg_meter_config.feeder_2.data_source, "feeder2 meter readings")
    ev_meter_readings = _get_meter_readings(mg_meter_config.ev_charger.data_source, "ev charger meter readings")

    logging.info("Loading plot meter readings...")
    plot_meter_readings, new_notices = get_plot_meter_readings(
        source=config.reporting.metering.plot_meters.data_source,
        start=time_index[0],
        end=time_index[-1],
        file_path_resolver_func=file_path_resolver_func,
        db_engine=flows_db_url,
        context="plot meter readings"
    )
    notices.extend(new_notices)
    plot_meter_readings = plot_meter_readings.set_index("time")

    logging.info("Loading bess readings...")
    bess_readings, new_notices = get_bess_readings(
        source=config.reporting.bess.data_source,
        start=time_index[0],
        end=time_index[-1],
        file_path_resolver_func=file_path_resolver_func,
        db_engine=flows_db_url,
        context="bess readings"
    )
    notices.extend(new_notices)
    bess_readings = bess_readings.set_index("time")

    # Create a dataframe to hold the 5-minutely site data
    df = pd.DataFrame(index=time_index)

    # The import/export nomenclature can be confusing for a battery, so prefer the "charge" and "discharge" terminology
    df["bess_discharge"] = bess_meter_readings["energy_imported_active_delta"]
    df["bess_charge"] = bess_meter_readings["energy_exported_active_delta"]
    df["bess_import_cum_reading"] = bess_meter_readings["energy_imported_active_min"]
    df["bess_export_cum_reading"] = bess_meter_readings["energy_exported_active_min"]
    df["soe"] = bess_readings["soe_avg"]
    df["soe"] = df["soe"].ffill(limit=get_steps_per_hh(step_size)-1)

    df["grid_import_cum_reading"] = grid_meter_readings["energy_imported_active_min"]
    df["grid_export_cum_reading"] = grid_meter_readings["energy_exported_active_min"]
    df["grid_import"] = grid_meter_readings["energy_imported_active_delta"]
    df["grid_export"] = grid_meter_readings["energy_exported_active_delta"]

    df, new_notices = synthesise_battery_inverter_if_needed(df, bess_readings["target_power_avg"])
    notices.extend(new_notices)

    df = calc_flows(
        df=df,
        ev_meter_readings=ev_meter_readings,
        feeder1_meter_readings=feeder1_meter_readings,
        feeder2_meter_readings=feeder2_meter_readings,
    )

    df, new_notices = calculate_missing_net_flows_in_junction(
        df,
        cols_with_direction=[
            ("grid_net", 1),
            ("bess_net", -1),
            ("feeder1_net", -1),
            ("feeder2_net", -1),
            ("ev_net", -1),
        ],
    )
    notices.extend(new_notices)
    notices.extend(missing_data_warnings(df, "Flows metering data for BESS"))

    logging.info("Approximating solar and load...")

    df, new_notices = approximate_solar_and_load(
        df=df,
        plot_meter_readings=plot_meter_readings,
        meter_config=config.reporting.metering.microgrid_meters,
        time_index_hh=time_index_hh
    )
    notices.extend(new_notices)

    df["solar"] = df[["solar_feeder1", "solar_feeder2"]].sum(axis=1)
    df["load"] = df[["plot_load_feeder1", "plot_load_feeder2"]].sum(axis=1)
    df["solar_to_load"] = np.minimum(df["solar"], df["load"])  # requires emlite data

    df["solar_to_load_property_level"] = np.nan  # TODO
    df["solar_to_load_microgrid_level"] = np.nan  # TODO
    df["bess_losses"] = np.nan  # TODO

    # These columns are required for the output CSV, and could be calculated, but there's not currently a need for it
    df["bess_max_charge"] = np.nan
    df["bess_max_discharge"] = np.nan
    df["imbalance_volume_final"] = np.nan
    df["imbalance_volume_predicted"] = np.nan

    df["osam_ncsp"], osam_df = calculate_osam_ncsp(
        df=df,
        index_to_calc_for=df.index,
        imp_bp_col="grid_import",
        exp_bp_col="grid_export",
        imp_stor_col="bess_charge",
        exp_stor_col="bess_discharge",
        imp_gen_col=None,
        exp_gen_col=None,
    )
    # Inform any OSAM rate objects about the NCSP
    osam_rates = []
    for rate in rates.mkt_vol.grid_to_batt:
        if isinstance(rate, OSAMFlatVolRate):
            osam_rates.append(rate)
            rate.add_ncsp(df["osam_ncsp"])

    # Generate dataframes with each individual p/kWh rate for each energy flow
    mkt_vol_rates_dfs, int_vol_rates_dfs = get_vol_rates_dfs(time_index, rates.mkt_vol)

    # Some of the reported results are shared with Skypro, and these are created in a library function here:
    breakdown = breakdown_microgrid_flows(
        df=df,
        int_vol_rates_dfs=int_vol_rates_dfs,
        mkt_vol_rates_dfs=mkt_vol_rates_dfs
    )

    notices.extend(missing_data_warnings(
        breakdown.int_vol_costs_dfs["bess_charge"],
        "Intermediate cost data (likely due to missing BESS meter readings)"
    ))

    # Calculate the various energy flows due to charging and discharging the BESS
    total_bess_discharged_1 = df["bess_import_cum_reading"].iloc[-1] - df["bess_import_cum_reading"].iloc[0]
    total_bess_discharged_2 = breakdown.total_flows["bess_discharge"]
    total_bess_discharged_3 = breakdown.total_flows["batt_to_grid"] + breakdown.total_flows["batt_to_load"]
    total_bess_charged_1 = df["bess_export_cum_reading"].iloc[-1] - df["bess_export_cum_reading"].iloc[0]
    total_bess_charged_2 = breakdown.total_flows["bess_charge"]
    total_bess_charged_3 = breakdown.total_flows["grid_to_batt"] + breakdown.total_flows["solar_to_batt"]

    # The total energy charged and discharged is calculated in three different ways and here we make sure that they are
    # approximately equal - to sanity check the data/calcs. There is often a small discrepancy - perhaps due to missing
    # telemetry, because we only have data at 5minute resolution, or sometimes if meters are offline we may make
    # assumptions like feeder1 == feeder2 in the config
    notices.extend(energy_discrepancy_warnings(
        total_bess_charged_1,
        total_bess_charged_2,
        "total energy charged"
    ))
    notices.extend(energy_discrepancy_warnings(
        total_bess_discharged_1,
        total_bess_discharged_2,
        "total energy discharged"
    ))
    notices.extend(energy_discrepancy_warnings(
        total_bess_charged_1,
        total_bess_charged_3,
        "bess charge split between solar and grid"
    ))
    notices.extend(energy_discrepancy_warnings(
        total_bess_discharged_1,
        total_bess_discharged_3,
        "bess discharge split between load and grid"
    ))

    # The total load is calculated in two ways: firstly at the feeder level and secondly at the plot-level from the emlite meters. Check the difference between the two readings:
    total_load_1 = df[["grid_to_load", "solar_to_load", "batt_to_load"]].sum().sum()
    total_load_2 = df["load"].sum()
    notices.extend(energy_discrepancy_warnings(
        total_load_1,
        total_load_2,
        "total load at microgrid-level meters vs plot-level meters"
    ))

    # Calculate the BESS round trip efficiency
    soe_start = df.iloc[0]["soe"]
    soe_end = df.iloc[-1]["soe"]
    soe_diff = soe_end - soe_start
    roundtrip_efficiency = (total_bess_discharged_1 + soe_diff) / total_bess_charged_1

    # Validity check of total grid energy imports and exports, calculated by two different ways
    total_grid_imports_1 = df["grid_import_cum_reading"].iloc[-1] - df["grid_import_cum_reading"].iloc[0]
    total_grid_exports_1 = df["grid_export_cum_reading"].iloc[-1] - df["grid_export_cum_reading"].iloc[0]
    total_grid_imports_2 = breakdown.total_flows["grid_to_batt"] + breakdown.total_flows["grid_to_load"]
    total_grid_exports_2 = breakdown.total_flows["batt_to_grid"] + breakdown.total_flows["solar_to_grid"]
    notices.extend(energy_discrepancy_warnings(total_grid_imports_1, total_grid_imports_2, "total grid imports"))
    notices.extend(energy_discrepancy_warnings(total_grid_exports_1, total_grid_exports_2, "total grid exports"))

    # There are two ways of calculating the total solar generation so warn of discrepancies between the two:
    # 1) The first method (done in `breakdown_microgrid_flows`) simply sums the 'solar' column, which is estimated from
    #    plot-level load data and microgrid feeder data.
    # 2) The second method sums all the individual solar flows, as below. This is different because the solar_to_grid
    #    and solar_to_batt flows are calculated purely from microgrid-level meter data.
    total_solar_1 = breakdown.total_flows["solar"]
    total_solar_2 = breakdown.total_flows["solar_to_load"] + breakdown.total_flows["solar_to_grid"] + breakdown.total_flows["solar_to_batt"]
    notices.extend(energy_discrepancy_warnings(total_solar_1, total_solar_2, "total solar generation"))

    num_days = (end - start).total_seconds() / (60*60*24)

    total_cycles = breakdown.total_flows["bess_discharge"] / config.reporting.bess.energy_capacity

    mkt_fixed_cost_dfs, _ = get_rates_dfs_by_type(
        time_index=time_index,
        rates_by_category=rates.mkt_fix,
        allow_vol_rates=False,
    )
    customer_fixed_cost_dfs, customer_vol_rates_dfs = get_rates_dfs_by_type(
        time_index=time_index,
        rates_by_category=rates.customer,
        allow_vol_rates=True,
    )

    return Report(
        df=df,

        mkt_vol_rates_dfs=mkt_vol_rates_dfs,
        int_vol_rates_dfs=int_vol_rates_dfs,
        mkt_fixed_cost_dfs=mkt_fixed_cost_dfs,
        customer_fixed_cost_dfs=customer_fixed_cost_dfs,
        customer_vol_rates_dfs=customer_vol_rates_dfs,

        breakdown=breakdown,

        osam_rates=osam_rates,
        osam_df=osam_df,

        num_days=num_days,

        total_bess_discharged=total_bess_discharged_1,
        total_bess_charged=total_bess_charged_1,

        roundtrip_efficiency=roundtrip_efficiency,
        total_cycles=total_cycles,

        notices=notices
    )


def log_report_summary(config: Config, result: Report):
    """
    Prints a summary of the results to screen
    """

    notice_df = pd.DataFrame(columns=["level_number", "Description"])
    for notice in result.notices:
        notice_df.loc[len(notice_df)] = [notice.level.value, notice.detail]
    notice_df = notice_df.sort_values("level_number", ascending=False)

    notice_df["Level"] = "N/A"
    notice_df.loc[notice_df["level_number"] == 1, "Level"] = "Info"
    notice_df.loc[notice_df["level_number"] == 2, "Level"] = "*Noteworthy*"
    notice_df.loc[notice_df["level_number"] == 3, "Level"] = "*Serious*"
    # Log all the notices
    print("\nNOTICES")
    print(tabulate(
        tabular_data=notice_df[["Level", "Description"]],
        headers='keys',
        tablefmt='presto',
        floatfmt=(None, ",.0f", ",.0f", ",.2f", ",.0f", ",.2f"),
        showindex=False
    ))

    # Only force the user to acknowledge the notices if there are any level 2 or above
    num_important_warnings = len(notice_df[notice_df["level_number"] >= 2])
    if num_important_warnings > 0:
        user_input = input(
            f"WARNING: There are {num_important_warnings} important notice(s) listed above. This may make the results invalid, would you like to continue anyway?")
        if user_input.lower() not in ['yes', 'y']:
            print("Exiting")
            exit(-1)

    # Friendly names to present to the user
    flow_name_map = {
        "solar_to_grid": "solarToGrid",
        "grid_to_load": "gridToLoad",
        "solar_to_load": "solarToLoad",
        "batt_to_load": "battToLoad",
        "batt_to_grid": "battToGrid",
        "solar_to_batt": "solarToBatt",
        "grid_to_batt": "gridToBatt",
        "bess_charge": "All batt charge",
        "bess_discharge": "All batt discharge",
        "solar": "All solar",
        "load": "All load"
    }

    # Friendly names to present to the user
    flow_summary_column_name_map = {
        "volume": "Volume (kWh)",
        "int_cost": "Int. Cost (£)",
        "int_avg_rate": "Int. Avg Rate (p/kWh)",
        "mkt_cost": "Mkt. Cost (£)",
        "mkt_avg_rate": "Mkt. Avg Rate (p/kWh)",
    }

    # Rename the flows to the externally-facing names
    result.breakdown.fundamental_flows_summary_df.index = result.breakdown.fundamental_flows_summary_df.index.map(flow_name_map)
    result.breakdown.derived_flows_summary_df.index = result.breakdown.derived_flows_summary_df.index.map(flow_name_map)

    print("")
    print(tabulate(
        tabular_data=result.breakdown.fundamental_flows_summary_df.rename(columns=flow_summary_column_name_map),
        headers='keys',
        tablefmt='presto',
        floatfmt=(None, ",.0f", ",.0f", ",.2f", ",.0f", ",.2f")
    ))
    print("")
    print("* The internal prices assigned to battery flows are signed from the perspective of the battery strategy")

    print("")
    print(tabulate(
        tabular_data=result.breakdown.derived_flows_summary_df.rename(columns=flow_summary_column_name_map),
        headers='keys',
        tablefmt='presto',
        floatfmt=(None, ",.0f", ",.0f", ",.2f", ",.0f", ",.2f")
    ))

    print("")
    print(f"Solar self-use (inc batt losses): {result.breakdown.solar_self_use:,.2f} kWh, {(result.breakdown.solar_self_use / result.breakdown.total_flows['solar']) * 100:.1f}% of the solar generation.")

    # Cycling

    print("")
    print(f"Total cycles over simulation: {result.total_cycles:.2f} cycles")
    print(f"Average cycles per day: {result.total_cycles / result.num_days:.2f} cycles/day")
    print(f"Roundtrip efficiency: {result.roundtrip_efficiency*100:.1f}%")

    print("")
    print(f"Total BESS gain over period: £{result.breakdown.total_int_bess_gain / 100:,.2f}")
    print(f"Average daily BESS gain over period: £{(result.breakdown.total_int_bess_gain / 100) / result.num_days:.2f}")

    if config.reporting.bill_match:
        if config.reporting.bill_match.import_direction:
            bill_match(
                grid_energy_flow=result.df["grid_import"],
                mkt_vol_grid_rates_df=result.mkt_vol_rates_dfs["grid_to_batt"],  # use the grid rates for grid_to_batt as these include info about any OSAM rates
                mkt_fixed_costs_df=result.mkt_fixed_cost_dfs["import"],
                osam_rates=result.osam_rates,
                osam_df=result.osam_df,
                cepro_mkt_vol_bill_total_expected=result.breakdown.total_mkt_vol_costs["grid_to_batt"] + result.breakdown.total_mkt_vol_costs["grid_to_load"],
                context="import",
                line_items=config.reporting.bill_match.import_direction.line_items,
            )
        if config.reporting.bill_match.export_direction:
            bill_match(
                grid_energy_flow=result.df["grid_export"],
                mkt_vol_grid_rates_df=result.mkt_vol_rates_dfs["batt_to_grid"],  # use the grid rates for grid_to_batt as these include info about any OSAM rates
                mkt_fixed_costs_df=result.mkt_fixed_cost_dfs["export"],
                osam_rates=result.osam_rates,
                osam_df=result.osam_df,
                cepro_mkt_vol_bill_total_expected=result.breakdown.total_mkt_vol_costs["batt_to_grid"] + result.breakdown.total_mkt_vol_costs["solar_to_grid"],
                context="export",
                line_items=config.reporting.bill_match.export_direction.line_items,
            )

    if len(result.osam_rates) > 0:
        print("")
        avg_ncsp = np.average(a=result.osam_df["ncsp"], weights=result.osam_df["bp_x_stor"])
        print(f"Weighted average OSAM NCSP: {avg_ncsp:.3f}")


def save_profiles(df: pd.DataFrame, save_dir: str, start: datetime):
    """
    Saves the load and solar profiles to disk separately, so they can be used for simulations by Skypro.
    """

    # To follow the naming convention used for other data repositories we need to extract the last part of the path
    # as that will go into the actual file name for each profile.
    profiles_save_dir = os.path.normpath(save_dir)  # remove any trailing slashes
    data_source = os.path.basename(profiles_save_dir)  # extract the last part of the path
    profiles_save_dir = profiles_save_dir.removesuffix(data_source)

    profiles_df = df[["solar", "load"]]
    profiles_df = profiles_df.resample("30min").sum()
    profiles_df = profiles_df.reset_index(names="ClockTime")
    profiles_df["UTCTime"] = profiles_df["ClockTime"].dt.tz_convert("UTC")

    logging.info(f"Saving load and solar profiles to {profiles_save_dir}...")
    solar_file_path = prepare_data_dir(profiles_save_dir, data_source, "solar", start)
    load_file_path = prepare_data_dir(profiles_save_dir, data_source, "load", start)
    profiles_df[["UTCTime", "ClockTime", "solar"]].rename(columns={"solar": "energy"}).to_csv(solar_file_path, index=False)
    profiles_df[["UTCTime", "ClockTime", "load"]].rename(columns={"load": "energy"}).to_csv(load_file_path, index=False)
