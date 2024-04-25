import logging
from datetime import timedelta
from os import listdir
from os.path import isfile, join
from typing import List, Optional

import numpy as np
import pandas as pd
import pytz

from simt_common.jsonconfig.rates import parse_rates, parse_supply_points, collate_import_and_export_rate_configurations
from skypro.commands.simulator.algorithm import run_imbalance_algorithm
from skypro.commands.simulator.config import parse_config
from skypro.commands.simulator.profiler import Profiler
from skypro.commands.simulator.results import explore_results


def simulate(config_file_path: str, output_file_path: Optional[str] = None):

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

    # Imbalance pricing can come from either Modo or Elexon files, Modo takes priority if both are given
    imbalance_source = config.simulation.imbalance_data_source
    if imbalance_source.modo is not None:
        by_sp = read_modo_imbalance_data(
            price_csv=imbalance_source.modo.price_csv,
            volume_csv=imbalance_source.modo.volume_csv
        )
    elif imbalance_source.elexon is not None:
        directory = imbalance_source.elexon.csv_directory
        files = []
        for f in listdir(directory):
            path = join(directory, f)
            if isfile(path):
                files.append(path)
        by_sp = read_elexon_imbalance_data(files)
    else:
        raise ValueError("Unknown imbalance data source")

    if config.simulation.start < by_sp.index[0]:
        raise ValueError(f"Simulation start time is outside of data range")
    if config.simulation.end >= by_sp.index[-1]:
        raise ValueError(f"Simulation end time is outside of data range")

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

    logging.info("Generating solar data...")
    solar_config = config.simulation.site.solar
    if solar_config.profile is not None:
        solarProfiler = Profiler(
            profile_csv_dir=solar_config.profile.profile_dir,
            scaling_factor=(solar_config.profile.scaled_size_kwp / solar_config.profile.profiled_size_kwp),
            value_col_name="Plot-08",
        )
        by_sp["solar"] = solarProfiler.get_for(by_sp.index.to_series()) * 2  # Multiply to convert kWh over HH to kW
    elif not np.isnan(solar_config.constant):
        by_sp["solar"] = solar_config.constant
    else:
        raise ValueError("Solar configuration must be either 'profile' or 'constant'")

    # Create domestic load data
    logging.info("Generating domestic load data...")
    load_config = config.simulation.site.load
    if load_config.profile is not None:
        loadProfiler = Profiler(
            profile_csv=load_config.profile.profile_csv_file,
            scaling_factor=(load_config.profile.scaled_num_plots / load_config.profile.profiled_num_plots),
            value_col_name="total"
        )
        by_sp["load"] = loadProfiler.get_for(by_sp.index.to_series()) * 2  # Multiply to convert kWh over HH to kW
    elif not np.isnan(load_config.constant):
        by_sp["load"] = load_config.constant
    else:
        raise ValueError("Load configuration must be either 'profile' or 'constant'")

    df = run_imbalance_algorithm(
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
        niv_chase_periods=config.simulation.strategy.niv_chase_periods,
        full_discharge_when_export_rate_applies=config.simulation.strategy.do_full_discharge_when_export_rate_applies,
    )

    if output_file_path:
        logging.info("Saving output file")
        df.to_csv(output_file_path)

    explore_results(
        df=df,
        battery_energy_capacity=config.simulation.site.bess.energy_capacity,
        battery_nameplate_power=config.simulation.site.bess.energy_capacity,
        site_import_limit=config.simulation.site.grid_connection.import_limit,
        site_export_limit=config.simulation.site.grid_connection.export_limit,
        import_rates=import_rates_final,
        export_rates=export_rates_final,
    )


def read_modo_imbalance_data(price_csv: str, volume_csv: str) -> pd.DataFrame:
    """
    Reads in the CSV files of Modo data and returns a dataframe with a row per settlement period.
    """

    imbalance_price = pd.read_csv(price_csv)
    imbalance_volume = pd.read_csv(volume_csv)

    def clean_modo_imbalance_df(df: pd.DataFrame):
        london_tz = pytz.timezone('Europe/London')
        df["datetime"] = (pd.to_datetime(df["date"]).dt.tz_localize(london_tz).dt.tz_convert("UTC") +
                          pd.to_timedelta((df["settlement_period"] - 1) * 30, unit='minutes'))
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)

    clean_modo_imbalance_df(imbalance_price)
    clean_modo_imbalance_df(imbalance_volume)

    all_sps = pd.concat([
        imbalance_price["datetime"],
        imbalance_volume["datetime"]
    ]).sort_values(ascending=True).unique()

    by_sp = pd.DataFrame(index=all_sps)

    # Run through the imbalance prices and extract the '10m price' and the 'final price'
    for grouping_tuple, sub_df in imbalance_price[["created_at", "datetime", "imbalance_price"]].groupby(["datetime"]):
        sp = grouping_tuple[0]  # SP is short for settlement period
        final_price = sub_df.iloc[-1]["imbalance_price"]  # The last imbalance price that Modo published for this SP
        by_sp.loc[sp, "imbalance_price_final"] = final_price
        try:
            price_forecast_10m = sub_df[  # The imbalance price calculation that Modo published ~10m into the SP
                sub_df["created_at"] < sp + timedelta(minutes=10, seconds=10)
                # Give another 10 secs as the data pull is run on a minute-aligned cron job
                ].iloc[-1]["imbalance_price"]
        except IndexError:
            price_forecast_10m = np.NaN  # Occasionally Modo doesn't publish a price within 10m
        try:
            price_forecast_20m = sub_df[
                sub_df["created_at"] < sp + timedelta(minutes=20, seconds=10)
                # Give another 10 secs as the data pull is run on a minute-aligned cron job
                ].iloc[-1]["imbalance_price"]
        except IndexError:
            price_forecast_20m = np.NaN # Occasionally Modo doesn't publish a price within 10m

        by_sp.loc[sp, "imbalance_price_10m"] = price_forecast_10m
        by_sp.loc[sp, "imbalance_price_20m"] = price_forecast_20m

    # Do the same thing for imbalance *volumes*
    for grouping_tuple, sub_df in imbalance_volume[["created_at", "datetime", "imbalance_volume"]].groupby(
            ["datetime"]):
        sp = grouping_tuple[0]
        final_volume = sub_df.iloc[-1]["imbalance_volume"]
        by_sp.loc[sp, "imbalance_volume_final"] = final_volume
        try:
            volume_forecast_10m = sub_df[
                sub_df["created_at"] < sp + timedelta(minutes=10, seconds=10)
                ].iloc[-1]["imbalance_volume"]
        except IndexError:
            volume_forecast_10m = np.NaN
        try:
            volume_forecast_20m = sub_df[
                sub_df["created_at"] < sp + timedelta(minutes=20, seconds=10)
                ].iloc[-1]["imbalance_volume"]
        except IndexError:
            volume_forecast_20m = np.NaN
        by_sp.loc[sp, "imbalance_volume_10m"] = volume_forecast_10m
        by_sp.loc[sp, "imbalance_volume_20m"] = volume_forecast_20m

    by_sp["imbalance_price_10m"] = by_sp["imbalance_price_10m"] / 10  # convert £/MW to p/kW
    by_sp["imbalance_price_20m"] = by_sp["imbalance_price_20m"] / 10  # convert £/MW to p/kW
    by_sp["imbalance_price_final"] = by_sp["imbalance_price_final"] / 10  # convert £/MW to p/kW

    return by_sp


def read_elexon_imbalance_data(csv_files: List[str]) -> pd.DataFrame:
    """
    Reads the given list of CSV files of Elexon-published imbalance and creates a single
    dataframe with a row per settlement period
    """
    imbalance_df = pd.DataFrame()
    for csv_file in csv_files:
        file_df = pd.read_csv(csv_file)
        imbalance_df = pd.concat([imbalance_df, file_df], ignore_index=True)

    imbalance_df["datetime"] = pd.to_datetime(imbalance_df["datetime"])

    imbalance_df = imbalance_df.sort_values(by="datetime",ascending=True)

    # Elexon only has a single final SSP price that we need to use for 10m and 20m. This means we are feeding in
    # a 'perfect price forecast' so the results should be taken with a pinch of salt.
    logging.warning("When using Elexon imbalance data, perfect hindsight is used for price and volume predictions."
                    "Expected profitability should be reduced by approx 12.5%")

    by_sp = pd.DataFrame(index=imbalance_df["datetime"])
    imbalance_df = imbalance_df.set_index("datetime")
    by_sp["imbalance_price_10m"] = imbalance_df["Price"] / 10  # convert £/MW to p/kW
    by_sp["imbalance_price_20m"] = imbalance_df["Price"] / 10  # convert £/MW to p/kW
    by_sp["imbalance_price_final"] = imbalance_df["Price"] / 10  # convert £/MW to p/kW
    by_sp["imbalance_volume_10m"] = imbalance_df["Volume"]
    by_sp["imbalance_volume_20m"] = imbalance_df["Volume"]
    by_sp["imbalance_volume_final"] = imbalance_df["Volume"]

    return by_sp


if __name__ == "__main__":
    Fire(main)
