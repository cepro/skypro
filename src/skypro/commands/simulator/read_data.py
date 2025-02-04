import logging
from datetime import timedelta, datetime
from typing import Dict

import numpy as np
import pandas as pd
import sqlalchemy
from simt_common.cli_utils.cliutils import read_directory_of_csvs
from simt_common.timeutils.math import floor_hh

from skypro.cli_utils.cli_utils import get_user_ack_of_warning_or_exit
from skypro.commands.simulator.config.config_common import DataSource
from skypro.commands.simulator.config.path_field import resolve_file_path


def read_data_source(source: DataSource, start: datetime, end: datetime, env_vars: Dict, flows_db_url: str) \
        -> pd.DataFrame:
    """
    Reads data from the given source for the given time range and returns as a dataframe.
    """

    logging.info(f"Reading data source '{source.source_str}'...")

    components = source.source_str.split(":")
    source_type = components[0]
    source_locator = components[1]

    if source_type == "csv-dir":
        return read_csv_dir(
            csv_dir=resolve_file_path(source_locator, env_vars),
            is_predictive=source.is_predictive,
            start=start,
            end=end
        )

    if source_type == "flowsdb-market-data":
        return read_flows_db_market_data(
            market_data_type_name=source_locator,
            is_predictive=source.is_predictive,
            start=start,
            end=end,
            flows_db_url=flows_db_url
        )

    raise ValueError(f"Unknown data source type: '{source_type}'")


def read_flows_db_market_data(
        market_data_type_name: str,
        is_predictive: bool,
        start: datetime,
        end: datetime,
        flows_db_url: str
) -> pd.DataFrame:
    """
    """

    # The system is quite flexible to allow interpretation of predictive datasets as non-predictive, and vice-versa, but
    # here there is a sanity check to make sure the user isn't doing something quite wrong:
    if is_predictive and market_data_type_name in ["elexon-imbalance-price", "elexon-imbalance-volume"]:
        raise ValueError(f"The {market_data_type_name} source was configured as predictive, this probably wont work")
    if not is_predictive and market_data_type_name in ["modo-imbalance-price-forecast", "modo-imbalance-volume-forecast"]:
        raise ValueError(f"The {market_data_type_name} source was configured as non-predictive, this probably isn't what you want")

    if is_predictive:
        # If the data is 'predictive' then we need to pull not just the latest values, but all the updates that happened
        # along the way.
        query = (
            "SELECT time, created_at, value FROM flows.market_data "
            "JOIN flows.market_data_types on market_data.type = market_data_types.id "
            f"WHERE "
            f"  time >= '{start.isoformat()}' AND "
            f"  time <= '{end.isoformat()}' AND "
            f"  market_data_types.name = '{market_data_type_name}' "
            f"order by time asc, created_at asc"
        )
    else:
        # If the data is not 'predictive' then we extract just the latest values for each settlement period via the
        # SELECT DISTINCT ON clause.
        query = (
            " WITH data AS ( "
            "   SELECT time, created_at, value FROM flows.market_data "
            "   JOIN flows.market_data_types on market_data.type = market_data_types.id "
            f"  WHERE "
            f"    time >= '{start.isoformat()}' AND "
            f"    time <= '{end.isoformat()}' AND "
            f"    market_data_types.name = '{market_data_type_name}' "
            " )"
            " SELECT DISTINCT ON (time) time, value FROM data "
            " ORDER BY time asc, created_at desc"
        )

    engine = sqlalchemy.create_engine(flows_db_url)
    df = pd.read_sql(query, con=engine)

    if len(df) == 0:
        raise ValueError(f"No rows were returned from flows DB for market data '{market_data_type_name}'")

    if is_predictive:
        df = df.rename(columns={"created_at": "predicted_at"})

    return df


def read_csv_dir(
        csv_dir: str,
        is_predictive: bool,
        start: datetime,
        end: datetime
) -> pd.DataFrame:
    df = read_directory_of_csvs(csv_dir)

    # Parse the settlement period time column
    df["spUTCTime"] = pd.to_datetime(df["spUTCTime"], format="ISO8601")

    predictions_available = "predictionUTCTime" in df.columns
    if is_predictive != predictions_available:
        raise ValueError("The data source does not match the isPredictive flag")
    if is_predictive:
        df["predictionUTCTime"] = pd.to_datetime(df["predictionUTCTime"], format="ISO8601")

    # Use better column names to be consistent with the flows db data source
    df = df.rename(columns={
        "spUTCTime": "time",
        "predictionUTCTime": "predicted_at",
        "volume": "value",
        "price": "value"
    })

    end_floor_hh = floor_hh(end)

    if start < min(df["time"]):
        raise ValueError("Simulation start time is outside of data range. Do you need to download more data?")
    if end_floor_hh > max(df["time"]):
        raise ValueError("Simulation end time is outside of data range. Do you need to download more data?")

    # Remove data out of the time range of interest
    # TODO: only read in CSVs for the months that are required in the first place
    df = df[(df["time"] >= start) & (df["time"] <= end)]

    return df


def normalise_final_imbalance_data(
        time_index: pd.DatetimeIndex,
        price_df: pd.DataFrame,
        volume_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Puts the imbalance data into the 'final' format, where there is a value for each row/timeslot
    """

    if "predicted_at" in price_df.columns or "predicted_at" in volume_df.columns:
        # We used to just take the last prediction from Modo and use it as the final pricing, but that's deprecated as
        # we now have support for specifying the final pricing separately.
        raise ValueError("Cannot use modo data for final rates pricing")

    # Drop columns we aren't interested in
    price_df = price_df[["time", "value"]]
    volume_df = volume_df[["time", "value"]]

    price_df = price_df.set_index("time")
    price_df.index.name = None
    volume_df = volume_df.set_index("time")
    volume_df.index.name = None

    df = pd.DataFrame(index=time_index)

    steps_per_sp = int(timedelta(minutes=30) / pd.to_timedelta(time_index.freq))
    df["imbalance_price_final"] = price_df["value"]
    df["imbalance_volume_final"] = volume_df["value"]
    df["imbalance_price_final"] = df["imbalance_price_final"].ffill(limit=(steps_per_sp-1))
    df["imbalance_volume_final"] = df["imbalance_volume_final"].ffill(limit=(steps_per_sp-1))

    return df


def normalise_live_imbalance_data(
        time_index: pd.DatetimeIndex,
        price_df: pd.DataFrame,
        volume_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Puts the imbalance data into the 'live' format, where the values are discovered as the settlement period progresses,
    and so there are NaNs at the beginning of the settlement period where predictions may not be available.
    """
    price_has_predictions = "predicted_at" in price_df.columns
    volume_has_predictions = "predicted_at" in volume_df.columns
    if price_has_predictions != volume_has_predictions:
        raise ValueError("Price and volume data must either both be predictive, or both be non-predictive.")
    has_predictions = price_has_predictions

    if has_predictions:
        # If the data has predictions then it's Modo data
        price_df = normalise_modo_data_for_live(
            modo_df=price_df,
            time_index=time_index,
            col="value",
            col_prediction="imbalance_price_live",
        )
        volume_df = normalise_modo_data_for_live(
            modo_df=volume_df,
            time_index=time_index,
            col="value",
            col_prediction="imbalance_volume_live",
        )
        df = pd.merge(
            left=price_df,
            right=volume_df,
            left_index=True,
            right_index=True
        )
    else:
        # If the data doesn't have predictions then it's Elexon data
        df = normalise_elexon_data_for_live(
            time_index=time_index,
            price_df=price_df,
            volume_df=volume_df,
        )

    return df


def normalise_elexon_data_for_live(
        time_index: pd.DatetimeIndex,
        price_df: pd.DataFrame,
        volume_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Puts the non-predictive/Elexon imbalance price and volume data into a normalised 'live' format, returning a df
    with a row per time step and columns for live values. We can use Elexon data for 'live' data by unveiling the
    imbalance value 20 minutes into the settlement period. This means we are feeding in a 'perfect price forecast' so the
    results should be taken with a pinch of salt.
    """

    # The final data format is used as an intermediate step for this function
    final_df = normalise_final_imbalance_data(time_index, price_df, volume_df)

    # Drop columns we aren't interested in
    price_df = price_df[["time", "value"]]
    volume_df = volume_df[["time", "value"]]

    # Elexon only has a single final SSP price, this means we are feeding in a 'perfect price forecast' so the results
    # should be taken with a pinch of salt.
    get_user_ack_of_warning_or_exit("When using Elexon imbalance data, perfect hindsight is used for imbalance price "
                                    "and volume predictions. Modo data may not be as reliable or accurate so real-world"
                                    " profitability may be less")

    price_df = price_df.set_index("time")
    price_df.index.name = None
    volume_df = volume_df.set_index("time")
    volume_df.index.name = None

    df = pd.DataFrame(index=time_index)

    df["imbalance_price_live"] = np.nan
    df["sp"] = df.index.to_series().apply(lambda t: floor_hh(t))
    df["time_into_sp"] = df.index.to_series() - df["sp"]

    # Predictions are currently hard-coded to become available 20 minutes into the SP - this duration could come from
    # config instead.
    predictions_available = df["time_into_sp"] >= timedelta(minutes=20)
    df.loc[predictions_available, "imbalance_price_live"] = final_df["imbalance_price_final"]
    df.loc[predictions_available, "imbalance_volume_live"] = final_df["imbalance_volume_final"]

    return df[[
        "imbalance_price_live",
        "imbalance_volume_live",
    ]]


def normalise_modo_data_for_live(
        modo_df: pd.DataFrame,
        time_index: pd.DatetimeIndex,
        col: str,
        col_prediction: str,
) -> pd.DataFrame:
    """
    Modo imbalance data (for price and volume) has many rows for each settlement period (SP), because many
    predictions are made during the course of each SP. This function organises the data so that there is a row
    for each time_index, with columns for the predicted/live value at that time, and the final value (which is only
    known after the SP has ended).
    :param modo_df: dataframe of modo price or volume imbalance data
    :param time_index: the timestamps to index the return dataframe to
    :param col: the col of interest in `df`, e.g. 'price' or 'volume'
    :param col_prediction: the col name to use for the prediction values
    :return: dataframe organised by time_index.
    """

    df = pd.DataFrame(index=time_index)

    # Run through the imbalance prices/volumes and extract the 'predictive price' and the 'final price'
    for grouping_tuple, sub_df in modo_df.groupby(["time"]):
        sp_start = grouping_tuple[0]  # 'SP' is short for settlement period
        sp_end = sp_start + timedelta(minutes=30)
        sub_df = sub_df.sort_values(by="predicted_at", ascending=True)

        # The calculation that Modo published ~10m into the SP
        times_of_interest = time_index[(time_index >= sp_start) & (time_index < sp_end)]
        for time in times_of_interest:
            # Give another 10 secs as the data pull is run on a minute-aligned cron job
            prediction_cutoff_time = time + timedelta(seconds=10)
            try:
                predictive_val = sub_df[sub_df["predicted_at"] <= prediction_cutoff_time].iloc[-1][col]
            except IndexError:
                predictive_val = np.nan

            df.loc[time, col_prediction] = predictive_val

    return df
