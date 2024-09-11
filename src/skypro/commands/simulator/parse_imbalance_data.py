import logging
from datetime import timedelta

import numpy as np
import pandas as pd
from simt_common.dataparse.dataparse import read_directory_of_csvs
from simt_common.timeutils.hh_math import floor_hh

from skypro.cli_utils.cli_utils import get_user_ack_of_warning_or_exit


def read_imbalance_data(time_index: pd.DatetimeIndex, price_dir: str, volume_dir: str) -> pd.DataFrame:
    """
    Reads the imbalance price and volume CSV directories and returns a timeseries dataframe with the following columns:
    - `imbalance_price_predicted`
    - `imbalance_price_final`
    - `imbalance_volume_predicted`
    - `imbalance_volume_final`

    The predicted values either relate directly to the predictions present in the underlying data, or if underlying data
    is 'non-predictive' then it is assumed that a perfect prediction is made 20 minutes into each settlement period.
    """
    logging.info("Reading imbalance files...")

    price_df = read_directory_of_csvs(price_dir)
    volume_df = read_directory_of_csvs(volume_dir)

    logging.info("Processing imbalance files...")

    price_is_predictive = "predictionUTCTime" in price_df.columns
    volume_is_predictive = "predictionUTCTime" in volume_df.columns
    if price_is_predictive != volume_is_predictive:
        raise ValueError("Price and volume data must either both be predictive, or both be non-predictive.")
    is_predictive = price_is_predictive

    # Parse the date columns
    price_df["spUTCTime"] = pd.to_datetime(price_df["spUTCTime"], format="ISO8601")
    volume_df["spUTCTime"] = pd.to_datetime(volume_df["spUTCTime"], format="ISO8601")
    if is_predictive:
        price_df["predictionUTCTime"] = pd.to_datetime(price_df["predictionUTCTime"], format="ISO8601")
        volume_df["predictionUTCTime"] = pd.to_datetime(volume_df["predictionUTCTime"], format="ISO8601")

    if time_index[0] < min(price_df["spUTCTime"]) or time_index[0] < min(volume_df["spUTCTime"]):
        raise ValueError("Simulation start time is outside of imbalance data range. Do you need to download more imbalance data?")
    if time_index[-1] > max(price_df["spUTCTime"]) or time_index[-1] > max(volume_df["spUTCTime"]):
        raise ValueError("Simulation end time is outside of imbalance volume data range. Do you need to download more imbalance data?")

    # Remove data out of the time range of interest
    # TODO: only read in CSVs for the months that are required in the first place
    price_df = price_df[(price_df["spUTCTime"] >= time_index[0]) & (price_df["spUTCTime"] <= time_index[-1])]
    volume_df = volume_df[(volume_df["spUTCTime"] >= time_index[0]) & (volume_df["spUTCTime"] <= time_index[-1])]

    if is_predictive:
        df = normalise_predictive_imbalance_data(time_index, price_df, volume_df)
    else:
        df = normalise_non_predictive_imbalance_data(time_index, price_df, volume_df)

    df = df.sort_index()

    breakpoint()

    return df


def normalise_predictive_imbalance_data(
        time_index: pd.DatetimeIndex,
        price_df: pd.DataFrame,
        volume_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Puts the predictive/Modo imbalance price and volume data into a 'normalised' format, returning a dataframe with rows
    aligning with time_index
    """

    # Drop columns we aren't interested in
    price_df = price_df[["spUTCTime", "predictionUTCTime", "price"]]
    volume_df = volume_df[["spUTCTime", "predictionUTCTime", "volume"]]

    price_df = split_modo_data_by_settlement_period(
        modo_df=price_df,
        time_index=time_index,
        col="price",
        col_prediction="imbalance_price_predicted",
        col_final="imbalance_price_final",
    )

    volume_df = split_modo_data_by_settlement_period(
        modo_df=volume_df,
        time_index=time_index,
        col="volume",
        col_prediction="imbalance_volume_predicted",
        col_final="imbalance_volume_final",
    )
    df = pd.merge(
        left=price_df,
        right=volume_df,
        left_index=True,
        right_index=True
    )

    return df


def normalise_non_predictive_imbalance_data(time_index: pd.DatetimeIndex, price_df: pd.DataFrame, volume_df: pd.DataFrame) -> pd.DataFrame:
    """
    Puts the non-predictive/Elexon imbalance price and volume data into a 'normalised' format, returning a dataframe
    with a row per time step, period and columns for final values and predicted values (copied from final values).
    """

    # Drop columns we aren't interested in
    price_df = price_df[["spUTCTime", "price"]]
    volume_df = volume_df[["spUTCTime", "volume"]]

    # Elexon only has a single final SSP price, this means we are feeding in a 'perfect price forecast' so the results
    # should be taken with a pinch of salt.
    get_user_ack_of_warning_or_exit("When using Elexon imbalance data, perfect hindsight is used for imbalance price "
                                    "and volume predictions. Modo data may not be as reliable or accurate so real-world"
                                    " profitability may be less")

    price_df = price_df.set_index("spUTCTime")
    price_df.index.name = None
    volume_df = volume_df.set_index("spUTCTime")
    volume_df.index.name = None

    df = pd.DataFrame(index=time_index)

    steps_per_sp = int(timedelta(minutes=30) / pd.to_timedelta(time_index.freq))
    df["imbalance_price_final"] = price_df["price"]
    df["imbalance_volume_final"] = volume_df["volume"]
    df["imbalance_price_final"] = df["imbalance_price_final"].ffill(limit=(steps_per_sp-1))
    df["imbalance_volume_final"] = df["imbalance_volume_final"].ffill(limit=(steps_per_sp-1))

    df["imbalance_price_predicted"] = np.nan
    df["sp"] = df.index.to_series().apply(lambda t: floor_hh(t))
    df["time_into_sp"] = df.index.to_series() - df["sp"]

    # TODO: predictions are currently hard-coded to become available 20 minutes into the SP - this should be from config
    predictions_available = df["time_into_sp"] >= timedelta(minutes=20)
    df.loc[predictions_available, "imbalance_price_predicted"] = df["imbalance_price_final"]
    df.loc[predictions_available, "imbalance_volume_predicted"] = df["imbalance_volume_final"]

    return df[[
        "imbalance_price_predicted",
        "imbalance_price_final",
        "imbalance_volume_predicted",
        "imbalance_volume_final",
    ]]


def split_modo_data_by_settlement_period(
        modo_df: pd.DataFrame,
        time_index: pd.DatetimeIndex,
        col: str,
        col_prediction: str,
        col_final: str
) -> pd.DataFrame:
    """
    Modo imbalance data (for price and volume) has many rows for each settlement period (SP), because many
    predictions are made for during the course of each SP. This function organises the data so that there is a row
    for each time_index, with columns for the predicted value at that time, and the final value (which is only known
    after the SP has ended).
    :param modo_df: dataframe of modo price or volume imbalance data
    :param time_index: the timestamps to index the return dataframe to
    :param col: the col of interest in `df`, e.g. 'price' or 'volume'
    :param col_prediction: the col name to use for the prediction values
    :param col_final: the col name to use for the final values
    :return: dataframe organised by time_index.
    """

    df = pd.DataFrame(index=time_index)

    # Run through the imbalance prices/volumes and extract the 'predictive price' and the 'final price'
    for grouping_tuple, sub_df in modo_df.groupby(["spUTCTime"]):
        sp_start = grouping_tuple[0]  # SP is short for settlement period
        sp_end = sp_start + timedelta(minutes=30)
        sub_df = sub_df.sort_values(by="predictionUTCTime", ascending=True)
        final_val = sub_df.iloc[-1][col]  # The last imbalance price/volume that Modo published for this SP

        # The calculation that Modo published ~10m into the SP
        times_of_interest = time_index[(time_index >= sp_start) & (time_index < sp_end)]
        for time in times_of_interest:
            # Give another 10 secs as the data pull is run on a minute-aligned cron job
            prediction_cutoff_time = time + timedelta(seconds=10)
            try:
                predictive_val = sub_df[sub_df["predictionUTCTime"] <= prediction_cutoff_time].iloc[-1][col]
            except IndexError:
                predictive_val = np.nan

            df.loc[time, col_prediction] = predictive_val
            df.loc[time, col_final] = final_val

    return df
