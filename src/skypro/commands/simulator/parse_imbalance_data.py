import logging
from datetime import timedelta, datetime

import numpy as np
import pandas as pd
from simt_common.dataparse.dataparse import read_directory_of_csvs

from skypro.cli_utils.cli_utils import get_user_ack_of_warning_or_exit


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
                predictive_val = np.NaN

            df.loc[time, col_prediction] = predictive_val
            df.loc[time, col_final] = final_val

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


def normalise_non_predictive_imbalance_data(price_df: pd.DataFrame, volume_df: pd.DataFrame) -> pd.DataFrame:
    """
    Puts the non-predictive/Elexon imbalance price and volume data into a 'normalised' format, returning a dataframe
    with a row per settlement period and columns for 10m, 20m, and final values.
    """

    # TODO: UPDATE THIS TO USE time_index
    # Elexon only has a single final SSP price that we need to use for 10m and 20m. This means we are feeding in
    # a 'perfect price forecast' so the results should be taken with a pinch of salt.
    get_user_ack_of_warning_or_exit("When using Elexon imbalance data, perfect hindsight is used for imbalance price "
                                    "and volume predictions. Expected profitability should be reduced by approx 12.5%")

    price_df = price_df.set_index("spUTCTime")
    price_df.index.name = None
    volume_df = volume_df.set_index("spUTCTime")
    volume_df.index.name = None

    df = pd.merge(
        left=price_df,
        right=volume_df,
        left_index=True,
        right_index=True,
    )

    df["imbalance_price_10m"] = df["price"]
    df["imbalance_price_20m"] = df["price"]
    df["imbalance_price_final"] = df["price"]
    df["imbalance_volume_10m"] = df["volume"]
    df["imbalance_volume_20m"] = df["volume"]
    df["imbalance_volume_final"] = df["volume"]

    return df[[
        "imbalance_price_10m",
        "imbalance_price_20m",
        "imbalance_price_final",
        "imbalance_volume_10m",
        "imbalance_volume_20m",
        "imbalance_volume_final"
    ]]


def read_imbalance_data(time_index: pd.DatetimeIndex, price_dir: str, volume_dir: str) -> pd.DataFrame:

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

    # Remove data out of the time range of interest
    # TODO: only read in CSVs for the months that are required in the first place
    price_df = price_df[(price_df["spUTCTime"] >= time_index[0]) & (price_df["spUTCTime"] <= time_index[-1])]
    volume_df = volume_df[(volume_df["spUTCTime"] >= time_index[0]) & (volume_df["spUTCTime"] <= time_index[-1])]

    if is_predictive:
        df = normalise_predictive_imbalance_data(time_index, price_df, volume_df)
    else:
        df = normalise_non_predictive_imbalance_data(time_index, price_df, volume_df)

    df = df.sort_index()

    return df
