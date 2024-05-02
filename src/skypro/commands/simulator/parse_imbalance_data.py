from datetime import timedelta

import numpy as np
import pandas as pd

from skypro.cli_utils.cli_utils import get_user_ack_of_warning_or_exit, read_directory_of_csvs


def split_modo_data_by_settlement_period(df, col: str, col_10m: str, col_20m: str, col_final: str):
    """
    Modo imbalance data (for price and volume) has many rows for each settlement period (SP), because many
    predictions are made for during the course of each SP. This function organises the data so that there is a single
    row for each SP, with columns for the predictions at 10 minutes and 20 minutes into the SP, as well as the final
    prediction.
    :param df: dataframe of modo price or volume imbalance data
    :param col: the col of interest in `df`, e.g. 'price' or 'volume'
    :param col_10m: the col name to use for the 10minute prediction
    :param col_20m: the col name to use for the 20minute prediction
    :param col_final: the col name to use for the final prediction
    :return: dataframe organised by settlement period.
    """

    by_sp = pd.DataFrame(index=df["spUTCTime"].sort_values(ascending=True).unique())

    # Run through the imbalance prices/volumes and extract the '10m price' and the 'final price'
    for grouping_tuple, sub_df in df.groupby(["spUTCTime"]):
        sp = grouping_tuple[0]  # SP is short for settlement period
        sub_df = sub_df.sort_values(by="predictionUTCTime", ascending=True)
        final_val = sub_df.iloc[-1][col]  # The last imbalance price/volume that Modo published for this SP
        try:
            val_10m = sub_df[  # The calculation that Modo published ~10m into the SP
                sub_df["predictionUTCTime"] < sp + timedelta(minutes=10, seconds=10)
                # Give another 10 secs as the data pull is run on a minute-aligned cron job
                ].iloc[-1][col]
        except IndexError:
            val_10m = np.NaN  # Occasionally Modo doesn't publish a price/volume within 10m
        try:
            val_20m = sub_df[
                sub_df["predictionUTCTime"] < sp + timedelta(minutes=20, seconds=10)
                # Give another 10 secs as the data pull is run on a minute-aligned cron job
                ].iloc[-1][col]
        except IndexError:
            val_20m = np.NaN  # Occasionally Modo doesn't publish a price/volume within 10m

        by_sp.loc[sp, col_10m] = val_10m
        by_sp.loc[sp, col_20m] = val_20m
        by_sp.loc[sp, col_final] = final_val

    return by_sp


def normalise_predictive_imbalance_data(price_df: pd.DataFrame, volume_df: pd.DataFrame) -> pd.DataFrame:
    """
    Puts the predictive/Modo imbalance price and volume data into a 'normalised' format, returning a dataframe with a
    row per settlement period and columns for 10m, 20m, and final values.
    """

    # Drop columns we aren't interested in
    price_df = price_df[["spUTCTime", "predictionUTCTime", "price"]]
    volume_df = volume_df[["spUTCTime", "predictionUTCTime", "volume"]]

    price_df = split_modo_data_by_settlement_period(
        df=price_df,
        col="price",
        col_10m="imbalance_price_10m",
        col_20m="imbalance_price_20m",
        col_final="imbalance_price_final",
    )

    volume_df = split_modo_data_by_settlement_period(
        df=volume_df,
        col="volume",
        col_10m="imbalance_volume_10m",
        col_20m="imbalance_volume_20m",
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

    df["imbalance_price_10m"] = df["price"] / 10  # convert £/MW to p/kW
    df["imbalance_price_20m"] = df["price"] / 10  # convert £/MW to p/kW
    df["imbalance_price_final"] = df["price"] / 10  # convert £/MW to p/kW
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


def read_imbalance_data(price_dir: str, volume_dir: str) -> pd.DataFrame:

    # TODO: only read in the months that are required
    price_df = read_directory_of_csvs(price_dir)
    volume_df = read_directory_of_csvs(volume_dir)

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

    if is_predictive:
        df = normalise_predictive_imbalance_data(price_df, volume_df)
    else:
        df = normalise_non_predictive_imbalance_data(price_df, volume_df)

    df = df.sort_index()

    return df
