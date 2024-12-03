import logging
from datetime import timedelta, datetime

import numpy as np
import pandas as pd
from simt_common.cli_utils.cliutils import read_directory_of_csvs
from simt_common.timeutils.math import floor_hh

from skypro.cli_utils.cli_utils import get_user_ack_of_warning_or_exit


def read_imbalance_data(
        start: datetime,
        end: datetime,
        price_dir: str,
        volume_dir: str
) -> (pd.DataFrame, pd.DataFrame):
    """
    Reads the imbalance price and volume data from the given directories
    """
    logging.info("Reading imbalance files...")

    price_df = read_directory_of_csvs(price_dir)
    volume_df = read_directory_of_csvs(volume_dir)

    logging.info("Processing imbalance files...")

    # Parse the date columns
    price_df["spUTCTime"] = pd.to_datetime(price_df["spUTCTime"], format="ISO8601")
    volume_df["spUTCTime"] = pd.to_datetime(volume_df["spUTCTime"], format="ISO8601")

    if "predictionUTCTime" in price_df.columns:
        price_df["predictionUTCTime"] = pd.to_datetime(price_df["predictionUTCTime"], format="ISO8601")
    if "predictionUTCTime" in volume_df.columns:
        volume_df["predictionUTCTime"] = pd.to_datetime(volume_df["predictionUTCTime"], format="ISO8601")

    end_floor_hh = floor_hh(end)

    if start < min(price_df["spUTCTime"]) or start < min(volume_df["spUTCTime"]):
        raise ValueError("Simulation start time is outside of imbalance data range. Do you need to download more imbalance data?")
    if end_floor_hh > max(price_df["spUTCTime"]) or end_floor_hh > max(volume_df["spUTCTime"]):
        raise ValueError("Simulation end time is outside of imbalance volume data range. Do you need to download more imbalance data?")

    # Remove data out of the time range of interest
    # TODO: only read in CSVs for the months that are required in the first place
    price_df = price_df[(price_df["spUTCTime"] >= start) & (price_df["spUTCTime"] <= end)]
    volume_df = volume_df[(volume_df["spUTCTime"] >= start) & (volume_df["spUTCTime"] <= end)]

    return price_df.sort_index(), volume_df.sort_index()


def normalise_final_imbalance_data(
        time_index: pd.DatetimeIndex,
        price_df: pd.DataFrame,
        volume_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Puts the imbalance data into the 'final' format, where there is a value for each row/timeslot
    """

    if "predictionUTCTime" in price_df.columns or "predictionUTCTime" in volume_df.columns:
        # We used to just take the last prediction from Modo and use it as the final pricing, but that's deprecated as
        # we now have support for specifying the final pricing separately.
        raise ValueError("Cannot use modo data for final rates pricing")

    # Drop columns we aren't interested in
    price_df = price_df[["spUTCTime", "price"]]
    volume_df = volume_df[["spUTCTime", "volume"]]

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

    return df


def normalise_live_imbalance_data(
        time_index: pd.DatetimeIndex,
        price_df: pd.DataFrame,
        volume_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Puts the imbalance data into the 'live' format, where the values are discovered as the settlement period progresses,
    and so there are NaNs at the beginning of the settlement period.
    """
    price_has_predictions = "predictionUTCTime" in price_df.columns
    volume_has_predictions = "predictionUTCTime" in volume_df.columns
    if price_has_predictions != volume_has_predictions:
        raise ValueError("Price and volume data must either both be predictive, or both be non-predictive.")
    has_predictions = price_has_predictions

    if has_predictions:
        # If the data has predictions then it's Modo data
        price_df = normalise_modo_data_for_live(
            modo_df=price_df,
            time_index=time_index,
            col="price",
            col_prediction="imbalance_price_live",
        )
        volume_df = normalise_modo_data_for_live(
            modo_df=volume_df,
            time_index=time_index,
            col="volume",
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
    for grouping_tuple, sub_df in modo_df.groupby(["spUTCTime"]):
        sp_start = grouping_tuple[0]  # 'SP' is short for settlement period
        sp_end = sp_start + timedelta(minutes=30)
        sub_df = sub_df.sort_values(by="predictionUTCTime", ascending=True)

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

    return df
