import logging

import numpy as np
import pandas as pd
from typing import Optional

from skypro.common.timeutils.math import floor_day
from skypro.common.timeutils.math_wallclock import add_wallclock_days

# TODO: This is an assumed figure - the docs say that "it will be set by the BSC Panel from time to time", but I haven't
#  been able to find the actual number online.
NCSP_DEFAULT = 0.5


def calculate_osam_ncsp(
        df: pd.DataFrame,
        index_to_calc_for: pd.DatetimeIndex,
        imp_bp_col: str,
        exp_bp_col: str,
        imp_stor_col: str,
        exp_stor_col: str,
        imp_gen_col: Optional[str],
        exp_gen_col: Optional[str]
) -> (pd.Series, Optional[pd.DataFrame]):
    """
    Calculates the on-site allocation methodologies' non-chargeable proportion using the volumes in `df` and returns it
    as a pandas Series with a row for each index given in `index_to_calc_for`.
    It also returns a dataframe containing more details of the OSAM calculations, which has a row per half-hour (OSAM
    is based on hh settlement periods) and contains the NCSP as well as the OSAM microgrid flows (which use a different
    methodology to Cepro's microgrid flows calculations).

    The `imp_bp_col`, `exp_bp_col`, etc., arguments specify the columns in `df` that correspond with the OSAM volumes
    as specified in the methodology document. E.g. `imp_bp` is the metered import volume at the boundary point.
    https://bscdocs.elexon.co.uk/category-3-documents/on-site-energy-allocation-methodology-document
    """

    if imp_bp_col not in df.columns:
        # If we aren't given the requisite columns (perhaps because a simulation has just begun) then there are no
        # energy flows to calculate the NCSP, so we must assume that it's the default value.
        return pd.Series(index=index_to_calc_for, data=NCSP_DEFAULT), None

    # The OSAM doc defines that we should use the previous 7 days of data
    N_SETTLEMENT_DAYS = 7

    # Limit the backing data range to that required for the OSAM calculation
    data_start = add_wallclock_days(floor_day(index_to_calc_for[0]), -N_SETTLEMENT_DAYS)
    data_end = floor_day(index_to_calc_for[-1])

    df = df[df.index >= data_start]
    df = df[df.index < data_end]

    osam_df = pd.DataFrame(index=df.index)
    osam_df["imp_bp"] = df[imp_bp_col]
    osam_df["exp_bp"] = df[exp_bp_col]
    osam_df["imp_stor"] = df[imp_stor_col]
    osam_df["exp_stor"] = df[exp_stor_col]
    osam_df["imp_gen"] = 0.0 if imp_gen_col is None else df[imp_gen_col]
    osam_df["exp_gen"] = 0.0 if exp_gen_col is None else df[exp_gen_col]

    # OSAM calcs are done based on half-hourly settlement periods
    osam_df = osam_df.resample("30min").sum()

    # The below calcs are pulled directly from the OSAM documentation, and variable naming should match the docs

    osam_df["net_other"] = (osam_df["imp_bp"] - osam_df["imp_stor"] - osam_df["imp_gen"]) - (
                osam_df["exp_bp"] - osam_df["exp_stor"] - osam_df["exp_gen"])
    osam_df["imp_other"] = np.maximum(0, osam_df["net_other"])
    osam_df["exp_other"] = -np.minimum(0, osam_df["net_other"])

    osam_df["surplus"] = np.maximum(0, osam_df["exp_gen"] + osam_df["exp_stor"] - osam_df["exp_bp"])

    osam_df["gen_x_stor"] = np.minimum(osam_df["exp_gen"], osam_df["imp_stor"], osam_df["surplus"])

    osam_df["remaining"] = osam_df["surplus"] - osam_df["gen_x_stor"]

    osam_df["gen_x_other"] = np.minimum(osam_df["exp_gen"] - osam_df["gen_x_stor"], osam_df["imp_other"],
                                        osam_df["remaining"])

    osam_df["gen_x_bp"] = osam_df["exp_gen"] - osam_df["gen_x_stor"] - osam_df["gen_x_other"]

    osam_df["stor_x_bp"] = np.maximum(0, np.minimum(osam_df["exp_bp"] - osam_df["gen_x_bp"], osam_df["exp_stor"]))

    osam_df["stor_x_gen"] = np.maximum(0, np.minimum(osam_df["imp_gen"], osam_df["exp_stor"] - osam_df["stor_x_bp"]))

    osam_df["stor_x_other"] = np.maximum(0, osam_df["exp_stor"] - osam_df["stor_x_bp"] - osam_df["stor_x_gen"])

    osam_df["bp_x_stor"] = np.minimum(osam_df["imp_bp"], osam_df["imp_stor"] - osam_df["gen_x_stor"])

    osam_df["other_x_stor"] = osam_df["imp_stor"] - osam_df["gen_x_stor"] - osam_df["bp_x_stor"]

    osam_df["bp_x_gen"] = np.minimum(osam_df["imp_gen"] - osam_df["stor_x_gen"],
                                     osam_df["imp_bp"] - osam_df["bp_x_stor"])

    osam_df["other_x_gen"] = np.minimum(osam_df["imp_other"],
                                        osam_df["imp_gen"] - osam_df["stor_x_gen"] - osam_df["bp_x_gen"])

    osam_df["bp_x_other"] = osam_df["imp_bp"] - osam_df["bp_x_stor"] - osam_df["bp_x_gen"]

    osam_df["other_x_bp"] = osam_df["exp_bp"] - osam_df["gen_x_bp"] - osam_df["stor_x_bp"]

    osam_daily_df = osam_df.groupby(pd.Grouper(freq='D')).sum()

    # TODO: The rolling sum should use NCSP_DEFAULT for periods where there is missing data (e.g. if we have just
    #  started a simulation, however, the NCSP_DEFAULT value is a guess at the moment (see above) so it's probably more
    #  accurate for now to just to use only the data that is available.

    osam_daily_df = pd.concat([
        osam_daily_df,
        osam_daily_df.rolling(f"{N_SETTLEMENT_DAYS}D").sum().add_prefix("roll_sum_")
    ], axis=1)

    # Extend the index by a day so that we can hold the NSCP result for the day after the 7 day calculation window
    osam_daily_df = osam_daily_df.reindex(pd.date_range(
        start=osam_daily_df.index[0],
        end=add_wallclock_days(osam_daily_df.index[-1], 1),
        freq='D'
    ))

    # Calculate the NSCP, which is 'shifted' to apply it to the day after the backing data
    osam_daily_df["ncsp"] = ((osam_daily_df["roll_sum_exp_stor"] - osam_daily_df["roll_sum_stor_x_other"]) / osam_daily_df["roll_sum_exp_stor"]).shift(1)

    num_nans = np.isnan(osam_daily_df["ncsp"]).sum()
    if num_nans != 1:
        logging.warning(f"OSAM calculation had {num_nans - 1} unexpected NaN values")
    osam_daily_df["ncsp"] = osam_daily_df["ncsp"].fillna(NCSP_DEFAULT)

    result_df = pd.DataFrame(index=index_to_calc_for)

    result_df["ncsp"] = osam_daily_df["ncsp"]

    # If the index_to_calc_for doesn't start at midnight then we may miss the NCSP value on the first day:
    if np.isnan(result_df.iloc[0]["ncsp"]):
        result_df.loc[result_df.index[0], "ncsp"] = osam_daily_df.loc[floor_day(result_df.index[0]), "ncsp"]

    result_df["ncsp"] = result_df["ncsp"].ffill()

    # Also include the NCSP in the detailed calculation dataframe
    osam_df["ncsp"] = result_df["ncsp"]

    return result_df["ncsp"], osam_df[["ncsp", "bp_x_stor", "bp_x_other", "bp_x_gen"]]


def calculate_osam_rate_cost(osam_df: pd.DataFrame, base_rate: pd.Series) -> float:
    """
    Returns the total cost of a given p/kWh rate when OSAM is applicable to it.
    - When the rate is applied to flows from boundary point -> 'other' then it is always applied at 100%
    - When the rate is applied to flows from boundary point -> 'storage' then it is applied according to the non-chargeable proportion
    """
    return ((osam_df["bp_x_other"] * base_rate) + (osam_df["bp_x_stor"] * base_rate * (1 - osam_df["ncsp"]))).sum()

