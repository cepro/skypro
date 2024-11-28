from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
from simt_common.rates.microgrid import VolRatesForEnergyFlows, get_vol_rates_dfs
from simt_common.rates.osam import calculate_osam_ncsp
from simt_common.rates.rates import OSAMFlatVolRate
from simt_common.timeutils.math import add_wallclock_days

from skypro.commands.simulator.microgrid import calculate_microgrid_flows


def run_osam_calcs_for_day(
    df: pd.DataFrame,
    t: datetime,
) -> Tuple[pd.DataFrame, pd.DatetimeIndex]:
    """
    Does the calculations necessary to calculate the OSAM NCSP for one day and adds the results to the dataframe which
    is returned.
    It also returns the datetime index associated for the day for convenience
    """
    df = df.copy()

    start_of_tomorrow = add_wallclock_days(t, 1)
    todays_index = df.loc[t:start_of_tomorrow].index

    # The above `loc` includes the `start_of_tomorrow` if we are simulating the whole day, which de don't want, so
    # remove it:
    if todays_index[-1] == start_of_tomorrow:
        todays_index = todays_index[:-1]

    start_of_yesterday = add_wallclock_days(t, -1)

    # Calculate the microgrid flows for yesterday, as these impact the OSAM NCSP
    mg_flow_calc_start = start_of_yesterday
    if mg_flow_calc_start < df.index[0]:
        mg_flow_calc_start = df.index[0]
    if mg_flow_calc_start < t:

        # To calculate OSAM rates we first need to work out the microgrid energy flows for yesterday given the
        # simulated actions
        df_with_mg_flows = calculate_microgrid_flows(df.loc[mg_flow_calc_start:t])

        # The below loc command doesn't work unless all the columns are already present.
        _match_columns(df, df_with_mg_flows)
        df.loc[mg_flow_calc_start:t] = df_with_mg_flows
    else:
        # We haven't simulated yesterday, so skip the calculation of microgrid flows
        pass

    # Next we can calculate the OSAM NCSP factor for today
    df.loc[todays_index, "osam_ncsp"], _ = calculate_osam_ncsp(
        df=df,
        index_to_calc_for=todays_index,
        imp_bp_col="grid_import",
        exp_bp_col="grid_export",
        imp_stor_col="bess_charge",
        exp_stor_col="bess_discharge",
        imp_gen_col=None,
        exp_gen_col="solar",
    )

    return df, todays_index


def add_total_vol_rates_to_df(
    df: pd.DataFrame,
    index_to_add_for: pd.DatetimeIndex,
    vol_rates: VolRatesForEnergyFlows,
    live_or_final: str,  # TODO: this isn't helpful for the LP optimiser? as it's neither live nor final
) -> pd.DataFrame:
    """
    Adds the total p/kWh rates for each flow to the dataframe for the period specified by `index_to_add_for` and returns
    the dataframe.
    """
    df = df.copy()

    # Inform any OSAM rate objects about the NCSP for today
    for rate in vol_rates.bess_charge_from_grid:
        if isinstance(rate, OSAMFlatVolRate):
            rate.add_ncsp(df.loc[index_to_add_for, "osam_ncsp"])

    # Next we can calculate the individual p/kWh rates that apply for today
    ext_vol_rates_dfs, int_vol_rates_dfs = get_vol_rates_dfs(index_to_add_for, vol_rates, log=False)

    # Then we sum up the individual rates to create a total for each flow
    for set_name, vol_rates_df in ext_vol_rates_dfs.items():
        df.loc[index_to_add_for, f"vol_rate_{live_or_final}_{set_name}"] = vol_rates_df.sum(axis=1, skipna=False)
    for set_name, vol_rates_df in int_vol_rates_dfs.items():
        df.loc[index_to_add_for, f"int_vol_rate_{live_or_final}_{set_name}"] = vol_rates_df.sum(axis=1, skipna=False)

    return df


def _match_columns(target_df, source_df):
    """
    Makes sure that all the columns in `source_df` are also present in `target_df`, by creating the columns with NaN
    values if they are not present.
    """
    for col in source_df.columns:
        if col not in target_df:
            target_df[col] = np.nan
