from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class MicrogridBreakdown:
    """Summarises key info about a microgrid."""

    # Dictionary, keyed by energy flow name, containing the total energy for each flow, in kWh.
    total_flows: Dict[str, float]

    # Dictionaries, keyed by energy flow name, containing a dataframe of costs, with a column for each associated rate.
    int_vol_costs_dfs: Dict[str, pd.DataFrame]
    mkt_vol_costs_dfs: Dict[str, pd.DataFrame]

    # Dictionaries, keyed by energy flow name, containing the total cost, in pence, associated with that flow.
    total_int_vol_costs: Dict[str, float]
    total_mkt_vol_costs: Dict[str, float]

    # Dictionaries, keyed by energy flow name, containing the average rates, in p/kWh, associated with each energy flow
    avg_int_vol_rates: Dict[str, float]
    avg_mkt_vol_rates: Dict[str, float]

    # The benefit, in pence, of having the battery
    total_int_bess_gain: float

    # How much solar energy we use on-site (including battery flows and battery losses)
    solar_self_use: float

    avg_osam_ncsp: float  # Weighted average OSAM NCSP

    # These tables contain some of the above data in a handy dataframe summary, for printing to the user or storing as
    # a CSV etc. The 'fundamental flows' are the seven flows for which there are rates defined, whereas the 'derived
    # flows' are derived from the fundamental flows, but may be interesting to the user.
    fundamental_flows_summary_df: pd.DataFrame
    derived_flows_summary_df: pd.DataFrame


def breakdown_microgrid_flows(
        df: pd.DataFrame,
        int_vol_rates_dfs: Dict[str, pd.DataFrame],
        mkt_vol_rates_dfs: Dict[str, pd.DataFrame]
) -> MicrogridBreakdown:
    """
    This function uses the energy flows defined in `df` and the rates associated with each energy flow in
    `int_vol_rates_dfs` and `mkt_vol_rates_dfs` to give a breakdown of costs and other useful info about the microgrid.

    :param df: dataframe containing a column for each energy flow
    :param int_vol_rates_dfs: dictionary, keyed by energy flow name, containing a dataframe of internal rates for that flow
    :param mkt_vol_rates_dfs: dictionary, keyed by energy flow name, containing a dataframe of supplier/market rates for that flow
    """

    result = MicrogridBreakdown(
        total_flows={},
        int_vol_costs_dfs={},
        mkt_vol_costs_dfs={},
        total_int_vol_costs={},
        total_mkt_vol_costs={},
        avg_int_vol_rates={},
        avg_mkt_vol_rates={},
        total_int_bess_gain=np.nan,
        solar_self_use=np.nan,
        avg_osam_ncsp=np.nan,
        fundamental_flows_summary_df=pd.DataFrame(),
        derived_flows_summary_df=pd.DataFrame()
    )

    # Calculate both the internal and market cost associated with each energy flow
    for flow_name, int_vol_rate_df in int_vol_rates_dfs.items():
        result.int_vol_costs_dfs[flow_name] = int_vol_rate_df.mul(df[flow_name], axis=0)
    for flow_name, vol_rate_df in mkt_vol_rates_dfs.items():
        result.mkt_vol_costs_dfs[flow_name] = vol_rate_df.mul(df[flow_name], axis=0)
    # Also calculate some 'derived' total cost of bess charges; discharges; solar and load, summing up from all sources.
    result.int_vol_costs_dfs["bess_charge"] = pd.concat([
        result.int_vol_costs_dfs["grid_to_batt"].add_prefix("from_grid_"),
        result.int_vol_costs_dfs["solar_to_batt"].add_prefix("from_solar_")
    ], axis=1)
    result.mkt_vol_costs_dfs["bess_charge"] = pd.concat([
        result.mkt_vol_costs_dfs["grid_to_batt"].add_prefix("from_grid_"),
        result.mkt_vol_costs_dfs["solar_to_batt"].add_prefix("from_solar_")
    ], axis=1)
    result.int_vol_costs_dfs["bess_discharge"] = pd.concat([
        result.int_vol_costs_dfs["batt_to_grid"].add_prefix("to_grid_"),
        result.int_vol_costs_dfs["batt_to_load"].add_prefix("to_load_")
    ], axis=1)
    result.mkt_vol_costs_dfs["bess_discharge"] = pd.concat([
        result.mkt_vol_costs_dfs["batt_to_grid"].add_prefix("to_grid_"),
        result.mkt_vol_costs_dfs["batt_to_load"].add_prefix("to_load_")
    ], axis=1)
    result.int_vol_costs_dfs["solar"] = pd.concat([
        result.int_vol_costs_dfs["solar_to_grid"].add_prefix("to_grid_"),
        result.int_vol_costs_dfs["solar_to_load"].add_prefix("to_load_"),
        -1 * result.int_vol_costs_dfs["solar_to_batt"].add_prefix("to_bess_")
    ], axis=1)
    result.mkt_vol_costs_dfs["solar"] = pd.concat([
        result.mkt_vol_costs_dfs["solar_to_grid"].add_prefix("to_grid_"),
        result.mkt_vol_costs_dfs["solar_to_load"].add_prefix("to_load_"),
        result.mkt_vol_costs_dfs["solar_to_batt"].add_prefix("to_bess_")
    ], axis=1)
    result.int_vol_costs_dfs["load"] = pd.concat([
        result.int_vol_costs_dfs["grid_to_load"].add_prefix("from_grid_"),
        -1 * result.int_vol_costs_dfs["solar_to_load"].add_prefix("from_solar_"),
        -1 * result.int_vol_costs_dfs["batt_to_load"].add_prefix("from_bess_")
    ], axis=1)
    result.mkt_vol_costs_dfs["load"] = pd.concat([
        result.mkt_vol_costs_dfs["grid_to_load"].add_prefix("from_grid_"),
        result.mkt_vol_costs_dfs["solar_to_load"].add_prefix("from_solar_"),
        result.mkt_vol_costs_dfs["batt_to_load"].add_prefix("from_bess_")
    ], axis=1)

    # Calculate the total energy over the period for each energy flow of interest.
    for flow_name in list(result.int_vol_costs_dfs.keys()):
        if df[flow_name].isnull().all():
            # Some flows may not be supported (e.g. solar before emlite data is available)
            result.total_flows[flow_name] = np.nan
        else:
            result.total_flows[flow_name] = df[flow_name].sum()
    # Also include bess losses in the total flows
    result.total_flows["bess_losses"] = df['bess_losses'].sum()  # TODO: what are we doing with this info?
    # The charge/discharge flows are representative of the flows into and out of the BESS. Losses are modelled as
    # 'internal to the battery'. So the total bess charge (from all sources) is larger than the total bess discharge
    # (to all loads).

    # Calculate the total cost associated with each energy flow.
    for flow_name, cost_df in result.int_vol_costs_dfs.items():
        if np.isnan(result.total_flows[flow_name]):
            result.total_int_vol_costs[flow_name] = np.nan
        else:
            result.total_int_vol_costs[flow_name] = cost_df.sum(skipna=False).sum(skipna=False)
    for flow_name, cost_df in result.mkt_vol_costs_dfs.items():
        if np.isnan(result.total_flows[flow_name]):
            result.total_mkt_vol_costs[flow_name] = np.nan
        else:
            result.total_mkt_vol_costs[flow_name] = cost_df.sum(skipna=False).sum(skipna=False)

    result.total_int_bess_gain = - result.total_int_vol_costs["bess_discharge"] - result.total_int_vol_costs["bess_charge"]

    result.solar_self_use = result.total_flows["solar"] - result.total_flows["solar_to_grid"]

    if "osam_ncsp" in df.columns:
        if df["grid_to_batt"].sum() == 0:
            result.avg_osam_ncsp = np.nan  # np.average raises if the weights sum to zero
        else:
            result.avg_osam_ncsp = np.average(a=df["osam_ncsp"], weights=df["grid_to_batt"])

    # Calculate the average p/kWh rates associated with the various energy flows
    for flow_name, total_cost in result.total_int_vol_costs.items():
        result.avg_int_vol_rates[flow_name] = total_cost / result.total_flows[flow_name]
    for flow_name, total_cost in result.total_mkt_vol_costs.items():
        result.avg_mkt_vol_rates[flow_name] = total_cost / result.total_flows[flow_name]

    # Here we put some of the above data into summary dataframes for convenience.
    fundamental_flow_names = list(int_vol_rates_dfs.keys())
    result.fundamental_flows_summary_df.index.name = "flow"
    result.derived_flows_summary_df.index.name = "flow"
    for flow_name in result.int_vol_costs_dfs.keys():
        # Separate the more fundamental flows from the derived flows as they are presented to the user separately.
        if flow_name in fundamental_flow_names:
            flow_summary_df = result.fundamental_flows_summary_df
        else:
            flow_summary_df = result.derived_flows_summary_df

        flow_summary_df.loc[flow_name, "volume"] = result.total_flows[flow_name]
        flow_summary_df.loc[flow_name, "int_vol_cost"] = result.total_int_vol_costs[flow_name] / 100  # pence to £
        flow_summary_df.loc[flow_name, "int_avg_vol_rate"] = result.avg_int_vol_rates[flow_name]
        flow_summary_df.loc[flow_name, "mkt_vol_cost"] = result.total_mkt_vol_costs[flow_name] / 100  # pence to £
        flow_summary_df.loc[flow_name, "mkt_avg_vol_rate"] = result.avg_mkt_vol_rates[flow_name]

    for flow in ["solar", "load"]:
        # The internal rates for solar_to_grid, solar_to_load and grid_to_load are zero which makes the total internal
        # rates for solar and load somewhat meaningless. So set them to NaN to avoid misinterpretation.
        result.derived_flows_summary_df.loc[flow, "int_vol_cost"] = np.nan
        result.derived_flows_summary_df.loc[flow, "int_avg_vol_rate"] = np.nan

        # The market rates for solar_to_batt, solar_to_load and batt_to_load are zero which makes the total market
        # rates for solar and load somewhat meaningless. So set them to NaN to avoid misinterpretation.
        result.derived_flows_summary_df.loc[flow, "mkt_vol_cost"] = np.nan
        result.derived_flows_summary_df.loc[flow, "mkt_avg_vol_rate"] = np.nan

    return result
