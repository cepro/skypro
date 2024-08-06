from datetime import timedelta
from typing import Dict, Optional

import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tabulate import tabulate

from simt_common.analysis.daily_gains import plot_daily_gains

from skypro.commands.simulator.config.config_v4 import OutputSummary


def explore_results(
        df: pd.DataFrame,
        final_rates_dfs: Dict[str, pd.DataFrame],
        final_int_rates_dfs: Dict[str, pd.DataFrame],
        do_plots: bool,
        battery_energy_capacity: float,
        battery_nameplate_power: float,
        site_import_limit: float,
        site_export_limit: float,
        summary_output_config: Optional[OutputSummary]
) -> pd.DataFrame:
    """
    Generally explores/plots the results, including logging the weighted average prices, cycling statistics, and
    benchmark £/kW and £/kWh values for the simulation. Returns a dataframe summarising the results
    """

    df = df.copy()
    sim_start = df.iloc[0].name
    sim_end = df.iloc[-1].name
    sim_days = get_24hr_days(sim_end - sim_start)

    # Calculate both the internal and external cost associated with each energy flow
    int_costs_dfs: Dict[str, pd.DataFrame] = {}
    for flow_name, int_rate_df in final_int_rates_dfs.items():
        int_costs_dfs[flow_name] = int_rate_df.mul(df[flow_name], axis=0)
    ext_costs_dfs: Dict[str, pd.DataFrame] = {}
    for flow_name, rate_df in final_rates_dfs.items():
        ext_costs_dfs[flow_name] = rate_df.mul(df[flow_name], axis=0)

    # Also calculate some 'derived' total cost of bess charges; discharges; solar and load, summing up from all sources.
    int_costs_dfs["bess_charge"] = pd.concat([
        int_costs_dfs["bess_charge_from_grid"].add_prefix("from_grid_"),
        int_costs_dfs["bess_charge_from_solar"].add_prefix("from_solar_")
    ], axis=1)
    ext_costs_dfs["bess_charge"] = pd.concat([
        ext_costs_dfs["bess_charge_from_grid"].add_prefix("from_grid_"),
        ext_costs_dfs["bess_charge_from_solar"].add_prefix("from_solar_")
    ], axis=1)
    int_costs_dfs["bess_discharge"] = pd.concat([
        int_costs_dfs["bess_discharge_to_grid"].add_prefix("to_grid_"),
        int_costs_dfs["bess_discharge_to_load"].add_prefix("to_load_")
    ], axis=1)
    ext_costs_dfs["bess_discharge"] = pd.concat([
        ext_costs_dfs["bess_discharge_to_grid"].add_prefix("to_grid_"),
        ext_costs_dfs["bess_discharge_to_load"].add_prefix("to_load_")
    ], axis=1)
    int_costs_dfs["solar"] = pd.concat([
        int_costs_dfs["solar_to_grid"].add_prefix("to_grid_"),
        int_costs_dfs["solar_to_load"].add_prefix("to_load_"),
        -1 * int_costs_dfs["bess_charge_from_solar"].add_prefix("to_bess_")
    ], axis=1)
    ext_costs_dfs["solar"] = pd.concat([
        ext_costs_dfs["solar_to_grid"].add_prefix("to_grid_"),
        ext_costs_dfs["solar_to_load"].add_prefix("to_load_"),
        ext_costs_dfs["bess_charge_from_solar"].add_prefix("to_bess_")
    ], axis=1)
    int_costs_dfs["load"] = pd.concat([
        int_costs_dfs["load_from_grid"].add_prefix("from_grid_"),
        -1 * int_costs_dfs["solar_to_load"].add_prefix("from_solar_"),
        -1 * int_costs_dfs["bess_discharge_to_load"].add_prefix("from_bess_")
    ], axis=1)
    ext_costs_dfs["load"] = pd.concat([
        ext_costs_dfs["load_from_grid"].add_prefix("from_grid_"),
        ext_costs_dfs["solar_to_load"].add_prefix("from_solar_"),
        ext_costs_dfs["bess_discharge_to_load"].add_prefix("from_bess_")
    ], axis=1)

    # Calculate the total cost associated with each energy flow.
    total_int_costs: Dict[str, float] = {}
    total_ext_costs: Dict[str, float] = {}
    for flow_name, cost_df in int_costs_dfs.items():
        total_int_costs[flow_name] = cost_df.sum().sum()
    for flow_name, cost_df in ext_costs_dfs.items():
        total_ext_costs[flow_name] = cost_df.sum().sum()

    total_int_bess_gain = - total_int_costs["bess_discharge"] - total_int_costs["bess_charge"]

    # Calculate the total energy over the period for each energy flow of interest.
    total_flows: Dict[str, float] = {}
    for flow_name in list(int_costs_dfs.keys()):
        total_flows[flow_name] = df[flow_name].sum()

    # The above charge/discharge flows are representative of the flows into and out of the BESS. Losses are modelled as
    # 'internal to the battery'. So the total bess charge (from all sources) is larger than the total bess discharge
    # (to all loads).
    total_bess_losses = df['bess_losses'].sum()

    solar_self_use = total_flows["solar"] - total_flows["solar_to_grid"]

    # Calculate the average p/kWh rates associated with the various energy flows
    avg_int_rates: Dict[str, float] = {}
    for flow_name, total_cost in total_int_costs.items():
        avg_int_rates[flow_name] = total_cost / total_flows[flow_name]
    avg_ext_rates: Dict[str, float] = {}
    for flow_name, total_cost in total_ext_costs.items():
        avg_ext_rates[flow_name] = total_cost / total_flows[flow_name]

    # Friendly names to present to the user
    flow_name_map = {
        "solar_to_grid": "solarToGrid",
        "load_from_grid": "gridToLoad",
        "solar_to_load": "solarToLoad",
        "bess_discharge_to_load": "battToLoad",
        "bess_discharge_to_grid": "battToGrid",
        "bess_charge_from_solar": "solarToBatt",
        "bess_charge_from_grid": "gridToBatt",
        "bess_charge": "All batt charge",
        "bess_discharge": "All batt discharge",
        "solar": "All solar",
        "load": "All load"
    }

    # Separate the more fundamental flows from the derived flows as they are presented to the user in separate tables.
    fundamental_flow_names = list(final_int_rates_dfs.keys())
    fundamental_flow_summary_df = pd.DataFrame()
    derived_flow_summary_df = pd.DataFrame()
    fundamental_flow_summary_df.index.name = "flow"
    derived_flow_summary_df.index.name = "flow"
    for flow_name in int_costs_dfs.keys():
        output_flow_name = flow_name_map[flow_name]

        if flow_name in fundamental_flow_names:
            flow_summary_df = fundamental_flow_summary_df
        else:
            flow_summary_df = derived_flow_summary_df

        flow_summary_df.loc[output_flow_name, "volume"] = total_flows[flow_name]
        flow_summary_df.loc[output_flow_name, "int_cost"] = total_int_costs[flow_name] / 100  # pence to £
        flow_summary_df.loc[output_flow_name, "int_avg_rate"] = avg_int_rates[flow_name]
        flow_summary_df.loc[output_flow_name, "ext_cost"] = total_ext_costs[flow_name] / 100  # pence to £
        flow_summary_df.loc[output_flow_name, "ext_avg_rate"] = avg_ext_rates[flow_name]

    # Friendly names to present to the user
    flow_summary_column_name_map = {
        "volume": "Volume (kWh)",
        "int_cost": "Int. Cost (£)",
        "int_avg_rate": "Int. Avg Rate (p/kWh)",
        "ext_cost": "Ext. Cost (£)",
        "ext_avg_rate": "Ext. Avg Rate (p/kWh)",
    }

    print(tabulate(
        tabular_data=fundamental_flow_summary_df.rename(columns=flow_summary_column_name_map),
        headers='keys',
        tablefmt='presto',
        floatfmt=(None, ",.0f", ",.0f", ",.2f", ",.0f", ",.2f")
    ))
    print("")
    print("* The internal prices assigned to battery flows are signed from the perspective of the battery strategy")

    print("")
    print(tabulate(
        tabular_data=derived_flow_summary_df.rename(columns=flow_summary_column_name_map),
        headers='keys',
        tablefmt='presto',
        floatfmt=(None, ",.0f", ",.0f", ",.2f", ",.0f", ",.2f")
    ))

    print("")
    print(f"Solar self-use (inc batt losses): {solar_self_use:,.2f} kWh, {(solar_self_use/total_flows['solar'])*100:.1f}% of the solar generation.")

    # Cycling
    total_cycles = total_flows["bess_discharge"] / battery_energy_capacity
    cycles_per_day = total_cycles / sim_days
    print("")
    print(f"Total cycles over simulation: {total_cycles:.2f} cycles")
    print(f"Average cycles per day: {cycles_per_day:.2f} cycles/day")

    print("")
    print(f"Total BESS gain over period: £{total_int_bess_gain/100:,.2f}")
    print(f"Average daily BESS gain over period: £{(total_int_bess_gain / 100)/sim_days:.2f}")

    total_ext_cost = 0.0
    for flow_name in fundamental_flow_names:
        total_ext_cost += total_ext_costs[flow_name]

    print("")
    print(f"Total external costs: £{total_ext_cost / 100:,.2f}")

    # Output a CSV file summarising the energy flows and costs
    if summary_output_config:
        fundamental_flow_summary_df.to_csv(summary_output_config.csv, index=True)

    # TODO: print warning if cycling is low - charge efficiency changes

    # Plot energy flows with charge / discharge limits
    if do_plots:
        plot_hh_strategy(df)
        plot_microgrid_energy_flows(
            df, site_import_limit, site_export_limit, battery_nameplate_power
        )
        # plot_costs_by_grouping(costs_dfs["bess_charge"], costs_dfs["bess_discharge"])
        plot_daily_gains(int_costs_dfs)

    return fundamental_flow_summary_df


def plot_hh_strategy(df: pd.DataFrame):

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # fig.add_trace(go.Scatter(x=df.index, y=df["rate_import_10m"], name="Import Price 10m (SSP plus DUoS)", line=dict(color="rgba(89, 237, 131, 1)")))
    # fig.add_trace(go.Scatter(x=df.index, y=df["rate_import_20m"], name="Import Price 20m (SSP plus DUoS)", line=dict(color="rgba(40, 189, 82, 1)")))
    fig.add_trace(
        go.Scatter(x=df.index, y=df["rate_final_bess_charge_from_grid"], name="Import Price", line=dict(color="rgba(0, 141, 40, 1)")))
    # fig.add_trace(go.Scatter(x=df.index, y=df["rate_export_10m"], name="Export Price 10m (SSP plus DUoS)", line=dict(color="rgba(185, 102, 247, 1)")))
    # fig.add_trace(go.Scatter(x=df.index, y=df["rate_export_20m"], name="Export Price 20m (SSP plus DUoS)", line=dict(color="rgba(153, 59, 224, 1)")))
    fig.add_trace(
        go.Scatter(x=df.index, y=df["rate_final_bess_discharge_to_grid"]*-1, name="Export Price", line=dict(color="rgba(102, 0, 178, 1)")))

    # if "notional_spread" in df.columns:
    #     fig.add_trace(
    #         go.Scatter(x=df.index, y=df["notional_spread"], name="Notional Spread", mode="markers", line=dict(color="green"))
    #     )
    #
    if "red_approach_distance" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["red_approach_distance"], name="red_approach_distance", mode="markers", line=dict(color="red")),
            secondary_y=True
        )

    if "amber_approach_distance" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["amber_approach_distance"], name="amber_approach_distance", mode="markers", line=dict(color="orange")),
            secondary_y=True
        )

    if "spread_algo_energy" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["spread_algo_energy"], name="spread_algo_energy", mode="markers", line=dict(color="yellow")),
            secondary_y=True
        )

    if "microgrid_algo_energy" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["microgrid_algo_energy"], name="microgrid_algo_energy", mode="markers", line=dict(color="green")),
            secondary_y=True
        )

    # fig.add_trace(go.Scatter(x=df.index, y=df["imbalance_volume_final"], name="Imbalance volume final", line=dict(color="red")), secondary_y=True)
    fig.add_trace(go.Scatter(x=df.index, y=df["soe"], name="Battery SoE", line=dict(color="orange")),
                  secondary_y=True)

    fig.update_yaxes(title_text="Price (p/kW)", range=[-10, 40], secondary_y=False, row=1, col=1)
    fig.update_yaxes(title_text="SoE (kWh)", range=[0, 200], secondary_y=True, row=1, col=1)
    fig.update_layout(title="Optimisation strategy")
    fig.show()


def plot_microgrid_energy_flows(df, site_import_limit, site_export_limit, battery_nameplate_power):
    """
    This plots the various power flows in teh microgrid with site import/export limits.
    """
    time_step_hours = pd.to_timedelta(df.index.freq).total_seconds() / 3600
    df_tmp = df[["solar_power", "load_power", "bess_max_power_charge", "bess_max_power_discharge"]].copy()
    df_tmp["solar_power"] = -df_tmp["solar_power"]
    df_tmp["bess_max_power_discharge"] = -df_tmp["bess_max_power_discharge"]
    df_tmp["bess_power"] = df["energy_delta"] / time_step_hours
    df_tmp["solar_to_grid_power"] = -df["solar_to_grid"] / time_step_hours
    df_tmp["load_from_grid_power"] = df["load_from_grid"] / time_step_hours

    fig = px.line(df_tmp, line_shape='hv')
    fig.add_hline(y=site_import_limit, line_dash="dot", annotation_text="Site import limit")
    fig.add_hline(y=-site_export_limit, line_dash="dot", annotation_text="Site export limit")
    fig.add_hline(y=battery_nameplate_power, line_dash="dot", annotation_text="Battery nameplate power")
    fig.add_hline(y=-battery_nameplate_power, line_dash="dot", annotation_text="Battery nameplate power")
    fig.update_layout(title="Constraints and powers")
    fig.show()


def report_dropped_rows(orig, filtered, data_name):
    pct_dropped = ((len(orig) - len(filtered)) / len(orig)) * 100
    if pct_dropped > 3:
        user_input = input(
            f"Warning: dropped {pct_dropped:.1f}% of rows for '{data_name}', which may make the associated results "
            f"invalid, would you like to continue anyway? ")
        if user_input.lower() not in ['yes', 'y']:
            print("Exiting")
            exit(-1)
    elif pct_dropped > 0:
        print(f"Warning: dropped {pct_dropped:.1f}% of rows for '{data_name}.")


def get_24hr_days(duration: timedelta) -> float:
    """
    Returns the duration in number of days, assuming each day is 24hrs (which is not always true with daylight saving
    transitions) with decimal places if required.
    """
    return (duration.total_seconds() / 3600) / 24.0
