from datetime import timedelta
from typing import Dict, List, Optional

import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from simt_common.microgrid_analysis.breakdown import breakdown_microgrid_flows
from simt_common.microgrid_analysis.daily_gains import plot_daily_gains
from simt_common.rates.bill_match import bill_match
from simt_common.rates.rates import OSAMFlatVolRate
from tabulate import tabulate


def explore_results(
        df: pd.DataFrame,
        final_mkt_vol_rates_dfs: Dict[str, pd.DataFrame],
        final_int_vol_rates_dfs: Dict[str, pd.DataFrame],
        mkt_fixed_costs_dfs: Optional[Dict[str, pd.DataFrame]],
        do_plots: bool,
        battery_energy_capacity: float,
        battery_nameplate_power: float,
        osam_rates: List[OSAMFlatVolRate],
        osam_df: pd.DataFrame,
):
    """
    Generally explores/plots the results, including logging the weighted average prices, cycling statistics, and
    benchmark £/kW and £/kWh values for the simulation.
    """

    df = df.copy()
    sim_start = df.iloc[0].name
    sim_end = df.iloc[-1].name
    sim_days = get_24hr_days(sim_end - sim_start)

    breakdown = breakdown_microgrid_flows(
        df=df,
        int_vol_rates_dfs=final_int_vol_rates_dfs,
        mkt_vol_rates_dfs=final_mkt_vol_rates_dfs
    )

    # Friendly names to present to the user
    flow_name_map = {
        "solar_to_grid": "solarToGrid",
        "grid_to_load": "gridToLoad",
        "solar_to_load": "solarToLoad",
        "batt_to_load": "battToLoad",
        "batt_to_grid": "battToGrid",
        "solar_to_batt": "solarToBatt",
        "grid_to_batt": "gridToBatt",
        "bess_charge": "All batt charge",
        "bess_discharge": "All batt discharge",
        "solar": "All solar",
        "load": "All load"
    }

    # Friendly names to present to the user
    flow_summary_column_name_map = {
        "volume": "Volume (kWh)",
        "int_cost": "Int. Cost (£)",
        "int_avg_rate": "Int. Avg Rate (p/kWh)",
        "mkt_cost": "Ext. Cost (£)",
        "mkt_avg_rate": "Ext. Avg Rate (p/kWh)",
    }

    # Rename the flows to the externally-facing names
    breakdown.fundamental_flows_summary_df.index = breakdown.fundamental_flows_summary_df.index.map(flow_name_map)
    breakdown.derived_flows_summary_df.index = breakdown.derived_flows_summary_df.index.map(flow_name_map)

    print(tabulate(
        tabular_data=breakdown.fundamental_flows_summary_df.rename(columns=flow_summary_column_name_map),
        headers='keys',
        tablefmt='presto',
        floatfmt=(None, ",.0f", ",.0f", ",.2f", ",.0f", ",.2f")
    ))
    print("")
    print("* The internal prices assigned to battery flows are signed from the perspective of the battery strategy")

    print("")
    print(tabulate(
        tabular_data=breakdown.derived_flows_summary_df.rename(columns=flow_summary_column_name_map),
        headers='keys',
        tablefmt='presto',
        floatfmt=(None, ",.0f", ",.0f", ",.2f", ",.0f", ",.2f")
    ))

    print("")
    print(f"Solar self-use (inc batt losses): {breakdown.solar_self_use:,.2f} kWh, {(breakdown.solar_self_use/breakdown.total_flows['solar'])*100:.1f}% of the solar generation.")

    print("")
    print(f"Weighted average OSAM NCSP: {breakdown.avg_osam_ncsp:,.3f}")

    # Cycling
    total_cycles = breakdown.total_flows["bess_discharge"] / battery_energy_capacity
    cycles_per_day = total_cycles / sim_days
    print("")
    print(f"Total cycles over simulation: {total_cycles:.2f} cycles")
    print(f"Average cycles per day: {cycles_per_day:.2f} cycles/day")

    print("")
    print(f"Total BESS gain over period: £{breakdown.total_int_bess_gain/100:,.2f}")
    print(f"Average daily BESS gain over period: £{(breakdown.total_int_bess_gain / 100)/sim_days:.2f}")

    total_mkt_vol_cost = 0.0
    for flow_name in final_mkt_vol_rates_dfs.keys():
        total_mkt_vol_cost += breakdown.total_mkt_vol_costs[flow_name]

    print("")
    print(f"Total market vol costs: £{total_mkt_vol_cost / 100:,.2f}")

    bill_match(
        grid_energy_flow=df["grid_import"],
        # use the grid rates for grid_to_batt as these include info about any OSAM rates
        mkt_vol_grid_rates_df=final_mkt_vol_rates_dfs["grid_to_batt"],
        mkt_fixed_costs_df=mkt_fixed_costs_dfs["import"] if "import" in mkt_fixed_costs_dfs else None,
        osam_rates=osam_rates,
        osam_df=osam_df,
        cepro_mkt_vol_bill_total_expected=breakdown.total_mkt_vol_costs["grid_to_batt"] + breakdown.total_mkt_vol_costs["grid_to_load"],
        context="import",
        line_items=None,
    )

    # Plot energy flows with charge / discharge limits
    if do_plots:
        plot_hh_strategy(df)
        plot_constraints(
            df, battery_nameplate_power
        )
        # plot_costs_by_grouping(costs_dfs["bess_charge"], costs_dfs["bess_discharge"])
        plot_daily_gains(breakdown.int_vol_costs_dfs)

    return


def plot_hh_strategy(df: pd.DataFrame):

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # fig.add_trace(go.Scatter(x=df.index, y=df["rate_import_10m"], name="Import Price 10m (SSP plus DUoS)", line=dict(color="rgba(89, 237, 131, 1)")))
    # fig.add_trace(go.Scatter(x=df.index, y=df["rate_import_20m"], name="Import Price 20m (SSP plus DUoS)", line=dict(color="rgba(40, 189, 82, 1)")))
    fig.add_trace(
        go.Scatter(x=df.index, y=df["mkt_vol_rate_final_grid_to_batt"], name="Import Price", line=dict(color="rgba(0, 141, 40, 1)")))
    # fig.add_trace(go.Scatter(x=df.index, y=df["rate_export_10m"], name="Export Price 10m (SSP plus DUoS)", line=dict(color="rgba(185, 102, 247, 1)")))
    # fig.add_trace(go.Scatter(x=df.index, y=df["rate_export_20m"], name="Export Price 20m (SSP plus DUoS)", line=dict(color="rgba(153, 59, 224, 1)")))
    fig.add_trace(
        go.Scatter(x=df.index, y=df["mkt_vol_rate_final_batt_to_grid"]*-1, name="Export Price", line=dict(color="rgba(102, 0, 178, 1)")))

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

    # fig.add_trace(go.Scatter(x=df.index, y=df["imbalance_volume_final"], name="Imbalance volume final", line=dict(color="red")), secondary_y=True)
    fig.add_trace(go.Scatter(x=df.index, y=df["soe"], name="Battery SoE", line=dict(color="orange")),
                  secondary_y=True)

    fig.update_yaxes(title_text="Price (p/kW)", range=[-10, 40], secondary_y=False, row=1, col=1)
    fig.update_yaxes(title_text="SoE (kWh)", range=[0, 200], secondary_y=True, row=1, col=1)
    fig.update_layout(title="Optimisation strategy")
    fig.show()


def plot_constraints(df, battery_nameplate_power):
    """
    This plots the various power flows in teh microgrid with site import/export limits.
    """
    time_step_hours = pd.to_timedelta(df.index.freq).total_seconds() / 3600
    df_tmp = df[["solar_power", "load_power", "bess_max_power_charge", "bess_max_power_discharge"]].copy()
    df_tmp["solar_power"] = -df_tmp["solar_power"]
    df_tmp["bess_max_power_discharge"] = -df_tmp["bess_max_power_discharge"]
    df_tmp["bess_power"] = df["energy_delta"] / time_step_hours
    df_tmp["solar_to_grid_power"] = -df["solar_to_grid"] / time_step_hours
    df_tmp["grid_to_load_power"] = df["grid_to_load"] / time_step_hours
    df_tmp["grid_net"] = df_tmp["load_power"] + df_tmp["solar_power"] + df_tmp["bess_power"]

    fig = px.line(df_tmp, line_shape='hv')
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["grid_connection_import_limit_power"],
            mode='lines',
            line=dict(dash='dot'),
            name='Microgrid import limit'
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=-df["grid_connection_export_limit_power"],
            mode='lines',
            line=dict(dash='dot'),
            name='Microgrid export limit'
        )
    )

    fig.add_hline(y=battery_nameplate_power, line_dash="dot", annotation_text="Battery nameplate charge power")
    fig.add_hline(y=-battery_nameplate_power, line_dash="dot", annotation_text="Battery nameplate discharge power")
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
