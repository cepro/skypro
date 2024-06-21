from datetime import timedelta
from typing import Dict

import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from simt_common.analysis.costs_by_group import plot_costs_by_grouping
from simt_common.analysis.daily_gains import plot_daily_gains
from simt_common.rates.microgrid import breakdown_costs


def explore_results(
        df: pd.DataFrame,
        final_rates_dfs: Dict[str, pd.DataFrame],
        do_plots: bool,
        battery_energy_capacity: float,
        battery_nameplate_power: float,
        site_import_limit: float,
        site_export_limit: float,
):
    """
    Generally explores/plots the results, including logging the weighted average prices, cycling statistics, and
    benchmark £/kW and £/kWh values for the simulation.
    """

    df = df.copy()
    sim_start = df.iloc[0].name
    sim_end = df.iloc[-1].name
    sim_days = get_24hr_days(sim_end - sim_start)

    df["solar_n"] = df["solar"] * -1
    df["bess_discharge_to_load_n"] = df["bess_discharge_to_load"] * -1
    df["bess_discharge_to_grid_n"] = df["bess_discharge_to_grid"] * -1
    df["bess_discharge_n"] = df["bess_discharge"] * -1

    costs_dfs = breakdown_costs(rates_dfs=final_rates_dfs, df=df)

    total_bess_charged_1 = df["bess_charge"].sum()
    total_bess_charge_from_grid = df['bess_charge_from_grid'].sum()
    total_bess_charge_from_solar = df['bess_charge_from_solar'].sum()
    total_bess_discharged_1 = df["bess_discharge"].sum()
    total_bess_discharge_to_grid = df['bess_discharge_to_grid'].sum()
    total_bess_discharge_to_load = df['bess_discharge_to_load'].sum()

    # Calculate total costs of the BESS charging and discharging
    total_cost_bess_charge = costs_dfs["bess_charge"].sum().sum()
    total_cost_bess_discharge = costs_dfs["bess_discharge"].sum().sum()
    total_bess_gain = - total_cost_bess_discharge - total_cost_bess_charge
    total_cost_bess_charge_from_grid = costs_dfs["bess_charge_from_grid"].sum().sum()
    total_cost_bess_charge_from_solar = costs_dfs["bess_charge_from_solar_inc_opp"].sum().sum()
    total_cost_bess_discharge_to_grid = costs_dfs["bess_discharge_to_grid"].sum().sum()
    total_cost_bess_discharge_to_load = costs_dfs["bess_discharge_to_load_inc_opp"].sum().sum()

    # Calculate the summaries/costs of non-BESS microgrid imports/exports
    total_cost_solar_to_grid = costs_dfs["solar_to_grid"].sum().sum()
    total_cost_load_from_grid = costs_dfs["load_from_grid"].sum().sum()
    total_solar_to_grid = df["solar_to_grid"].sum()
    total_load_from_grid = df["load_from_grid"].sum()

    # Calculate the average p/kWh rates associated with the various energy flows
    avg_rate_bess_charge = total_cost_bess_charge / total_bess_charged_1
    avg_rate_bess_discharge = total_cost_bess_discharge / total_bess_discharged_1
    avg_rate_bess_charge_from_grid = total_cost_bess_charge_from_grid / total_bess_charge_from_grid
    avg_rate_bess_charge_from_solar = total_cost_bess_charge_from_solar / total_bess_charge_from_solar
    avg_rate_bess_discharge_to_grid = total_cost_bess_discharge_to_grid / total_bess_discharge_to_grid
    avg_rate_bess_discharge_to_load = total_cost_bess_discharge_to_load / total_bess_discharge_to_load

    avg_rate_solar_to_grid = total_cost_solar_to_grid / total_solar_to_grid
    avg_rate_load_from_grid = total_cost_load_from_grid / total_load_from_grid

    print("")
    print(f"Total BESS charge: {total_bess_charged_1:.1f} kWh, £{total_cost_bess_charge/100:.2f}, {avg_rate_bess_charge:.2f} p/kWh")
    print(f"  Charge from grid: {total_bess_charge_from_grid:.1f} kWh, £{total_cost_bess_charge_from_grid/100:.2f}, {avg_rate_bess_charge_from_grid:.2f} p/kWh")
    print(f"  Charge from solar: {total_bess_charge_from_solar:.1f} kWh, £{total_cost_bess_charge_from_solar/100:.2f}, {avg_rate_bess_charge_from_solar:.2f} p/kWh")
    print(f"Total BESS discharge: {total_bess_discharged_1:.1f} kWh, £{total_cost_bess_discharge/100:.2f}, {avg_rate_bess_discharge:.2f} p/kWh")
    print(f"  Discharge to grid: {total_bess_discharge_to_grid:.1f} kWh, £{total_cost_bess_discharge_to_grid/100:.2f}, {avg_rate_bess_discharge_to_grid:.2f} p/kWh")
    print(f"  Discharge to load: {total_bess_discharge_to_load:.1f} kWh, £{total_cost_bess_discharge_to_load/100:.2f}, {avg_rate_bess_discharge_to_load:.2f} p/kWh")

    print("")
    print(f"Surplus solar to grid: {total_solar_to_grid:.1f} kWh, £{total_cost_solar_to_grid / 100:.2f}, {avg_rate_solar_to_grid:.2f} p/kWh")
    print(f"Load from grid: {total_load_from_grid:.1f} kWh, £{total_cost_load_from_grid / 100:.2f}, {avg_rate_load_from_grid:.2f} p/kWh")

    print("")
    print(f"Total BESS gain over period: £{total_bess_gain/100:.2f}")
    print(f"Average daily BESS gain over period: £{(total_bess_gain / 100)/sim_days:.2f}")

    # Cycling
    total_cycles = total_bess_discharged_1 / battery_energy_capacity
    cycles_per_day = total_cycles / sim_days
    print("")
    print("- - CYCLING - - ")
    print(f"Total cycles over simulation: {total_cycles:.2f} cycles")
    print(f"Average cycles per day: {cycles_per_day:.2f} cycles/day")

    # TODO: print warning if cycling is low - charge efficiency changes

    # Plot energy flows with charge / discharge limits
    if do_plots:
        plot_hh_strategy(df)
        plot_microgrid_energy_flows(
            df, site_import_limit, site_export_limit, battery_nameplate_power
        )
        plot_costs_by_grouping(costs_dfs["bess_charge"], costs_dfs["bess_discharge"])
        plot_daily_gains(costs_dfs)


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
    # if "red_approach_distance" in df.columns:
    #     fig.add_trace(
    #         go.Scatter(x=df.index, y=df["red_approach_distance"], name="red_approach_distance", mode="markers", line=dict(color="red")),
    #         secondary_y=True
    #     )
    #
    # if "amber_approach_distance" in df.columns:
    #     fig.add_trace(
    #         go.Scatter(x=df.index, y=df["amber_approach_distance"], name="amber_approach_distance", mode="markers", line=dict(color="orange")),
    #         secondary_y=True
    #     )
    #
    # if "spread_algo_energy" in df.columns:
    #     fig.add_trace(
    #         go.Scatter(x=df.index, y=df["spread_algo_energy"], name="spread_algo_energy", mode="markers", line=dict(color="yellow")),
    #         secondary_y=True
    #     )

    # fig.add_trace(go.Scatter(x=df.index, y=df["imbalance_volume_final"], name="Imbalance volume final", line=dict(color="red")), secondary_y=True)
    fig.add_trace(go.Scatter(x=df.index, y=df["soe"], name="Battery SoE", line=dict(color="orange")),
                  secondary_y=True)

    fig.update_yaxes(title_text="Price (p/kW)", range=[-10, 40], secondary_y=False, row=1, col=1)
    fig.update_yaxes(title_text="SoE (kWh)", range=[0, 200], secondary_y=True, row=1, col=1)
    fig.update_layout(title="Typical optimisation strategy")
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
