from dataclasses import dataclass
from datetime import timedelta
from typing import List, Tuple, Dict

import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from simt_common.analysis.costs_by_categories import plot_costs_by_categories
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
    time_step_hours = (pd.to_timedelta(df.index.freq).total_seconds() / 3600)

    if do_plots:
        plot_hh_strategy(df)

    cols_of_interest = [
        "energy_delta",
        "rate_final_bess_charge_from_solar",
        "rate_final_bess_charge_from_grid",
        "rate_final_bess_discharge_to_load",
        "rate_final_bess_discharge_to_grid",
        "rate_final_solar_to_grid",
        "rate_final_load_from_grid",
    ]

    df["bess_discharge"] = -df["energy_delta"][df["energy_delta"] < 0]
    df["bess_discharge"] = df["bess_discharge"].fillna(0)
    df["bess_charge"] = df["energy_delta"][df["energy_delta"] > 0]
    df["bess_charge"] = df["bess_charge"].fillna(0)

    # Calculate load and solar energies from the power
    df["solar"] = df["solar_power"] * time_step_hours
    df["load"] = df["load_power"] * time_step_hours

    df["solar_to_load"] = df[["solar", "load"]].min(axis=1)
    df["solar_to_grid"] = df["solar"] - df["solar_to_load"]  # TODO: this is wrong?

    df["load_not_supplied_by_solar"] = df["load"] - df["solar_to_load"]

    df["bess_discharge_to_load"] = df[["bess_discharge", "load_not_supplied_by_solar"]].min(axis=1)
    df["bess_discharge_to_grid"] = df["bess_discharge"] - df["bess_discharge_to_load"]

    df["bess_charge_from_solar"] = df[["bess_charge", "solar"]].min(axis=1)
    df["bess_charge_from_grid"] = df["bess_charge"] - df["bess_charge_from_solar"]

    df["load_from_grid"] = df["load"] - df["solar_to_load"] - df["bess_discharge_to_load"] # TODO: this is wrong?

    df["solar_n"] = df["solar"] * -1
    df["bess_discharge_to_load_n"] = df["bess_discharge_to_load"] * -1
    df["bess_discharge_to_grid_n"] = df["bess_discharge_to_grid"] * -1
    df["bess_discharge_n"] = df["bess_discharge"] * -1

    costs_dfs = breakdown_costs(rates_dfs=final_rates_dfs, df=df)

    mg_total_cost_bess_charge = costs_dfs["bess_charge"].sum().sum()
    mg_total_cost_bess_discharge = costs_dfs["bess_discharge"].sum().sum()
    mg_total_bess_gain = - mg_total_cost_bess_discharge - mg_total_cost_bess_charge


    # Calculate the actual prices achieved (using pricing data from after the end of the SP)
    df_for_avg_prices = df[[
        "bess_charge", "bess_discharge", "rate_final_bess_charge_from_grid", "rate_final_bess_discharge_to_grid"
    ]]
    df_for_avg_prices_no_nan = df_for_avg_prices.dropna()
    report_dropped_rows(df_for_avg_prices, df_for_avg_prices_no_nan, "Final prices for average")
    avg_import_price_final = np.average(
        a=df_for_avg_prices_no_nan["rate_final_bess_charge_from_grid"],
        weights=df_for_avg_prices_no_nan["bess_charge"]
    )
    avg_export_price_final = np.average(
        a=df_for_avg_prices_no_nan["rate_final_bess_discharge_to_grid"],
        weights=df_for_avg_prices_no_nan["bess_discharge"]
    )
    print("\n- - AVERAGE PRICES - - ")
    # print(f"10m expected average import price: {avg_import_price_10m:.2f} p/kW")
    # print(f"10m Expected average export price: {avg_export_price_10m:.2f} p/kW")
    print(f"Final average import price: {avg_import_price_final:.2f} p/kW")
    print(f"Final average export price: {avg_export_price_final:.2f} p/kW")

    # Cycling
    total_export = df["bess_discharge"].sum()
    total_cycles = total_export / battery_energy_capacity
    sim_start = df.iloc[0].name
    sim_end = df.iloc[-1].name
    sim_days = get_24hr_days(sim_end - sim_start)
    cycles_per_day = total_cycles / sim_days
    print("\n- - CYCLING - - ")
    print(f"Total energy discharge over simulation: {total_export:.2f} kWh")
    print(f"Total cycles over simulation: {total_cycles:.2f} cycles")
    print(f"Average cycles per day: {cycles_per_day:.2f} cycles/day")

    # TODO: print warning if cycling is low - charge efficiency changes

    # £/kW and £/kWh benchmarks
    df["cost_charge"] = df["rate_final_bess_charge_from_grid"] * df["bess_charge"]
    df["cost_discharge"] = - df["rate_final_bess_discharge_to_grid"] * df["bess_discharge"]
    df["cost"] = df["cost_charge"] + df["cost_discharge"]
    total_import_cost = df["cost_charge"].sum() / 100  # convert p to £
    total_export_cost = df["cost_discharge"].sum() / 100  # convert p to £
    total_gain = total_export_cost - total_import_cost
    average_gain_per_day = total_gain / sim_days
    annualized_per_kwh = (average_gain_per_day * 365) / battery_energy_capacity
    annualized_per_kw_nameplate = (average_gain_per_day * 365) / battery_nameplate_power
    # annualized_per_kw_usable = (average_gain_per_day * 365) / ((battery_charge_limit + battery_discharge_limit)/2)
    print("\n- - BENCHMARKING - - ")
    print(f"Total import cost over simulation: £{total_import_cost:.2f}")
    print(f"Total export revenue over simulation: £{total_export_cost:.2f}")
    print(f"Total gain over simulation: £{total_gain:.2f}")
    print(f"Average gain per day: £{average_gain_per_day:.2f}")
    print(f"Annualised per kWh: £{annualized_per_kwh:.2f} £/kWh")
    # print(f"Annualised per kW usable: £{annualized_per_kw_usable:.2f} £/kW")
    print(f"Annualised per kW nameplate: £{annualized_per_kw_nameplate:.2f} £/kW")

    # print("\n- - WITH MICROGRID EFFECTS - - ")
    # additional_import_cost = mg["additional_imports_cost"].sum() / 100
    # avoided_export_cost = -mg["avoided_exports_cost"].sum() / 100
    # additional_export_cost = mg["additional_exports_cost"].sum() / 100
    # avoided_import_cost = -mg["avoided_imports_cost"].sum() / 100
    # total_charge_cost = additional_import_cost + avoided_export_cost
    # total_discharge_cost = additional_export_cost + avoided_import_cost
    #
    # print(f"Total BESS charge cost over simulation: £{total_charge_cost:.2f}")
    # print(f"  From additional imports: £{additional_import_cost:.2f}")
    # print(f"  From avoided exports: £{avoided_export_cost:.2f}")
    # print(f"Total BESS discharge cost over simulation: £{total_discharge_cost:.2f}")
    # print(f"  From additional exports: £{additional_export_cost:.2f}")
    # print(f"  From avoided imports: £{avoided_import_cost:.2f}")
    # mg_total_gain = total_discharge_cost - total_charge_cost
    # mg_average_gain_per_day = mg_total_gain / sim_days
    # print(f"Total gain over simulation: £{mg_total_gain:.2f}")
    # print(f"Average gain per day: £{mg_average_gain_per_day:.2f}")


    # Plot cumulative profit
    if do_plots:
        px.line(-df["cost"].cumsum()).show()

    # imports["cost"] = imports["rate_import_final"] * imports["energy_delta"]
    # exports["cost"] = exports["rate_export_final"] * exports["energy_delta"]

    # TODO: support this with new microgrid flows
    # if do_plots:
    #     import_rates_df = df[[col for col in df if col.startswith('rate_import_final_')]]
    #     export_rates_df = df[[col for col in df if col.startswith('rate_export_final_')]]
    #
    #     import_charges_df = import_rates_df.mul(imports_with_final_prices["energy_delta"], axis=0)
    #     export_charges_df = export_rates_df.mul(exports_with_final_prices["energy_delta"], axis=0)
    #     plot_costs_by_categories(
    #         import_rates=import_rates,
    #         export_rates=export_rates,
    #         import_charges_df=import_charges_df,
    #         export_charges_df=export_charges_df,
    #         import_col_name_prefix="rate_import_final_",
    #         export_col_name_prefix="rate_export_final_"
    #     ).show()

    # Plot bess charge / discharge limits
    if do_plots:
        time_step_hours = pd.to_timedelta(df.index.freq).total_seconds() / 3600
        df_tmp = df[["solar_power", "load_power", "bess_max_power_charge", "bess_max_power_discharge"]].copy()
        df_tmp["solar_power"] = -df_tmp["solar_power"]
        df_tmp["bess_max_power_discharge"] = -df_tmp["bess_max_power_discharge"]
        df_tmp["bess_power"] = df["energy_delta"] / time_step_hours
        fig = px.line(df_tmp, line_shape='hv')
        fig.add_hline(y=site_import_limit, line_dash="dot", annotation_text="Site import limit")
        fig.add_hline(y=-site_export_limit, line_dash="dot", annotation_text="Site export limit")
        fig.add_hline(y=battery_nameplate_power, line_dash="dot", annotation_text="Battery nameplate power")
        fig.add_hline(y=-battery_nameplate_power, line_dash="dot", annotation_text="Battery nameplate power")
        fig.show()


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
    # fig.add_trace(go.Scatter(x=df.index, y=df["imbalance_volume_final"], name="Imbalance volume final", line=dict(color="red")), secondary_y=True)
    fig.add_trace(go.Scatter(x=df.index, y=df["soe"], name="Battery SoE", line=dict(color="orange")),
                  secondary_y=True)

    fig.update_yaxes(title_text="Price (p/kW)", range=[-10, 40], secondary_y=False, row=1, col=1)
    fig.update_yaxes(title_text="SoE (kWh)", range=[0, 200], secondary_y=True, row=1, col=1)
    fig.update_layout(title="Typical optimisation strategy")
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
