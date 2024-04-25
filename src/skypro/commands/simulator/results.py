from typing import List

import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from simt_common.analysis.costs_by_categories import plot_costs_by_categories
from skypro.commands.simulator.algorithm import get_days


def explore_results(
        df: pd.DataFrame,
        battery_energy_capacity: float,
        battery_nameplate_power: float,
        site_import_limit: float,
        site_export_limit: float,
        import_rates: List,
        export_rates: List,
):
    plot_hh_strategy(df)

    """
    Generally explores/plots the results, including logging the weighted average prices, cycling statistics, and
    benchmark £/kW and £/kWh values for the simulation.
    """
    cols_of_interest = [
        "energy_delta",
        "imbalance_price_10m",
        "imbalance_price_20m",
        "imbalance_price_final",
        "rate_import_10m",
        "rate_import_20m",
        "rate_import_final",
        "rate_export_10m",
        "rate_export_20m",
        "rate_export_final",
    ]
    imports_inc_nan = df[df["energy_delta"] > 0][cols_of_interest]
    exports_inc_nan = df[df["energy_delta"] < 0][cols_of_interest]

    # Calculate the expected prices when 10m into the SP
    imports_with_10m_prices = imports_inc_nan[["energy_delta", "rate_import_10m"]].dropna().copy()
    exports_with_10m_prices = exports_inc_nan[["energy_delta", "rate_export_10m"]].dropna().copy()
    report_dropped_rows(imports_inc_nan, imports_with_10m_prices, "imports for 10m expected average price")
    report_dropped_rows(exports_inc_nan, exports_with_10m_prices, "exports for 10m expected average price")
    avg_import_price_10m = np.average(a=imports_with_10m_prices["rate_import_10m"], weights=imports_with_10m_prices["energy_delta"])
    avg_export_price_10m = np.average(a=exports_with_10m_prices["rate_export_10m"], weights=exports_with_10m_prices["energy_delta"])

    # Calculate the actual prices achieved (using pricing data from after the end of the SP)
    imports_with_final_prices = imports_inc_nan[["energy_delta", "rate_import_final"]].dropna().copy()
    exports_with_final_prices = exports_inc_nan[["energy_delta", "rate_export_final"]].dropna().copy()
    report_dropped_rows(imports_inc_nan, imports_with_final_prices, "imports final price")
    report_dropped_rows(exports_inc_nan, exports_with_final_prices, "exports final price")
    avg_import_price_final = np.average(a=imports_with_final_prices["rate_import_final"], weights=imports_with_final_prices["energy_delta"])
    avg_export_price_final = np.average(a=exports_with_final_prices["rate_export_final"], weights=exports_with_final_prices["energy_delta"])

    print("\n- - AVERAGE PRICES - - ")
    print(f"10m expected average import price: {avg_import_price_10m:.2f} p/kW")
    print(f"10m Expected average export price: {avg_export_price_10m:.2f} p/kW")
    print(f"Final average import price: {avg_import_price_final:.2f} p/kW")
    print(f"Final average export price: {avg_export_price_final:.2f} p/kW")

    # Cycling
    total_export = -exports_inc_nan["energy_delta"].sum()
    total_cycles = total_export / battery_energy_capacity
    sim_start = df.iloc[0].name
    sim_end = df.iloc[-1].name
    sim_days = get_days(sim_end - sim_start)
    cycles_per_day = total_cycles / sim_days
    print("\n- - CYCLING - - ")
    print(f"Total energy discharge over simulation: {total_export:.2f} kWh")
    print(f"Total cycles over simulation: {total_cycles:.2f} cycles")
    print(f"Average cycles per day: {cycles_per_day:.2f} cycles/day")

    # TODO: print warning if cycling is low - charge efficiency changes

    # £/kW and £/kWh benchmarks
    imports_with_final_prices["cost"] = imports_with_final_prices["rate_import_final"] * imports_with_final_prices["energy_delta"]
    exports_with_final_prices["cost"] = exports_with_final_prices["rate_export_final"] * exports_with_final_prices["energy_delta"]
    total_import_cost = imports_with_final_prices["cost"].sum() / 100  # convert p to £
    total_export_cost = exports_with_final_prices["cost"].sum() / 100  # convert p to £
    total_gain = -total_export_cost - total_import_cost
    average_gain_per_day = total_gain / sim_days
    annualized_per_kwh = (average_gain_per_day * 365) / battery_energy_capacity
    annualized_per_kw_nameplate = (average_gain_per_day * 365) / battery_nameplate_power
    # annualized_per_kw_usable = (average_gain_per_day * 365) / ((battery_charge_limit + battery_discharge_limit)/2)
    print("\n- - BENCHMARKING - - ")
    print(f"Total import cost over simulation: £{total_import_cost:.2f}")
    print(f"Total export revenue over simulation: £{-total_export_cost:.2f}")
    print(f"Total gain over simulation: £{total_gain:.2f}")
    print(f"Average gain per day: £{average_gain_per_day:.2f}")
    print(f"Annualised per kWh: £{annualized_per_kwh:.2f} £/kWh")
    # print(f"Annualised per kW usable: £{annualized_per_kw_usable:.2f} £/kW")
    print(f"Annualised per kW nameplate: £{annualized_per_kw_nameplate:.2f} £/kW")

    import_rates_df = df[[col for col in df if col.startswith('rate_import_final_')]]
    export_rates_df = df[[col for col in df if col.startswith('rate_export_final_')]]

    import_charges_df = import_rates_df.mul(imports_with_final_prices["energy_delta"], axis=0)
    export_charges_df = export_rates_df.mul(exports_with_final_prices["energy_delta"], axis=0)

    # Plot cummulative profit
    total_import_charges = import_charges_df.sum(axis=1)
    total_export_charges = export_charges_df.sum(axis=1)
    total_charges = total_export_charges - total_import_charges
    px.line(total_charges.cumsum()).show()

    # imports["cost"] = imports["rate_import_final"] * imports["energy_delta"]
    # exports["cost"] = exports["rate_export_final"] * exports["energy_delta"]

    plot_costs_by_categories(
        import_rates=import_rates,
        export_rates=export_rates,
        import_charges_df=import_charges_df,
        export_charges_df=export_charges_df,
        import_col_name_prefix="rate_import_final_",
        export_col_name_prefix="rate_export_final_"
    ).show()

    # Plot bess charge / discharge limits
    df["battery_charge_power"] = imports_inc_nan["energy_delta"] * 2
    df["battery_discharge_power"] = exports_inc_nan["energy_delta"] * -2
    fig = px.line(df[["solar", "load", "battery_max_power_charge", "battery_max_power_discharge", "battery_charge_power", "battery_discharge_power"]])
    fig.add_hline(y=site_import_limit, line_dash="dot", annotation_text="Site import limit")
    fig.add_hline(y=site_export_limit, line_dash="dot", annotation_text="Site export limit")
    fig.add_hline(y=battery_nameplate_power, line_dash="dot", annotation_text="Battery nameplate power")
    fig.show()


def plot_hh_strategy(df: pd.DataFrame):

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # fig.add_trace(go.Scatter(x=df.index, y=df["rate_import_10m"], name="Import Price 10m (SSP plus DUoS)", line=dict(color="rgba(89, 237, 131, 1)")))
    # fig.add_trace(go.Scatter(x=df.index, y=df["rate_import_20m"], name="Import Price 20m (SSP plus DUoS)", line=dict(color="rgba(40, 189, 82, 1)")))
    fig.add_trace(
        go.Scatter(x=df.index, y=df["rate_import_final"], name="Import Price", line=dict(color="rgba(0, 141, 40, 1)")))
    # fig.add_trace(go.Scatter(x=df.index, y=df["rate_export_10m"], name="Export Price 10m (SSP plus DUoS)", line=dict(color="rgba(185, 102, 247, 1)")))
    # fig.add_trace(go.Scatter(x=df.index, y=df["rate_export_20m"], name="Export Price 20m (SSP plus DUoS)", line=dict(color="rgba(153, 59, 224, 1)")))
    fig.add_trace(
        go.Scatter(x=df.index, y=df["rate_export_final"], name="Export Price", line=dict(color="rgba(102, 0, 178, 1)")))
    # fig.add_trace(go.Scatter(x=df.index, y=df["imbalance_volume_final"], name="Imbalance volume final", line=dict(color="red")), secondary_y=True)
    fig.add_trace(go.Scatter(x=df.index, y=df["soe"], name="Battery charge", line=dict(color="orange")),
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
