import math
from collections import namedtuple
from datetime import timedelta
from typing import Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from skypro.commands.report.config.config import SpreadScenario

DAYS_IN_MS = 24 * 60 * 60 * 1000
BESS_CHARGE_EFFICIENCY = 0.85


def plot_load_and_solar(df: pd.DataFrame):
    px.line(df[[
        "plot_load_feeder1",
        "plot_load_feeder2",
        "feeder1_net",
        "feeder2_net",
        "solar_feeder1",
        "solar_feeder2",
    ]]).show()


def plot_spread_report(int_rates_dfs: pd.DataFrame, scenario_configs: Dict[str, SpreadScenario]):
    """
    Plots a chart showing how much spread is available in the market for different sized batteries
    """
    rates_pkwh_df = pd.DataFrame(index=int_rates_dfs["grid_to_batt"].index)
    rates_pkwh_df["grid_to_batt"] = int_rates_dfs["grid_to_batt"].sum(axis=1)
    rates_pkwh_df["batt_to_grid"] = int_rates_dfs["batt_to_grid"].sum(axis=1)

    # Parse the configuration and extract the "spread scenarios" that we are to analyse
    spread_scenarios = []
    SpreadScenario = namedtuple("SpreadScenario", ["name", "import_duration", "export_duration", "highlight"])
    for name, scenario_config in scenario_configs.items():
        spread_scenarios.append((SpreadScenario(
            name=name,
            import_duration=scenario_config.import_duration,
            export_duration=scenario_config.export_duration,
            highlight=scenario_config.highlight
        )))

    # Work out the best prices that would be available on each day for each scenario
    spread_scenarios_df = pd.DataFrame(index=np.unique(rates_pkwh_df.index.date))
    for name, day_df in rates_pkwh_df.groupby(rates_pkwh_df.index.date):  # Run over each days worth of data
        day_df.index.name = "time"
        import_prices = day_df.sort_values(by=["grid_to_batt", "time"])["grid_to_batt"]
        export_prices = day_df.sort_values(by=["batt_to_grid", "time"])["batt_to_grid"]
        for scenario in spread_scenarios:
            spread_scenarios_df.loc[name, f"{scenario.name}_import"] = calculate_best_average_price(
                duration=timedelta(hours=scenario.import_duration),
                sorted_prices=import_prices,
                prices_timebase=pd.to_timedelta(rates_pkwh_df.index.freq)
            )
            spread_scenarios_df.loc[name, f"{scenario.name}_export"] = calculate_best_average_price(
                duration=timedelta(hours=scenario.export_duration),
                sorted_prices=export_prices,
                prices_timebase=pd.to_timedelta(rates_pkwh_df.index.freq)
            )

    fig = go.Figure()

    for i, scenario in enumerate(spread_scenarios):

        import_col_name = f"{scenario.name}_import"
        export_col_name = f"{scenario.name}_export"

        if scenario.highlight:
            dash = "solid"
        else:
            dash = "dash"

        fig.add_trace(
            go.Scatter(
                x=spread_scenarios_df.index,
                y=spread_scenarios_df[import_col_name],
                name=import_col_name,
                line=dict(color=px.colors.sequential.Blues[-i - 1], dash=dash)),
        )
        fig.add_trace(
            go.Scatter(
                x=spread_scenarios_df.index,
                y=spread_scenarios_df[export_col_name]*-1,
                name=export_col_name,
                line=dict(color=px.colors.sequential.Greens[-i - 1], dash=dash)),
        )

        if scenario.highlight:
            spread = (spread_scenarios_df[export_col_name] * -1) - spread_scenarios_df[import_col_name]
            spread_adj = (spread_scenarios_df[export_col_name]*-1) - (spread_scenarios_df[import_col_name]/BESS_CHARGE_EFFICIENCY)
            # fig.add_trace(
            #     go.Scatter(
            #         x=spread.index,
            #         y=spread,
            #         name="Best-case spread",
            #         line=dict(color="purple", dash="dash")),
            # )
            fig.add_trace(
                go.Scatter(
                    x=spread_adj.index,
                    y=spread_adj,
                    name="Constrained spread, derated by efficiency",
                    line=dict(color="purple", dash="solid")),
            )
            print(f"Average best-case spread at 1 cycle: {spread.mean()}")
            print(f"Average best-case spread at 1 cycle, adjusted for efficiency: {spread_adj.mean()}")

    fig.update_layout(
        title="Perfect hindsight spread at 1.0 cycle per day (without microgrid effects)",
        xaxis_title="Date",
        yaxis_title="Price (p/kWh)",
    )
    fig.update_xaxes(dtick=DAYS_IN_MS)
    fig.show()


def calculate_best_average_price(duration: timedelta, sorted_prices: pd.Series, prices_timebase: timedelta) -> float:
    """
    Calculates and returns the best price that could be achieved over the given duration.
    duration is the duration of the battery - e.g. 2hr
    sorted_prices are the prices for each time slot, sorted with the best prices first
    prices_timebase is the length of each time slot. E.g. if there is a price for every 5 minutes then this is 5mins.
    """

    weights = pd.Series(index=sorted_prices.index, data=0.0)
    total_steps = duration / prices_timebase  # how many time steps/rows for this duration of battery
    for i, step in enumerate(range(1, math.ceil(total_steps) + 1)):
        if step < total_steps:
            # we use steps price for the entire step duration
            step_duration = prices_timebase
        else:
            # we use this half-hours price for less than the step duration:
            step_duration = (total_steps % 1) * prices_timebase
        try:
            weights.iloc[i] = step_duration.total_seconds()
        except IndexError:
            # If we are looking at periods of time where we don't have enough half-hours to do this calculation just
            # bail and return NaN
            return np.nan

    return np.average(sorted_prices, weights=weights)


def plot_cycle_rate(df):

    bess_daily_discharge = pd.DataFrame()
    bess_daily_discharge["discharge"] = df["bess_discharge"].resample("1d").sum()
    bess_daily_discharge["cycle_rate"] = bess_daily_discharge["discharge"] / 1280
    fig_cycle_rate = px.line(bess_daily_discharge["cycle_rate"])
    fig_cycle_rate.add_hline(bess_daily_discharge["cycle_rate"].mean())
    fig_cycle_rate.show()
