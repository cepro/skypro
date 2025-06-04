import pandas as pd
import plotly.graph_objects as go

DAYS_IN_MS = 24 * 60 * 60 * 1000


def plot_daily_gains(costs_dfs):

    # Plot bar chart of import vs export costs daily
    bess_daily_summed_charges = pd.DataFrame()
    bess_daily_summed_charges["charge"] = costs_dfs["bess_charge"].resample("1d").sum().sum(axis=1)
    bess_daily_summed_charges["discharge"] = costs_dfs["bess_discharge"].resample("1d").sum().sum(axis=1) * -1
    bess_daily_summed_charges["gain"] = bess_daily_summed_charges["discharge"] - bess_daily_summed_charges["charge"]
    bess_daily_summed_charges = bess_daily_summed_charges / 100  # Convert pence to £

    # Plot a chart of import and export charges usage over the time span
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=bess_daily_summed_charges.index,
        y=bess_daily_summed_charges["charge"],
        name="Daily charge costs"  # this sets its legend entry
    ))
    fig2.add_trace(go.Bar(
        x=bess_daily_summed_charges.index,
        y=bess_daily_summed_charges["discharge"],
        name="Daily discharge revenue"  # this sets its legend entry
    ))
    fig2.add_trace(go.Bar(
        x=bess_daily_summed_charges.index,
        y=bess_daily_summed_charges["gain"],
        name="Daily gain"  # this sets its legend entry
    ))
    fig2.update_layout(
        title="Daily Outcome",
        xaxis_title="Day",
        yaxis_title="£",
    )
    fig2.update_xaxes(dtick=DAYS_IN_MS)
    fig2.show()
