import pandas as pd
import plotly.express as px


def plot_load_and_solar(df: pd.DataFrame):
    px.line(df[[
        "plot_load_feeder1",
        "plot_load_feeder2",
        "feeder1_net",
        "feeder2_net",
        "solar_feeder1",
        "solar_feeder2",
    ]]).show()


def plot_cycle_rate(df):

    bess_daily_discharge = pd.DataFrame()
    bess_daily_discharge["discharge"] = df["bess_discharge"].resample("1d").sum()
    bess_daily_discharge["cycle_rate"] = bess_daily_discharge["discharge"] / 1280
    fig_cycle_rate = px.line(bess_daily_discharge["cycle_rate"])
    fig_cycle_rate.add_hline(bess_daily_discharge["cycle_rate"].mean())
    fig_cycle_rate.show()
