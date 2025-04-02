import datetime
import requests
from typing import Optional

import pandas as pd
from simt_common.timeutils.settlement_periods import date_and_sp_num_to_utc_datetime


def pull_modo_api(url: str, modo_api_key: str, params: Optional[dict] = None) -> pd.DataFrame:
    df = pd.DataFrame()
    headers = {"X-Token": modo_api_key}
    while url is not None:
        print("Pulling...")
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            raise ConnectionError(f"{response.status_code}: {response.reason}")
        df = pd.concat([df, pd.DataFrame(response.json()["results"])])
        url = response.json()["next"]
    return df


# Write function to calculate weighted averages
def weighted_average(value: pd.Series, weight: pd.Series) -> float:
    return (value * weight).sum() / weight.sum() if weight.sum() != 0 else 0


# Pull data from the API
url = "https://api.modoenergy.com/pub/v1/gb/modo/markets/detailed-system-price-live"
modo_api_key = "1166e47af8c51fbbda68def9188b4150d2b170a5059ec8dce6f938606a79"
params = {
    "date_from": datetime.date(2025, 3, 1),
    "date_to": datetime.date(2025, 3, 10),
}
# det_sys_price = pull_modo_api(url, modo_api_key, params)
det_sys_price = pd.read_csv("det_sys_price.csv")


def read_mtk_data_csv(file: str) -> pd.DataFrame:
    mkt_df = pd.read_csv(file)
    mkt_df.index = pd.to_datetime(mkt_df["time"], utc=True)
    mkt_df = mkt_df.drop("time", axis=1)
    return mkt_df

outturn_price_df = read_mtk_data_csv("elexon_imbalance_price.csv")
outturn_vol_df = read_mtk_data_csv("elexon_imbalance_volume.csv")

# SO flagged actions do not form part of the imbalance price calc
# We are missing a fiar bit of nuance here, but this gets us a lot closer
det_sys_price = det_sys_price[det_sys_price["so_flag"] == False]


# Calculate volume weighted average prices for bids and offers by fuel type
reference_price_df = (
    det_sys_price
    .groupby(["date", "settlement_period", "boa_type"])
    .apply(lambda x: weighted_average(value=x["price"], weight=x["volume"]), include_groups=False)
    .rename("reference_price")
    .reset_index()
)

reference_price_df["time"] = date_and_sp_num_to_utc_datetime(reference_price_df["date"], reference_price_df["settlement_period"], "Europe/London")
reference_price_df = reference_price_df.set_index("time")
reference_price_df = reference_price_df.drop(["date", "settlement_period"], axis=1)

bids_df = reference_price_df[reference_price_df["boa_type"] == "BID"][["reference_price"]].rename(columns={"reference_price": "bid"})
offer_df = reference_price_df[reference_price_df["boa_type"] == "OFFER"][["reference_price"]].rename(columns={"reference_price": "offer"})

df = pd.concat([bids_df, offer_df], axis=1)

df["bid"] = df["bid"] / 10
df["offer"] = df["offer"] / 10
df["volume"] = outturn_vol_df["value"] / 1000
df.loc[df["volume"] >= 0, "prediction"] = df["offer"]
df.loc[df["volume"] < 0, "prediction"] = df["bid"]
df["outturn"] = outturn_price_df["value"]

import plotly.express as px
px.line(df).show()

df[["bid", "offer"]].to_csv("/Users/marcuswood/Desktop/all/repos/skypro-cli/reference_price.csv")
breakpoint()