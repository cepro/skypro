import logging

import pandas as pd


def save_output(df: pd.DataFrame, output_file_path: str):

    output_df = pd.DataFrame(index=df.index)

    output_df["clocktime"] = df.index.tz_convert("Europe/London")  # TODO: check this

    output_df["m:battSoe"] = df["soe"]
    output_df["m:battCharge"] = df[df["energy_delta"] > 0]["energy_delta"]
    output_df["m:battCharge"] = output_df["m:battCharge"].fillna(0)
    output_df["m:battDischarge"] = df[df["energy_delta"] < 0]["energy_delta"] * -1
    output_df["m:battDischarge"] = output_df["m:battDischarge"].fillna(0)

    # Convert kW to kWh in a HH
    output_df["c:limitMaxBattCharge"] = df["battery_max_power_charge"] / 2.0
    output_df["c:limitMaxBattDischarge"] = df["battery_max_power_discharge"] / 2.0

    output_df["other:imbalanceVolume.final"] = df["imbalance_volume_final"]

    for col in df.columns:
        if col.startswith("rate_import_final_"):
            rateName = col.removeprefix("rate_import_final_")
            output_df[f"rate:import.final.{rateName}"] = df[col]
        if col.startswith("rate_export_final_"):
            rateName = col.removeprefix("rate_export_final_")
            output_df[f"rate:export.final.{rateName}"] = df[col]

    output_df["rate:import.final"] = df["rate_import_final"]
    output_df["rate:export.final"] = df["rate_export_final"]

    logging.info(f"Saving output to {output_file_path}...")

    breakpoint()

    output_df.to_csv(
        output_file_path,
        index_label="utctime"
    )