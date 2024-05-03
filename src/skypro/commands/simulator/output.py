import importlib.metadata
import logging
from typing import List, Tuple, Any

import pandas as pd

from skypro.commands.simulator.config.config import Config


def with_config_entries(df: pd.DataFrame, entries: List[Tuple[str, Any]]) -> pd.DataFrame:
    for key, value in entries:
        df[key] = ""
        value_str = str(value)
        df.iloc[0, df.columns.get_loc(key)] = value_str

    return df


def save_output(df: pd.DataFrame, config: Config, output_file_path: str):

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

    # Add the configuration input to the output file - this is stretching the use of the CSV format a bit, but it means
    # that there is a single output file with full traceability as to all the inputs.
    output_df = with_config_entries(
        df=output_df,
        entries=[
            ("skypro.version", importlib.metadata.version('skypro')),
            ("start", config.simulation.start.isoformat()),
            ("end", config.simulation.end.isoformat()),
            ("site.gridConnection.importLimit", config.simulation.site.grid_connection.import_limit),
            ("site.gridConnection.exportLimit", config.simulation.site.grid_connection.export_limit),
            ("site.solar.constant", config.simulation.site.solar.constant),
            ("site.solar.profile", config.simulation.site.solar.profile),
            ("site.load.constant", config.simulation.site.load.constant),
            ("site.load.profile", config.simulation.site.load.profile),
            ("site.bess.energyCapacity", config.simulation.site.bess.energy_capacity),
            ("site.bess.nameplatePower", config.simulation.site.bess.nameplate_power),
            ("site.bess.chargeEfficiency", config.simulation.site.bess.charge_efficiency),
            ("strategy.priceCurveAlgo", config.simulation.strategy.price_curve_algo),
            ("imbalanceDataSource", config.simulation.imbalance_data_source),
            ("rates", config.simulation.rates),
        ]
    )

    logging.info(f"Saving output to {output_file_path}...")

    output_df.to_csv(
        output_file_path,
        index_label="utctime"
    )


