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
    output_df["m:battCharge"] = df["bess_charge"]
    output_df["m:battDischarge"] = df["bess_discharge"]
    # output_df["c:battLosses"] = df["bess_losses"]
    output_df["c:limitMaxBattCharge"] = df["bess_max_charge"]
    output_df["c:limitMaxBattDischarge"] = df["bess_max_discharge"]

    output_df["agd:solar"] = df["solar"]
    output_df["agd:load"] = df["load"]
    output_df["solarToLoad"] = df["solar_to_load"]  # TODO: what to call this as it's match at adg level plus m level
    output_df["loadNotSuppliedBySolar"] = df["load_not_supplied_by_solar"]
    output_df["solarNotSupplyingLoad"] = df["solar_not_supplying_load"]
    output_df["battDischargeToLoad"] = df["bess_discharge_to_load"]
    output_df["battDischargeToGrid"] = df["bess_discharge_to_grid"]
    output_df["battChargeFromSolar"] = df["bess_charge_from_solar"]
    output_df["battChargeFromGrid"] = df["bess_charge_from_grid"]
    output_df["loadFromGrid"] = df["load_from_grid"]
    output_df["solarToGrid"] = df["solar_to_grid"]

    output_df["other:imbalanceVolume.final"] = df["imbalance_volume_final"]
    output_df["other:imbalanceVolume.predicted"] = df["imbalance_volume_predicted"]

    for col in df.columns:
        if col.startswith("rate_final_"):
            flow_name = col.removeprefix("rate_final_")
            output_df[f"rate:{flow_name}.final"] = df[col]
        if col.startswith("rate_predicted_"):
            flow_name = col.removeprefix("rate_predicted_")
            output_df[f"rate:{flow_name}.predicted"] = df[col]

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


