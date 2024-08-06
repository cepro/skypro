import importlib.metadata
import logging
from typing import List, Tuple, Any, Dict

import pandas as pd

from skypro.commands.simulator.config.config_v3 import SimulationCaseV3
from skypro.commands.simulator.config.config_v4 import SimulationCaseV4, OutputSimulation


def with_config_entries(df: pd.DataFrame, entries: List[Tuple[str, Any]]) -> pd.DataFrame:
    for key, value in entries:
        df[key] = ""
        value_str = str(value)
        df.iloc[0, df.columns.get_loc(key)] = value_str

    return df


def save_simulation_output(
        df: pd.DataFrame,
        final_rates_dfs: Dict[str, pd.DataFrame],
        load_energy_breakdown_df: pd.DataFrame,
        sim_config: SimulationCaseV3 | SimulationCaseV4,
        output_config: OutputSimulation
):
    """
    Saves a detailed timeseries of the simulation to CSV file.
    The `case_config` is used to retrieve some configuration which is also saved into the CSV file
    """

    # TODO: this could do with a bit of a refactor to make it cleaner, but waiting on finalized column naming from Damon

    output_df = pd.DataFrame(index=df.index)

    output_df["clocktime"] = df.index.tz_convert("Europe/London")  # TODO: check this

    output_df["agd:load"] = df["load"]
    for col in load_energy_breakdown_df.columns:
        output_df[f"agd:load.{col}"] = load_energy_breakdown_df[col]

    output_df["agd:solar"] = df["solar"]

    # For now, assume that all the match happens at the property level, and none happens at the microgrid level
    output_df["agd:match"] = df["solar_to_load"]

    output_df["m:battSoe"] = df["soe"]
    output_df["m:battCharge"] = df["bess_charge"]
    output_df["m:battDischarge"] = df["bess_discharge"]
    output_df["m:match"] = 0.0  # Assume that no solar->load matching happens at the microgrid level

    output_df["c:battLosses"] = df["bess_losses"]
    output_df["c:limitMaxBattCharge"] = df["bess_max_charge"]
    output_df["c:limitMaxBattDischarge"] = df["bess_max_discharge"]

    # Maps the column names used internally in Skypro to the column names we want in outputs
    output_flow_name_map = {
        "solar_to_grid": "solarToGrid",
        "load_from_grid": "gridToLoad",
        "solar_to_load": "solarToLoad",
        "bess_discharge_to_load": "battToLoad",
        "bess_discharge_to_grid": "battToGrid",
        "bess_charge_from_solar": "solarToBatt",
        "bess_charge_from_grid": "gridToBatt"
    }

    for internal_flow_name, csv_flow_name in output_flow_name_map.items():
        output_df[f"c:{csv_flow_name}"] = df[internal_flow_name]

    # Put the total predicted/final rate for each energy flow into the CSV
    for col in df.columns:
        if col.startswith("rate_final_"):
            internal_flow_name = col.removeprefix("rate_final_")
            csv_flow_name = output_flow_name_map[internal_flow_name]
            output_df[f"rate:{csv_flow_name}.final"] = df[col]
        if col.startswith("rate_predicted_"):
            internal_flow_name = col.removeprefix("rate_predicted_")
            csv_flow_name = output_flow_name_map[internal_flow_name]
            output_df[f"rate:{csv_flow_name}.predicted"] = df[col]

    # If the command-line option for detailed rate info is specified then put each individual rate in the CSV
    if output_config.rate_detail:

        rates_of_interest = []
        if output_config.rate_detail != "all":
            # parse the string like a comma-seperated list of rates to include detail for
            rates_of_interest = output_config.rate_detail.split(",")

        for internal_flow_name, rates_df in final_rates_dfs.items():
            csv_flow_name = output_flow_name_map[internal_flow_name]
            for col in rates_df.columns:
                if output_config.rate_detail == "all" or col in rates_of_interest:
                    output_df[f"rate:{csv_flow_name}.{col}.final"] = rates_df[col]

    output_df["other:imbalanceVolume.final"] = df["imbalance_volume_final"]
    output_df["other:imbalanceVolume.predicted"] = df["imbalance_volume_predicted"]

    if output_config.aggregate:
        if output_config.aggregate == "30min":

            # Here we need to define how each column in the output_df is aggregated over time. The order of this
            # dictionary defines the order of the output.
            agg_rules = {
                "clocktime": "first",
                "agd:load": "sum"
            }

            for col in output_df.columns:
                if col.startswith("agd:load."):
                    agg_rules[col] = "sum"

            agg_rules = agg_rules | {
                "agd:solar": "sum",
                "agd:match": "sum",

                "m:battSoe": "first",
                "m:battCharge": "sum",
                "m:battDischarge": "sum",
                "m:match": "sum",

                "c:battLosses": "sum",
                "c:limitMaxBattCharge": "sum",  # the limits are in kWh over the period
                "c:limitMaxBattDischarge": "sum",
            }

            for _, csv_flow_name in output_flow_name_map.items():
                agg_rules[f"c:{csv_flow_name}"] = "sum"

            for col in output_df.columns:
                if col.startswith("rate:") and col.endswith(".final"):
                    agg_rules[col] = first_ensure_equal
                # There doesn't seem to be a sensible way to aggregate *predicted* rates or volumes, so leave them out

            agg_rules = agg_rules | {
                "other:imbalanceVolume.final": first_ensure_equal,
            }

            # Do the actual aggregation
            output_df = output_df.resample("30min").agg(agg_rules)
        else:
            raise ValueError(f"Unknown aggregate option: '{output_config.aggregate}'")

    # Add the configuration input to the output file - this is stretching the use of the CSV format a bit, but it means
    # that there is a single output file with full traceability as to all the inputs.
    output_df = with_config_entries(
        df=output_df,
        entries=[
            ("skypro.version", importlib.metadata.version('skypro')),
            ("start", sim_config.start.isoformat()),
            ("end", sim_config.end.isoformat()),
            ("site.gridConnection.importLimit", sim_config.site.grid_connection.import_limit),
            ("site.gridConnection.exportLimit", sim_config.site.grid_connection.export_limit),
            ("site.solar.constant", sim_config.site.solar.constant),
            ("site.solar.profile", sim_config.site.solar.profile),
            ("site.load.constant", sim_config.site.load.constant),
            ("site.load.profile", sim_config.site.load.profile),
            ("site.bess.energyCapacity", sim_config.site.bess.energy_capacity),
            ("site.bess.nameplatePower", sim_config.site.bess.nameplate_power),
            ("site.bess.chargeEfficiency", sim_config.site.bess.charge_efficiency),
            ("strategy.priceCurveAlgo", sim_config.strategy.price_curve_algo),
            ("imbalanceDataSource", sim_config.imbalance_data_source),
            ("rates", sim_config.rates),
        ]
    )

    logging.info(f"Saving output to {output_config.csv}...")

    output_df.to_csv(
        output_config.csv,
        index_label="utctime"
    )


def first_ensure_equal(series: pd.Series):
    """
    Aggregation function that returns the first element in the series, but also ensures that all elements
    of the series are equal.
    """
    if (series.iloc[0] == series).all():
        return series.iloc[0]
    else:
        raise ValueError("Not all elements of series are equal")
