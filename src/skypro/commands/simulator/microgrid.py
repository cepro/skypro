import pandas as pd


def calculate_microgrid_flows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the individual flows of energy around the microgrid, for a simulation, and adds them to a copy of the dataframe which is returned.
    """
    df = df.copy()

    # BESS discharges are whenever the energy flow is negative
    df["bess_discharge"] = -df["energy_delta"][df["energy_delta"] < 0]
    df["bess_discharge"] = df["bess_discharge"].fillna(0)

    # BESS charges are whenever the energy flow is negative
    df["bess_charge"] = df["energy_delta"][df["energy_delta"] > 0]
    df["bess_charge"] = df["bess_charge"].fillna(0)

    # Calculate load and solar energies from the power
    df["solar_to_load"] = df[["solar", "load"]].min(axis=1)
    df["load_not_supplied_by_solar"] = df["load"] - df["solar_to_load"]
    df["solar_not_supplying_load"] = df["solar"] - df["solar_to_load"]

    df["batt_to_load"] = df[["bess_discharge", "load_not_supplied_by_solar"]].min(axis=1)
    df["batt_to_grid"] = df["bess_discharge"] - df["batt_to_load"]

    df["solar_to_batt"] = df[["bess_charge", "solar_not_supplying_load"]].min(axis=1)
    df["grid_to_batt"] = df["bess_charge"] - df["solar_to_batt"]

    df["grid_to_load"] = df["load_not_supplied_by_solar"] - df["batt_to_load"]
    df["solar_to_grid"] = df["solar_not_supplying_load"] - df["solar_to_batt"]

    # For now, assume that all the 'solar matching' happens at the property level, and none happens at the microgrid level
    df["solar_to_load_property_level"] = df["solar_to_load"]
    df["solar_to_load_microgrid_level"] = 0.0

    # The microgrid boundary flows are calculated here from the individual flows. These are needed for reporting in CSV
    # output files, although they aren't used directly in Skypro at the moment.
    df["grid_import"] = df["grid_to_batt"] + df["grid_to_load"]
    df["grid_export"] = df["batt_to_grid"] + df["solar_to_grid"]

    return df
