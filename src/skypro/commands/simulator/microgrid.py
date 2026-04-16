import pandas as pd


def calculate_microgrid_flows(df: pd.DataFrame, allow_remote_flow_to_site: bool = False) -> pd.DataFrame:
    """
    Calculates the individual flows of energy around the microgrid, for a simulation, and adds them to a copy of the dataframe which is returned.

    If a remote (sleeved) generation asset is configured and `allow_remote_flow_to_site` is True, the remote solar
    is first attributed to any residual load, then to any residual battery charging demand, and whatever is left
    flows directly to the grid. When the flag is False (or no remote site is configured), the remote solar does
    not feed load/battery and is fully attributed to grid export.

    Expects a `remote_solar` column on the input dataframe (zero-filled when no remote site is configured).
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

    # Remote (sleeved) solar attribution:
    # - If flow-to-site is allowed, remote solar first serves any residual onsite load (after onsite solar and
    #   battery discharge have been attributed), then supplies any residual battery charging demand, and whatever
    #   is left flows out to the grid.
    # - If flow-to-site is not allowed, all remote solar is attributed to grid export.
    if allow_remote_flow_to_site and "remote_solar" in df.columns:
        df["load_not_supplied_by_site"] = df["load_not_supplied_by_solar"] - df["batt_to_load"]
        df["remote_solar_to_load"] = df[["load_not_supplied_by_site", "remote_solar"]].min(axis=1)
        df["remote_solar_not_supplying_load"] = df["remote_solar"] - df["remote_solar_to_load"]
        df["batt_not_supplied_by_site"] = df["bess_charge"] - df["solar_to_batt"]
        df["remote_solar_to_batt"] = df[["remote_solar_not_supplying_load", "batt_not_supplied_by_site"]].min(axis=1)
        df["remote_solar_to_grid"] = df["remote_solar"] - df["remote_solar_to_load"] - df["remote_solar_to_batt"]
    else:
        df["remote_solar_to_load"] = 0.0
        df["remote_solar_to_batt"] = 0.0
        df["remote_solar_to_grid"] = df["remote_solar"] if "remote_solar" in df.columns else 0.0

    df["grid_to_batt"] = df["bess_charge"] - df["solar_to_batt"] - df["remote_solar_to_batt"]
    df["grid_to_load"] = df["load"] - df["solar_to_load"] - df["batt_to_load"] - df["remote_solar_to_load"]
    df["solar_to_grid"] = df["solar_not_supplying_load"] - df["solar_to_batt"]

    # For now, assume that all the 'solar matching' happens at the property level, and none happens at the microgrid level
    df["solar_to_load_property_level"] = df["solar_to_load"]
    df["solar_to_load_microgrid_level"] = 0.0

    # The microgrid boundary flows are calculated here from the individual flows. These are needed for reporting in CSV
    # output files, although they aren't used directly in Skypro at the moment.
    df["grid_import"] = df["grid_to_batt"] + df["grid_to_load"]
    df["grid_export"] = df["batt_to_grid"] + df["solar_to_grid"] + df["remote_solar_to_grid"]

    return df
