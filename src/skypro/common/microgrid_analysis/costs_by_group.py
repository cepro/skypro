from typing import Dict

import pandas as pd
import plotly.express as px


def plot_costs_by_grouping(
        costs_bess_charge_df: pd.DataFrame,
        costs_bess_discharge_df: pd.DataFrame,
        cost_groupings: Dict[str, str]
) -> None:
    """
    Plots a bar chart showing the costs grouped by the `cost_groupings` given.
    `cost_groupings` is a dictionary where the key is the rate name, and the value is the group to which the cost is
    assigned, e.g: {
        "duosRed": "duos",
        "duosAmber": "duos",
        "brytAccountManagement": "supplier"
    }
    """

    grouped_charge_costs = group_costs(costs_bess_charge_df, cost_groupings)
    grouped_discharge_costs = group_costs(costs_bess_discharge_df, cost_groupings)

    charge_summed = grouped_charge_costs.sum()
    charge_summed_df = pd.DataFrame()
    charge_summed_df["Description"] = charge_summed.index
    charge_summed_df["Value"] = charge_summed.values / 100
    charge_summed_df["Direction"] = "Charge"

    discharge_summed = grouped_discharge_costs.sum()
    discharge_summed_df = pd.DataFrame()
    discharge_summed_df["Description"] = discharge_summed.index
    discharge_summed_df["Value"] = discharge_summed.values / -100
    discharge_summed_df["Direction"] = "Discharge"

    df = pd.concat([charge_summed_df, discharge_summed_df])

    px.bar(
        df,
        x="Direction",
        y="Value",
        color="Description",
        color_discrete_map={  # This allows the colouring to be consistent across simt-cli and skypro-cli
            "duos": "green",
            "supplier": "purple",
            "imbalance": "blue"
        },
        text="Description",
        title="Breakdown of BESS charge and discharge costs (including opportunity cost)",
        labels={
            "Value": "Cost / Revenue (£)",
            "Description": "Legend"
        },
    ).show()


def group_costs(costs_df: pd.DataFrame, groupings: Dict[str, str]):
    """
    Takes a dataframe with many columns and sums the columns into groups, as defined by the groupings dictionary.

    For example, if there are columns of: duosRed, duosAmber, bryt, statkraft
    And a grouping dictionary of: {
      "duosRed": "duos",
      "duosAmber": "duos",
      "bryt": "supplier",
      "statkraft": "supplier"
    }
    Then a dataframe with two columns- "duos" and "supplier" would be returned.
    """

    found_cols = 0
    costs_by_group = pd.DataFrame(index=costs_df.index)
    for col in costs_df.columns:
        for cost_name, group_name in groupings.items():
            if col.endswith(cost_name):
                if group_name not in costs_by_group.columns:
                    costs_by_group[group_name] = 0
                costs_by_group[group_name] = costs_by_group[group_name] + costs_df[col]
                found_cols += 1
                break

    if found_cols != len(costs_df.columns):
        raise ValueError(f"Not all costs were successfully grouped into: {groupings}")

    return costs_by_group
