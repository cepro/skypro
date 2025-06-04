from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from skypro.common.config.bill_match import BillMatchLineItem
from skypro.common.rates.osam import calculate_osam_rate_cost
from skypro.common.rates.rates import OSAMFlatVolRate
from tabulate import tabulate


@dataclass
class BillMatch:
    """
    Holds the estimated bills from a supplier
    """
    bill_total: float  # The total bill
    mkt_fixed_bill: float # The total bill for market fixed costs
    mkt_vol_bill: float  # The total bill for market volume costs according to "OSAM flows"
    cepro_mkt_vol_bill: float  # The total bill for market volume costs according to "Cepro flows"
    error_pct_mkt_vol_bill: float  # The percentage between the above two numbers (OSAM vs Cepro flow bill calculations)

    bill_by_line_items_df: Optional[pd.DataFrame]  # The bill broken down by line items (which is useful for humans to compare the actual invoice with our estimate)


def bill_match(
    grid_energy_flow: pd.Series,
    mkt_vol_grid_rates_df: pd.DataFrame,
    mkt_fixed_costs_df: Optional[pd.DataFrame],
    osam_rates: List[OSAMFlatVolRate],
    osam_df: pd.DataFrame,
    cepro_mkt_vol_bill_total_expected: float,
    context: str,
    line_items: Optional[Dict[str, BillMatchLineItem]]
) -> BillMatch:
    """
    Calculates the total expected bill using the grid meter and any OSAM flows.

    If `line_items` are specified then the bill is broken into those categories. This is useful to match against the suppliers bill.

    Using the grid meter and OSAM calcs give slightly different results compared to Cepro's calculations based on the 7
    'fundamental flows' because Cepro has its own methodology for determining the flows and uses 5 or 10 minute
    granularity, whereas OSAM has a separate methodology for calculating the flows from meter data and uses 30 minute
    granularity.
    """

    osam_df = osam_df.copy()
    billed_costs_vol_df = pd.DataFrame(columns=mkt_vol_grid_rates_df.columns, index=[0])
    billed_costs_fixed_df = pd.DataFrame(columns=[], index=[0])
    osam_rates_dict = {rate.name: rate for rate in osam_rates}
    total_energy = grid_energy_flow.sum()
    total_days = ((grid_energy_flow.index[-1] - grid_energy_flow.index[0]).total_seconds() / 3600) / 24

    # Run through all the various volumetric rates and construct a billed_costs_vol_df which contains the totals
    for rate_name in mkt_vol_grid_rates_df.columns:
        if rate_name in osam_rates_dict.keys():
            # If this is an OSAM rate then we need to calculate the cost appropriately
            billed_costs_vol_df[rate_name] = calculate_osam_rate_cost(osam_df, osam_rates_dict[rate_name].get_per_kwh_base_rate_series(osam_df.index))
        else:
            billed_costs_vol_df[rate_name] = (grid_energy_flow * mkt_vol_grid_rates_df[rate_name]).sum()

    # Now do the same for the fixed market rates like standing and capacity charges
    if mkt_fixed_costs_df is not None:
        for rate_name in mkt_fixed_costs_df:
            billed_costs_fixed_df[rate_name] = mkt_fixed_costs_df[rate_name].sum()

    mkt_vol_bill = billed_costs_vol_df.sum().sum()
    mkt_fixed_bill = billed_costs_fixed_df.sum().sum()
    bill_total = mkt_vol_bill + mkt_fixed_bill  # # bill_total = vol_bill_total + fixed_bill_total
    error_pct_mkt_vol_bill = ((mkt_vol_bill - cepro_mkt_vol_bill_total_expected) / mkt_vol_bill) * 100

    print("")
    print(f"Cepro expected vs OSAM expected bill error: {error_pct_mkt_vol_bill:.1f}% (£{cepro_mkt_vol_bill_total_expected/100:.2f} vs £{mkt_vol_bill/100:.2f}, volumetric rates only)")

    # If bill 'line items' are specified then we can print the bill as it appears by the supplier company
    bill_df = None
    if line_items:
        bill_df = pd.DataFrame(index=list(line_items.keys()), columns=["Quantity", "Unit Rate", "Cost"], dtype=object)

        all_rates = []
        for line_item_name, line_item in line_items.items():
            all_rates.extend(line_item.rate_names)
            line_item_total = 0.0
            for rate_name in line_item.rate_names:
                # Pull the cost out of either the volumetric or fixed cost dataframe
                is_in_vol = rate_name in billed_costs_vol_df.columns
                is_in_fixed = rate_name in billed_costs_fixed_df.columns
                if is_in_vol and not is_in_fixed:
                    line_item_total += billed_costs_vol_df.loc[0, rate_name]
                elif is_in_fixed and not is_in_vol:
                    line_item_total += billed_costs_fixed_df.loc[0, rate_name]
                elif is_in_fixed and is_in_vol:
                    raise ValueError(f"Found rate named '{rate_name}' in both fixed and volumetric rates.")
                else:
                    raise ValueError(f"Could not find rate '{rate_name}' that was specified in bill match line items.")

            if line_item.unit.lower() == "p/kwh":
                unit_rate = line_item_total / total_energy
                bill_df.loc[line_item_name, "Quantity"] = f"{total_energy:.0f} kWh (MSP)"
                bill_df.loc[line_item_name, "Unit Rate"] = f"{unit_rate:.2f} p/kWh"
                # TODO: we may want to support different supply points like MSP / NSP here, but at the moment we report everything against the MSP
            elif line_item.unit.lower() in ["p/day", "p/kva/day"]:
                unit_rate = line_item_total / total_days
                bill_df.loc[line_item_name, "Quantity"] = f"{total_days:.1f} days"
                bill_df.loc[line_item_name, "Unit Rate"] = f"{unit_rate/100:.2f} £/day"
                # TODO: support p/kVA/day charges properly (at the moment they are configured as p/day

            bill_df.loc[line_item_name, "Cost"] = f"£{line_item_total/100:.2f}"

        # Check that all the rates are accounted for properly
        if len(set(all_rates)) != len(all_rates):
            raise ValueError(f"At least one bill match market rate for {context} are specified twice.")
        all_rates.sort()
        all_rates_2 = list(billed_costs_vol_df.columns) + list(billed_costs_fixed_df.columns)
        all_rates_2.sort()
        if all_rates != all_rates_2:
            raise ValueError(f"At least one bill match market rate for {context} is missing: "
                             f"{(set(all_rates) - set(all_rates_2)) | (set(all_rates_2) - set(all_rates))}")

        print("")
        print(f"{context} bill")
        print(tabulate(
            tabular_data=bill_df,
            headers='keys',
            tablefmt='presto',
            # floatfmt=(None, ",.2f")
        ))

    return BillMatch(
        bill_total=bill_total,
        mkt_fixed_bill=mkt_fixed_bill,
        mkt_vol_bill=mkt_vol_bill,
        cepro_mkt_vol_bill=cepro_mkt_vol_bill_total_expected,
        error_pct_mkt_vol_bill=error_pct_mkt_vol_bill,
        bill_by_line_items_df=bill_df,
    )
