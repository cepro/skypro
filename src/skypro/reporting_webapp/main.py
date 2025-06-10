import calendar
import logging
import os
from datetime import datetime, time, timedelta

import hmac
from typing import List

import pandas as pd
import pytz
import streamlit as st
import plotly.graph_objects as go
from dateutil.relativedelta import relativedelta
from skypro.common.cli_utils.cli_utils import read_yaml_file
from skypro.commands.report.config.config import parse_config
from skypro.commands.report.main import report, Report
from skypro.common.cli_utils.cli_utils import substitute_vars
from skypro.common.microgrid_analysis.output import generate_output_df
from skypro.common.microgrid_analysis.bill_match import bill_match
from skypro.common.notice.notice import Notice

TIMEZONE = pytz.timezone("Europe/London")


"""
This is a Streamlit web app that uses the reporting functionality from `skypro/commands/report`
"""


def main():
    logging.basicConfig(level=logging.INFO)  # Set to logging.INFO for non-debug mode

    skip_password = os.environ.get("SKIP_PASSWORD")
    if skip_password == "true":
        logging.info("Skipping password gate...")
    else:
        password = os.environ.get("PASSWORD")
        if password is None:
            raise ValueError("No password defined")
        if not check_password(password):
            st.stop()  # Do not continue if check_password is not True.

    flows_db_url = get_db_url("FLOWS_DB_URL", "~/.simt/env.json", "flows")
    rates_db_url = get_db_url("RATES_DB_URL", "~/.simt/env.json", "rates")

    config_path = os.environ.get("CONFIG_FILE", "./config/config.yaml")
    config = read_yaml_file(config_path)

    st.title("Simtricity Reporting")

    # Draw the first row, which contains three columns:
    # 1. A scenario selection (the scenarios are defined in the config file)
    # 2. A month selector
    # 3. A custom date range selector
    # Either the month selector or custom date range selector will be used to set the time range of the reporting
    now = datetime.now()
    _, num_days = calendar.monthrange(now.year, now.month-1)
    col1, col2, col3 = st.columns(3)
    with col1:
        scenario_name = st.selectbox(
            "Select Scenario",
            config["reportScenarios"].keys(),
        )
    with col2:
        months_df = get_previous_months_df(24)
        month_select_start = st.selectbox(
            label="Select month",
            options=months_df["start"],
            index=None,  # select nothing by default
            placeholder="Choose an option",
            format_func=lambda x: x.strftime("%Y %B")
        )
        if month_select_start is None:
            month_select_end = None
        else:
            i = months_df.index[months_df["start"] == month_select_start].to_list()[0]
            month_select_end = months_df.loc[i, "end"]

    with col3:
        custom_selected = st.date_input(
            label="Or select a custom time range",
            value=[],
            format="DD/MM/YYYY",
        )
        if isinstance(custom_selected, tuple) and len(custom_selected) == 2:
            custom_selected_start = custom_selected[0]
            custom_selected_end = custom_selected[1]
        else:
            custom_selected_start = None
            custom_selected_end = None

    start = month_select_start
    end = month_select_end
    if custom_selected_start is not None:
        start = custom_selected_start
        end = custom_selected_end

    if start is None:
        st.stop()

    # Convert the start and end dates to aware datetimes
    step_size = timedelta(minutes=5)
    start = TIMEZONE.localize(datetime.combine(date=start, time=time()))
    end = TIMEZONE.localize(datetime.combine(date=end, time=time())) + timedelta(days=1) - step_size
    st.write(f"Reporting for period: {start} -> {end}")

    # Read in the reporting configuration for this scenario
    report_config_path = config["reportScenarios"][scenario_name]["config"]
    report_config = parse_config(report_config_path, env_vars=config["vars"])

    with st.spinner("Running report..."):
        result = report(
            config=report_config,
            flows_db_url=flows_db_url,
            rates_db_url=rates_db_url,
            start=start,
            end=end,
            step_size=step_size,
            file_path_resolver_func=lambda file: os.path.expanduser(substitute_vars(file, config["vars"])),  # Substitutes env vars and resolves `~` in file paths. This captures the `config` variable.
        )

    # The reporting run has produced a set of 'notices' for the user - display them in a table, highlighting any severe notices
    notice_df = get_notice_df(result.notices)
    num_important_notices = len(notice_df[notice_df["level_number"] >= 2])
    num_notices = len(notice_df)
    with st.expander(f"{num_important_notices} important notices ({num_notices} in total)", icon="⚠️" if num_important_notices > 0 else None):
        st.dataframe(notice_df[["Level", "Description"]], hide_index=True)

    # Present the import and export invoice estimates using 'bill matching' so that they are formatted like we see from our suppliers
    import_bill = bill_match(
        grid_energy_flow=result.df["grid_import"],
        mkt_vol_grid_rates_df=result.mkt_vol_rates_dfs["grid_to_batt"],  # use the grid rates for grid_to_batt as these include info about any OSAM rates
        mkt_fixed_costs_df=result.mkt_fixed_cost_dfs["import"],
        osam_rates=result.osam_rates,
        osam_df=result.osam_df,
        cepro_mkt_vol_bill_total_expected=result.breakdown.total_mkt_vol_costs["grid_to_batt"] + result.breakdown.total_mkt_vol_costs["grid_to_load"],
        context="import",
        line_items=report_config.reporting.bill_match.import_direction.line_items,
    )

    export_bill = bill_match(
        grid_energy_flow=result.df["grid_export"],
        mkt_vol_grid_rates_df=result.mkt_vol_rates_dfs["batt_to_grid"],  # we have to pick one set of rates for all exports, so use batt_to_grid here, although solar_to_grid should also be the same
        mkt_fixed_costs_df=result.mkt_fixed_cost_dfs["export"],
        osam_rates=result.osam_rates,
        osam_df=result.osam_df,
        cepro_mkt_vol_bill_total_expected=result.breakdown.total_mkt_vol_costs["batt_to_grid"] + result.breakdown.total_mkt_vol_costs["solar_to_grid"],
        context="export",
        line_items=report_config.reporting.bill_match.export_direction.line_items,
    )

    with st.expander(f"Import Bill:  £{import_bill.bill_total / 100:.2f}"):
        st.write("Note that TNUoS, Ripple, and Capacity Market are not currently accounted for.")
        st.dataframe(import_bill.bill_by_line_items_df, hide_index=False)

    with st.expander(f"Export Bill:  £{export_bill.bill_total / 100:.2f}"):
        st.dataframe(export_bill.bill_by_line_items_df, hide_index=False)

    with st.expander("Remote Generator:  £N/A"):
        st.write("Remote generators (e.g. Ripple) are not currently accounted for")

    with st.expander("Customer Income:  £N/A"):
        st.write("Detailed breakdown not yet available")

    st.write(f"Average cycles per day: {result.total_cycles/result.num_days:.1f}")

    draw_sankey(result)

    # Provide an option to download a CSV file for further analysis
    output_df = generate_output_df(
        df=result.df,
        int_final_vol_rates_dfs=result.int_vol_rates_dfs,
        mkt_final_vol_rates_dfs=result.mkt_vol_rates_dfs,
        int_live_vol_rates_dfs=None,
        mkt_live_vol_rates_dfs=None,
        mkt_fixed_costs_dfs=result.mkt_fixed_cost_dfs,
        customer_fixed_cost_dfs=result.customer_fixed_cost_dfs,
        customer_vol_rates_dfs=result.customer_vol_rates_dfs,
        load_energy_breakdown_df=None,
        aggregate_timebase="30min",
        rate_detail="all",
        config_entries=[],
    )
    output_csv = output_df.to_csv(
        index_label="utctime"
    ).encode("utf-8")

    st.download_button(
        "Download CSV",
        output_csv,
        f"{scenario_name.lower().replace(' ', '_')}_report_{start.date().isoformat().replace('-', '_')}_{end.date().isoformat().replace('-', '_')}.csv",
        "text/csv",
        on_click="ignore",
        key='download-csv'
    )


def get_db_url(env_var_name: str, env_config_path: str, env_config_section: str):
    """
    Get database URL from environment variable or environment config file.
    """
    db_url = os.environ.get(env_var_name)
    if db_url is None:
        logging.info(f"No {env_var_name} defined, trying {env_config_path}...")
        try:
            env_config = read_yaml_file(env_config_path)
        except FileNotFoundError:
            raise ValueError(f"Failed to find {env_var_name} in either environment variables or {env_config_path}")

        db_url = env_config[env_config_section]["dbUrl"]
        if db_url is None:
            raise ValueError(f"Failed to find {env_var_name} in either environment variables or {env_config_path}")

    return db_url


def get_previous_months_df(num_months: int) -> pd.DataFrame:
    """
    Returns a dataframe containing the start and end dates of the previous months.
    """
    starts = []
    ends = []
    current_date = datetime.now()

    # Generate the previous months
    for i in range(1, num_months):
        # Subtract i months from current date
        past_date = current_date - relativedelta(months=i)

        # Create first day of the month
        start_of_month = datetime(past_date.year, past_date.month, 1)

        # Create last day of the month (first day of next month - 1 day)
        if past_date.month == 12:
            end_of_month = datetime(past_date.year + 1, 1, 1) - timedelta(days=1)
        else:
            end_of_month = datetime(past_date.year, past_date.month + 1, 1) - timedelta(days=1)

        starts.append(start_of_month.date())
        ends.append(end_of_month.date())

    # Create and return DataFrame
    df = pd.DataFrame({
        'start': starts,
        'end': ends
    })

    return df


def check_password(password: str):
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], password):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


def draw_sankey(result: Report):
    """
    Plots a Sankey diagram of the energy flows in the microgrid
    """
    color_blue = "rgba(38, 70, 83, 0.5)"
    color_orange = "rgba(231, 111, 81, 0.5)"
    color_teal = "rgba(42, 157, 143, 0.5)"
    color_yellow = "rgba(233, 196, 106, 0.5)"
    color_grey = "rgba(224, 225, 221, 0.7)"
    sankey_df = pd.DataFrame.from_dict(
        {
            0: ["Imports", "Battery", result.breakdown.total_flows["grid_to_batt"], color_grey, f"{result.breakdown.avg_mkt_vol_rates['grid_to_batt']:.2f} p/kWh"],
            1: ["Imports", "Loads", result.breakdown.total_flows["grid_to_load"], color_grey, f"{result.breakdown.avg_mkt_vol_rates['grid_to_load']:.2f} p/kWh"],
            2: ["Solar", "Battery", result.breakdown.total_flows["solar_to_batt"], color_yellow, f"{result.breakdown.avg_int_vol_rates['solar_to_batt']:.2f} p/kWh (displaced export rate)"],
            3: ["Solar", "Exports", result.breakdown.total_flows["solar_to_grid"], color_yellow, f"{result.breakdown.avg_mkt_vol_rates['solar_to_grid']:.2f} p/kWh"],
            4: ["Solar", "Loads", result.breakdown.total_flows["solar_to_load"], color_yellow, f"{result.breakdown.avg_int_vol_rates['solar_to_load']:.2f} p/kWh (displaced import rate)"],
            5: ["Battery", "Exports", result.breakdown.total_flows["batt_to_grid"], color_blue, f"{result.breakdown.avg_mkt_vol_rates['batt_to_grid']:.2f} p/kWh"],
            6: ["Battery", "Loads", result.breakdown.total_flows["batt_to_load"], color_blue, f"{result.breakdown.avg_int_vol_rates['batt_to_load']:.2f} p/kWh (displaced import rate)"],
            7: ["Battery", "Losses", result.breakdown.total_flows["bess_losses"], color_orange, ""],
            # 8: ["Microgrid", "Power", 1, color_teal, ""],
            # 9: ["Microgrid", "EVs", 1, color_teal, ""],
            # 10: ["Microgrid", "Heat", 1, color_teal, ""],
        },
        orient="index"
    )
    sankey_df = sankey_df.rename(columns={
        0: "source",
        1: "target",
        2: "value",
        3: "color",
        4: "label"
    })

    unique_source_target = list(pd.unique(sankey_df[['source', 'target']].values.ravel('K')))
    mapping_dict = {k: v for v, k in enumerate(unique_source_target)}
    sankey_df['source'] = sankey_df['source'].map(mapping_dict)
    sankey_df['target'] = sankey_df['target'].map(mapping_dict)
    links_dict = sankey_df.to_dict(orient='list')

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=15,
            thickness=15,
            line=dict(color="black", width=0.5),
            label=unique_source_target,
            color=[color_grey, color_yellow, color_blue, color_teal, color_grey, color_orange, color_teal, color_teal,
                   color_teal]
        ),
        link=dict(
            source=links_dict["source"],
            target=links_dict["target"],
            value=links_dict["value"],
            label=links_dict["label"],
            color=links_dict["color"]
        ))])

    fig.update_layout(font_size=10)

    st.write("")
    st.write("")
    st.subheader("Energy Flows")
    st.plotly_chart(fig)


def get_notice_df(notices: List[Notice]) -> pd.DataFrame:
    """
    Returns a dataframe summarising the Notices, which can then be presented to the user.
    """
    df = pd.DataFrame(columns=["level_number", "Description"])
    for notice in notices:
        df.loc[len(df)] = [notice.level.value, notice.detail]
    df = df.sort_values("level_number", ascending=False)

    df["Level"] = "N/A"
    df.loc[df["level_number"] == 1, "Level"] = "Info"
    df.loc[df["level_number"] == 2, "Level"] = "⚠️ Noteworthy"
    df.loc[df["level_number"] == 3, "Level"] = "⚠️ Serious"

    df = df.reset_index(drop=True)

    return df


if __name__ == '__main__':
    main()
