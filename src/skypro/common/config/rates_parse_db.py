from datetime import time, datetime, timedelta
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

from skypro.common.rates.microgrid import VolRatesForEnergyFlows
from skypro.common.rates.rates import RegularFixedRate, MultiplierVolRate, ShapedVolRate, FlatVolRate, OSAMFlatVolRate, PeriodicFlatVolRate, FixedRate, VolRate
from skypro.common.rates.supply_point import SupplyPoint
from skypro.common.rates.time_varying_value import PeriodicValue
from skypro.common.timeutils import ClockTimePeriod
from skypro.common.timeutils.dayed_period import DayedPeriod
from skypro.common.timeutils.days import Days

"""
This file handles parsing of rates from a database 
"""


def get_rates_from_db(
        supply_points_name: str,
        site_region: str,
        site_bands: List[str],
        import_bundle_names: List[str],
        export_bundle_names: List[str],
        db_engine,
        imbalance_pricing: pd.Series,
        import_grid_capacity: Optional[float],
        export_grid_capacity: Optional[float],
        future_offset: timedelta
) -> Tuple[VolRatesForEnergyFlows, List[FixedRate], List[FixedRate]]:
    """
    Reads the Rates database and returns the volumetric rates per flow, and the import and export Fixed rates in turn.
    If `future_offset` is set positive, then rates will be 'brought forwards' so that rates from the future will be used. This enables simulations that
    use historical datasets whilst using rates from the future.
    """

    supply_points = _get_supply_points_from_db(supply_points_name, db_engine=db_engine)

    """
    The Rates DB stores rates a little differently to how they are modelled in software. The Rates DB has the notion of
    Import and Export rates, whereas this code models rates with more with granularity- with rates against each flow of
    energy in the microgrid. For example, the code can model independent rates for the "solar_to_grid" flow and
    "batt_to_grid" flows, whereas the database can't as it only has knowledge of "Import" and "Export" rates.
    
    The Rates DB flags a given rate as "OSAM", in the code below the OSAM rates are applied only to the "grid_to_batt"
    flows, and OSAM is not applied to the "grid_to_load" flows (which pay full price under P395/OSAM).    
    """

    import_vol_rates_with_osam, import_fixed_rates = _combine_rates(
        [
            _get_site_specific_from_db(
                region=site_region,
                band=site_band,
                direction="import",
                db_engine=db_engine,
                imbalance_pricing=imbalance_pricing,
                supply_points=supply_points,
                grid_connection_capacity=import_grid_capacity,
                future_offset=future_offset,
                osam_support="yes"
            )
            for site_band in site_bands
        ] +
        [
            _get_bundle_from_db(
                bundle_name=bundle_name,
                db_engine=db_engine,
                imbalance_pricing=imbalance_pricing,
                supply_points=supply_points,
                grid_connection_capacity=import_grid_capacity,
                future_offset=future_offset,
                osam_support="yes"
            )
            for bundle_name in import_bundle_names
        ]
    )

    import_vol_rates_without_osam, _ = _combine_rates(
        [
            _get_site_specific_from_db(
                region=site_region,
                band=site_band,
                direction="import",
                db_engine=db_engine,
                imbalance_pricing=imbalance_pricing,
                supply_points=supply_points,
                grid_connection_capacity=import_grid_capacity,
                future_offset=future_offset,
                osam_support="convert"
            )
            for site_band in site_bands
        ] +
        [
            _get_bundle_from_db(
                bundle_name=bundle_name,
                db_engine=db_engine,
                imbalance_pricing=imbalance_pricing,
                supply_points=supply_points,
                grid_connection_capacity=import_grid_capacity,
                future_offset=future_offset,
                osam_support="convert"
            )
            for bundle_name in import_bundle_names
        ]
    )

    export_vol_rates, export_fixed_rates = _combine_rates(
        [
            _get_site_specific_from_db(
                region=site_region,
                band=site_band,
                direction="export",
                db_engine=db_engine,
                imbalance_pricing=imbalance_pricing * -1,
                supply_points=supply_points,
                grid_connection_capacity=export_grid_capacity,
                future_offset=future_offset,
                osam_support="no"
            )
            for site_band in site_bands
        ] +
        [
            _get_bundle_from_db(
                bundle_name=bundle_name,
                db_engine=db_engine,
                imbalance_pricing=imbalance_pricing * -1,
                supply_points=supply_points,
                grid_connection_capacity=export_grid_capacity,
                future_offset=future_offset,
                osam_support="no"
            )
            for bundle_name in export_bundle_names
        ]
    )

    all_vol_rates = VolRatesForEnergyFlows(
        grid_to_batt=import_vol_rates_with_osam,
        grid_to_load=import_vol_rates_without_osam,
        solar_to_grid=export_vol_rates,
        batt_to_grid=export_vol_rates,
        batt_to_load=[],
        solar_to_load=[],
        solar_to_batt=[],
    )

    return all_vol_rates, import_fixed_rates, export_fixed_rates


def _get_supply_points_from_db(name: str, db_engine) -> Dict[str, SupplyPoint]:
    """
    Queries the Rates DB for the supply point set with the given name.
    """
    query = (
        '''
        select id, name, msp_loss_factor, gsp_loss_factor, nsp_loss_factor from rates.supply_points where name = '{name}'
        '''.format(name=name)
    )

    df = pd.read_sql(query, con=db_engine)

    if len(df) < 1:
        raise ValueError(f"No supply points were found for the name '{name}'")
    elif len(df) > 1:
        raise ValueError(f"Multiple supply points were found for the name '{name}'")  # this should never happen due to DB constraints

    row = df.iloc[0]

    return {
        "msp": SupplyPoint(
            name="msp",
            line_loss_factor=row["msp_loss_factor"]
        ),
        "gsp": SupplyPoint(
            name="gsp",
            line_loss_factor=row["gsp_loss_factor"]
        ),
        "nsp": SupplyPoint(
            name="nsp",
            line_loss_factor=row["nsp_loss_factor"]
        )
    }


def _combine_rates(rate_groups: List[Tuple[List[VolRate], List[FixedRate]]]) -> Tuple[List[VolRate], List[FixedRate]]:
    """
    This takes the many groups of rates that have been retrieved from the database and combines them into one list, keeping
    the distinction between volumetric and fixed rates.
    """
    vol_rates: List[VolRate] = []
    fixed_rates: List[FixedRate] = []
    for group_vol_rates, group_fixed_rates in rate_groups:
        vol_rates.extend(group_vol_rates)
        fixed_rates.extend(group_fixed_rates)

    # This runs through all the rates and if there is a multiplier rate present then it will be
    # informed of all the other rates in the set, so that it knows what rates to multiply.
    for rate in vol_rates:
        if isinstance(rate, MultiplierVolRate):
            rate.set_all_rates_in_set(vol_rates)

    return vol_rates, fixed_rates


def _get_bundle_from_db(
        bundle_name: str,
        db_engine,
        imbalance_pricing: pd.Series,
        supply_points: Dict[str, SupplyPoint],
        grid_connection_capacity: Optional[float],  # kVA rating of the grid connection is needed for some rates
        future_offset: timedelta,
        osam_support: str
) -> Tuple[List[VolRate], List[FixedRate]]:
    """
    Queries the Rates DB for the given named bundle and returns the rates objects, split into volumetric and fixed rates.
    See `get_rates_from_db` for an explanation of the `future_offset` argument.
    """

    df = _query_db_for_named_bundle(bundle_name, db_engine)
    return _process_db_query_into_rate_objects(
        df=df,
        imbalance_pricing=imbalance_pricing,
        supply_points=supply_points,
        grid_connection_capacity=grid_connection_capacity,
        future_offset=future_offset,
        osam_support=osam_support,
    )


def _get_site_specific_from_db(
        region: str,
        band: str,
        direction: str,
        db_engine,
        imbalance_pricing: pd.Series,
        supply_points: Dict[str, SupplyPoint],
        grid_connection_capacity: Optional[float],  # kVA rating of the grid connection is needed for some rates
        future_offset: timedelta,
        osam_support: str
) -> Tuple[List[VolRate], List[FixedRate]]:
    """
    Queries the Rates DB for rates that effect a site at the given region and banding and returns the rates objects, split into volumetric and fixed rates.
    See `get_rates_from_db` for an explanation of the `future_offset` argument.
    """

    df = _query_db_for_site_specific_rates(region=region, band=band, direction=direction, db_engine=db_engine)
    return _process_db_query_into_rate_objects(
        df=df,
        imbalance_pricing=imbalance_pricing,
        supply_points=supply_points,
        grid_connection_capacity=grid_connection_capacity,
        future_offset=future_offset,
        osam_support=osam_support
    )


def _process_db_query_into_rate_objects(
        df: pd.DataFrame,
        imbalance_pricing: pd.Series,
        supply_points: Dict[str, SupplyPoint],
        grid_connection_capacity: Optional[float],  # kVA rating of the grid connection is needed for some rates
        future_offset: timedelta,
        osam_support: str
) -> Tuple[List[VolRate], List[FixedRate]]:
    """
    Takes the result of a rates DB query (as a dataframe) and converts it into various Rate instances.
    The volume and fixed rates are returned separately.
    OSAM support is optional, and is specified as a string:
    - "yes" allows OSAM rates to be interpreted normally
    - "convert" interprets OSAM rates as their non-OSAM equivalent
    - "no" raises an exception if OSAM rates are seen
    """
    vol_rates: List[VolRate] = []
    fixed_rates: List[FixedRate] = []

    # Each rate can have different values that become active at different points in time (e.g. yearly changes to DUoS).
    # The big query above pulled all this information into a flat dataframe. Here we loop over each rate and configure
    # a Rate instance to handle it.
    for ident, ident_df in df.groupby("ident"):
        ident = str(ident)

        rate_type = ident_df.iloc[0]["type"].lower()
        rate_supply_point_name = ident_df.iloc[0]["supply_point"]
        is_osam_rate = ident_df.iloc[0]["osam"] if "osam" in ident_df.columns else False
        is_vol_rate = False
        is_periodic = False
        handled_osam = False
        handled_periodic = False

        # Here we loop over how this rate changes over time - some rates will be configured with only a single value for all time, while
        # others may have yearly changes etc. This info is held as a list of (start_time, value) pairs.
        # Some types of rates also support 'periodicity' so DayedPeriod's are held alongside the value.
        values_without_periods: List[Tuple[datetime, float]] = []
        values_with_periods: List[Tuple[datetime, PeriodicValue]] = []
        for i, row in ident_df.iterrows():
            periods = _parse_periods_from_db_row(row)
            start = row["start"]
            if not pd.isnull(start):
                start = start - future_offset  # shift the start time of the rates so that simulations can be run with historical data and future rates.

            if len(periods) > 0:
                is_periodic = True
            if np.isnan(row["value"]) and rate_type != "imbalance":
                raise ValueError(f"rate '{ident}' is missing a value")
            values_with_periods.append((start, PeriodicValue(value=row["value"], periods=periods)))
            values_without_periods.append((start, row["value"]))

        if rate_type == "p/kwh":
            is_vol_rate = True
            if is_periodic:
                rate = PeriodicFlatVolRate(
                    name=ident,
                    periodic_values=values_with_periods,
                    supply_point=supply_points[rate_supply_point_name],
                )
                handled_periodic = True
            else:
                if is_osam_rate:
                    if osam_support == "yes":
                        rate = OSAMFlatVolRate(
                            name=ident,
                            rates=values_without_periods,
                            supply_point=supply_points[rate_supply_point_name]
                        )
                        handled_osam = True
                    elif osam_support == "convert":
                        # Interpret this as the equivalent non-OSAM rate
                        rate = FlatVolRate(
                            name=ident,
                            values=values_without_periods,
                            supply_point=supply_points[rate_supply_point_name]
                        )
                        handled_osam = True
                    elif osam_support == "no":
                        raise ValueError("OSAM rates are not supported in this context")
                else:
                    rate = FlatVolRate(
                        name=ident,
                        values=values_without_periods,
                        supply_point=supply_points[rate_supply_point_name]
                    )
        elif rate_type == "imbalance":
            is_vol_rate = True
            if rate_supply_point_name.lower() != "nsp":
                raise ValueError(f"Imbalance pricing should probably be specified at the nsp, but {rate_supply_point_name} was specified.")
            rate = ShapedVolRate(
                name=ident,
                pricing=imbalance_pricing,
                supply_point=supply_points[rate_supply_point_name],
            )
        elif rate_type == "multiplier":
            is_vol_rate = True
            rate = MultiplierVolRate(
                name=ident,
                mode="all-in-this-direction",
                factors=values_without_periods,
            )
        elif rate_type == "p/day":
            rate = RegularFixedRate(
                name=ident,
                daily_costs=values_without_periods
            )
        elif rate_type == "p/kva/day":
            # For the 'per kVA' charges we just use the regular fixed rate class and multiply the value by the configured grid connection size
            if grid_connection_capacity is None:
                raise ValueError("Grid connection capacity must be specified when p/kVA/day rates are used")

            rate = RegularFixedRate(
                name=ident,
                daily_costs=[(start, value * grid_connection_capacity) for start, value in values_without_periods]
            )
        else:
            raise ValueError(f"Rate of type '{rate_type}' is not supported")

        if is_osam_rate and not handled_osam:
            raise NotImplementedError(f"OSAM not supported for '{ident}' of type '{rate_type}'")
        if is_periodic and not handled_periodic:
            raise NotImplementedError(f"Periods not supported for '{ident}' of type '{rate_type}'")

        if is_vol_rate:
            vol_rates.append(rate)
        else:
            fixed_rates.append(rate)

    return vol_rates, fixed_rates


def _query_db_for_named_bundle(bundle_name: str, db_engine) -> pd.DataFrame:
    """
    Queries the DB to give a single, flat joined result dataframe with all the rate information for this bundle.
    """

    # Bundles can import 'child bundles' so the query is recursive.
    # The period information is queried as JSON (`array_to_json`) so that it can be flattened, and SQLAlchemy/pandas
    # handles the subsequent conversion to Python lists.
    query = (
        '''
        WITH RECURSIVE bundle_hierarchy AS (
            -- Base case: the starting bundle
            SELECT 
                id, 
                name
            FROM 
                rates.bundle
            WHERE 
                name = '{bundle_name}'

            UNION

            -- Recursive case: include all child bundles
            SELECT 
                b.id, 
                b.name
            FROM 
                rates.bundle b
            JOIN 
                rates.bundle_bundle_links bbl ON b.id = bbl.child_bundle_id
            JOIN 
                bundle_hierarchy bh ON bbl.parent_bundle_id = bh.id
        )

        SELECT 
            r.ident,
            r.direction,
            r.supply_point,
            brl.osam,
            r.type,
            rv.start,
            rv.value,
            array_to_json(ARRAY_AGG(p.start ORDER BY p.id)) AS period_starts,
            array_to_json(ARRAY_AGG(p.end ORDER BY p.id)) AS period_ends,
            array_to_json(ARRAY_AGG(p.days ORDER BY p.id)) AS period_days,
            array_to_json(ARRAY_AGG(p.timezone ORDER BY p.id)) AS period_timezones
        FROM
            rates.rates r
        JOIN
            rates.bundle_rate_links brl ON r.id = brl.rate_id
        JOIN
            bundle_hierarchy bh ON brl.bundle_id = bh.id
        LEFT JOIN
            rates.rate_value rv ON r.id = rv.rate_id
        LEFT JOIN
            rates.period_set ps ON rv.period_set_id = ps.id
        LEFT JOIN
            rates.period p ON ps.id = p.period_set_id
        GROUP BY 
            r.ident,
            r.direction,
            r.supply_point,
            brl.osam,
            r.type,
            rv.start,
            rv.value
        ORDER BY 
            r.ident, rv.start;
        '''.format(bundle_name=bundle_name)
    )

    df = pd.read_sql(query, con=db_engine)

    if len(df) == 0:
        raise ValueError(f"No rates were found for the bundle '{bundle_name}'")

    return df


def _query_db_for_site_specific_rates(region: str, band: str, direction: str, db_engine) -> pd.DataFrame:
    """
    Queries the DB to give a single, flat joined result dataframe with all the rate information corresponding to the given
    region and band.
    """

    if region == "" or region.lower() == "all":
        raise ValueError(f"Region must be specific, got '{region}'")
    if band == "" or band.lower() == "all":
        raise ValueError(f"Band must be specific, got '{band}'")

    # The period information is queried as JSON (`array_to_json`) so that it can be flattened, and SQLAlchemy/pandas
    # handles the subsequent conversion to Python lists.
    query = (
        '''
        SELECT 
            r.ident,
            r.direction,
            r.supply_point,
            r.type,
            rv.start,
            rv.value,
            array_to_json(ARRAY_AGG(p.start ORDER BY p.id)) AS period_starts,
            array_to_json(ARRAY_AGG(p.end ORDER BY p.id)) AS period_ends,
            array_to_json(ARRAY_AGG(p.days ORDER BY p.id)) AS period_days,
            array_to_json(ARRAY_AGG(p.timezone ORDER BY p.id)) AS period_timezones
        FROM
            rates.rates r
        LEFT JOIN
            rates.rate_value rv ON r.id = rv.rate_id
        LEFT JOIN
            rates.period_set ps ON rv.period_set_id = ps.id
        LEFT JOIN
            rates.period p ON ps.id = p.period_set_id
        WHERE
            r.region = '{region}'
            AND r.band = '{band}'
            AND r.direction = '{direction}'
        GROUP BY 
            r.ident,
            r.direction,
            r.supply_point,
            r.type,
            rv.start,
            rv.value
        ORDER BY 
            r.ident, rv.start;
        '''.format(region=region, band=band, direction=direction)
    )

    df = pd.read_sql(query, con=db_engine)

    return df


def _parse_periods_from_db_row(row) -> List[DayedPeriod]:
    """
    Takes a row from a SQL query of the rates database, and returns the periods for which the rate is active (e.g. "weekdays 5pm - 7pm").
    """

    if len(row.period_starts) != len(row.period_ends) != len(row.period_days) != len(row.period_timezones):
        # The DB is queried so that the period info is flattened into lists, which should all be the same length
        raise ValueError(f"Inconsistency in period definition of rate {row.ident}")

    periods = []
    for i in range(len(row.period_starts)):

        if row.period_starts[i] is None:
            continue  # If there are no periods defined then we seem to get a list with a single `None` value from the query

        periods.append(
            DayedPeriod(
                days=Days(name=row.period_days[i], tz_str=row.period_timezones[i]),
                period=ClockTimePeriod(
                    start=time.fromisoformat(row.period_starts[i]),
                    end=time.fromisoformat(row.period_ends[i]),
                    tz_str=row.period_timezones[i]
                )
            )
        )

    return periods
