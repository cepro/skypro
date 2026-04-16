"""
Fetch P442 Total Exempt Supply Volumes from Elexon BMRS API (dataset S0621).

Pulls half-hourly exempt supply volumes for a date range and summarises
daily/monthly totals. No API key required.

Usage:
    python fetch_p442_volumes.py                          # Last 30 days
    python fetch_p442_volumes.py --scan                   # Walk backwards from yesterday until data runs out
    python fetch_p442_volumes.py --start 2025-02-01       # From date to today
    python fetch_p442_volumes.py --start 2025-02-01 --end 2025-12-31
    python fetch_p442_volumes.py --start 2025-02-01 --end 2025-12-31 --csv output.csv
"""

import argparse
import sys
from datetime import date, timedelta

import pandas as pd
import requests

API_BASE = "https://data.elexon.co.uk/bmrs/api/v1/saa/datasets/total-exempt-volume"


def fetch_day(settlement_date: date) -> list[dict]:
    """Fetch all settlement periods for a single date."""
    url = f"{API_BASE}/{settlement_date.isoformat()}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json().get("data", [])


def fetch_backwards(from_date: date, max_empty_streak: int = 14) -> pd.DataFrame:
    """Walk backwards from from_date, collecting data until we hit max_empty_streak
    consecutive days with no data or all-zero volumes."""
    all_records = []
    empty_streak = 0
    d = from_date
    days_checked = 0

    print(f"  Scanning backwards from {from_date}...", file=sys.stderr)
    while empty_streak < max_empty_streak:
        try:
            records = fetch_day(d)
            has_volume = any(
                float(r.get("totalExemptSupplyVolume", 0)) > 0 for r in records
            )
            if records and has_volume:
                all_records.extend(records)
                empty_streak = 0
            else:
                empty_streak += 1
        except requests.RequestException as e:
            print(f"  Warning: failed to fetch {d}: {e}", file=sys.stderr)
            empty_streak += 1

        days_checked += 1
        if days_checked % 30 == 0:
            print(f"  ...checked back to {d} ({days_checked} days)", file=sys.stderr)

        d -= timedelta(days=1)

    first_active = d + timedelta(days=max_empty_streak + 1)
    print(f"  Scanned {days_checked} days. Data runs from ~{first_active} to {from_date}", file=sys.stderr)

    if not all_records:
        print("No non-zero data found.", file=sys.stderr)
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df["settlementDate"] = pd.to_datetime(df["settlementDate"])
    df["totalExemptSupplyVolume"] = pd.to_numeric(df["totalExemptSupplyVolume"])
    return df.sort_values("settlementDate")


def fetch_range(start: date, end: date) -> pd.DataFrame:
    """Fetch exempt supply volumes for a date range, with progress output."""
    total_days = (end - start).days + 1
    all_records = []

    for i in range(total_days):
        d = start + timedelta(days=i)
        if i % 10 == 0:
            print(f"  Fetching {d} ... ({i+1}/{total_days})", file=sys.stderr)
        try:
            records = fetch_day(d)
            all_records.extend(records)
        except requests.RequestException as e:
            print(f"  Warning: failed to fetch {d}: {e}", file=sys.stderr)

    if not all_records:
        print("No data returned.", file=sys.stderr)
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df["settlementDate"] = pd.to_datetime(df["settlementDate"])
    df["totalExemptSupplyVolume"] = pd.to_numeric(df["totalExemptSupplyVolume"])
    return df


def summarise(df: pd.DataFrame):
    """Print daily and monthly summaries to stdout."""
    if df.empty:
        return

    # Daily totals
    daily = (
        df.groupby("settlementDate")["totalExemptSupplyVolume"]
        .agg(["sum", "min", "max", "mean"])
        .rename(columns={"sum": "daily_total_MWh", "min": "min_HH", "max": "max_HH", "mean": "avg_HH"})
    )

    # Filter to days with actual activity
    active_days = daily[daily["daily_total_MWh"] > 0]
    zero_days = daily[daily["daily_total_MWh"] == 0]

    print("\n=== P442 Exempt Supply Volumes (Elexon S0621) ===\n")
    print(f"Date range: {daily.index.min().date()} to {daily.index.max().date()}")
    print(f"Days with data: {len(daily)}")
    print(f"Days with non-zero volume: {len(active_days)}")
    print(f"Days with zero volume: {len(zero_days)}")

    if not active_days.empty:
        print(f"\nTotal exempt supply volume: {active_days['daily_total_MWh'].sum():,.1f} MWh")
        print(f"Average daily volume (active days): {active_days['daily_total_MWh'].mean():,.1f} MWh")
        print(f"Peak daily volume: {active_days['daily_total_MWh'].max():,.1f} MWh")
        print(f"Peak half-hourly volume: {active_days['max_HH'].max():,.2f} MWh")

        # Monthly summary
        monthly = active_days.copy()
        monthly.index = monthly.index.to_period("M")
        monthly_totals = monthly.groupby(level=0)["daily_total_MWh"].agg(["sum", "count"])
        monthly_totals.columns = ["total_MWh", "active_days"]

        print("\n--- Monthly Totals ---")
        print(f"{'Month':<12} {'Total MWh':>12} {'Active Days':>13} {'Avg MWh/day':>13}")
        print("-" * 52)
        for period, row in monthly_totals.iterrows():
            avg = row["total_MWh"] / row["active_days"] if row["active_days"] > 0 else 0
            print(f"{str(period):<12} {row['total_MWh']:>12,.1f} {int(row['active_days']):>13} {avg:>13,.1f}")

        # Annualised estimate
        if len(active_days) >= 7:
            avg_daily = active_days["daily_total_MWh"].mean()
            print(f"\nAnnualised estimate (based on active-day avg): {avg_daily * 365:,.0f} MWh/year")

            # Levy context: ~6p/kWh = £60/MWh
            levy_rate = 60  # £/MWh approximate
            print(f"Estimated annual levy avoidance at ~£60/MWh: £{avg_daily * 365 * levy_rate:,.0f}")

    # Show last 14 days detail
    print("\n--- Last 14 Active Days ---")
    recent = active_days.tail(14)
    if not recent.empty:
        print(f"{'Date':<12} {'Daily MWh':>11} {'Peak HH':>9} {'Avg HH':>9}")
        print("-" * 43)
        for dt, row in recent.iterrows():
            print(f"{dt.date()!s:<12} {row['daily_total_MWh']:>11,.1f} {row['max_HH']:>9,.2f} {row['avg_HH']:>9,.2f}")


def main():
    parser = argparse.ArgumentParser(description="Fetch P442 exempt supply volumes from Elexon BMRS")
    parser.add_argument("--scan", action="store_true",
                        help="Walk backwards from yesterday until data runs out")
    parser.add_argument("--start", type=date.fromisoformat, default=None,
                        help="Start date (YYYY-MM-DD). Default: 30 days ago")
    parser.add_argument("--end", type=date.fromisoformat, default=None,
                        help="End date (YYYY-MM-DD). Default: yesterday")
    parser.add_argument("--csv", type=str, default=None,
                        help="Save half-hourly data to CSV file")
    args = parser.parse_args()

    end = args.end or (date.today() - timedelta(days=1))

    if args.scan:
        print(f"Scanning backwards from {end} for P442 exempt supply volumes", file=sys.stderr)
        df = fetch_backwards(end)
    else:
        start = args.start or (end - timedelta(days=29))
        print(f"Fetching P442 exempt supply volumes: {start} to {end}", file=sys.stderr)
        df = fetch_range(start, end)

    if df.empty:
        sys.exit(1)

    summarise(df)

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nHalf-hourly data saved to {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
