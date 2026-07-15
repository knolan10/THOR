#!/usr/bin/env python3
"""
Fetch and visualise LSST alerts for a given date range.

Usage
-----
    python summarize_rubin_alerts.py                        # previous night
    python summarize_rubin_alerts.py 07-01-2026 07-02-2026
"""

import argparse
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

load_dotenv()

from thor.utils.fetch_alerts import babamul_get_alerts
from thor.utils.rubin_stats_visualizations import plot_skymap, summarize_night


def _previous_night():
    """Return (start, end) strings for the previous UTC night."""
    today = datetime.now(timezone.utc).date()
    yesterday = today - timedelta(days=1)
    return yesterday.strftime("%m-%d-%Y"), today.strftime("%m-%d-%Y")


def main():
    parser = argparse.ArgumentParser(description="Fetch and summarise LSST alerts.")
    parser.add_argument("start", nargs="?", help="Start date MM-DD-YYYY (default: yesterday)")
    parser.add_argument("end",   nargs="?", help="End date   MM-DD-YYYY (default: today)")
    args = parser.parse_args()

    if args.start and args.end:
        start, end = args.start, args.end
    elif args.start or args.end:
        parser.error("Provide both start and end dates, or neither.")
    else:
        start, end = _previous_night()

    print(f"Fetching LSST alerts from {start} to {end}...")

    alerts = babamul_get_alerts(
        survey="LSST",
        start_time=start,
        end_time=end,
        save=False,
    )

    if not alerts:
        print("No alerts returned.")
        return

    summarize_night(alerts)

    title = f"LSST Alert Skymap  {start} to {end}"
    plot_skymap(alerts, heatmap=True, plot_ddf=True, title=title, save=True)


if __name__ == "__main__":
    main()
