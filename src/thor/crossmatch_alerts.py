import argparse
import json
import math
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import dotenv

from thor.utils import filter_functions
from thor.utils.fetch_alerts import babamul_get_alerts

Z_COLS = {"z", "Z_BEST", "ZPHOT", "zfinal", "zpdf_med"}


def _launch_scan_notebook(object_ids):
    """Write a temporary notebook pre-loaded with candidates and open it."""
    object_ids_repr = json.dumps(list(object_ids))
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": [
            {
                "cell_type": "code",
                "id": "scan-setup",
                "metadata": {},
                "outputs": [],
                "source": (
                    "import dotenv\n"
                    "dotenv.load_dotenv()\n"
                    "\n"
                    "import babamul.api as _api\n"
                    "from babamul import LsstAlert\n"
                    "from babamul.jupyter import scan_alerts\n"
                    "\n"
                    f"object_ids = {object_ids_repr}\n"
                    "\n"
                    "alerts = []\n"
                    "for oid in object_ids:\n"
                    "    try:\n"
                    "        alerts.append(_api.get_object('LSST', oid))\n"
                    "    except Exception as e:\n"
                    "        print(f'Failed {oid}: {e}')\n"
                    "\n"
                    "print(f'Loaded {len(alerts)} alerts.')\n"
                    "scan_alerts(alerts)"
                ),
            }
        ],
    }

    tmp = tempfile.NamedTemporaryFile(
        suffix=".ipynb", prefix="thor_scan_", delete=False, mode="w"
    )
    json.dump(nb, tmp)
    tmp.close()
    print(f"\nOpening scan notebook — press Ctrl+C in this terminal when done to close and clean up.")
    try:
        subprocess.run(["jupyter", "notebook", tmp.name], env=os.environ.copy())
    finally:
        os.unlink(tmp.name)


def _print_match_report(crossmatched_objects):
    n = len(crossmatched_objects)
    print(f"\nMatched Object IDs: {n}")

    if n > 100:
        print(f"More than 100 matches ({n} total) — skipping per-object summary.")
        return

    # column widths
    id_w = max(len("LSST Object ID"), max(len(str(oid)) for oid in crossmatched_objects))
    cat_w = 30
    z_w = 8
    sep_w = 10

    header = (
        f"{'LSST Object ID':<{id_w}}  "
        f"{'Catalog':<{cat_w}}  "
        f"{'z':>{z_w}}  "
        f"{'Sep (\")':>{sep_w}}"
    )
    divider = "-" * len(header)
    print(divider)
    print(header)
    print(divider)

    for obj_id, obj in crossmatched_objects.items():
        first = True
        for catalog, data in obj.items():
            if catalog == "LSST" or data is None:
                continue
            z_val = next((data[c] for c in Z_COLS if c in data and data[c] is not None), None)
            sep_val = data.get("conesearch_arcsecs")
            z_str = f"{z_val:.3f}" if z_val is not None else "—"
            sep_str = f"{sep_val:.2f}" if sep_val is not None else "—"
            id_str = str(obj_id) if first else ""
            print(
                f"{id_str:<{id_w}}  "
                f"{catalog:<{cat_w}}  "
                f"{z_str:>{z_w}}  "
                f"{sep_str:>{sep_w}}"
            )
            first = False

    print(divider)


def _print_prost_report(results_df):
    matched = results_df[results_df['best_cat'].notna()]
    n = len(matched)
    print(f"\nMatched Object IDs: {n}")

    if n > 100:
        print(f"More than 100 matches ({n} total) — skipping per-object summary.")
        return

    if n == 0:
        return

    def _isnan(v):
        try:
            return math.isnan(v)
        except (TypeError, ValueError):
            return v is None

    id_w   = max(len("LSST Object ID"), matched['name'].astype(str).str.len().max())
    cat_w  = 30
    z_w    = 8
    sep_w  = 10
    post_w = 10

    header = (
        f"{'LSST Object ID':<{id_w}}  "
        f"{'Catalog':<{cat_w}}  "
        f"{'z':>{z_w}}  "
        f"{'Sep (\")'  :>{sep_w}}  "
        f"{'Posterior':>{post_w}}"
    )
    divider = "-" * len(header)
    print(divider)
    print(header)
    print(divider)

    for _, row in matched.iterrows():
        first = True
        for prefix in ("host", "host_2"):
            objid = row.get(f"{prefix}_objID")
            post_val = row.get(f"{prefix}_total_posterior")
            if objid is None or _isnan(objid):
                continue
            if _isnan(post_val) or post_val < 0.3:
                continue
            z_val   = row.get(f"{prefix}_redshift_mean")
            sep_val = row.get(f"{prefix}_offset_mean")
            cat     = row.get("best_cat", "—")

            z_str    = f"{z_val:.3f}"    if not _isnan(z_val)    else "—"
            sep_str  = f"{sep_val:.2f}"  if not _isnan(sep_val)  else "—"
            post_str = f"{post_val:.3f}" if not _isnan(post_val) else "—"
            id_str   = str(row['name']) if first else ""

            print(
                f"{id_str:<{id_w}}  "
                f"{cat:<{cat_w}}  "
                f"{z_str:>{z_w}}  "
                f"{sep_str:>{sep_w}}  "
                f"{post_str:>{post_w}}"
            )
            first = False

    print(divider)


def main():
    dotenv.load_dotenv()
    parser = argparse.ArgumentParser(description="Fetch LSST alerts and crossmatch against catalogs.")
    parser.add_argument("--start", required=True, help="Start date (MM-DD-YYYY).")
    parser.add_argument("--end", required=True, help="End date (MM-DD-YYYY).")
    parser.add_argument(
        "--additional_filtering",
        default=None,
        choices=["tde_filter"],
        help="Optional additional filter to apply after crossmatch (default: none).",
    )
    parser.add_argument(
        "--save_raw_alerts",
        action="store_true",
        help="Save fetched alerts to data/lsst_alert_download/raw_files/ (default: off).",
    )
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="Save crossmatch results to data/lsst_alert_download/ (default: off).",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Open a temporary Jupyter notebook to scan candidates with scan_alerts (default: off).",
    )
    parser.add_argument(
        "--method",
        default="conesearch",
        choices=["conesearch", "prost"],
        help="Crossmatch method: conesearch (default) or prost (probabilistic host association).",
    )
    args = parser.parse_args()

    # ── Fetch alerts ──────────────────────────────────────────────────────────
    loaded_alerts = babamul_get_alerts(
        survey="LSST",
        start_time=args.start,
        end_time=args.end,
        min_drb=0.4,
        is_rock=False,
        is_star=False,
        is_near_brightstar=False,
        is_stationary=True,
        save=args.save_raw_alerts,
    )
    print(f"Loaded {len(loaded_alerts):,} alerts.")

    # ── Generic cuts ──────────────────────────────────────────────────────────
    filtered_alerts = filter_functions.filter_alerts(
        loaded_alerts,
        filter_functions.generic_filter,
    )

    # ── Deduplicate to unique objects ─────────────────────────────────────────
    filtered_objects = filter_functions.deduplicate_alerts(filtered_alerts)

    # ── Optional additional filtering (alert-based, method-independent) ───────
    if args.additional_filtering == "tde_filter":
        filtered_objects = filter_functions.filter_alerts(
            filtered_objects,
            filter_functions.tde_filter,
        )

    # ── Crossmatch against all available catalogs ─────────────────────────────
    if args.method == "prost":
        # Pre-filter with a fast cone search to avoid running prost on alerts
        # with no catalog coverage at all.
        cone_matches = filter_functions.catalog_crossmatch(
            alerts=filtered_objects,
            method="conesearch",
            radius_arcsec=10.0,
        )
        n_before = len(filtered_objects)
        filtered_objects = [a for a in filtered_objects if a.objectId in cone_matches]
        n_after = len(filtered_objects)
        print(f"10\" pre-filter: cut {n_before - n_after} candidates, running prost on {n_after}.")

    import time
    _t0 = time.monotonic()
    crossmatched_objects = filter_functions.catalog_crossmatch(
        alerts=filtered_objects,
        method=args.method,
    )
    _elapsed_min = (time.monotonic() - _t0) / 60
    print(f"\nRan crossmatch in {_elapsed_min:.1f} minutes on {len(filtered_objects):,} alerts from {args.start} to {args.end}.")

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "data" / "lsst_alert_download"

    # ── prost returns a DataFrame; handle separately ──────────────────────────
    if args.method == "prost":
        _print_prost_report(crossmatched_objects)
        if args.save_result:
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"crossmatch_candidates_{timestamp}.csv"
            crossmatched_objects.to_csv(out_file, index=False)
            print(f"\nSaved {len(crossmatched_objects):,} prost results to {out_file}")
        if args.scan:
            _launch_scan_notebook(crossmatched_objects['name'].tolist())
        return

    # ── Report ────────────────────────────────────────────────────────────────
    if not crossmatched_objects:
        print("\nNo crossmatch candidates found.")
        return

    _print_match_report(crossmatched_objects)

    # ── Optionally scan in Jupyter ────────────────────────────────────────────
    if args.scan:
        _launch_scan_notebook(list(crossmatched_objects.keys()))

    # ── Optionally save results ───────────────────────────────────────────────
    if not args.save_result:
        return

    serializable = {
        obj_id: {k: v for k, v in obj.items() if k != "LSST"}
        for obj_id, obj in crossmatched_objects.items()
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"crossmatch_candidates_{timestamp}.json"

    with open(out_file, "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    print(f"\nSaved {len(serializable):,} crossmatch candidates to {out_file}")


if __name__ == "__main__":
    main()
