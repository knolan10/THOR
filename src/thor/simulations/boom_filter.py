import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "alert_stream"))

import filter_functions
from rubin_stats_functions import load_alerts


def main():
    parser = argparse.ArgumentParser(description="Run TDE filter pipeline on local alert data.")
    parser.add_argument(
        "--test_data",
        required=True,
        help="Path to directory containing .json.gz alert files.",
    )
    args = parser.parse_args()

    data_dir = Path(args.test_data)
    if not data_dir.is_dir():
        raise ValueError(f"--test_data must be a directory, got: {data_dir}")

    # ── Load alerts from all .json.gz files in the directory ─────────────────
    alert_files = sorted(data_dir.glob("*.json.gz"))
    if not alert_files:
        raise FileNotFoundError(f"No .json.gz files found in {data_dir}")

    loaded_alerts = []
    for f in alert_files:
        loaded_alerts.extend(load_alerts(f, survey="LSST"))
    print(f"Loaded {len(loaded_alerts):,} alerts from {len(alert_files)} file(s).")

    # ── Generic cuts (drb, rock, star, near_brightstar, stationary, PSF) ─────
    filtered_alerts = filter_functions.filter_alerts(
        loaded_alerts,
        filter_functions.generic_filter,
    )

    # ── TDE-specific cuts (>=5 detections, no Milliquas match, rising) ────────
    tde_candidates = filter_functions.filter_alerts(
        filtered_alerts,
        filter_functions.tde_filter,
    )

    # ── Save results ──────────────────────────────────────────────────────────
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("filter_results") / f"filter_test_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "timestamp": timestamp,
        "test_data": str(data_dir),
        "n_loaded": len(loaded_alerts),
        "n_after_generic": len(filtered_alerts),
        "n_tde_candidates": len(tde_candidates),
        "candidates": [a.objectId for a in tde_candidates],
    }
    out_file = out_dir / "results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out_file}")
    print(f"  Loaded:          {results['n_loaded']:,}")
    print(f"  After generic:   {results['n_after_generic']:,}")
    print(f"  TDE candidates:  {results['n_tde_candidates']:,}")


if __name__ == "__main__":
    main()
