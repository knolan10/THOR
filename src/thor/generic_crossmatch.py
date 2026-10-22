import json
import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from thor.utils import filter_functions


def crossmatch_objects(
    path_objects: str,
    xmatch_method: str = "conesearch",
    save: bool = False,
) -> dict:
    """
    Crossmatch a CSV of objects against all available catalogs.

    Parameters
    ----------
    path_objects : str or Path
        Path to a CSV with at least 'ra' and 'dec' columns (degrees).
        An 'objid' column is used as the object identifier; if absent,
        'id' or 'name' are tried, then the row index.
    xmatch_method : str
        'conesearch' (default) or 'prost'.
    save : bool
        If True, save results to data/crossmatch_results/.

    Returns
    -------
    dict
        Keys are objids from the input catalog. Each value is a dict
        keyed by catalog name, with the matched row as the value.
        Only catalogs with a match are included per object.
    """
    df = pd.read_csv(path_objects)

    missing = [c for c in ("ra", "dec") if c not in df.columns]
    if missing:
        raise ValueError(f"Input catalog is missing required columns: {missing}")

    # Determine object ID column
    for candidate in ("objid", "id", "name"):
        if candidate in df.columns:
            objid_col = candidate
            break
    else:
        objid_col = None  # will use row index

    df = df.dropna(subset=["ra", "dec"])

    ra_list  = df["ra"].tolist()
    dec_list = df["dec"].tolist()
    objids   = df[objid_col].tolist() if objid_col else list(df.index)

    print(f"Loaded {len(df):,} objects from {path_objects} (using '{objid_col or 'index'}' as objid).")

    if xmatch_method == "prost":
        # prost path: pass objids as names so the result DataFrame uses them directly
        results_df = filter_functions.catalog_crossmatch(
            ra=ra_list,
            dec=dec_list,
            method="prost",
        )
        # Build dict from prost DataFrame: keyed by objid, value is flat row dict per catalog
        result = {}
        for idx, (_, row) in enumerate(results_df.iterrows()):
            objid = objids[idx]
            best_cat = row.get("best_cat")
            if pd.isna(best_cat) or best_cat is None:
                continue
            row_dict = row.to_dict()
            result[objid] = {best_cat: row_dict}
        _print_summary(result, xmatch_method)
    else:
        # conesearch path: returns dict keyed by integer index
        raw = filter_functions.catalog_crossmatch(
            ra=ra_list,
            dec=dec_list,
            method="conesearch",
        )
        # Remap integer keys to actual objids; strip the "LSST" placeholder key
        result = {}
        for int_key, cat_entries in raw.items():
            objid = objids[int_key]
            result[objid] = {k: v for k, v in cat_entries.items() if k != "LSST"}
        _print_summary(result, xmatch_method)

    if save:
        repo_root = Path(__file__).resolve().parents[2]
        out_dir = repo_root / "data" / "crossmatch_results"
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        stem = Path(path_objects).stem
        out_file = out_dir / f"{stem}_crossmatch_{timestamp}.json"
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nSaved {len(result):,} crossmatch results to {out_file}")

    return result


def _print_summary(result: dict, method: str) -> None:
    n_total = len(result)
    print(f"\n{'='*50}")
    print(f"Crossmatch summary ({method}): {n_total} objects matched in at least one catalog.")

    # Count matches per catalog
    catalog_counts: dict[str, int] = {}
    for obj in result.values():
        for cat, val in obj.items():
            if val is not None:
                catalog_counts[cat] = catalog_counts.get(cat, 0) + 1

    if catalog_counts:
        cat_w = max(len(c) for c in catalog_counts)
        print(f"\n  {'Catalog':<{cat_w}}  Matches")
        print(f"  {'-'*cat_w}  -------")
        for cat, count in sorted(catalog_counts.items(), key=lambda x: -x[1]):
            print(f"  {cat:<{cat_w}}  {count:,}")
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Crossmatch a CSV of objects against all available catalogs."
    )
    parser.add_argument("path_objects", help="Path to input CSV (must have ra, dec columns).")
    parser.add_argument(
        "--method",
        default="conesearch",
        choices=["conesearch", "prost"],
        help="Crossmatch method (default: conesearch).",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to data/crossmatch_results/ (default: off).",
    )
    args = parser.parse_args()

    crossmatch_objects(
        path_objects=args.path_objects,
        xmatch_method=args.method,
        save=args.save,
    )


if __name__ == "__main__":
    main()
