"""
catalog_crossmatch_from_coords.py
----------------------------------
Run a catalog crossmatch for one or more RA/Dec coordinates and print results.

Usage
-----
Single coordinate:
    python -m thor.catalog_crossmatch_from_coords --coords 150.1234,2.5678

Multiple coordinates:
    python -m thor.catalog_crossmatch_from_coords --coords 150.1234,2.5678 34.5678,-5.1234

Options
-------
--coords RA,DEC [RA,DEC ...]
    One or more RA,Dec pairs in decimal degrees (comma-separated).
--radius ARCSEC
    Match radius in arcseconds (default: 5.0).
--catalog FILENAME
    Restrict to a specific catalog file (e.g. COSMOS2025_cut.fits).
    If omitted, all .fits files in data/catalogs/ are searched.
"""

import argparse
import math

from thor.utils import filter_functions
from thor.utils.fetch_alerts import angular_sep_to_parsecs_with_error

Z_COLS     = {"z", "Z_BEST", "ZPHOT", "zfinal", "zpdf_med"}
Z_UNC_COLS = {"z_unc", "z_err", "z_phot_err", "zphot_err", "ez_best", "redshift_err"}


def _print_match_report(crossmatched_objects, ra_list, dec_list):
    """Print a formatted table of crossmatch results keyed by coord index."""
    n_with_matches = sum(
        1 for obj in crossmatched_objects.values()
        if any(k != "LSST" and v is not None for k, v in obj.items())
    )
    print(f"\nCoordinates with matches: {n_with_matches} / {len(ra_list)}")

    id_w   = max(len("Coord (RA, Dec)"), max(
        len(f"({ra_list[i]:.4f}, {dec_list[i]:.4f})")
        for i in range(len(ra_list))
    ))
    cat_w  = 30
    z_w    = 14
    sep_w  = 10
    dist_w = 18

    header = (
        f"{'Coord (RA, Dec)':<{id_w}}  "
        f"{'Catalog':<{cat_w}}  "
        f"{'z':>{z_w}}  "
        f"{'Sep (\")':>{sep_w}}  "
        f"{'Dist (kpc)':>{dist_w}}"
    )
    divider = "-" * len(header)
    print(divider)
    print(header)
    print(divider)

    for idx, obj in crossmatched_objects.items():
        coord_label = f"({ra_list[idx]:.4f}, {dec_list[idx]:.4f})"
        any_match = False
        first = True
        for catalog, data in obj.items():
            if catalog == "LSST" or data is None:
                continue
            z_val      = next((data[c] for c in Z_COLS     if c in data and data[c] is not None), None)
            z_unc_val  = next((data[c] for c in Z_UNC_COLS if c in data and data[c] is not None), None)
            sep_val    = data.get("conesearch_arcsecs")
            sep_pc     = data.get("sep_pc")
            sep_err_pc = data.get("sep_err_pc")

            if z_val is not None and z_unc_val is not None:
                z_str = f"{z_val:.3f}±{z_unc_val:.3f}"
            elif z_val is not None:
                z_str = f"{z_val:.3f}"
            else:
                z_str = "—"

            sep_str = f"{sep_val:.2f}" if sep_val is not None else "—"
            if sep_pc is not None and sep_err_pc is not None:
                dist_str = f"{sep_pc/1000:.2f}±{sep_err_pc/1000:.2f} kpc"
            else:
                dist_str = "—"

            id_str = coord_label if first else ""
            print(
                f"{id_str:<{id_w}}  "
                f"{catalog:<{cat_w}}  "
                f"{z_str:>{z_w}}  "
                f"{sep_str:>{sep_w}}  "
                f"{dist_str:>{dist_w}}"
            )
            first = False
            any_match = True

        if not any_match:
            print(f"{coord_label:<{id_w}}  {'(no match)':<{cat_w}}")

    print(divider)


def main():
    parser = argparse.ArgumentParser(
        description="Crossmatch RA/Dec coordinates against catalogs and print results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--coords",
        nargs="+",
        metavar="RA,DEC",
        required=True,
        help="One or more RA,Dec pairs in decimal degrees (comma-separated, e.g. 150.12,2.56).",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=5.0,
        metavar="ARCSEC",
        help="Match radius in arcseconds (default: 5.0).",
    )
    parser.add_argument(
        "--catalog",
        default=None,
        metavar="FILENAME",
        help="Restrict to a specific catalog file (default: all catalogs).",
    )
    args = parser.parse_args()

    try:
        pairs = [c.split(",") for c in args.coords]
        ra_list  = [float(p[0]) for p in pairs]
        dec_list = [float(p[1]) for p in pairs]
    except (ValueError, IndexError):
        parser.error("--coords values must be comma-separated RA,Dec pairs, e.g. --coords 150.12,2.56")

    print(f"Crossmatching {len(ra_list)} coordinate(s) within {args.radius}\"...")

    results = filter_functions.catalog_crossmatch(
        ra=ra_list,
        dec=dec_list,
        catalog_name=args.catalog,
        radius_arcsec=args.radius,
        method="conesearch",
    )

    if not results:
        print("\nNo crossmatch candidates found.")
        return

    _print_match_report(results, ra_list, dec_list)


if __name__ == "__main__":
    main()
