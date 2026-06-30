import base64
import gzip
import json
import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import rcParams
rcParams["font.family"] = "Liberation Serif"
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from collections import Counter, defaultdict
import requests
import pandas as pd
from io import StringIO
import babamul
from babamul import LsstAlert, ZtfAlert
from pydantic import ValidationError

# ── Constants ─────────────────────────────────────────────────────────────────

BAND_COLORS = {
    "u": "#7b2d8b",
    "g": "#2ca02c",
    "r": "#d62728",
    "i": "#ff7f0e",
    "z": "#9467bd",
    "y": "#8c564b",
    "?": "#aec7e8",
}

# FIXME: verify and cite field coords
_DDF_CENTERS = {
    "COSMOS":    SkyCoord(150.1191,  2.2058,  unit="deg"),
    "XMM-LSS":  SkyCoord( 35.708,  -4.750,   unit="deg"),
    "ELAIS-S1": SkyCoord(  9.450,  -44.000,  unit="deg"),
    "ECDFS":    SkyCoord( 53.125,  -28.100,  unit="deg"),
    "EDFS-a":   SkyCoord( 58.900,  -49.315,  unit="deg"),
    "EDFS-b":   SkyCoord( 63.600,  -47.600,  unit="deg"),
}
_DDF_RADIUS   = 1.75  # degrees, Rubin FOV radius

_CERRO_PACHON = {"latitude": -30.2447, "longitude": -70.7494}
_CHILE_TZ     = ZoneInfo("America/Santiago")

_WEATHER_VARS = [
    "cloud_cover",
    "precipitation",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_gusts_10m",
    "dew_point_2m",
    "temperature_2m",
]

# ── Alert fetching ─────────────────────────────────────────────────────────────

def babamul_get_alerts(
    survey="LSST",
    *,
    object_id: str | None = None,
    ra: float | None = None,
    dec: float | None = None,
    radius_arcsec: float | None = None,
    start_time: float | str | None = None,
    end_time: float | str | None = None,
    min_magpsf: float | None = None,
    max_magpsf: float | None = None,
    min_drb: float | None = None,
    max_drb: float | None = None,
    is_rock: bool | None = None,
    is_star: bool | None = None,
    is_near_brightstar: bool | None = None,
    is_stationary: bool | None = None,
    checkpoint_path: str | None = None,
    checkpoint_every: int = 10,
):
    """
    Thin wrapper around babamul.api.get_alerts.

    Parameters
    ----------
    survey : str
        Survey to query ("ZTF" or "LSST"), default "LSST".
    object_id : str | None
        Filter by object ID.
    ra : float | None
        Right Ascension in degrees (requires dec and radius_arcsec).
    dec : float | None
        Declination in degrees (requires ra and radius_arcsec).
    radius_arcsec : float | None
        Cone search radius in arcseconds (max 600).
    start_jd : float | str | None
        Start filter as a JD float or date string "MM-DD-YYYY".
    end_jd : float | str | None
        End filter as a JD float or date string "MM-DD-YYYY".
    min_magpsf : float | None
        Minimum PSF magnitude filter.
    max_magpsf : float | None
        Maximum PSF magnitude filter.
    min_drb : float | None
        Minimum DRB (reliability) score filter.
    max_drb : float | None
        Maximum DRB score filter.
    is_rock : bool | None
        Filter for likely solar system objects.
    is_star : bool | None
        Filter for likely stellar sources.
    is_near_brightstar : bool | None
        Filter for sources near bright stars.
    is_stationary : bool | None
        Filter for likely stationary sources.

    Returns
    -------
    list of ZtfAlert | LsstAlert
    """
    def _to_jd(val):
        if isinstance(val, str):
            return Time(datetime.strptime(val, "%m-%d-%Y"), format="datetime", scale="utc").jd
        return val

    if start_time is None:
        start_time = "04-01-2026"
    if end_time is None:
        end_time = datetime.utcnow().strftime("%m-%d-%Y")

    alert_model = ZtfAlert if survey == "ZTF" else LsstAlert
    _base_params = {k: v for k, v in {
        "object_id": object_id, "ra": ra, "dec": dec,
        "radius_arcsec": radius_arcsec,
        "min_magpsf": min_magpsf, "max_magpsf": max_magpsf,
        "min_drb": min_drb, "max_drb": max_drb,
        "is_rock": is_rock, "is_star": is_star,
        "is_near_brightstar": is_near_brightstar, "is_stationary": is_stationary,
    }.items() if v is not None}

    def _fetch_raw(start_jd, end_jd, chunk_size=1/96):
        """Fetch raw alerts, splitting into 15-min then 5-min chunks if the 100k cap is hit."""
        params = {**_base_params, **({} if start_jd is None else {"start_jd": start_jd}), **({} if end_jd is None else {"end_jd": end_jd})}
        response = babamul.api._request("GET", f"/surveys/{survey}/alerts", params=params)
        raw = response.get("data", [])
        if len(raw) == 100_000:
            window = end_jd - start_jd
            min_chunk = 1/288  # 5 minutes
            if window <= min_chunk:
                print(f"  WARNING: JD {start_jd:.4f}–{end_jd:.4f} ({window*24*60:.0f} min) still at cap — results may be truncated")
                return raw
            next_chunk = min_chunk if window <= chunk_size else chunk_size
            print(f"  cap hit JD {start_jd:.4f}–{end_jd:.4f} ({window*24*60:.0f} min), splitting into {next_chunk*24*60:.0f}-min chunks")
            results = []
            t = start_jd
            while t < end_jd:
                t_end = min(t + next_chunk, end_jd)
                results.extend(_fetch_raw(t, t_end, chunk_size))
                t = t_end
            return results
        return raw

    def _fetch_and_validate(start_jd, end_jd):
        start_str = f"{start_jd:.5f}" if start_jd is not None else "any"
        end_str = f"{end_jd:.5f}" if end_jd is not None else "any"
        print(f"Fetching alerts for JD {start_str} to {end_str}...")
        raw_alerts = _fetch_raw(start_jd, end_jd)
        valid, skipped = [], []
        for raw in raw_alerts:
            try:
                valid.append(alert_model.model_validate(raw))
            except ValidationError:
                skipped.append(raw)
        return valid, skipped

    def _jd_to_datestr(jd):
        return Time(jd, format="jd").strftime("%m-%d-%Y")

    def _save_chunk(alerts, chunk_start_jd, chunk_end_jd, base_path):
        path = Path(base_path) / f"alerts_{survey}_{_jd_to_datestr(chunk_start_jd)}_to_{_jd_to_datestr(chunk_end_jd)}.json.gz"
        save_alerts(alerts, path)
        return path

    start = _to_jd(start_time)
    end   = _to_jd(end_time)

    base_path = checkpoint_path or "data/lsst_alert_download/raw_files"
    Path(base_path).mkdir(parents=True, exist_ok=True)

    all_alerts, all_skipped = [], []

    if start is None or end is None or math.ceil(end) - math.floor(start) <= 1:
        all_alerts, all_skipped = _fetch_and_validate(start, end)
    else:
        chunk_alerts = []
        chunk_start_jd = math.floor(start)
        last_saved_jd = None
        night_start = math.floor(start)

        try:
            while night_start < math.ceil(end):
                valid, skipped = _fetch_and_validate(
                    float(night_start), float(min(night_start + 1, end))
                )
                chunk_alerts.extend(valid)
                all_skipped.extend(skipped)
                night_start += 1

                nights_in_chunk = night_start - chunk_start_jd
                if nights_in_chunk >= checkpoint_every:
                    _save_chunk(chunk_alerts, chunk_start_jd, night_start, base_path)
                    last_saved_jd = night_start
                    all_alerts.extend(chunk_alerts)
                    chunk_alerts = []
                    chunk_start_jd = night_start

        except (KeyboardInterrupt, Exception) as e:
            if chunk_alerts:
                path = _save_chunk(chunk_alerts, chunk_start_jd, night_start, base_path)
                all_alerts.extend(chunk_alerts)
                print(f"\nInterrupted. Saved partial chunk to {path}.")
            if last_saved_jd:
                print(f"Last completed checkpoint: up to {_jd_to_datestr(last_saved_jd)} (JD {last_saved_jd}).")
                print(f"To resume, set start_time=\"{_jd_to_datestr(last_saved_jd)}\".")
            if not isinstance(e, KeyboardInterrupt):
                print(f"Error: {e}")
            return all_alerts

        # save any remaining nights
        if chunk_alerts:
            _save_chunk(chunk_alerts, chunk_start_jd, night_start, base_path)
            all_alerts.extend(chunk_alerts)

    if all_skipped:
        skipped_jds = [
            r.get("candidate", {}).get("jd") or r.get("candidate", {}).get("midpointMjdTai")
            for r in all_skipped
            if (r.get("candidate", {}).get("jd") or r.get("candidate", {}).get("midpointMjdTai"))
        ]
        print(f"Skipped {len(all_skipped):,} invalid alerts.")

    return all_alerts

# ── I/O ───────────────────────────────────────────────────────────────────────

def save_alerts(alerts, path="alerts.json.gz"):
    """
    Save a list of ZtfAlert or LsstAlert objects to a gzip-compressed JSON file.
    If the file already exists, appends new alerts, skipping any whose objectId
    is already present.

    Parameters
    ----------
    alerts : list of ZtfAlert | LsstAlert
    path : str or Path
        Destination file path. Defaults to "alerts.json.gz".
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _object_hook(obj):
        if "__bytes__" in obj:
            return base64.b64decode(obj["__bytes__"])
        return obj

    def _default(obj):
        if isinstance(obj, bytes):
            return {"__bytes__": base64.b64encode(obj).decode()}
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    existing = []
    existing_ids = set()
    if path.exists():
        try:
            with gzip.open(path, "rt") as f:
                existing = json.load(f, object_hook=_object_hook)
            existing_ids = {r.get("objectId") for r in existing}
        except (json.JSONDecodeError, EOFError, OSError) as e:
            print(f"Warning: could not read existing file ({e}), starting fresh.")
            existing = []

    new_records = [a.model_dump() for a in alerts if a.objectId not in existing_ids]
    skipped = len(alerts) - len(new_records)

    with gzip.open(path, "wt") as f:
        json.dump(existing + new_records, f, default=_default)

    print(f"Saved {len(new_records):,} new alerts to {path} (skipped {skipped:,} duplicates, {len(existing):,} already existed).")


def save_objects(ids, path="objects.json.gz"):
    """
    Save a list of object IDs to a gzip-compressed JSON file.
    If the file already exists, appends new IDs, skipping duplicates.

    Parameters
    ----------
    ids : list of str
    path : str or Path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing_ids = set()
    if path.exists():
        try:
            with gzip.open(path, "rt") as f:
                existing_ids = set(json.load(f))
        except (json.JSONDecodeError, EOFError, OSError) as e:
            print(f"Warning: could not read existing file ({e}), starting fresh.")

    new_ids = [i for i in ids if i not in existing_ids]
    skipped = len(ids) - len(new_ids)

    with gzip.open(path, "wt") as f:
        json.dump(list(existing_ids) + new_ids, f)

    print(f"Saved {len(new_ids):,} new object IDs to {path} (skipped {skipped:,} duplicates, {len(existing_ids):,} already existed).")


def load_objects(path):
    """
    Load a list of object IDs saved by save_objects.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    list of str
    """
    with gzip.open(Path(path), "rt") as f:
        ids = json.load(f)
    print(f"Loaded {len(ids):,} object IDs from {path}.")
    return ids


def fetch_latest_alerts(object_ids, survey="LSST"):
    """
    Fetch the most recent alert for each object ID.

    Parameters
    ----------
    object_ids : list of str
    survey : str
        "LSST" or "ZTF". Default "LSST".

    Returns
    -------
    list of ZtfAlert | LsstAlert
        One alert per object ID (the most recent by JD/MJD).
    """
    latest = []
    for oid in object_ids:
        alerts = babamul_get_alerts(survey=survey, object_id=oid)
        if not alerts:
            continue
        most_recent = max(
            alerts,
            key=lambda a: a.candidate.jd if hasattr(a.candidate, "jd") else a.candidate.midpointMjdTai,
        )
        latest.append(most_recent)
    print(f"Fetched latest alert for {len(latest):,}/{len(object_ids):,} objects.")
    return latest


def load_alerts(path, survey="LSST"):
    """
    Load alerts from a gzip-compressed JSON file saved by save_alerts.

    Parameters
    ----------
    path : str or Path
    survey : str
        "LSST" or "ZTF", used to select the correct model class.

    Returns
    -------
    list of ZtfAlert | LsstAlert
    """
    def _object_hook(obj):
        if "__bytes__" in obj:
            return base64.b64decode(obj["__bytes__"])
        return obj

    from babamul import LsstAlert, ZtfAlert
    model = ZtfAlert if survey == "ZTF" else LsstAlert
    with gzip.open(Path(path), "rt") as f:
        data = json.load(f, object_hook=_object_hook)
    print(f"Loaded {len(data):,} alerts from {path}.")
    return [model.model_validate(a) for a in data]


def combine_alert_files(input_dir, output_path, pattern="*.json.gz", input_files=None, delete_raw=False):
    """
    Combine alert chunk files into a single .json.gz file.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing chunk files (e.g. "data/lsst_alert_download/raw_files").
    output_path : str or Path
        Destination file for the combined output.
    pattern : str
        Glob pattern to match chunk files. Default "*.json.gz". Ignored if input_files is given.
    input_files : list of str, optional
        Specific filenames (within input_dir) to combine. If provided, pattern is ignored.
    delete_raw : bool
        If True, delete the input chunk files after combining. Default False.
    """
    input_dir = Path(input_dir)
    if input_files is not None:
        files = [input_dir / f for f in input_files]
        missing = [f for f in files if not f.exists()]
        if missing:
            raise FileNotFoundError(f"Missing files: {[str(f) for f in missing]}")
    else:
        files = sorted(input_dir.glob(pattern))
    if not files:
        print(f"No files matching '{pattern}' found in {input_dir}.")
        return

    combined = []
    for f in files:
        with gzip.open(f, "rt") as fh:
            combined.extend(json.load(fh))
        print(f"  {f.name}: {len(combined):,} alerts total")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt") as fh:
        json.dump(combined, fh)
    print(f"Saved {len(combined):,} alerts to {output_path}.")

    if delete_raw:
        for f in files:
            f.unlink()
        print(f"Deleted {len(files)} raw chunk files from {input_dir}.")


# ── Summaries ──────────────────────────────────────────────────────────────────

def summarize_night(alerts):
    """
    Print summary statistics for a list of LsstAlerts from a single night.

    Covers: total alerts, unique objects, visit count, UTC time window,
    and per-filter breakdown of alert count, visits, and magnitude range.
    """
    if not alerts:
        print("No alerts to summarize.")
        return

    def band_str(a):
        b = a.candidate.band
        return (b.value if hasattr(b, "value") else str(b)) if b is not None else "?"

    n_alerts  = len(alerts)
    n_objects = len({a.objectId for a in alerts})
    n_visits  = len({a.candidate.visit for a in alerts})

    jds     = [a.candidate.jd for a in alerts]
    t_start = Time(min(jds), format="jd", scale="utc").to_datetime().strftime("%Y-%m-%d %H:%M")
    t_end   = Time(max(jds), format="jd", scale="utc").to_datetime().strftime("%Y-%m-%d %H:%M")

    bands       = [band_str(a) for a in alerts]
    band_counts = Counter(bands)
    band_visits, band_mags = {}, {}
    for band in band_counts:
        ba = [a for a in alerts if band_str(a) == band]
        band_visits[band] = len({a.candidate.visit for a in ba})
        mags = [a.candidate.magpsf for a in ba if a.candidate.magpsf is not None]
        band_mags[band] = (np.median(mags), min(mags), max(mags)) if mags else (None, None, None)

    print("=== Night Summary ===")
    print(f"  Alerts:         {n_alerts:>7,}")
    print(f"  Unique objects: {n_objects:>7,}")
    print(f"  Unique visits:  {n_visits:>7,}")
    print(f"  Time (UTC):     {t_start}  →  {t_end}")
    print()
    print(f"  {'Band':<6}  {'Alerts':>7}  {'Visits':>7}  {'Med mag':>8}  {'Mag range'}")
    print(f"  {'----':<6}  {'------':>7}  {'------':>7}  {'-------':>8}  {'---------'}")
    for band in sorted(band_counts):
        med, lo, hi = band_mags[band]
        mag_str = f"{med:6.2f}      {lo:.1f} – {hi:.1f}" if med is not None else "  —"
        print(f"  {band:<6}  {band_counts[band]:>7,}  {band_visits[band]:>7,}  {mag_str}")


def summarize_ddf_alerts(alerts, radius_deg=_DDF_RADIUS): #FIXME: naive crossmatch
    if not alerts:
        print("No alerts.")
        return

    def band_str(a):
        b = a.candidate.band
        return (b.value if hasattr(b, "value") else str(b)) if b is not None else "?"

    coords = SkyCoord(
        ra  = [a.candidate.ra  for a in alerts],
        dec = [a.candidate.dec for a in alerts],
        unit="deg",
    )

    print(f"  {'Field':<12}  {'Alerts':>7}  Bands")
    print(f"  {'-----':<12}  {'------':>7}  -----")
    for name, center in _DDF_CENTERS.items():
        in_field = [a for a, m in zip(alerts, coords.separation(center) < radius_deg * u.deg) if m]
        if not in_field:
            print(f"  {name:<12}  {'—':>7}")
        else:
            bands = ", ".join(sorted({band_str(a) for a in in_field}))
            print(f"  {name:<12}  {len(in_field):>7,}  {bands}")

# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_alert_property(alerts, field, bins=30, log_scale=True):
    """
    Plot a histogram of a candidate property from a list of alerts.

    Parameters
    ----------
    field : str
        Attribute name, e.g. "magpsf", "snr", "psfFlux".
        Looks on alert.candidate first, then the alert itself.
    """
    _MISSING = object()

    def get_val(a):
        v = getattr(a.candidate, field, _MISSING)
        if v is _MISSING:
            v = getattr(a, field, _MISSING)
        return None if v is _MISSING else v

    values = [v for a in alerts if (v := get_val(a)) is not None]
    if not values:
        print(f"No non-null values found for field '{field}'.")
        return

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins, alpha=0.7)
    if log_scale:
        plt.yscale("log")
    plt.title(f"Distribution of {field}", fontsize=20)
    plt.xlabel(field, fontsize=16)
    plt.ylabel("Count", fontsize=16)
    plt.show()


_DDF_LABEL_OFFSETS = {
    "COSMOS":    (6, -14),  # shifted down to avoid overlap with RA axis labels
    "XMM-LSS":  (6,   4),
    "ELAIS-S1": (6,   4),
    "ECDFS":    (6,   4),
    "EDFS-a":   (10, -18),  # shifted further down from EDFS-b
    "EDFS-b":   (6,   4),
}

def _draw_ddf_markers(ax, mollweide=True):
    """Draw DDF field centres as labelled circles on ax."""
    for name, center in _DDF_CENTERS.items():
        ra_deg  = center.ra.deg
        dec_deg = center.dec.deg
        if mollweide:
            ra_rad = np.radians(ra_deg)
            ra_rad = ra_rad - 2 * np.pi if ra_rad > np.pi else ra_rad
            x, y = -ra_rad, np.radians(dec_deg)
        else:
            x, y = ra_deg, dec_deg
        ax.scatter(x, y, s=120, marker="o", facecolors="none",
                   edgecolors="black", linewidths=1.8, alpha=1.0, zorder=3)
        ax.scatter(x, y, s=120, marker="o", facecolors="none",
                   edgecolors="white", linewidths=0.9, alpha=1.0, zorder=4)
        offset = _DDF_LABEL_OFFSETS.get(name, (6, 4))
        ax.annotate(name, (x, y), textcoords="offset points", xytext=offset,
                    color="white", fontsize=17, fontweight="bold", alpha=0.9, zorder=3)


def plot_skymap(alerts, title=None, heatmap=False, bin_size_deg=3.5, plot_ddf=False):
    """
    Full-sky Mollweide projection with zoomed footprint inset and band legend.
    Colored by filter, sized by brightness (larger = brighter).

    If heatmap=True, the main Mollweide panel shows a 2-D density heatmap of
    alert counts binned into bin_size_deg × bin_size_deg cells instead of
    individual scatter points.  The zoom inset and band legend are hidden in
    this mode.

    If plot_ddf=True, DDF field centres are overlaid as labelled open circles
    on both the Mollweide panel and the zoom inset.
    """
    def band_str(a):
        b = a.candidate.band
        return (b.value if hasattr(b, "value") else str(b)) if b is not None else "?"

    if not alerts:
        print("No alerts to plot.")
        return

    all_ras  = np.array([a.candidate.ra  for a in alerts])
    all_decs = np.array([a.candidate.dec for a in alerts])

    if heatmap:
        fig = plt.figure(figsize=(14, 7), facecolor="#0f0f1a")
        ax_sky = fig.add_subplot(111, projection="mollweide")
        ax_sky.set_facecolor("#0f0f1a")

        # bin edges in degrees
        ra_bins  = np.arange(0,   360 + bin_size_deg, bin_size_deg)
        dec_bins = np.arange(-90, 90  + bin_size_deg, bin_size_deg)

        counts, _, _ = np.histogram2d(all_ras, all_decs, bins=[ra_bins, dec_bins])

        # convert bin edges to radians for Mollweide, wrap RA to [-pi, pi]
        ra_edge_rad  = np.radians(ra_bins)
        ra_edge_rad  = np.where(ra_edge_rad > np.pi, ra_edge_rad - 2 * np.pi, ra_edge_rad)
        dec_edge_rad = np.radians(dec_bins)

        ra_edge_grid, dec_edge_grid = np.meshgrid(ra_edge_rad, dec_edge_rad, indexing="ij")

        # mask zero-count cells so they stay transparent
        plot_counts = np.ma.masked_where(counts == 0, counts)

        sc = ax_sky.pcolormesh(
            -ra_edge_grid, dec_edge_grid, plot_counts,
            cmap="YlOrRd", alpha=0.85, shading="flat",
            norm=mcolors.LogNorm(vmin=1, vmax=plot_counts.max()),
        )
        cbar = fig.colorbar(sc, ax=ax_sky, orientation="horizontal",
                            pad=0.05, fraction=0.04, aspect=30)
        cbar.set_label("alerts per bin", color="#aaaaaa", fontsize=23)
        cbar.ax.tick_params(colors="#aaaaaa", labelsize=20)

        if plot_ddf:
            _draw_ddf_markers(ax_sky, mollweide=True)

        ax_sky.tick_params(colors="#bbbbbb", labelsize=17)
        # hide default equatorial RA labels and redraw at 15° dec
        ax_sky.set_xticklabels([])
        ra_label_deg = [d for d in np.arange(30, 360, 30) if d != 180]  # skip 12h
        ra_label_rad = np.radians(ra_label_deg)
        ra_label_rad = np.where(ra_label_rad > np.pi, ra_label_rad - 2 * np.pi, ra_label_rad)
        label_y = np.radians(15)
        for ra_r, ra_d in zip(ra_label_rad, ra_label_deg):
            hours = int(ra_d / 15)
            ax_sky.annotate(f"{hours}h", (-ra_r, label_y), ha="center", va="bottom",
                            color="#bbbbbb", fontsize=17, zorder=5)
        ax_sky.grid(True, alpha=0.3, color="white", linewidth=0.65)
        ax_sky.set_title(
            title or f"Alert density heatmap  ({bin_size_deg}° bins)",
            color="white", pad=14, fontsize=32,
        )
        plt.tight_layout()
        plt.show()
        return

    # --- original scatter plot ---
    all_mags  = np.array([a.candidate.magpsf for a in alerts])
    mag_max   = all_mags.max()
    mag_range = (all_mags.max() - all_mags.min()) or 1.0

    by_band = defaultdict(list)
    for a in alerts:
        by_band[band_str(a)].append(a)

    fig = plt.figure(figsize=(15, 7), facecolor="#0f0f1a")
    gs  = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[5, 2],
                           hspace=0.4, wspace=0.12)

    ax_sky  = fig.add_subplot(gs[:, 0], projection="mollweide")
    ax_zoom = fig.add_subplot(gs[0, 1])
    ax_leg  = fig.add_subplot(gs[1, 1])

    for ax in (ax_sky, ax_zoom, ax_leg):
        ax.set_facecolor("#0f0f1a")
    ax_leg.axis("off")
    for spine in ax_zoom.spines.values():
        spine.set_edgecolor("#444444")

    all_ras_list, all_decs_list = [], []

    for band in sorted(by_band):
        ba    = by_band[band]
        ras   = np.array([a.candidate.ra  for a in ba])
        decs  = np.array([a.candidate.dec for a in ba])
        mags  = np.array([a.candidate.magpsf for a in ba])
        color = BAND_COLORS.get(band, "#aec7e8")
        sizes = (mag_max - mags) / mag_range

        all_ras_list.append(ras); all_decs_list.append(decs)

        ra_rad = np.radians(ras)
        ra_rad = np.where(ra_rad > np.pi, ra_rad - 2 * np.pi, ra_rad)

        ax_sky.scatter(-ra_rad, np.radians(decs),
                       s=1 + 12 * sizes, c=color, alpha=0.7, linewidths=0,
                       label=f"{band}  ({len(ba):,})")
        ax_zoom.scatter(ras, decs,
                        s=2 + 28 * sizes, c=color, alpha=0.5, linewidths=0)

    if plot_ddf:
        _draw_ddf_markers(ax_sky, mollweide=True)
        _draw_ddf_markers(ax_zoom, mollweide=False)

    ax_sky.tick_params(colors="#aaaaaa", labelsize=13)
    ax_sky.set_xlabel("Right Ascension", color="#aaaaaa", fontsize=18)
    ax_sky.set_ylabel("Declination", color="#aaaaaa", fontsize=18)
    ax_sky.grid(True, alpha=0.2, color="white", linewidth=0.5)
    ax_sky.set_title(title or "Alert skymap", color="white", pad=14, fontsize=24)

    all_ras  = np.concatenate(all_ras_list)
    all_decs = np.concatenate(all_decs_list)
    pad = max(0.5, 0.05 * (all_ras.max() - all_ras.min()))
    ax_zoom.set_xlim(all_ras.max() + pad, all_ras.min() - pad)
    ax_zoom.set_ylim(all_decs.min() - pad, all_decs.max() + pad)
    ax_zoom.set_xlabel("RA (°)",  fontsize=12, color="#aaaaaa")
    ax_zoom.set_ylabel("Dec (°)", fontsize=12, color="#aaaaaa")
    ax_zoom.tick_params(labelsize=8, colors="#aaaaaa")
    ax_zoom.set_title("zoomed footprint", fontsize=14, color="#aaaaaa")

    handles, labels = ax_sky.get_legend_handles_labels()
    leg = ax_leg.legend(handles, labels, loc="center", title="band",
                        framealpha=0.4, labelcolor="#aaaaaa", facecolor="#1a1a2e",
                        edgecolor="#555555", fontsize=14, markerscale=5,
                        title_fontsize=15)
    leg.get_title().set_color("#aaaaaa")

    plt.tight_layout()
    plt.show()

# ── Weather ────────────────────────────────────────────────────────────────────

def fetch_rubin_weather(date_str=None):
    """
    Fetch hourly observing-relevant weather for a night at Cerro Pachón.

    Source: Open-Meteo (ERA5/ECMWF reanalysis) — model-based, not direct
    telescope telemetry. For actual site data, use the Rubin EFD on the RSP.

    Parameters
    ----------
    date_str : str, optional
        Evening date, e.g. "4-13-2026". Defaults to last night.

    Returns
    -------
    dict with "times" (list of Chile-local datetimes) and one key per variable.
    """
    now_utc   = datetime.now(timezone.utc)
    now_chile = now_utc.astimezone(_CHILE_TZ)

    if date_str:
        night_date = datetime.strptime(date_str, "%m-%d-%Y").date()
    else:
        night_date = now_chile.date() - timedelta(days=1)

    next_day    = night_date + timedelta(days=1)
    night_start = datetime(night_date.year, night_date.month, night_date.day,
                           20, tzinfo=_CHILE_TZ)
    night_end   = datetime(next_day.year, next_day.month, next_day.day,
                           8,  tzinfo=_CHILE_TZ)

    start_utc = night_start.astimezone(timezone.utc)
    end_utc   = night_end.astimezone(timezone.utc)
    days_ago  = (now_utc.date() - start_utc.date()).days

    params = {
        **_CERRO_PACHON,
        "hourly":          ",".join(_WEATHER_VARS),
        "wind_speed_unit": "kmh",
        "timezone":        "UTC",
    }

    if 0 <= days_ago <= 92:
        url = "https://api.open-meteo.com/v1/forecast"
        params["past_days"]     = days_ago + 1
        params["forecast_days"] = 0
    else:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params["start_date"] = start_utc.date().isoformat()
        params["end_date"]   = end_utc.date().isoformat()

    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    hourly = resp.json()["hourly"]

    times = [datetime.fromisoformat(t).replace(tzinfo=timezone.utc) for t in hourly["time"]]
    mask  = [start_utc <= t <= end_utc for t in times]

    result = {"times": [t.astimezone(_CHILE_TZ) for t, m in zip(times, mask) if m]}
    for var in _WEATHER_VARS:
        result[var] = [v for v, m in zip(hourly[var], mask) if m]

    _print_weather_summary(night_date, result)
    return result


def _print_weather_summary(night_date, d):
    times = d["times"]
    if not times:
        print(f"No weather data found for night of {night_date}.")
        return

    def stat(key, fn):
        vals = [v for v in d[key] if v is not None]
        return fn(vals) if vals else float("nan")

    t0, t1 = times[0].strftime("%H:%M"), times[-1].strftime("%H:%M")

    print(f"=== Weather: night of {night_date} — Cerro Pachón ===")
    print(f"  Window (CLT):     {t0} – {t1}")
    print()
    print(f"  Cloud cover:      {stat('cloud_cover', np.mean):.0f}% mean"
          f"  /  {stat('cloud_cover', max):.0f}% max")
    print(f"  Precipitation:    {stat('precipitation', sum):.1f} mm total")
    print(f"  Humidity:         {stat('relative_humidity_2m', np.mean):.0f}% mean"
          f"  /  {stat('relative_humidity_2m', max):.0f}% max")
    print(f"  Dew point:        {stat('dew_point_2m', np.mean):.1f} °C mean")
    print(f"  Wind speed:       {stat('wind_speed_10m', np.mean):.1f} km/h mean")
    print(f"  Wind gusts:       {stat('wind_gusts_10m', max):.1f} km/h max")
    print(f"  Temperature:      {stat('temperature_2m', min):.1f}"
          f" – {stat('temperature_2m', max):.1f} °C")
    print()

    flags = []
    if stat("cloud_cover",          max) > 50: flags.append(f"clouds ({stat('cloud_cover', max):.0f}%)")
    if stat("precipitation",        sum) >  0: flags.append(f"precip ({stat('precipitation', sum):.1f} mm)")
    if stat("relative_humidity_2m", max) > 85: flags.append(f"humidity ({stat('relative_humidity_2m', max):.0f}%)")
    if stat("wind_gusts_10m",       max) > 60: flags.append(f"gusts ({stat('wind_gusts_10m', max):.0f} km/h)")

    if flags:
        print(f"  Flags:            {', '.join(flags)}")
    else:
        print("  Flags:            none — likely photometric")

# ── Rubin schedule (ObsLocTAP) ─────────────────────────────────────────────────

_RUBIN_TAP    = "https://usdf-rsp.slac.stanford.edu/obsloctap/tap"
_OBSTAP_TABLE = "ivoa.ObsLocTAP"   # verify with discover_tap_tables() if queries fail


def _to_mjd(dt_aware):
    return Time(
        dt_aware.astimezone(timezone.utc).replace(tzinfo=None),
        format="datetime", scale="utc",
    ).mjd


def _night_mjd(date_str):
    """Return (mjd_start, mjd_end) for the night beginning on date_str's evening."""
    night_date = datetime.strptime(date_str, "%m-%d-%Y").date()
    next_day   = night_date + timedelta(days=1)
    return (
        _to_mjd(datetime(night_date.year, night_date.month, night_date.day,
                         20, 0, 0, tzinfo=_CHILE_TZ)),
        _to_mjd(datetime(next_day.year, next_day.month, next_day.day,
                         8,  0, 0, tzinfo=_CHILE_TZ)),
    )


def _tap_query(query, token=None, timeout=60):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    resp = requests.post(
        f"{_RUBIN_TAP}/sync",
        data={"REQUEST": "doQuery", "LANG": "ADQL", "QUERY": query, "FORMAT": "csv"},
        headers=headers,
        timeout=timeout,
    )
    resp.raise_for_status()
    return pd.read_csv(StringIO(resp.text))


def discover_tap_tables(token=None):
    """
    Print the raw table schema from the ObsLocTAP service.
    Run this first to verify the correct table and column names.
    """
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    resp = requests.get(f"{_RUBIN_TAP}/tables", headers=headers, timeout=30)
    resp.raise_for_status()
    print(resp.text[:3000])


def fetch_rubin_schedule(start_date, end_date=None, token=None):
    """
    Fetch Rubin observations from the ObsLocTAP service at USDF-RSP.

    Parameters
    ----------
    start_date : str
        Evening date of the first night, e.g. "4-10-2026".
    end_date : str, optional
        Evening date of the last night. Defaults to start_date (single night).
    token : str, optional
        RSP bearer token. Required for authenticated access.
        Get from: USDF-RSP portal → User menu → Copy auth token.

    Returns
    -------
    pd.DataFrame with one row per observation.
    """
    mjd_start, _  = _night_mjd(start_date)
    _, mjd_end    = _night_mjd(end_date if end_date else start_date)

    query = f"""
        SELECT *
        FROM   {_OBSTAP_TABLE}
        WHERE  t_min >= {mjd_start}
        AND    t_max <= {mjd_end}
        ORDER BY t_min
    """
    df = _tap_query(query.strip(), token=token)
    label = f"{start_date}{' – ' + end_date if end_date else ''}"
    print(f"fetched {len(df):,} observations ({label}).")
    return df


def summarize_schedule(df, field_col="target_name"):
    """
    Print summary statistics for a set of Rubin observations.

    Parameters
    ----------
    df : pd.DataFrame
        From fetch_rubin_schedule.
    field_col : str
        Column to use as the field identifier.
        Common options: "target_name", "fieldId". Check df.columns if unsure.
    """
    if df.empty:
        print("No observations.")
        return

    print("=== Schedule Summary ===")
    print(f"  Total observations:  {len(df):,}")
    print()

    if "execution_status" in df.columns:
        print("  Execution status:")
        for status, n in df["execution_status"].value_counts().items():
            print(f"    {status:<20}  {n:>6,}")
        executed = df[df["execution_status"].str.lower() == "executed"]
    else:
        print("  (execution_status column not found — showing all rows)")
        executed = df

    if executed.empty:
        print("\n  No executed observations found.")
        return

    print(f"\n  Executed: {len(executed):,}")

    # Filter/band breakdown
    filter_col = next((c for c in ["filter", "band", "em_filter"] if c in executed.columns), None)
    if filter_col:
        print(f"\n  Filter breakdown (executed):")
        for f, n in executed[filter_col].value_counts().sort_index().items():
            print(f"    {str(f):<6}  {n:>6,}")

    # Visits per field
    if field_col in executed.columns:
        field_counts = executed[field_col].value_counts()
        print(f"\n  Visits per field — top 15 (col: '{field_col}'):")
        print(f"  {'Field':<30}  {'Visits':>7}")
        print(f"  {'-----':<30}  {'------':>7}")
        for field, n in field_counts.head(15).items():
            print(f"  {str(field):<30}  {n:>7,}")
    else:
        print(f"\n  Column '{field_col}' not found. Available: {list(df.columns)}")
