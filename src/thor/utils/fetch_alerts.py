import base64
import gzip
import json
import math
from datetime import datetime
from pathlib import Path

import babamul
from astropy.time import Time
from babamul import LsstAlert, ZtfAlert
from pydantic import ValidationError

_LIST_FIELDS = frozenset(("prv_candidates", "fp_hists"))
_FLOAT_SUFFIXES = ("chi2", "rate", "rate_error", "dt", "jd", "mag", "err", "Err", "flux", "Flux")
# Defaults injected for required fields added to the babamul schema after old alerts were saved.
_MISSING_BOOL_DEFAULTS = {"isNegative": False}


def _coerce_alert(obj):
    """Recursively fix API nulls that babamul models don't allow:
    - None list fields → []
    - None float fields (by name heuristic) → nan
    - Missing required bool fields (schema additions) → default value
    """
    if isinstance(obj, list):
        return [_coerce_alert(i) for i in obj]
    if not isinstance(obj, dict):
        return obj
    out = {}
    for k, v in obj.items():
        if v is None and k in _LIST_FIELDS:
            out[k] = []
        elif v is None and any(k.endswith(s) for s in _FLOAT_SUFFIXES):
            out[k] = float("nan")
        else:
            out[k] = _coerce_alert(v)
    for field, default in _MISSING_BOOL_DEFAULTS.items():
        if field not in out:
            out[field] = default
    return out


# ── Monkey-patch babamul.api.get_object to coerce nulls before model validation ──
# babamul's lazy photometry fetch calls get_object internally. Some LSST alerts
# return null for fields like psfFluxErr that the model requires to be floats.
# Wrapping get_object with _coerce_alert fixes these the same way we fix saved alerts.
import babamul.api as _babamul_api

_orig_get_object = _babamul_api.get_object

def _patched_get_object(survey, object_id):
    import base64 as _base64
    from babamul import LsstAlert as _LsstAlert, ZtfAlert as _ZtfAlert
    from babamul.api import _request, get_args, Survey
    response = _request("GET", f"/surveys/{survey}/objects/{object_id}")
    data = response.get("data", response)
    for key in ["cutoutScience", "cutoutTemplate", "cutoutDifference"]:
        if data.get(key) and isinstance(data[key], str):
            data[key] = _base64.b64decode(data[key])
    data = _coerce_alert(data)
    if survey == "ZTF":
        return _ZtfAlert.model_validate(data)
    elif survey == "LSST":
        return _LsstAlert.model_validate(data)
    else:
        valid_surveys = ", ".join(get_args(Survey))
        raise ValueError(f"Survey {survey} is not supported, must be one of: {valid_surveys}")

_babamul_api.get_object = _patched_get_object

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
    is_rock: bool | None = None,
    is_star: bool | None = None,
    is_near_brightstar: bool | None = None,
    is_stationary: bool | None = None,
    checkpoint_path: str | None = None,
    checkpoint_every: int = 10,
    save: bool = True,
):
    """
    Thin wrapper around babamul.api.get_alerts.
    Iterates to work around babamul's 100k alert cap, 1 day limit on time range

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
    is_rock : bool | None
        Filter for likely solar system objects.
    is_star : bool | None
        Filter for likely stellar sources.
    is_near_brightstar : bool | None
        Filter for sources near bright stars.
    is_stationary : bool | None
        Filter for likely stationary sources.
    save : bool
        If True (default), save alerts to disk in data/lsst_alert_download/raw_files/.
        For multi-day ranges, alerts are saved in chunks of checkpoint_every nights.
        For single-day fetches, all alerts are saved in one file on completion.
        Set to False to return alerts in memory only.

    Returns
    -------
    list of ZtfAlert | LsstAlert
    """
    # ── Helpers ───────────────────────────────────────────────────────────────

    def _to_jd(val):
        if isinstance(val, str):
            return Time(datetime.strptime(val, "%m-%d-%Y"), format="datetime", scale="utc").jd
        return val

    def _jd_to_datestr(jd):
        return Time(jd, format="jd").strftime("%m-%d-%Y")

    def _fetch_raw(start_jd, end_jd, chunk_size=1/96):
        """Fetch raw alerts, splitting into 15-min then 5-min chunks if the 100k cap is hit."""
        params = {**base_params}
        if start_jd is not None:
            params["start_jd"] = start_jd
        if end_jd is not None:
            params["end_jd"] = end_jd
        raw = babamul.api._request("get", f"/surveys/{survey.lower()}/alerts", params=params).get("data", [])
        if len(raw) < 100_000:
            return raw
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

    def _fetch_and_validate(start_jd, end_jd):
        start_str = f"{start_jd:.5f}" if start_jd is not None else "any"
        end_str = f"{end_jd:.5f}" if end_jd is not None else "any"
        print(f"Fetching alerts for JD {start_str} to {end_str}...")
        valid, skipped = [], []
        first_error = None
        for raw in _fetch_raw(start_jd, end_jd):
            try:
                valid.append(alert_model.model_validate(_coerce_alert(raw)))
            except ValidationError as e:
                skipped.append(raw)
                if first_error is None:
                    first_error = e
        if first_error is not None:
            print(f"  First validation error: {first_error}")
        return valid, skipped

    def _save_chunk(alerts, chunk_start_jd, chunk_end_jd):
        path = Path(base_path) / f"alerts_{survey}_{_jd_to_datestr(chunk_start_jd)}_to_{_jd_to_datestr(chunk_end_jd)}.json.gz"
        save_alerts(alerts, path)
        return path

    # ── Setup ─────────────────────────────────────────────────────────────────

    start = _to_jd(start_time) if start_time is not None else Time("2026-04-01").jd
    end   = _to_jd(end_time)   if end_time   is not None else Time.now().jd

    alert_model = ZtfAlert if survey == "ZTF" else LsstAlert
    base_params = {k: v for k, v in {
        "object_id": object_id, "ra": ra, "dec": dec,
        "radius_arcsec": radius_arcsec,
        "min_magpsf": min_magpsf, "max_magpsf": max_magpsf,
        "min_drb": min_drb,
        "is_rock": is_rock, "is_star": is_star,
        "is_near_brightstar": is_near_brightstar, "is_stationary": is_stationary,
    }.items() if v is not None}

    _repo_root = Path(__file__).resolve().parents[3]
    base_path = Path(checkpoint_path) if checkpoint_path else _repo_root / "data" / "lsst_alert_download" / "raw_files"
    if save:
        Path(base_path).mkdir(parents=True, exist_ok=True)

    # ── Fetch ─────────────────────────────────────────────────────────────────

    all_alerts, all_skipped = [], []

    if start is None or end is None or math.ceil(end) - math.floor(start) <= 1:
        all_alerts, all_skipped = _fetch_and_validate(start, end)
        if save and all_alerts:
            _save_chunk(all_alerts, math.floor(start), math.ceil(end))
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

                if save and night_start - chunk_start_jd >= checkpoint_every:
                    _save_chunk(chunk_alerts, chunk_start_jd, night_start)
                    last_saved_jd = night_start
                    all_alerts.extend(chunk_alerts)
                    chunk_alerts = []
                    chunk_start_jd = night_start

        except BaseException as e:
            if chunk_alerts:
                all_alerts.extend(chunk_alerts)
                if save:
                    path = _save_chunk(chunk_alerts, chunk_start_jd, night_start)
                    print(f"\nInterrupted. Saved partial chunk to {path}.")
                else:
                    print(f"\nInterrupted.")
            if last_saved_jd:
                print(f"Last completed checkpoint: up to {_jd_to_datestr(last_saved_jd)} (JD {last_saved_jd}).")
                print(f"To resume, set start_time=\"{_jd_to_datestr(last_saved_jd)}\".")
            if not isinstance(e, KeyboardInterrupt):
                print(f"Error: {e}")
            return all_alerts

        if chunk_alerts:
            if save:
                _save_chunk(chunk_alerts, chunk_start_jd, night_start)
            all_alerts.extend(chunk_alerts)

    if all_skipped:
        print(f"Skipped {len(all_skipped):,} invalid alerts.")

    print(f"Fetched {len(all_alerts):,} alerts total.")
    return all_alerts

# ── I/O ───────────────────────────────────────────────────────────────────────

def save_alerts(alerts, path="alerts.json.gz"):
    """
    Save a list of ZtfAlert or LsstAlert objects to a gzip-compressed JSON file.
    If the file already exists, appends new alerts, skipping any whose objectId
    is already present.
    Function called within babamul_get_alerts

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
    Used to save candidates locally, as alternative to working on Fritz.

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


def fetch_latest_alerts(object_ids, survey="LSST", save=True, verbose=True):
    """
    Fetch the current object record for each object ID using a single API call
    per object (get_object), which is much faster than pulling full alert history.

    Parameters
    ----------
    object_ids : list of str
    survey : str
        "LSST" or "ZTF". Default "LSST".
    save : bool
        If True (default), save the fetched alerts to
        data/lsst_alert_download/raw_files/latest_alerts_{survey}.json.gz.
    verbose : bool
        If True (default), print progress per object.

    Returns
    -------
    list of ZtfAlert | LsstAlert
        One alert per successfully fetched object ID.
    """
    latest = []
    failed = []
    for oid in object_ids:
        if verbose:
            print(f"Fetching {oid}...")
        try:
            alert = _babamul_api.get_object(survey, oid)
            latest.append(alert)
        except Exception as e:
            failed.append(oid)
            if verbose:
                print(f"  Failed for {oid}: {e}")
    print(f"Fetched {len(latest):,}/{len(object_ids):,} objects ({len(failed):,} failed).")
    if save and latest:
        _repo_root = Path(__file__).resolve().parents[3]
        out_path = _repo_root / "data" / "lsst_alert_download" / "raw_files" / f"latest_alerts_{survey}.json.gz"
        save_alerts(latest, out_path)
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

    model = ZtfAlert if survey == "ZTF" else LsstAlert
    with gzip.open(Path(path), "rt") as f:
        data = json.load(f, object_hook=_object_hook)
    print(f"Loaded {len(data):,} alerts from {path}.")

    alerts = [model.model_validate(_coerce_alert(a)) for a in data]
    # The main alerts API doesn't return photometry history inline — those fields
    # are fetched lazily by babamul when show() is called. _coerce_alert converts
    # the None values to [] so pydantic accepts them, but that tricks babamul into
    # thinking photometry was already fetched. Reset to None to re-enable lazy fetch.
    for alert in alerts:
        if not alert.prv_candidates and not alert.fp_hists:
            alert.prv_candidates = None
            alert.fp_hists = None
    return alerts


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