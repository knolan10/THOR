import os
from pathlib import Path
from babamul import LsstAlert, ZtfAlert
from babamul.models import add_cross_matches
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
import astropy.units as u
import pandas as pd
import numpy as np
from collections import defaultdict

from thor.utils.fetch_alerts import babamul_get_alerts



def generic_filter(alerts: list[ZtfAlert | LsstAlert]) -> list[ZtfAlert | LsstAlert]:
    """
    Filter with high level cuts to select generically for astrophysical sources.
    Used when fetching alerts from Babamul API and filtering locally
    https://sdm-schemas.lsst.io/apdb.html
    """
    def _keep(alert: ZtfAlert | LsstAlert) -> bool:
        if isinstance(alert, ZtfAlert):
            return False  # for now we read from the no ztf match topic
        if isinstance(alert, LsstAlert):
            if alert.candidate.isDipole:  # Source well fit by a dipole, ie image subtraction artifact
                return False
            if alert.candidate.psfFlux_flag:  # Failure to derive linear least-squares fit of psf model.
                return False
            # Only consider alerts with a reasonable PSF fit (using a threshold on the reduced chi2 of the PSF fit)
            if alert.candidate.psfChi2 / alert.candidate.psfNdata > 10.0:
                return False
            if alert.candidate.snr < 3.0:
                return False
            if alert.candidate.extendedness is None:  # 0-1 where 1=extended.
                return False
            if alert.candidate.shape_flag:  # set if anything went wrong when measuring the shape
                return False
            if alert.candidate.centroid_flag:  # set if anything went wrong when fitting the centroid
                return False

        if alert.drb < 0.4:  # Only consider alerts with a real-bogus score above 0.4
            return False
        if alert.properties.rock:  # Exclude alerts that are likely to be known SSOs
            return False
        if alert.properties.star:  # Exclude likely stars (PS1 PSC for ZTF, LSPSC for LSST)
            return False
        if alert.properties.near_brightstar:  # same here
            return False
        if not alert.candidate.isdiffpos:  # Only consider positive subtractions
            return False
        # Filter for bright mag in PSF photometry - currently don't do this, as our sources are dim
        # if alert.candidate.magpsf is None or alert.candidate.magpsf > 21.5:
        #     return False
        return alert.properties.stationary  # Must be detected at least twice with sufficient time separation

    return [a for a in alerts if _keep(a)]



def _band_mags_from_alerts(
    object_alerts: list[ZtfAlert | LsstAlert],
) -> dict[str, list[float]]:
    """
    Extract {band: [magnitudes]} from a list of per-epoch alerts.

    For ZTF uses candidate.fid + candidate.magpsf.
    For LSST converts candidate.psfFlux (nJy) to AB magnitude using zp=31.4.
    """
    ZTF_FID = {1: "g", 2: "r", 3: "i"}
    LSST_ZP = 31.4  # AB mag, flux in nJy

    band_mags: dict[str, list[float]] = {}
    for a in object_alerts:
        if isinstance(a, ZtfAlert):
            band = ZTF_FID.get(a.candidate.fid)
            mag = a.candidate.magpsf
        else:
            band = str(a.candidate.band) if a.candidate.band is not None else None
            flux = a.candidate.psfFlux
            mag = (-2.5 * np.log10(flux) + LSST_ZP) if (flux and flux > 0) else None
        if band and mag is not None:
            band_mags.setdefault(band, []).append(mag)
    return band_mags


def _band_series_from_alerts(
    object_alerts: list[ZtfAlert | LsstAlert],
) -> dict[str, list[tuple[float, float]]]:
    """
    Extract {band: [(time, mag), ...]} from a list of per-epoch alerts, sorted by time.

    Multiple detections in the same band on the same night (same integer day) are
    collapsed to a single point: the median magnitude at the median time.

    Time is JD for ZTF and MJD for LSST 
    """
    ZTF_FID = {1: "g", 2: "r", 3: "i"}
    LSST_ZP = 31.4

    # group raw detections by (band, night)
    nightly: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
    for a in object_alerts:
        if isinstance(a, ZtfAlert):
            band = ZTF_FID.get(a.candidate.fid)
            mag = a.candidate.magpsf
            t = a.candidate.jd
        else:
            band = str(a.candidate.band) if a.candidate.band is not None else None
            flux = a.candidate.psfFlux
            mag = (-2.5 * np.log10(flux) + LSST_ZP) if (flux and flux > 0) else None
            t = a.candidate.midpointMjdTai
        if band and mag is not None and t is not None:
            nightly[(band, int(t))].append((t, mag))

    # collapse each night to median time + median magnitude
    series: dict[str, list[tuple[float, float]]] = {}
    for (band, _night), pts in nightly.items():
        t_med = float(np.median([p[0] for p in pts]))
        m_med = float(np.median([p[1] for p in pts]))
        series.setdefault(band, []).append((t_med, m_med))

    for band in series:
        series[band].sort(key=lambda x: x[0])
    return series


def _is_rising(
    object_alerts: list[ZtfAlert | LsstAlert],
    min_separation_days: float = 1.0,
) -> bool:
    """
    Return True if at least one band has three detections each separated by at
    least ``min_separation_days``, and those three show a rising brightness trend.

    Requires the first such trio per band (greedy earliest selection).
    Returns False if no band can produce three sufficiently separated detections.
    """
    series = _band_series_from_alerts(object_alerts)

    for pts in series.values():
        # greedily pick detections at least min_separation_days apart
        separated = [pts[0]]
        for t, mag in pts[1:]:
            if t - separated[-1][0] >= min_separation_days:
                separated.append((t, mag))
            if len(separated) == 3:
                break

        if len(separated) < 3:
            continue

        ts = np.array([p[0] for p in separated])
        mags = np.array([p[1] for p in separated])
        slope = np.polyfit(ts - ts[0], mags, 1)[0]
        if slope < 0:  # magnitude decreasing = brightness rising
            return True

    return False


def tde_filter(
    alerts: list[ZtfAlert | LsstAlert],
    min_detections: int = 2,
    milliquas_radius_arcsec: float = 3.0,
) -> list[ZtfAlert | LsstAlert]:
    """
    Filter alerts for TDE candidates.

    Applies the following cuts in order:
    1. Minimum historical detections (ndethist) (note - could be covered by rising cut).
    2. Reject alerts with a Milliquas crossmatch within ``milliquas_radius_arcsec``
       (removes known/likely AGN/quasars).
    3. rising FIXME had bug with fetching past alerts, decided to remove for now

    Parameters
    ----------
    alerts : list of ZtfAlert | LsstAlert
    min_detections : int
        Minimum number of historical detections required. Default 5.
    milliquas_radius_arcsec : float
        Reject alerts with a Milliquas match closer than this. Default 3.0.

    Returns
    -------
    list of ZtfAlert | LsstAlert
    """
    # --- 1. Detection count cut ---
    passed = [
        a for a in alerts
        if a.candidate.ndethist is not None and a.candidate.ndethist >= min_detections
    ]
    print(f"After detection count filter (>={min_detections}), {len(passed)} remain.")

    # Fetch cross-matches in bulk for alerts that don't have them yet
    needs_xmatch = [a for a in passed if a.cross_matches is None]
    if needs_xmatch:
        add_cross_matches(needs_xmatch)

    # --- 2. Milliquas cut: reject known AGN/quasars ---
    def _has_milliquas_match(alert: ZtfAlert | LsstAlert) -> bool:
        xm = alert.cross_matches
        if xm is None or not xm.milliquasar:
            return False
        return any(
            m.distance_arcsec is not None and m.distance_arcsec <= milliquas_radius_arcsec
            for m in xm.milliquasar
        )

    passed = [a for a in passed if not _has_milliquas_match(a)]
    print(f"After Milliquas cut (<={milliquas_radius_arcsec}\"), {len(passed)} remain.")

    #FIXME currently not cutting on rising here - this function had bug
    # # Fetch all per-band alerts once for each surviving object
    # object_alerts_map = {a.objectId: babamul_get_alerts(a) for a in passed}

    # # --- 3. Rising brightness cut ---
    # passed = [a for a in passed if _is_rising(object_alerts_map[a.objectId])]
    # print(f"After rising cut, {len(passed)} remain.")

    return passed

def nearby_tde_filter(
    alerts: list[ZtfAlert | LsstAlert],
    min_detections: int = 5,
    milliquas_radius_arcsec: float = 3.0,
    wise_w1w2_max: float = 0.8,
    max_gr: float = 0.3,
) -> list[ZtfAlert | LsstAlert]:
    """
    Filter alerts for TDE candidates.

    Applies the following cuts in order:
    1. Minimum historical detections (ndethist).
    2. Reject alerts with a Milliquas crossmatch within ``milliquas_radius_arcsec``
       (removes known/likely AGN/quasars).
    3. Colour cut: fetch all per-band alerts for each object via the API and
       require median g-r < ``max_gr``. Alerts with only one band observed are
       passed through (colour cannot be measured).
    4. rising brightness cut: require at least one band to have three detections

    Parameters
    ----------
    alerts : list of ZtfAlert | LsstAlert
    min_detections : int
        Minimum number of historical detections required. Default 5.
    milliquas_radius_arcsec : float
        Reject alerts with a Milliquas match closer than this. Default 3.0.
    wise_w1w2_max : float
        (Reserved) Maximum allowed w1-w2 from the closest CatWISE match. Default 0.8.
    max_gr : float
        Maximum allowed median g-r colour. Default 0.3.

    Returns
    -------
    list of ZtfAlert | LsstAlert
    """
    # --- 1. Detection count cut ---
    passed = [
        a for a in alerts
        if a.candidate.ndethist is not None and a.candidate.ndethist >= min_detections
    ]
    print(f"After detection count filter (>={min_detections}), {len(passed)} remain.")

    # Fetch cross-matches in bulk for alerts that don't have them yet
    needs_xmatch = [a for a in passed if a.cross_matches is None]
    if needs_xmatch:
        add_cross_matches(needs_xmatch)

    # --- 2. Milliquas cut: reject known AGN/quasars ---
    def _has_milliquas_match(alert: ZtfAlert | LsstAlert) -> bool:
        xm = alert.cross_matches
        if xm is None or not xm.milliquasar:
            return False
        return any(
            m.distance_arcsec is not None and m.distance_arcsec <= milliquas_radius_arcsec
            for m in xm.milliquasar
        )

    passed = [a for a in passed if not _has_milliquas_match(a)]
    print(f"After Milliquas cut (<={milliquas_radius_arcsec}\"), {len(passed)} remain.")

    # Fetch all per-band alerts once for each surviving object
    object_alerts_map = {a.objectId: babamul_get_alerts(survey=a.survey, object_id=a.objectId) for a in passed}

    # --- 3. g-r colour cut ---
    def _is_blue(alert: ZtfAlert | LsstAlert) -> bool:
        band_mags = _band_mags_from_alerts(object_alerts_map[alert.objectId])
        if "g" not in band_mags or "r" not in band_mags:
            return True  # can't compute colour, pass through
        return float(np.median(band_mags["g"]) - np.median(band_mags["r"])) < max_gr

    passed = [a for a in passed if _is_blue(a)]
    print(f"After g-r < {max_gr} colour cut, {len(passed)} remain.")

    # --- 4. Rising brightness cut ---
    passed = [a for a in passed if _is_rising(object_alerts_map[a.objectId])]
    print(f"After rising cut, {len(passed)} remain.")

    return passed


def filter_alerts(
    alerts: list[ZtfAlert | LsstAlert] | dict,
    *filters,
) -> list[ZtfAlert | LsstAlert] | dict:
    """
    Apply a sequence of filter functions to a list of alerts or crossmatch dict.

    Each filter must accept a list and return a list. Filters are applied in order.
    If a dict (output of catalog_crossmatch) is passed, alerts are extracted,
    filtered, and the dict is rebuilt keeping only surviving objects.

    Parameters
    ----------
    alerts : list of ZtfAlert | LsstAlert, or dict (catalog_crossmatch output)
    *filters : callable
        Filter functions to apply, e.g. generic_filter, tde_filter.
    """
    if isinstance(alerts, dict):
        alert_list = [v["LSST"] for v in alerts.values()]
        for f in filters:
            alert_list = f(alert_list)
        surviving_ids = {a.objectId for a in alert_list}
        result = {obj_id: obj for obj_id, obj in alerts.items() if obj_id in surviving_ids}
        print(f"After filtering, {len(result)} remain.")
        return result
    else:
        result = alerts
        for f in filters:
            result = f(result)
        print(f"After filtering, {len(result)} remain.")
        return result

def deduplicate_alerts(alerts: list[ZtfAlert | LsstAlert]) -> list[ZtfAlert | LsstAlert]:
    """
    deduplicate alerts by object objectId, keeping the most recent alert for each
    """
    alerts_to_scan = {}
    for a in alerts:
        if (
            a.objectId not in alerts_to_scan
            or a.candidate.jd > alerts_to_scan[a.objectId].candidate.jd
        ):
            alerts_to_scan[a.objectId] = a
    alerts_to_scan = list(alerts_to_scan.values())
    print(f"After deduplication, {len(alerts_to_scan)} unique alerts remain.")
    return alerts_to_scan


def catalog_filter(
    crossmatched_objects: dict,
    z_min: float = 0.2,
) -> dict:
    """
    Filter crossmatched objects dict by crossmatched catalog object features.
    
    Checks common redshift column names: z, Z_BEST, ZPHOT, zfinal, zpdf_med.
    An object passes if at least one matched catalog has a redshift >= z_min.

    Parameters
    ----------
    crossmatched_objects : dict
        Output of catalog_crossmatch.
    z_min : float
        Minimum redshift. Default 0.2.

    Returns
    -------
    dict
        Filtered subset of crossmatched_objects.
    """
    Z_COLS = {'z', 'Z_BEST', 'ZPHOT', 'zfinal', 'zpdf_med'}

    result = {}
    for obj_id, obj in crossmatched_objects.items():
        for key, data in obj.items():
            if key == "LSST" or data is None:
                continue
            z_val = next((data[c] for c in Z_COLS if c in data and data[c] is not None), None)
            if z_val is not None and z_val >= z_min:
                result[obj_id] = obj
                break

    print(f"After catalog_filter (z>={z_min}): {len(result)}/{len(crossmatched_objects)} objects remain.")
    return result


def catalog_crossmatch(
    alerts: list[ZtfAlert | LsstAlert] | None = None,
    ra: float | list[float] | None = None,
    dec: float | list[float] | None = None,
    catalog_name: str | list[str] | None = None,
    catalog_path: str | None = None,
    radius_arcsec: float = 5.0,
) -> pd.DataFrame:
    """
    Crossmatch alerts or coordinates against one or all .fits catalogs in catalog_path.
    Using a cone search with defined radius, keeping the closest match only.

    Parameters
    ----------
    alerts : list of ZtfAlert | LsstAlert, optional
        Pre-filtered alerts.
    ra : float or list of float, optional
        RA(s) in degrees. Used instead of alerts when provided.
    dec : float or list of float, optional
        Dec(s) in degrees. Used instead of alerts when provided.
    catalog_name : str or None
        Filename of a specific catalog to use (e.g. 'COSMOS2025_cut.fits').
        If None, crossmatches against all .fits files in catalog_path.
    catalog_path : str
        Directory containing .fits catalogs. Defaults to data/catalogs/ at the repo root.
    radius_arcsec : float
        Match radius in arcseconds. Default 5.0.

    Returns
    -------
    df : pd.DataFrame
        One row per input with at least one catalog match. Columns: LSST_objectID,
        then one bool column per catalog named by catalog stem.
    """
    if alerts is None and ra is None:
        raise ValueError("Provide either alerts or ra/dec coordinates.")

    if catalog_path is None:
        catalog_path = Path(__file__).resolve().parents[3] / "data" / "catalogs"

    fits_files = sorted([f for f in os.listdir(catalog_path) if f.endswith('.fits')])
    if not fits_files:
        raise FileNotFoundError(f"No .fits files found in {catalog_path}")

    if catalog_name is not None:
        fits_files = [catalog_name] if isinstance(catalog_name, str) else list(catalog_name)

    if alerts is not None:
        input_coords = SkyCoord(
            ra=[a.candidate.ra for a in alerts],
            dec=[a.candidate.dec for a in alerts],
            unit="deg",
        )
    else:
        ra_list = [ra] if isinstance(ra, (int, float)) else list(ra)
        dec_list = [dec] if isinstance(dec, (int, float)) else list(dec)
        input_coords = SkyCoord(ra=ra_list, dec=dec_list, unit="deg")

    # stem -> (matched bool array, catalog row dicts)
    catalog_results = {}

    for fname in fits_files:
        stem = fname.replace('.fits', '')
        path = os.path.join(catalog_path, fname)
        cat = Table.read(path)
        names = [n for n in cat.colnames if len(cat[n].shape) <= 1]
        cat = cat[names]

        ra_col = next((c for c in cat.colnames if c.lower() == 'ra'), None)
        dec_col = next((c for c in cat.colnames if c.lower() == 'dec'), None)
        if ra_col is None or dec_col is None:
            print(f"Skipping {fname}: no ra/dec columns found.")
            continue

        cat_coords = SkyCoord(ra=cat[ra_col], dec=cat[dec_col], unit="deg")
        idx, sep, _ = input_coords.match_to_catalog_sky(cat_coords)
        sep_arcsec = sep.to(u.arcsec).value
        within = sep_arcsec <= radius_arcsec

        def _safe_val(v):
            import numpy.ma as ma
            if isinstance(v, ma.core.MaskedConstant):
                return None
            if hasattr(v, 'item'):
                return v.item()
            return v

        # pre-build row dicts for matched entries
        row_dicts = []
        for j, (i, matched) in enumerate(zip(idx, within)):
            if matched:
                row_dict = {c: _safe_val(cat[c][i])
                            for c in cat.colnames if c not in (ra_col, dec_col)}
                row_dict['conesearch_arcsecs'] = float(sep_arcsec[j])
                row_dicts.append((j, row_dict))
            else:
                row_dicts.append((j, None))

        catalog_results[stem] = row_dicts
        print(f"{fname}: {within.sum()}/{len(input_coords)} matched within {radius_arcsec}\".")

    result = {}
    inputs = alerts if alerts is not None else list(range(len(ra_list)))
    for j, inp in enumerate(inputs):
        obj_id = inp.objectId if alerts is not None else j
        cat_entries = {stem: rows[j][1] for stem, rows in catalog_results.items()}
        if any(v is not None for v in cat_entries.values()):
            result[obj_id] = {"LSST": inp, **cat_entries}

    print(f"Total: {len(result)}/{len(inputs)} inputs matched in at least one catalog.")
    return result

