"""
Prost-compatible builder functions for THOR's local FITS catalogs.

Usage
-----
Pre-load a catalog DataFrame, register it, then pass the catalog name to
associate_sample as normal::

    from astropy.table import Table
    from thor.utils.prost_catalogs import register_local_catalog

    df = Table.read("data/catalogs/COSMOS2025_cut.fits").to_pandas()
    register_local_catalog("cosmos2025", df)

    results = associate_sample(
        transient_catalog,
        catalogs=["cosmos2025"],
        ...
    )

``register_local_catalog`` patches ``GalaxyCatalog.__init__`` once so that
every new ``GalaxyCatalog`` instance automatically includes all registered
custom catalogs — no forking of prost required.

Notes
-----
Your FITS catalogs must have columns named ``id``, ``ra``, ``dec``, and ``z``
(or specify alternate names via ``z_col`` / ``z_std_col``).
Shape/morphology columns are not required; offset scoring uses a fixed size
floor (point-source approximation) for all candidates.

If your catalog has a redshift uncertainty column, pass its name as
``z_std_col`` to ``register_local_catalog``; otherwise a fixed fractional
uncertainty (``Z_STD_FRAC``) is applied.
"""

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from scipy.stats import norm

from astro_prost.helpers import (
    GalaxyCatalog,
    build_galaxy_array,
    calc_dlr,
    SIZE_FLOOR,
    SIGMA_SIZE_FLOOR,
    REDSHIFT_FLOOR,
)

# ---- Runtime patch so custom catalogs survive GalaxyCatalog.__init__ --------
# catalog_functions is set as an *instance* attribute inside __init__, so we
# cannot assign at the class level.  Instead we wrap __init__ once to merge
# our registry into every new instance.

# Fractional redshift uncertainty to assume when the catalog has no z_std column.
# 10% is a reasonable default for photometric redshifts.
Z_STD_FRAC = 0.10


def make_local_catalog_fn(df, catalog_label="local", z_col="z", z_std_col=None):
    """Return a prost-compatible builder function for a pre-loaded catalog DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Catalog data. Must have columns ``id``, ``ra``, ``dec``, and ``z``
        (or whatever name you pass as ``z_col``).
    catalog_label : str
        Human-readable name used in prost log messages (e.g. "COSMOS2025").
    z_col : str
        Column name for photometric redshift. Default ``"z"``.
    z_std_col : str or None
        Column name for redshift uncertainty. If None, uses ``Z_STD_FRAC * z``.

    Returns
    -------
    callable
        Builder function with the prost ``build_*_candidates`` signature.
    """

    def _builder(
        transient,
        search_rad,
        cosmo,
        logger,
        calc_host_props=("offset", "redshift"),
        n_samples=1000,
        cat_cols=False,
        release=None,
        shred_cut=False,
        glade_catalog=None,   # prost passes this kwarg; unused here
    ):
        transient_pos = transient.position
        transient_pos_samples = transient.position_samples

        # --- 1. Box pre-filter (fast) then cone search ---
        margin = search_rad.deg + 0.01
        box = df[
            (df["ra"]  >= transient_pos.ra.deg  - margin) &
            (df["ra"]  <= transient_pos.ra.deg  + margin) &
            (df["dec"] >= transient_pos.dec.deg - margin) &
            (df["dec"] <= transient_pos.dec.deg + margin)
        ].copy()

        if len(box) == 0:
            return None, []

        cat_coords = SkyCoord(box["ra"].values * u.deg, box["dec"].values * u.deg)
        sep_arcsec = transient_pos.separation(cat_coords).arcsec
        candidate_hosts = box[sep_arcsec < search_rad.arcsec].copy()
        sep_arcsec = sep_arcsec[sep_arcsec < search_rad.arcsec]

        if len(candidate_hosts) == 0:
            return None, []

        # prost expects an "objID" column; rename "id" to match
        candidate_hosts = candidate_hosts.rename(columns={"id": "objID", z_col: "redshift"})

        # --- 2. Build the base structured numpy array ---
        galaxies, cat_col_fields = build_galaxy_array(
            candidate_hosts, cat_cols, transient.name, catalog_label, release, logger
        )
        if galaxies is None:
            return None, []

        n_galaxies = len(galaxies)
        galaxies_pos = SkyCoord(galaxies["ra"] * u.deg, galaxies["dec"] * u.deg)

        # --- 3. Offset ---
        if "offset" in calc_host_props:
            # No shape columns available: treat every galaxy as a point source
            # at the prost SIZE_FLOOR (minimum half-light radius).
            temp_sizes     = np.full(n_galaxies, SIZE_FLOOR)
            temp_sizes_std = np.full(n_galaxies, SIGMA_SIZE_FLOOR * SIZE_FLOOR)
            a_over_b       = np.ones(n_galaxies)        # circular
            a_over_b_std   = np.zeros(n_galaxies)
            phi            = np.zeros(n_galaxies)
            phi_std        = np.zeros(n_galaxies)

            dlr_samples = calc_dlr(
                transient_pos_samples,
                galaxies_pos,
                temp_sizes,
                temp_sizes_std,
                a_over_b,
                a_over_b_std,
                phi,
                phi_std,
                n_samples=n_samples,
            )

            offset_samples = np.array([
                transient_pos_samples.separation(
                    SkyCoord(galaxies["ra"][i] * u.deg, galaxies["dec"][i] * u.deg)
                ).arcsec
                for i in range(n_galaxies)
            ])

            for i in range(n_galaxies):
                galaxies["offset_samples"][i] = offset_samples[i]
                galaxies["dlr_samples"][i]    = dlr_samples[i]

            galaxies["offset_mean"] = np.nanmean(offset_samples, axis=1)
            galaxies["offset_std"]  = np.nanstd(offset_samples,  axis=1)
            galaxies["offset_info"] = "ARCSEC"
            galaxies["dlr_mean"]    = np.nanmean(dlr_samples, axis=1)
            galaxies["dlr_std"]     = np.nanstd(dlr_samples,  axis=1)
            galaxies["size_mean"]   = temp_sizes
            galaxies["size_std"]    = temp_sizes_std

        # --- 4. Redshift ---
        if "redshift" in calc_host_props:
            redshift_mean = candidate_hosts["redshift"].values.astype(float)

            if z_std_col and z_std_col in candidate_hosts.columns:
                redshift_std = candidate_hosts[z_std_col].values.astype(float)
            else:
                redshift_std = np.maximum(Z_STD_FRAC * redshift_mean, 0.01)

            redshift_samples = np.maximum(
                REDSHIFT_FLOOR,
                norm.rvs(
                    loc=redshift_mean[:, np.newaxis],
                    scale=redshift_std[:, np.newaxis],
                    size=(n_galaxies, n_samples),
                ),
            )

            galaxies["redshift_mean"] = redshift_mean
            galaxies["redshift_std"]  = redshift_std
            galaxies["redshift_info"] = "PHOT"

            for i in range(n_galaxies):
                galaxies["redshift_samples"][i] = redshift_samples[i]

        # --- 5. Absolute magnitude (placeholder) ---
        # TODO: populate if your catalog has an apparent magnitude column.
        # Example with an r-band column "mag_r":
        #
        #   if "absmag" in calc_host_props and "mag_r" in candidate_hosts.columns:
        #       temp_mag = candidate_hosts["mag_r"].values.astype(float)
        #       temp_mag_std = np.maximum(0.05 * np.abs(temp_mag), 0.1)
        #       absmag_samples = (
        #           norm.rvs(loc=temp_mag[:, np.newaxis],
        #                    scale=temp_mag_std[:, np.newaxis],
        #                    size=(n_galaxies, n_samples))
        #           - cosmo.distmod(redshift_samples).value
        #       )
        #       galaxies["absmag_mean"] = temp_mag - cosmo.distmod(redshift_mean).value
        #       galaxies["absmag_std"]  = temp_mag_std
        #       galaxies["absmag_info"] = "r"
        #       for i in range(n_galaxies):
        #           galaxies["absmag_samples"][i] = absmag_samples[i]

        return galaxies, cat_col_fields

    return _builder


# ---- Runtime patch so custom catalogs survive GalaxyCatalog.__init__ --------
# catalog_functions is set as an *instance* attribute inside __init__, so we
# cannot assign at the class level.  Instead we wrap __init__ once to merge
# our registry into every new instance.

_CUSTOM_CATALOGS: dict = {}
_patched = False


def _patch_galaxy_catalog_init() -> None:
    global _patched
    if _patched:
        return
    _orig_init = GalaxyCatalog.__init__

    def _new_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        self.catalog_functions.update(_CUSTOM_CATALOGS)

    GalaxyCatalog.__init__ = _new_init
    _patched = True


def register_local_catalog(
    name: str,
    df,
    catalog_label: str | None = None,
    z_col: str = "z",
    z_std_col: str | None = None,
) -> None:
    """Register a pre-loaded catalog DataFrame with prost's GalaxyCatalog.

    Parameters
    ----------
    name : str
        The catalog key used in ``associate_sample(catalogs=[name])``.
    df : pandas.DataFrame
        Catalog data. Must have columns ``id``, ``ra``, ``dec``, and ``z``
        (or whatever name you pass as ``z_col``).
    catalog_label : str or None
        Human-readable name used in prost log messages. Defaults to ``name``.
    z_col : str
        Column name for photometric redshift. Default ``"z"``.
    z_std_col : str or None
        Column name for redshift uncertainty. If None, uses ``Z_STD_FRAC * z``.
    """
    _CUSTOM_CATALOGS[name] = make_local_catalog_fn(
        df,
        catalog_label=catalog_label or name,
        z_col=z_col,
        z_std_col=z_std_col,
    )
    _patch_galaxy_catalog_init()
