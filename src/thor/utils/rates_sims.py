import os
import astropy.units as u
from thor.catalogs_catalog import catalogs as _CATALOGS_META
import astropy.constants as const
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from astropy.cosmology import Planck18 as cosmo, z_at_value
from astropy.modeling.models import BlackBody
from extinction import fitzpatrick99
from scipy.stats import norm
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter, LogLocator, NullFormatter
rcParams["font.family"] = "Liberation Serif"
rcParams["text.usetex"] = False

# ── Color coding — one fixed color per transient type, used by all plotters ────────
TRANSIENT_COLORS = {
    'SNIa':   '#6B9E94',  # cold  — muted teal-blue
    'SNII':   '#B8622A',  # warm  — brown-orange
    'SNIbc':  '#FFCCD5',  # warm  — pale pink
    'CCSN':   '#DC7868',  # warm  — peach-pink
    'SLSN-I': '#0D1B2A',  # cold  — dark navy
    'SLSN':   '#7B8B35',  # warm  — muted army green
    'TDE':    '#004F5C',  # cold  — dark teal
}
_FALLBACK_COLORS = ['#888888', '#AAAAAA', '#555555']  # gray tones for unknown types

# ── Rates Calculations ─────────────────────────────────────────────────────────────
def luminosity_distance_from_mag(M, m_lim):
    """
    Given absolute magnitude of an object M, 
    what distance can it be seen at apparent magnitude limit m_lim?
    Returns luminosity distance in pc
    """
    mu = m_lim - M
    D_pc = 10 ** ((mu + 5) / 5)
    return D_pc * u.pc

def get_z_limit(objects_dictionary, m_lim=24.5):
    """
    Furthest redshift we could see these object based on abs magnitude
    Default to m=24.5 (rubin single exposure limit)
    objects dictionary should have key object (string)
    and value M absolute magnitude (float)
    """
    detectable_dict = {}
    for obj, M in objects_dictionary.items():
        D_L = luminosity_distance_from_mag(M, m_lim=m_lim)
        z_max = z_at_value(cosmo.luminosity_distance, D_L)
        print(f"{obj}: z_max = {z_max:.2f}")
        detectable_dict[obj] = z_max
    return detectable_dict

def tde_apparent_mags(M_abs, z, days_since_peak, fading_rate_intrinsic=0.023,
                      tde_colors=None):
    """
    Compute the apparent r-band magnitude of a TDE and g/r/i colors at a given
    observer-frame epoch.

    Parameters
    ----------
    M_abs : float
        Absolute r-band magnitude at peak.
    z : float
        Redshift of the source.
    days_since_peak : float
        Time since peak in observer-frame days.
    fading_rate_intrinsic : float
        Intrinsic (rest-frame) linear fading rate in mag/day. Default 0.023
        from Hinkle et al. 2020.
    tde_colors : dict, optional
        Color offsets relative to r-band, e.g. {'g': -0.3, 'i': 0.2}.
        Defaults to blue TDE-like colors: g-r = -0.3, r-i = 0.2.

    Returns
    -------
    dict with keys 'r', 'g', 'i' apparent magnitudes.

    Notes
    -----
    Time dilation stretches the observed fading by (1+z):
        fading_rate_observed = fading_rate_intrinsic / (1 + z)
    Distance modulus from Planck18 cosmology (no K-correction applied).
    """
    if tde_colors is None:
        tde_colors = {'g': -0.3, 'i': 0.2}  # g bluer, i redder than r

    mu = cosmo.distmod(z).value                      # distance modulus
    fading_rate_obs = fading_rate_intrinsic / (1 + z)
    delta_m = fading_rate_obs * days_since_peak

    m_r = M_abs + mu + delta_m
    m_g = m_r + tde_colors['g']
    m_i = m_r + tde_colors['i']

    print(f"z={z}, mu={mu:.2f}, fading_rate_obs={fading_rate_obs:.4f} mag/day, "
          f"delta_m={delta_m:.2f} over {days_since_peak}d | "
          f"g={m_g:.2f}, r={m_r:.2f}, i={m_i:.2f}")

    return {'r': m_r, 'g': m_g, 'i': m_i}


def get_volumetric_BTS_rates(object_mean_magnitudes, BTS_counts, m_lim=19.0, survey_duration_yr=25.5/12, verbose=True):
    """
    Returns a dictionary of objects and rate per Gpc^3 per year
    """
    # use distance to get volume for each class of object
    sensitive_volumes = {}

    for obj, M in object_mean_magnitudes.items():
        D_L = luminosity_distance_from_mag(M, m_lim)
        z_max = z_at_value(cosmo.luminosity_distance, D_L)
        if verbose:
            print(f"{obj}: furthest detectable = {z_max:.2f}")
        V = cosmo.comoving_volume(z_max).to(u.Gpc**3)
        sensitive_volumes[obj] = V

    # Compute volumetric rates
    BTS_volumetric_rates = {}

    for obj, N in BTS_counts.items():
        if obj not in sensitive_volumes:
            continue  # e.g. "other"
        rate = N / survey_duration_yr / sensitive_volumes[obj]
        BTS_volumetric_rates[obj] = rate
        if verbose:
            print(f"{obj:6s}: {rate:.2f}")
    
    return BTS_volumetric_rates

def get_BTS_rates_from_filtered(volumetric_obs_dict, efficiency_dict, verbose=True):
    BTSpoprates={}
    fraction_observed = np.prod(list(efficiency_dict.values()))
    for obj, rate in volumetric_obs_dict.items():
        pop_rate = rate / fraction_observed
        BTSpoprates[obj] = pop_rate
        if verbose:
            print(f"{obj:6s} population rate estimate: {pop_rate:.2f} per Gpc^3 per year")
    return BTSpoprates

def calculate_rates_zbin(object_rate, z_bin_min, z_bin_max):
    """Input: object_rate in Gpc^-3 yr^-1 (astropy Quantity or plain float)"""
    vol_bin_gpc3 = (cosmo.comoving_volume(z_bin_max) - cosmo.comoving_volume(z_bin_min)).to(u.Gpc**3).value
    rate_gpc3 = object_rate.to(u.Gpc**-3).value if hasattr(object_rate, 'unit') else float(object_rate)
    return rate_gpc3 * vol_bin_gpc3

def redshift_wavelength(z, rest_wavelength=1216):
    """
    Calculate observed wavelength given rest wavelength and redshift.
    Default: Lyman-alpha (1216 Å)
    """
    return rest_wavelength * (1 + z)

def calculate_rates_vs_redshift(object_rates, z_min=0.0, z_max=4.0, dz=0.5, n_z_per_bin=30):
    """
    Given a dictionary of z=0 volumetric rates (per Gpc^3 per year), compute
    expected events per redshift bin with PLASTiCC z-dependent rate evolution
    applied in one step (Kessler et al. 2019, arXiv:1903.11756, Table 2):

        N_i = R_0 × ∫_{z_lo}^{z_hi} f(z) (dV_c/dz) dz

    where f(z) = R(z)/R(z=0) from _plasticc_rate_shape.

    Supported transient types: SNIa, SNII, SNIbc, SLSN-I, SLSN, TDE, CCSN.
    Unknown types are set to 0 and a warning is printed.

    Accepts two input formats per transient:
      - plain rate: ``{'SNIa': 2.35e4 * u.Unit("1 / Gpc3"), ...}``
      - PLASTiCC nested dict: ``{'SNIa': {'z0rate': 25e-6 * u.Unit("1 / Mpc3"), ...}, ...}``
    Any astropy-compatible volume rate unit (Mpc^-3, Gpc^-3, …) is accepted.
    """
    KNOWN_TYPES = {'SNIa', 'SNII', 'SNIbc', 'SLSN-I', 'SLSN', 'TDE', 'CCSN'}

    z_edges   = np.arange(z_min, z_max + dz, dz)
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    rates_dict = {'center_z': z_centers, 'center_lya': redshift_wavelength(z_centers)}

    for transient, rate in object_rates.items():
        # Support PLASTiCC nested format: {'z0rate': <Quantity>, 'ratio_z_0_1': ...}
        if isinstance(rate, dict):
            rate = rate['z0rate']
        rate_val = rate.to(u.Gpc**-3).value if hasattr(rate, 'unit') else float(rate)
        counts   = np.zeros(len(z_centers))

        if transient not in KNOWN_TYPES:
            print(f"Warning: no PLASTiCC rate evolution equation for '{transient}'. Rates set to 0.")
            rates_dict[transient] = counts
            continue

        for i, (z_lo, z_hi) in enumerate(zip(z_edges[:-1], z_edges[1:])):
            z_lo_eff = max(z_lo, 1e-4)
            z_arr    = np.linspace(z_lo_eff, z_hi, n_z_per_bin)
            # 4π × dV/dz|sr = total comoving volume element per unit z
            dVdz     = 4 * np.pi * cosmo.differential_comoving_volume(z_arr).to(u.Gpc**3 / u.sr).value
            f_arr    = _plasticc_rate_shape(transient, z_arr)
            counts[i] = rate_val * np.trapz(f_arr * dVdz, z_arr)

        rates_dict[transient] = np.round(counts)

    return rates_dict

def _plasticc_rate_shape(transient, z):
    """
    Returns R(z) / R(z=0) for the given transient type using PLASTiCC
    rate prescriptions (Kessler et al. 2019, arXiv:1903.11756, Table 2).

    SNIa   : power-law break at z=1 — (1+z)^1.5 for z<1 (Dilday 2008),
             continuous join 4*(1+z)^-0.5 for z≥1 (Hounsell 2018)
    SNII   : Strolger et al. 2015 SFH — ψ(z)/ψ(0), A=0.015, B=1.5, C=5.0, D=6.1
    SNIbc  : same as SNII (Strolger 2015; lacking direct Ibc constraints)
    CCSN   : alias for SNII
    SLSN-I : Madau & Dickinson 2014 SFR — ψ(z)/ψ(0), A=0.015, B=2.9, C=2.7, D=5.6
    SLSN   : separate population from SLSN-I; shares same Madau & Dickinson rate shape
    TDE    : Kochanek 2016 — 10^(-5z/6)
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))

    if transient == 'SNIa':
        # continuous at z=1: (1+1)^1.5 = 2^1.5; 4*(1+1)^-0.5 = 4/√2 = 2^1.5 ✓
        factor = np.where(z < 1, (1 + z)**1.5, 4.0 * (1 + z)**-0.5)

    elif transient in ('SNII', 'SNIbc', 'CCSN'):
        # Strolger et al. 2015: ψ(z) = A*(1+z)^C / [1 + ((1+z)/B)^D]
        A, B, C, D = 0.015, 1.5, 5.0, 6.1
        def psi(zz):
            return A * (1 + zz)**C / (1 + ((1 + zz) / B)**D)
        factor = psi(z) / psi(0.0)

    elif transient in ('SLSN-I', 'SLSN'):
        # Madau & Dickinson 2014 (arXiv:1403.0007): R(z) = k * h^2 * psi(z)
        # psi(z) = A*(1+z)^C / [1 + ((1+z)/B)^D]; A=0.015, B=2.9, C=2.7, D=5.6
        # k ~ 1.34e-6 M_sol^-1 (manually fit to PLASTiCC SLSN-I z0rate); h=0.7
        A, B, C, D = 0.015, 2.9, 2.7, 5.6
        k, h = 1.34e-6, 0.7
        def psi(zz):
            return A * (1 + zz)**C / (1 + ((1 + zz) / B)**D)
        def R_slsn(zz):
            return k * h**2 * psi(zz)
        factor = R_slsn(z) / R_slsn(0.0)

    elif transient == 'TDE':
        # Kochanek 2016: R(z) = 10^(-5z/6) yr^-1 Mpc^-3; already R(0)=1
        factor = 10**(-5 * z / 6)

    else:
        factor = np.ones_like(z)

    return float(factor[0]) if factor.shape == (1,) else factor




def calculate_rubin_detectable_rates(rates_vs_z, m_lim=24.5, n_z_per_bin=30, ebv=0.0, rv=3.1, f_sky=0.436, verbose=False):
    """
    Compute Rubin-detectable event counts per redshift bin by convolving
    intrinsic rates with representative luminosity distributions from the
    literature, with optional Milky Way dust extinction.
    Finally, multiply rate by rubin sky coverage fraction (0.436) to get expected counts in Rubin survey.

    Luminosity distributions (Gaussian in peak absolute magnitude):
      SNIa : M = -19.3 ± 0.3  mag  [Betoule et al. 2014, A&A 568, A22]
      CCSN : M = -17.5 ± 1.5  mag  [Richardson et al. 2014, AJ 147, 118;
                                      covers IIP through Ic-BL, ~1 mag bright tail]
      TDE  : M = -19.0 ± 1.5  mag  [Yao et al. 2023, ApJ 955, 6 (ZTF BTS);
                                      van Velzen et al. 2021, ApJ 908, 4]

    A transient at redshift z with absolute magnitude M is detectable if:
        m = M + mu(z) < m_lim - A_lambda(z)
    where A_lambda(z) is Milky Way foreground extinction at the observed peak
    wavelength lambda_obs = lambda_rest * (1 + z), computed via Fitzpatrick (1999).

    The detectable fraction is volume-weighted over each bin:
        f_det = ∫ Φ(m_lim - A_lam(z) - mu(z); M_mean, M_sigma) (dV/dz) dz / ΔV_bin

    Rest-frame peak wavelengths used per type:
      SNIa : 4400 Å  (B-band; standardization band)
      CCSN : 5500 Å  (V-band; representative of IIP/IIb/Ib/Ic population)
      TDE  : 3000 Å  (near-UV optical peak; Yao+2023, van Velzen+2021)

    Parameters
    ----------
    rates_vs_z  : dict   same format as BTS_calculated_rates_vs_z
    m_lim       : float  apparent magnitude limit (default 24.5, Rubin single-visit)
    n_z_per_bin : int    quadrature points per bin (default 30)
    ebv         : float  E(B-V) for the line of sight (default 0.0 = no extinction)
                         e.g. 0.018 for COSMOS [Schlegel et al. 1998],
                              ~0.1  for a typical mid-latitude BTS-like field
    rv          : float  R_V = A_V / E(B-V) (default 3.1, standard MW diffuse ISM)

    Returns
    -------
    dict  same structure as input, counts replaced by detectable sub-counts
    """

    # Literature luminosity distributions (peak absolute magnitude, Gaussian)
    # SNIa  : Betoule et al. 2014, A&A 568, A22
    # SNII  : Richardson et al. 2014, AJ 147, 118 (IIP -16.75, IIL -17.98)
    # SNIbc : Richardson et al. 2014, AJ 147, 118 (Ib -17.45, Ic -17.66)
    # CCSN  : Richardson et al. 2014 — broad mix of IIP/IIb/Ib/Ic/Ic-BL
    # SLSN-I: Lunnan et al. 2018, ApJ 852, 81; De Cia et al. 2018, ApJ 860, 100
    # SLSN  : same as SLSN-I (separate population, same luminosity prior)
    # TDE   : Yao et al. 2023, ApJ 955, 6; van Velzen et al. 2021, ApJ 908, 4
    lum_distributions = {
        'SNIa':   {'M_mean': -19.3, 'M_sigma': 0.3,  'lambda_rest_aa': 4400.},
        'SNII':   {'M_mean': -17.0, 'M_sigma': 1.1,  'lambda_rest_aa': 5500.},
        'SNIbc':  {'M_mean': -17.5, 'M_sigma': 1.1,  'lambda_rest_aa': 5500.},
        'CCSN':   {'M_mean': -17.5, 'M_sigma': 1.5,  'lambda_rest_aa': 5500.},
        'SLSN-I': {'M_mean': -21.0, 'M_sigma': 1.0,  'lambda_rest_aa': 4000.},
        'SLSN':   {'M_mean': -21.0, 'M_sigma': 1.0,  'lambda_rest_aa': 4000.},
        'TDE':    {'M_mean': -19.0, 'M_sigma': 1.5,  'lambda_rest_aa': 3000.},
    }

    a_v = rv * ebv
    center_z = rates_vs_z['center_z']
    dz = center_z[1] - center_z[0]
    result = {'center_z': center_z, 'center_lya': rates_vs_z['center_lya']}

    transients = [k for k in rates_vs_z if k not in ('center_z', 'center_lya')]
    for transient in transients:
        if transient not in lum_distributions:
            print(f"Warning: no luminosity distribution for '{transient}', skipping.")
            continue
        dist = lum_distributions[transient]
        M_mean      = dist['M_mean']
        M_sigma     = dist['M_sigma']
        lam_rest    = dist['lambda_rest_aa']
        intrinsic   = rates_vs_z[transient]
        detectable  = np.zeros(len(center_z))

        for i, z_c in enumerate(center_z):
            z_lo = max(z_c - dz / 2, 1e-3)
            z_hi = z_c + dz / 2
            z_arr = np.linspace(z_lo, z_hi, n_z_per_bin)

            mu_arr = cosmo.distmod(z_arr).value

            # MW dust extinction at the observed peak wavelength for each z sample
            if ebv > 0:
                lam_obs = (lam_rest * (1 + z_arr)).astype(np.float64)
                a_lam_arr = fitzpatrick99(lam_obs, a_v, rv)
            else:
                a_lam_arr = 0.0

            # effective magnitude limit after extinction
            m_lim_eff = m_lim - a_lam_arr
            frac_arr  = norm.cdf(m_lim_eff - mu_arr, loc=M_mean, scale=M_sigma)

            # volume-weighted average (dV/dz; 4π sr cancels numerator/denominator)
            dVdz_arr = cosmo.differential_comoving_volume(z_arr).to(u.Gpc**3 / u.sr).value

            frac_avg      = np.trapz(frac_arr * dVdz_arr, z_arr) / np.trapz(dVdz_arr, z_arr)
            detectable[i] = intrinsic[i] * frac_avg

        result[transient] = detectable * f_sky  # scale by Rubin's sky coverage

    if verbose:
        processed = [k for k in result if k not in ('center_z', 'center_lya')]

        # detectable fraction table
        header = f"{'Transient':<8}  " + "  ".join(f"z={z:.2f}" for z in center_z)
        print(header)
        print("-" * len(header))
        print("Rubin-detectable fraction:")
        for transient in processed:
            dist = lum_distributions[transient]
            fracs = []
            for z in center_z:
                mu_z = cosmo.distmod(z).value
                fracs.append(norm.cdf(m_lim - mu_z, loc=dist['M_mean'], scale=dist['M_sigma']))
            print(f"{transient:<8}  " + "  ".join(f"{f:>6.3f}" for f in fracs))

        print()
        print("Rubin-detectable counts per bin:")
        for transient in processed:
            counts = result[transient]
            print(f"{transient:<8}  " + "  ".join(f"{c:>6.0f}" for c in counts))

    return result


_TOTAL_SKY_DEG2 = 41253.0  # full sky in deg²

def calculate_all_catalog_rates(intrinsic_rates_vs_z, m_lim=24.5):
    """
    Run calculate_rubin_detectable_rates for every catalog in catalogs_catalog.py.

    Uses per-field E(B-V) and coverage_deg2 from the catalog metadata.
    Assumes 100% catalog completeness within each field's coverage area.

    Parameters
    ----------
    intrinsic_rates_vs_z : dict
        Output of calculate_rates_vs_redshift (intrinsic or z-corrected rates).
    m_lim : float
        Apparent magnitude limit (default 24.5, Rubin single-visit).

    Returns
    -------
    per_catalog : dict
        {catalog_stem: rates_dict} — one rates dict per catalog, same structure
        as calculate_rubin_detectable_rates output.
    summed : dict
        Rates summed across all catalogs, compatible with plot_rates_vs_redshift.
    """
    per_catalog = {}
    print(f"{'Catalog':<40}  {'Coverage (deg²)':>16}  {'f_sky':>10}  {'E(B-V)':>8}")
    print("-" * 80)
    for fname, meta in _CATALOGS_META.items():
        coverage = meta.get("coverage_deg2")
        ebv      = meta.get("ebv")
        if coverage is None or ebv is None:
            print(f"  Skipping {fname}: missing coverage_deg2 or ebv")
            continue
        f_sky = coverage / _TOTAL_SKY_DEG2
        stem  = fname.replace(".fits", "")
        print(f"{stem:<40}  {coverage:>16.4f}  {f_sky:>10.2e}  {ebv:>8.3f}")
        per_catalog[stem] = calculate_rubin_detectable_rates(
            intrinsic_rates_vs_z,
            m_lim=m_lim,
            ebv=ebv,
            f_sky=f_sky,
        )

    # sum across catalogs → compatible with plot_rates_vs_redshift
    ref = next(iter(per_catalog.values()))
    transients = [k for k in ref if k not in ('center_z', 'center_lya')]
    summed = {
        'center_z':   ref['center_z'],
        'center_lya': ref['center_lya'],
    }
    for t in transients:
        summed[t] = np.sum([r[t] for r in per_catalog.values()], axis=0)

    return per_catalog, summed


# ── Plotting ─────────────────────────────────────────────────────────────
def plot_comoving_volume(nominal_rates, z_min=0.0, z_max=4.0, dz=0.5,
                         logy=True, custom_title="Comoving Volume Per Redshift Bin",
                         ymin=None, ax=None):
    """
    Plot comoving volume per redshift bin for one or more nominal rates (Gpc⁻³ yr⁻¹).
    Passing rate=1 makes the bar height equal to the comoving volume of each shell in Gpc³.

    nominal_rates : dict  {label: rate_gpc3_per_yr}
    """
    z_edges   = np.arange(z_min, z_max + dz, dz)
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    vols = np.array([
        (cosmo.comoving_volume(z_hi) - cosmo.comoving_volume(z_lo)).to(u.Gpc**3).value
        for z_lo, z_hi in zip(z_edges[:-1], z_edges[1:])
    ])
    rates_vs_z = {'center_z': z_centers, 'center_lya': redshift_wavelength(z_centers)}
    for label, rate in nominal_rates.items():
        rate_val = rate.to(u.Gpc**-3).value if hasattr(rate, 'unit') else float(rate)
        rates_vs_z[label] = np.round(rate_val * vols)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 6))

    center_z   = rates_vs_z['center_z']
    center_lya = rates_vs_z['center_lya']
    labels     = [k for k in rates_vs_z if k not in ('center_z', 'center_lya')]
    n_bins     = len(center_z)
    n_classes  = len(labels)
    width      = 0.8 / n_classes
    x          = np.arange(n_bins)

    for i, label in enumerate(labels):
        ax.bar(x + i * width, rates_vs_z[label], width=width, label=label, alpha=0.85, color='#004F5C')

    # Euclidean (flat, non-expanding) shell volume: V = (4π/3) * [(cz_hi/H0)³ - (cz_lo/H0)³]
    D_H = (const.c / cosmo.H0).to(u.Gpc).value  # Hubble distance in Gpc
    z_edges = np.arange(z_min, z_max + dz, dz)
    euclidean_vols = (4 * np.pi / 3) * D_H**3 * (z_edges[1:]**3 - z_edges[:-1]**3)
    ax.step(np.append(x - 0.4, x[-1] + 0.4), np.append(euclidean_vols, euclidean_vols[-1]),
            where='post', color='gray', lw=2, ls='--', label='Euclidean (no expansion)', zorder=5)

    ax.set_xticks(x + width * (n_classes - 1) / 2)
    ax.set_xticklabels([f"{z:.2f}" for z in center_z])
    ax.set_xlabel("Redshift (bin center)", fontsize=18)
    ax.set_ylabel("Gpc³ per Δz bin  (×R₀ yr)", fontsize=18)
    if logy:
        ax.set_yscale("log")
    if ymin is not None:
        ax.set_ylim(bottom=ymin)
    ax.set_title(custom_title, fontsize=24, pad=25)
    ax.legend(fontsize=14)

    comoving_dist = cosmo.comoving_distance(center_z).to(u.Gpc).value
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xticklabels([f"{d:.1f}" for d in comoving_dist])
    ax2.set_xlabel("Comoving Distance (Gpc)", fontsize=18)
    ax.grid(False)
    ax2.grid(False)

    if standalone:
        plt.tight_layout()
        plt.show()


def plot_rates_vs_redshift(rates_vs_z, logy=True, custom_title="BTS Classified Transient Rates vs Redshift", ax=None, ymin=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 6))

    center_z = rates_vs_z['center_z']
    center_lya = rates_vs_z['center_lya']

    transients = [k for k in rates_vs_z.keys() if k not in ['center_z', 'center_lya']]
    n_bins = len(center_z)
    n_classes = len(transients)
    width = 0.8 / n_classes
    x = np.arange(n_bins)

    _fallback = iter(_FALLBACK_COLORS)
    for i, transient in enumerate(transients):
        color = TRANSIENT_COLORS.get(transient, next(_fallback, '#888888'))
        ax.bar(x + i * width, rates_vs_z[transient], width=width, label=transient, alpha=0.85, color=color)

    ax.set_xticks(x + width * (n_classes - 1) / 2)
    ax.set_xticklabels([f"{z:.2f}" for z in center_z])
    ax.set_xlabel("Redshift (bin center)", fontsize=18)
    ax.set_ylabel("Events per year per Δz bin", fontsize=18)
    if logy:
        ax.set_yscale("log")
    if ymin is not None:
        ax.set_ylim(bottom=ymin)
    ax.set_title(custom_title or "Transient Rates vs Redshift", fontsize=24, pad=25)
    ax.legend(fontsize=14)

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xticklabels([f"{int(l)} Å" for l in center_lya])
    ax2.set_xlabel("Observed Lyα Wavelength", fontsize=18)
    ax.grid(False)
    ax2.grid(False)

    if standalone:
        plt.tight_layout()
        plt.show()


def plot_rates_grid(plot_configs, ncols=2, ymin=None):
    """
    Render multiple plot_rates_vs_redshift panels on a shared figure grid.

    plot_configs : list of dicts, each with:
        'rates_vs_z'   : required
        'custom_title' : optional str
        'logy'         : optional bool (default True)
    ymin         : float, applied to all panels (default None)

    Example
    -------
    plot_rates_grid([
        {'rates_vs_z': BTS_rates_vs_z_naive, 'custom_title': 'Naive rates'},
        {'rates_vs_z': BTS_rates_vs_z,       'custom_title': 'Z-corrected rates'},
    ], ymin=1)
    """
    n = len(plot_configs)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 6 * nrows))
    axes = np.array(axes).flatten()

    for ax, cfg in zip(axes, plot_configs):
        plot_rates_vs_redshift(
            cfg['rates_vs_z'],
            logy=cfg.get('logy', True),
            custom_title=cfg.get('custom_title', ''),
            ymin=ymin,
            ax=ax,
        )

    for ax in axes[n:]:       # hide any unused subplot slots
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_observed_vs_rubin(tde_z_obs, rubin_rates,
                           obs_dz=0.1, z_max=2.5, xmax=None,
                           custom_title='Previously Published & Rubin Detectable TDEs'):
    """
    Histogram of observed TDE redshifts overlaid with Rubin projected counts per year.

    Formatting follows the OTTER paper (Fig 8): stepfilled cornflowerblue, lw=4, black edge,
    log y scale, single y-axis. Rubin projection overlaid in lilac on the same axis.
    Inset (upper right): count vs log10(z), step outlines only, dashed median labels.

    Parameters
    ----------
    tde_z_obs    : array-like  redshifts from OTTER (Nones already filtered)
    rubin_rates  : dict        pre-computed detectable rates with keys 'TDE' and 'center_z'
                               (output of calculate_rubin_detectable_rates)
    obs_dz       : float       bin width — used for both histograms (default 0.1)
    z_max        : float       upper redshift limit for computation and bins (default 2.5)
    xmax         : float       x-axis display limit; defaults to z_max if None
    """
    tde_z_obs = np.asarray(tde_z_obs)
    tde_z_obs = tde_z_obs[(tde_z_obs > 0) & (tde_z_obs <= z_max)]

    obs_bins = np.arange(0, z_max + obs_dz, obs_dz)

    rubin_tde     = rubin_rates['TDE']
    rubin_centers = rubin_rates['center_z']
    rubin_dz      = rubin_centers[1] - rubin_centers[0]
    rubin_edges   = np.concatenate([[rubin_centers[0] - rubin_dz / 2],
                                     rubin_centers + rubin_dz / 2])

    # --- medians ---
    obs_median = np.median(tde_z_obs)
    cumsum = np.cumsum(rubin_tde)
    idx  = np.searchsorted(cumsum, cumsum[-1] / 2)
    frac = (cumsum[-1] / 2 - (cumsum[idx - 1] if idx > 0 else 0)) / rubin_tde[idx]
    rubin_median = rubin_centers[idx] - rubin_dz / 2 + frac * rubin_dz

    # --- main plot (single y-axis, OTTER style) ---
    fig, ax1 = plt.subplots(figsize=(13.5, 8.25))

    ax1.stairs(rubin_tde, rubin_edges, fill=True, color='#C39BD3', alpha=0.5, zorder=1)
    ax1.stairs(rubin_tde, rubin_edges, fill=False, color='k', linewidth=4, zorder=1.5)

    ax1.hist(tde_z_obs, bins=obs_bins, lw=4, histtype='stepfilled',
             color='cornflowerblue', edgecolor='k', label='Published Optical TDEs', zorder=2)

    ax1.set_yscale('log')
    ax1.set_xlabel('Redshift', fontsize=52)
    ax1.set_ylabel('Number of TDEs', fontsize=52)
    ax1.set_xlim(0, xmax if xmax is not None else z_max)


    otter_handle = Patch(facecolor='cornflowerblue', edgecolor='k', linewidth=2,
                         label='Published Optical TDEs')
    rubin_handle = Patch(facecolor='#C39BD3', edgecolor='k', linewidth=2,
                         label='Rubin WFD (1 year)')
    leg = ax1.legend(handles=[otter_handle, rubin_handle], fontsize=33,
                     loc='lower right')
    leg.set_zorder(20)

    # draw ticks and spines on top of histogram patches
    ax1.set_axisbelow(False)
    for spine in ax1.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')

    ax1.set_ylim(bottom=0.8)
    ax1.yaxis.set_major_locator(LogLocator(base=10, numticks=15))
    ax1.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
    ax1.yaxis.set_minor_formatter(NullFormatter())
    ax1.tick_params(which='major', direction='in',
                    bottom=True, top=True, left=True, right=True,
                    length=10, width=2.0, labelsize=42)
    ax1.tick_params(which='minor', direction='in',
                    bottom=True, top=True, left=True, right=True,
                    length=4, width=1.0)

    # --- inset: count vs log10(z), step outlines only ---
    ax_in = ax1.inset_axes([0.67, 0.46, 0.32, 0.50])

    log_bins = np.linspace(-2.5, np.log10(z_max), 35)
    ax_in.hist(np.log10(tde_z_obs), bins=log_bins,
               histtype='step', color='cornflowerblue', linewidth=2.0)

    rubin_edges_clipped = np.clip(rubin_edges, 1e-3, None)
    ax_in.stairs(rubin_tde, np.log10(rubin_edges_clipped), color='#7D3C98', linewidth=2.0)

    for median_z, color in [(obs_median, 'cornflowerblue'), (rubin_median, '#7D3C98')]:
        ax_in.axvline(np.log10(median_z), color=color, linestyle='--', linewidth=1.5)
        ax_in.text(np.log10(median_z) + 0.06, 0.0, f'Med.={median_z:.2f}',
                   color=color, fontsize=40, fontweight='bold', rotation=90, va='bottom',
                   transform=ax_in.get_xaxis_transform(),
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

    ax_in.set_xlim(-2.5, np.log10(z_max))
    ax_in.set_yscale('log')
    ax_in.set_xlabel('log₁₀(z)', fontsize=27)
    ax_in.tick_params(labelsize=16)

    plt.tight_layout()
    plt.show()


def plot_rates_layered(intrinsic_rates, rubin_rates, catalog_rates,
                       logy=True, ymin=None,
                       custom_title="Transient Rates: Intrinsic vs Observable",
                       select_transients=None):
    """
    Overlaid bar chart with three transparency layers per transient type:
      - Intrinsic rates        (most transparent, background)
      - Rubin-observable rates (medium alpha)
      - Catalog fields rates   (most opaque, foreground)

    Same color per transient type across all three layers.
    """
    center_z   = intrinsic_rates['center_z']
    center_lya = intrinsic_rates['center_lya']
    transients = [k for k in intrinsic_rates if k not in ('center_z', 'center_lya')]

    if select_transients is not None:
        invalid = set(select_transients) - set(transients)
        if invalid:
            raise ValueError(f"Transients not found in rates dict: {sorted(invalid)}. Available: {sorted(transients)}")
        transients = [t for t in transients if t in select_transients]

    n_bins    = len(center_z)
    n_classes = len(transients)
    width     = 0.8 / n_classes
    x         = np.arange(n_bins)

    _fallback = iter(_FALLBACK_COLORS)
    colors = [TRANSIENT_COLORS.get(t, next(_fallback, '#888888')) for t in transients]
    layers       = [intrinsic_rates, rubin_rates, catalog_rates]
    layer_alphas = [0.25,            0.55,        0.85]
    layer_labels = ['Intrinsic',     'Rubin observable', 'Catalog fields']

    fig, ax = plt.subplots(figsize=(10, 6))

    for j, (transient, color) in enumerate(zip(transients, colors)):
        for layer, alpha in zip(layers, layer_alphas):
            ax.bar(
                x + j * width,
                layer[transient],
                width=width,
                color=color,
                alpha=alpha,
                edgecolor='none',
            )

    # legend: colors for transient type, grey patches for layer meaning
    type_handles  = [Patch(facecolor=colors[j], label=t) for j, t in enumerate(transients)]
    layer_handles = [Patch(facecolor='grey', alpha=a, label=l)
                     for a, l in zip(layer_alphas, layer_labels)]
    ax.legend(handles=type_handles + layer_handles, fontsize=12, ncol=2)

    ax.set_xticks(x + width * (n_classes - 1) / 2)
    ax.set_xticklabels([f"{z:.2f}" for z in center_z])
    ax.set_xlabel("Redshift (bin center)", fontsize=18)
    ax.set_ylabel("Events per year per Δz bin", fontsize=18)
    if logy:
        ax.set_yscale("log")
    if ymin is not None:
        ax.set_ylim(bottom=ymin)
    ax.set_title(custom_title, fontsize=24, pad=25)

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xticklabels([f"{int(l)} Å" for l in center_lya])
    ax2.set_xlabel("Observed Lyα Wavelength", fontsize=18)
    ax.grid(False)
    ax2.grid(False)

    plt.tight_layout()
    plt.show()
