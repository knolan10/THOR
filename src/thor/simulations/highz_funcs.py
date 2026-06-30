# helper functions for my highztransients notebook

# general imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from astropy.cosmology import Planck18 as cosmo
import redback.transient_models.tde_models as redback_tde
import redback.transient_models.supernova_models as redback_sn
from astropy.cosmology import z_at_value
from astropy.modeling.models import BlackBody
import astropy.units as u
from scipy.stats import norm
from extinction import fitzpatrick99

#plotting
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
rcParams["font.family"] = "Liberation Serif"
rcParams["text.usetex"] = False


# rate functions
def luminosity_distance_from_mag(M, m_lim):
    """
    Given absolute magnitude of an object M, 
    what distance can it be seen at apparent magnitude limit m_lim?
    Returns luminosity distance in pc
    """
    mu = m_lim - M
    D_pc = 10 ** ((mu + 5) / 5)
    return D_pc * u.pc

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
    """Input:"""
    vol_bin = (cosmo.comoving_volume(z_bin_max) - cosmo.comoving_volume(z_bin_min)).to(u.Gpc**3) # convert to Gpc^3
    rate_in_bin = object_rate * vol_bin  # number of objects per year in this bin
    return rate_in_bin

def redshift_wavelength(z, rest_wavelength=1216):
    """
    Calculate observed wavelength given rest wavelength and redshift.
    Default: Lyman-alpha (1216 Å)
    """
    return rest_wavelength * (1 + z)

def calculate_rates_vs_redshift(object_rates,
                                z_min=0.0,
                                z_max=4.0,
                                dz=0.5
                                ):
    """
    Given dictionary of object rates (per Gpc^3 per year)
    Calculate rates over bins (with size dz) out to z=4
    """

    # Redshift bin edges and centers
    z_edges = np.arange(z_min, z_max + dz, dz)
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    rates_dict = {'center_z': z_centers,
                  'center_lya': redshift_wavelength(z_centers)}

    for transient, rate in object_rates.items():
        counts = np.zeros(len(z_centers))

        for i, (z_lo, z_hi) in enumerate(zip(z_edges[:-1], z_edges[1:])):
            counts[i] = calculate_rates_zbin(rate, z_lo, z_hi)

        rates_dict[transient] = np.round(counts)

    return rates_dict

# FIXME: clean up code
# remove repeat functions and hardcoding, so it works with any input with different objects
def apply_z_dependent_rates(data):
    """
    Apply redshift-dependent rate corrections to the given data.
    Z dependence of different object rates is taken from Plasticc paper (https://arxiv.org/pdf/1706.01859.pdf):

    Parameters:
        data (dict): Dictionary containing redshift bins ('center_z') and object rates ('SNIa', 'CCSN', 'TDE').

    Returns:
        dict: A dictionary with corrected rates for 'SNIa', 'CCSN', and 'TDE'.
    """
    center_z = data['center_z']
    center_lya = data['center_lya']
    corrected_data = {
        'center_z': center_z,
        'center_lya': center_lya,
        'SNIa': [],
        'CCSN': [],
        'TDE': []
    }

    # Apply corrections for SNIa
    for z, rate in zip(center_z, data['SNIa']):
        if z < 1:
            correction_factor = (1 + z)**1.5
        else:
            correction_factor = (1 + z)**-0.5
        corrected_data['SNIa'].append(rate * correction_factor)

    # Apply corrections for CCSN
    for z, rate in zip(center_z, data['CCSN']):
        correction_factor = (1 + z)**4.9
        corrected_data['CCSN'].append(rate * correction_factor)

    # Apply corrections for TDE
    for z, rate in zip(center_z, data['TDE']):
        correction_factor = 10**(-5 * z / 6)
        corrected_data['TDE'].append(rate * correction_factor)

    # Convert lists back to numpy arrays
    corrected_data['SNIa'] = np.array(corrected_data['SNIa'])
    corrected_data['CCSN'] = np.array(corrected_data['CCSN'])
    corrected_data['TDE'] = np.array(corrected_data['TDE'])

    return corrected_data



def calculate_rubin_detectable_rates(rates_vs_z, m_lim=24.5, n_z_per_bin=30, ebv=0.0, rv=3.1, f_sky = 0.436):
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
    lum_distributions = {
        'SNIa': {'M_mean': -19.3, 'M_sigma': 0.3,  'lambda_rest_aa': 4400.},
        'CCSN': {'M_mean': -17.5, 'M_sigma': 1.5,  'lambda_rest_aa': 5500.},
        'TDE':  {'M_mean': -19.0, 'M_sigma': 1.5,  'lambda_rest_aa': 3000.},
    }

    a_v = rv * ebv
    center_z = rates_vs_z['center_z']
    dz = center_z[1] - center_z[0]
    result = {'center_z': center_z, 'center_lya': rates_vs_z['center_lya']}

    for transient, dist in lum_distributions.items():
        if transient not in rates_vs_z:
            continue
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

            frac_avg      = np.trapezoid(frac_arr * dVdz_arr, z_arr) / np.trapezoid(dVdz_arr, z_arr)
            detectable[i] = intrinsic[i] * frac_avg

        result[transient] = detectable * f_sky  # scale by Rubin's sky coverage

    return result


# Plotting functions
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

    for i, transient in enumerate(transients):
        ax.bar(x + i * width, rates_vs_z[transient], width=width, label=transient, alpha=0.85)

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


def plot_observed_vs_rubin(tde_z_obs, volumetric_rates, m_lim=24.5, ebv=0.1, f_sky=0.436,
                           obs_dz=0.1, z_max=2.5,
                           custom_title='TDE Redshift Distribution: Observed vs Rubin 1-year projection'):
    """
    Histogram of observed TDE redshifts overlaid with Rubin projected counts per year.

    Formatting follows the OTTER paper (Fig 8): stepfilled cornflowerblue, lw=4, black edge,
    log y scale, single y-axis. Rubin projection overlaid in lilac on the same axis.
    Inset (upper right): count vs log10(z), step outlines only, dashed median labels.

    Parameters
    ----------
    tde_z_obs        : array-like  redshifts from OTTER (Nones already filtered)
    volumetric_rates : dict        raw volumetric rates per Gpc^3/yr (e.g. BTSvolumetricrates_calculated)
    m_lim            : float       Rubin limiting magnitude (default 24.5)
    ebv              : float       E(B-V) for dust extinction (default 0.1)
    f_sky            : float       Rubin sky fraction (default 0.436)
    obs_dz           : float       bin width — used for both histograms (default 0.1)
    z_max            : float       upper redshift limit (default 2.5)
    """
    tde_z_obs = np.asarray(tde_z_obs)
    tde_z_obs = tde_z_obs[(tde_z_obs > 0) & (tde_z_obs <= z_max)]

    obs_bins = np.arange(0, z_max + obs_dz, obs_dz)

    # compute Rubin rates at obs_dz bin width directly
    rates_naive = calculate_rates_vs_redshift(volumetric_rates, z_max=z_max, dz=obs_dz)
    rates_vs_z  = apply_z_dependent_rates(rates_naive)
    rubin_rates = calculate_rubin_detectable_rates(rates_vs_z, m_lim=m_lim, ebv=ebv, f_sky=f_sky)

    rubin_tde     = rubin_rates['TDE']
    rubin_centers = rubin_rates['center_z']
    rubin_edges   = np.concatenate([[rubin_centers[0] - obs_dz / 2],
                                     rubin_centers + obs_dz / 2])

    # --- medians ---
    obs_median = np.median(tde_z_obs)
    cumsum = np.cumsum(rubin_tde)
    idx  = np.searchsorted(cumsum, cumsum[-1] / 2)
    frac = (cumsum[-1] / 2 - (cumsum[idx - 1] if idx > 0 else 0)) / rubin_tde[idx]
    rubin_median = rubin_centers[idx] - obs_dz / 2 + frac * obs_dz

    # --- main plot (single y-axis, OTTER style) ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.stairs(rubin_tde, rubin_edges, fill=True, color='#C39BD3', alpha=0.5, zorder=1)
    ax1.stairs(rubin_tde, rubin_edges, fill=False, color='k', linewidth=4, zorder=1.5)

    ax1.hist(tde_z_obs, bins=obs_bins, lw=4, histtype='stepfilled',
             color='cornflowerblue', edgecolor='k', label='OTTER TDEs', zorder=2)

    ax1.set_yscale('log')
    ax1.set_xlabel('Redshift', fontsize=16)
    ax1.set_ylabel('Number of TDEs', fontsize=16)
    ax1.set_xlim(0, z_max)
    ax1.set_title(custom_title, fontsize=22, pad=15)

    otter_handle = Patch(facecolor='cornflowerblue', edgecolor='k', linewidth=2,
                         label='OTTER TDEs')
    rubin_handle = Patch(facecolor='#C39BD3', edgecolor='k', linewidth=2,
                         label='Rubin projected (1 yr)')
    leg = ax1.legend(handles=[otter_handle, rubin_handle], fontsize=12,
                     loc='upper left', bbox_to_anchor=(0.01, 0.93))
    leg.set_zorder(20)

    # draw ticks and spines on top of histogram patches
    ax1.set_axisbelow(False)
    for spine in ax1.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')

    ax1.minorticks_on()
    ax1.tick_params(which='major', direction='in',
                    bottom=True, top=True, left=True, right=True,
                    length=8, width=1.5)
    ax1.tick_params(which='minor', direction='in',
                    bottom=True, top=True, left=True, right=True,
                    length=4, width=1.0)

    # --- inset: count vs log10(z), step outlines only ---
    ax_in = ax1.inset_axes([0.72, 0.56, 0.27, 0.41])

    log_bins = np.linspace(-2.5, np.log10(z_max), 35)
    ax_in.hist(np.log10(tde_z_obs), bins=log_bins,
               histtype='step', color='cornflowerblue', linewidth=2.0)

    rubin_edges_clipped = np.clip(rubin_edges, 1e-3, None)
    ax_in.stairs(rubin_tde, np.log10(rubin_edges_clipped), color='#7D3C98', linewidth=2.0)

    for median_z, color, y_pos in [(obs_median,   'cornflowerblue', 0.65),
                                    (rubin_median, '#7D3C98',        0.65)]:
        ax_in.axvline(np.log10(median_z), color=color, linestyle='--', linewidth=1.5)
        ax_in.text(np.log10(median_z) + 0.06, y_pos, f'Med.={median_z:.2f}',
                   color=color, fontsize=10, fontweight='bold', rotation=90, va='center',
                   transform=ax_in.get_xaxis_transform())

    ax_in.set_xlim(-2.5, np.log10(z_max))
    ax_in.set_yscale('log')
    ax_in.set_xlabel('log₁₀(z)', fontsize=10)
    ax_in.tick_params(labelsize=8)

    plt.tight_layout()
    plt.show()


def plot_rates_layered(intrinsic_rates, rubin_rates, cosmos_rates,
                       logy=True, ymin=None,
                       custom_title="Transient Rates: Intrinsic vs Observable"):
    """
    Overlaid bar chart with three transparency layers per transient type:
      - Intrinsic rates        (most transparent, background)
      - Rubin-observable rates (medium alpha)
      - COSMOS field rates     (most opaque, foreground)

    Same color per transient type across all three layers.
    """
    center_z   = intrinsic_rates['center_z']
    center_lya = intrinsic_rates['center_lya']
    transients = [k for k in intrinsic_rates if k not in ['center_z', 'center_lya']]

    n_bins    = len(center_z)
    n_classes = len(transients)
    width     = 0.8 / n_classes
    x         = np.arange(n_bins)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:n_classes]
    layers       = [intrinsic_rates, rubin_rates, cosmos_rates]
    layer_alphas = [0.25,            0.55,        0.85]
    layer_labels = ['Intrinsic',     'Rubin observable', 'COSMOS field']

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


def igm_laf_transmission(wavelengths_obs_aa, z_source):
    """
    Mean IGM transmission due to Lyman-alpha forest absorption,
    using the Madau (1995) prescription.

    For observed wavelengths in [912*(1+z_s), 1216*(1+z_s)], applies the
    cumulative Lyman-alpha forest opacity.  Wavelengths below the redshifted
    Lyman limit (912*(1+z_s)) are completely absorbed (T = 0).  Wavelengths
    above the redshifted Lyman-alpha (1216*(1+z_s)) are unattenuated (T = 1).

    Parameters
    ----------
    wavelengths_obs_aa : array-like
        Observed wavelengths in Angstroms.
    z_source : float
        Source redshift.

    Returns
    -------
    ndarray  Transmission T in [0, 1], same shape as input.
    """
    lam = np.asarray(wavelengths_obs_aa, dtype=float)
    transmission = np.ones_like(lam)

    lya   = 1216.0   # Lyman-alpha rest wavelength [AA]
    lylim = 912.0    # Lyman limit rest wavelength [AA]

    laf_lo = lylim * (1 + z_source)
    laf_hi = lya   * (1 + z_source)

    # Lyman-alpha forest: Madau (1995) cumulative opacity
    laf_mask = (lam >= laf_lo) & (lam < laf_hi)
    transmission[laf_mask] = np.exp(-0.0036 * (lam[laf_mask] / lya) ** 3.46)

    # Lyman continuum: complete absorption below the redshifted Lyman limit
    transmission[lam < laf_lo] = 0.0

    return transmission


def plot_tde_sed_with_filters(temperature_k, redshifts, solid_angle_sr=np.pi,
                              ebv=0.1, rv=3.1, apply_igm=True):
    """
    Plot the TDE SED as spectral flux density F_λ = B_λ * solid_angle_sr,
    with optional Milky Way foreground dust extinction.

    Parameters
    ----------
    solid_angle_sr : float
        Solid angle in steradians to integrate over. Default is π sr, which
        corresponds to hemisphere-integrating an isotropic emitter:
        F_λ = ∫ B_λ cos θ dΩ = π B_λ.
        For a physical source of radius R at luminosity distance d_L,
        use solid_angle_sr = np.pi * (R / d_L)**2.
    ebv : float
        E(B-V) for MW foreground dust extinction (default 0.1, representative
        mid-latitude average). Set to 0 to disable.
    rv : float
        R_V = A_V / E(B-V) (default 3.1, standard MW diffuse ISM).
    apply_igm : bool
        If True (default), apply mean IGM Lyman-alpha forest absorption using
        the Madau (1995) prescription.  Observed wavelengths below the
        redshifted Lyman limit (912*(1+z)) are set to zero; wavelengths in
        [912*(1+z), 1216*(1+z)] are attenuated by the cumulative LAF opacity
        tau = 0.0036 * (lambda_obs / 1216)^3.46.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    bb = BlackBody(temperature=temperature_k * u.K)
    wavelengths_rest = np.linspace(500, 15000, 1000) * u.AA

    # 1. Rubin Filter Regions
    rubin_filters = {
        'u': (3200, 4000), 'g': (4000, 5520), 'r': (5520, 6910),
        'i': (6910, 8180), 'z': (8180, 9220), 'y': (9220, 10600)
    }

    for i, (band, (w_min, w_max)) in enumerate(rubin_filters.items()):
        alpha_val = 0.05 if i % 2 == 0 else 0.1
        ax.axvspan(w_min, w_max, color='gray', alpha=alpha_val)
        ax.text((w_min + w_max)/2, 0.98, band, color='dimgray',
                transform=ax.get_xaxis_transform(),
                fontweight='bold', ha='center', va='top', fontsize=14)

    # 2. Setup Colormap (Cool-to-Warm: Blue for z=0, Red for high z)
    cmap = plt.get_cmap('RdBu_r')
    colors = cmap(np.linspace(0, .8, len(redshifts)))

    # 3. Plot the SED for each redshift
    for redshift, color in zip(redshifts, colors):
        b_nu = bb(wavelengths_rest)
        radiance_unit = u.erg / (u.cm**2 * u.s * u.AA * u.sr)
        b_lambda = b_nu.to(radiance_unit, equivalencies=u.spectral_density(wavelengths_rest))

        # Integrate over solid angle to get spectral flux density F_λ [erg/s/cm²/Å]
        f_lambda = b_lambda * solid_angle_sr * u.sr

        obs_w = wavelengths_rest * (1 + redshift)
        obs_f = f_lambda / (1 + redshift)

        # MW foreground dust extinction (applied at observed wavelengths)
        if ebv > 0:
            a_v = rv * ebv
            a_lam = fitzpatrick99(obs_w.value.astype(np.float64), a_v, rv)
            obs_f = obs_f * 10**(-0.4 * a_lam)

        # IGM Lyman-alpha forest absorption
        if apply_igm and redshift > 0:
            igm_trans = igm_laf_transmission(obs_w.value, redshift)
            obs_f = obs_f * igm_trans

        ax.plot(obs_w.value, obs_f.value, label=f'z = {redshift}', color=color, lw=3)

    # Formatting
    ax.set_xlabel('Observed Wavelength (Å)', fontsize=14)
    ax.set_ylabel(r'Spectral Flux Density $F_\lambda$ ($erg/s/cm^2/\AA$)', fontsize=14)
    igm_label = ' + IGM Lyα Forest' if apply_igm else ''
    ax.set_title(f'TDE SED ({temperature_k} K) Redshifted through Rubin Bands{igm_label}', fontsize=16, pad=25)
    
    # Force Scientific Notation
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    ax.legend(loc='upper right', frameon=True, fontsize=12)
    ax.set_xlim(1000, 12000)
    ax.set_ylim(0, None)

    plt.tight_layout()
    plt.show()


def _plot_sed_with_filters(transient_label, temperature_k, redshifts,
                            solid_angle_sr=np.pi, ebv=0.1, rv=3.1, apply_igm=True):
    """Shared SED plotting logic for all transient types."""
    fig, ax = plt.subplots(figsize=(12, 8))
    bb = BlackBody(temperature=temperature_k * u.K)
    wavelengths_rest = np.linspace(500, 15000, 1000) * u.AA

    rubin_bands = {
        'u': (3200, 4000), 'g': (4000, 5520), 'r': (5520, 6910),
        'i': (6910, 8180), 'z': (8180, 9220), 'y': (9220, 10600)
    }
    for i, (band, (w_min, w_max)) in enumerate(rubin_bands.items()):
        ax.axvspan(w_min, w_max, color='gray', alpha=0.05 if i % 2 == 0 else 0.1)
        ax.text((w_min + w_max) / 2, 0.98, band, color='dimgray',
                transform=ax.get_xaxis_transform(),
                fontweight='bold', ha='center', va='top', fontsize=14)

    cmap   = plt.get_cmap('RdBu_r')
    colors = cmap(np.linspace(0, .8, len(redshifts)))

    for redshift, color in zip(redshifts, colors):
        b_nu     = bb(wavelengths_rest)
        rad_unit = u.erg / (u.cm**2 * u.s * u.AA * u.sr)
        b_lambda = b_nu.to(rad_unit, equivalencies=u.spectral_density(wavelengths_rest))
        f_lambda = b_lambda * solid_angle_sr * u.sr

        obs_w = wavelengths_rest * (1 + redshift)
        obs_f = f_lambda / (1 + redshift)

        if ebv > 0:
            a_lam = fitzpatrick99(obs_w.value.astype(np.float64), rv * ebv, rv)
            obs_f = obs_f * 10 ** (-0.4 * a_lam)

        if apply_igm and redshift > 0:
            obs_f = obs_f * igm_laf_transmission(obs_w.value, redshift)

        ax.plot(obs_w.value, obs_f.value, label=f'z = {redshift}', color=color, lw=3)

    ax.set_xlabel('Observed Wavelength (Å)', fontsize=14)
    ax.set_ylabel(r'Spectral Flux Density $F_\lambda$ ($erg/s/cm^2/\AA$)', fontsize=14)
    igm_label = ' + IGM Lyα Forest' if apply_igm else ''
    ax.set_title(f'{transient_label} SED ({temperature_k} K) Redshifted through Rubin Bands{igm_label}',
                 fontsize=16, pad=25)
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.legend(loc='upper right', frameon=True, fontsize=12)
    ax.set_xlim(1000, 12000)
    ax.set_ylim(0, None)
    plt.tight_layout()
    plt.show()


def plot_snia_sed_with_filters(temperature_k=10000, redshifts=None,
                                solid_angle_sr=np.pi, ebv=0.1, rv=3.1, apply_igm=True):
    """
    Plot the SNIa SED (blackbody at ~10000 K) redshifted through Rubin bands,
    with MW dust and IGM Lyα forest absorption.

    Parameters
    ----------
    temperature_k  : float  blackbody temperature [K] (default 10000)
    redshifts      : list   redshifts to plot (default [0, 0.5, 1.0, 1.5, 2.0, 2.5])
    solid_angle_sr : float  solid angle [sr] (default π)
    ebv            : float  MW E(B-V) (default 0.1)
    rv             : float  R_V (default 3.1)
    apply_igm      : bool   apply IGM Lyα absorption (default True)
    """
    if redshifts is None:
        redshifts = [0, 0.5, 1.0, 1.5, 2.0, 2.5]
    _plot_sed_with_filters('SNIa', temperature_k, redshifts,
                           solid_angle_sr=solid_angle_sr, ebv=ebv, rv=rv, apply_igm=apply_igm)


def plot_ccsn_sed_with_filters(temperature_k=6000, redshifts=None,
                                solid_angle_sr=np.pi, ebv=0.1, rv=3.1, apply_igm=True):
    """
    Plot the CCSN SED (blackbody at ~6000 K) redshifted through Rubin bands,
    with MW dust and IGM Lyα forest absorption.

    Parameters
    ----------
    temperature_k  : float  blackbody temperature [K] (default 6000)
    redshifts      : list   redshifts to plot (default [0, 0.5, 1.0, 1.5, 2.0, 2.5])
    solid_angle_sr : float  solid angle [sr] (default π)
    ebv            : float  MW E(B-V) (default 0.1)
    rv             : float  R_V (default 3.1)
    apply_igm      : bool   apply IGM Lyα absorption (default True)
    """
    if redshifts is None:
        redshifts = [0, 0.5, 1.0, 1.5, 2.0, 2.5]
    _plot_sed_with_filters('CCSN', temperature_k, redshifts,
                           solid_angle_sr=solid_angle_sr, ebv=ebv, rv=rv, apply_igm=apply_igm)


# ── LSST lightcurve simulation ────────────────────────────────────────────────

# LSST WFD single-visit 5σ depths (AB mag) and photometric systematic floor
_LSST_M5      = {'u': 23.9, 'g': 25.0, 'r': 24.7, 'i': 24.0, 'z': 23.3, 'y': 22.1}

# WFD per-band mean inter-visit interval [days] for a single field
# (u/g/r/i/z/y from user spec + Rubin WFD documentation)
_WFD_MEAN_INTERVAL = {'u': 60, 'g': 36, 'r': 18, 'i': 18, 'z': 20, 'y': 20}

# Maximum lunar illumination fraction [0=new, 1=full] for scheduling each band
# u: dark time only; g: dark+gray; r/i/z/y: any time
_WFD_MAX_MOON_ILL  = {'u': 0.25, 'g': 0.50, 'r': 1.0, 'i': 1.0, 'z': 1.0, 'y': 1.0}
_SIGMA_SYS    = 0.005   # systematic photometric floor [mag]
_C_AA_S       = 2.998e18  # speed of light [Å/s]
_SIGMA_SB     = 5.6704e-5  # Stefan-Boltzmann [erg/s/cm²/K⁴]
_BAND_COLORS  = {'u': '#7b2d8b', 'g': '#4daf4a', 'r': '#e41a1c',
                 'i': '#ff7f00', 'z': '#377eb8', 'y': '#a65628'}


# ── Transient energy / temperature distributions ──────────────────────────────

def _draw_tde_params(rng):
    """
    Draw (L_peak_erg_s, T_peak_k) for a TDE from observed distributions.

    log10(L_peak) ~ N(44.0, 0.6)  clipped to [43.0, 45.5]
        Based on optically-selected TDE luminosity functions
        (van Velzen et al. 2020; Hammerstein et al. 2023).
    T_peak        ~ N(30000, 10000) K  clipped to [15000, 60000]
        UV/optical blackbody temperatures from ZTF TDE sample.
    """
    log_l = float(np.clip(rng.normal(44.0, 0.6), 43.0, 45.5))
    T_k   = float(np.clip(rng.normal(30000.0, 10000.0), 15000.0, 60000.0))
    return 10.0 ** log_l, T_k


def _draw_snia_params(rng):
    """
    Draw (L_peak_erg_s, T_peak_k) for a Type Ia SN.

    log10(L_peak) ~ N(43.36, 0.15)  clipped to [42.9, 43.8]
        Narrow distribution reflecting the standardisable-candle nature of SNIa;
        σ ≈ 0.15 dex matches the uncorrected scatter (Phillips et al. 1999;
        Betoule et al. 2014).  Central value corresponds to M_B ≈ -19.3.
    T_peak        ~ N(10000, 1000) K  clipped to [8000, 13000]
        Near-maximum photospheric temperature (Nugent et al. 2002).
    """
    log_l = float(np.clip(rng.normal(43.36, 0.15), 42.9, 43.8))
    T_k   = float(np.clip(rng.normal(10000.0, 1000.0), 8000.0, 13000.0))
    return 10.0 ** log_l, T_k


def _draw_ccsn_params(rng):
    """
    Draw (L_peak_erg_s, T_peak_k) for a core-collapse SN.

    log10(L_peak) ~ N(42.7, 0.5)  clipped to [41.5, 43.8]
        Broad distribution spanning Type IIP/IIb/Ib/Ic subtypes
        (Anderson et al. 2014; Richardson et al. 2014).
    T_peak        ~ N(6000, 1500) K  clipped to [4000, 10000]
        Photospheric temperature during the plateau / near-peak phase
        (Faran et al. 2014).
    """
    log_l = float(np.clip(rng.normal(42.7, 0.5), 41.5, 43.8))
    T_k   = float(np.clip(rng.normal(6000.0, 1500.0), 4000.0, 10000.0))
    return 10.0 ** log_l, T_k


def fetch_rubin_filters(data_dir):
    """
    Download Rubin/LSST total throughput curves (atmosphere + optics + detector)
    from the official lsst/throughputs GitHub repository, saving to data_dir.
    Files are only fetched if not already present.

    Parameters
    ----------
    data_dir : str  local directory to store the .dat files

    Returns
    -------
    str  data_dir (for chaining)
    """
    os.makedirs(data_dir, exist_ok=True)
    base_url = "https://raw.githubusercontent.com/lsst/throughputs/main/baseline"
    for band in ('u', 'g', 'r', 'i', 'z', 'y'):
        fpath = os.path.join(data_dir, f"total_{band}.dat")
        if not os.path.exists(fpath):
            url  = f"{base_url}/total_{band}.dat"
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            with open(fpath, 'wb') as f:
                f.write(resp.content)
    return data_dir


def load_rubin_filters(data_dir):
    """
    Load Rubin/LSST total throughput curves (downloads if not present).

    Returns
    -------
    dict  band -> (wavelengths_aa, throughput) as 1-D numpy arrays.
          Wavelengths are in Angstroms; throughput is dimensionless [0, 1].
    """
    fetch_rubin_filters(data_dir)
    filters = {}
    for band in ('u', 'g', 'r', 'i', 'z', 'y'):
        fpath = os.path.join(data_dir, f"total_{band}.dat")
        data  = np.loadtxt(fpath, comments='#')
        wl_aa = data[:, 0] * 10.0   # nm → Å
        thru  = data[:, 1]
        filters[band] = (wl_aa, thru)
    return filters


def _tde_luminosity_scale(t_arr, t_peak_mjd, t_rise_days):
    """
    Returns L(t) / L_peak for a simple phenomenological TDE model:
      - Pre-peak  : Gaussian rise  exp(-0.5 × (Δt / t_rise)²)
      - Post-peak : fallback decay (1 + Δt / t_rise)^(−5/3)

    Both branches equal 1 at t = t_peak.
    """
    dt = np.asarray(t_arr, dtype=float) - t_peak_mjd
    return np.where(
        dt <= 0,
        np.exp(-0.5 * (dt / t_rise_days) ** 2),
        (1.0 + dt / t_rise_days) ** (-5.0 / 3.0),
    )


def _synthetic_ab_mag(f_lambda, wl_aa, filt_wl_aa, filt_thru):
    """
    Compute synthetic AB magnitude using the photon-counting bandpass formula:

        f_ν_eff = ∫ f_ν(λ) T(λ) λ dλ / ∫ T(λ) λ dλ,   f_ν = f_λ λ²/c
        m_AB    = −2.5 log10(f_ν_eff) − 48.6

    Parameters
    ----------
    f_lambda   : array  F_λ [erg/s/cm²/Å] on grid wl_aa
    wl_aa      : array  wavelengths [Å]
    filt_wl_aa : array  filter wavelength grid [Å]
    filt_thru  : array  filter throughput [0, 1]

    Returns
    -------
    float  AB magnitude, or np.inf if no measurable flux in band
    """
    T = np.interp(wl_aa, filt_wl_aa, filt_thru, left=0.0, right=0.0)
    if T.sum() == 0 or not np.any(f_lambda > 0):
        return np.inf
    f_nu  = f_lambda * wl_aa ** 2 / _C_AA_S   # erg/s/cm²/Hz
    num   = np.trapezoid(f_nu * T * wl_aa, wl_aa)
    denom = np.trapezoid(T * wl_aa, wl_aa)
    if denom <= 0 or num <= 0:
        return np.inf
    return -2.5 * np.log10(num / denom) - 48.6


def _lunar_illumination(mjd):
    """
    Approximate lunar illumination fraction [0 = new moon, 1 = full moon].
    Uses a simple sinusoidal phase model anchored to a known new moon.
    """
    lunation    = 29.530589          # synodic month [days]
    new_moon_ref = 58849.0           # MJD of new moon 2020-01-24
    phase = ((np.asarray(mjd, dtype=float) - new_moon_ref) % lunation) / lunation
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * phase))


def generate_wfd_visit_mjds(t_start_mjd, t_end_mjd, rng,
                             season_length_days=180,
                             weather_loss_frac=0.15,
                             intranight_gap_min=20.0):
    """
    Generate realistic LSST WFD visit times for a single field.

    Cadence model
    -------------
    - **Seasonal visibility**: field is observable for `season_length_days` per
      year (~180 d), creating annual gaps.
    - **Weather**: `weather_loss_frac` of observable nights are randomly lost.
    - **Moon phase**: u band scheduled only in dark time (illumination < 0.25);
      g band in dark + gray time (< 0.50); r/i/z/y at any moon phase.
    - **Same-night pairs**: every scheduled visit produces a second exposure
      `intranight_gap_min` minutes later — the standard Rubin pair strategy.
    - **Per-band mean intervals** from _WFD_MEAN_INTERVAL:
      u=60 d, g=36 d, r=18 d, i=18 d, z=20 d, y=20 d.

    Parameters
    ----------
    t_start_mjd, t_end_mjd : float  survey window in MJD
    rng                    : numpy.random.Generator
    season_length_days     : float  days visible per year (default 180)
    weather_loss_frac      : float  fraction of nights lost to weather (default 0.15)
    intranight_gap_min     : float  same-night pair separation [min] (default 20)

    Returns
    -------
    dict  band -> sorted 1-D numpy array of visit MJDs
    """
    gap_days = intranight_gap_min / 1440.0

    # Integer night grid
    nights = np.arange(int(np.floor(t_start_mjd)),
                       int(np.ceil(t_end_mjd)) + 1, dtype=float)

    # Seasonal mask: 180-day window repeating every 365.25 days
    day_in_yr = (nights - t_start_mjd) % 365.25
    in_season = day_in_yr < season_length_days

    # Weather mask
    clear = rng.random(len(nights)) > weather_loss_frac

    observable = nights[in_season & clear]

    visits = {}
    moon_ill = _lunar_illumination(observable)

    for band in ('u', 'g', 'r', 'i', 'z', 'y'):
        max_moon   = _WFD_MAX_MOON_ILL[band]
        candidates = observable[moon_ill <= max_moon]

        if len(candidates) == 0:
            visits[band] = np.array([])
            continue

        # Target: (survey_duration / mean_interval) total visits, in pairs → N_nights
        # Probability per eligible night is set so expected nights = N_target_nights,
        # regardless of how many nights are lost to season gaps, weather, or moon.
        survey_duration  = t_end_mjd - t_start_mjd
        n_target_nights  = (survey_duration / _WFD_MEAN_INTERVAL[band]) / 2.0
        prob             = np.clip(n_target_nights / len(candidates), 0.0, 1.0)

        scheduled = candidates[rng.random(len(candidates)) < prob]

        if len(scheduled) == 0:
            visits[band] = np.array([])
            continue

        # Random time within the night (evening/night hours: +0.1 to +0.4 d)
        t_first  = scheduled + rng.uniform(0.1, 0.4, len(scheduled))
        t_second = t_first + gap_days

        band_mjds = np.sort(np.concatenate([t_first, t_second]))
        band_mjds = band_mjds[(band_mjds >= t_start_mjd) & (band_mjds <= t_end_mjd)]
        visits[band] = band_mjds

    return visits


def simulate_lsst_tde_lightcurve(
    redshift,
    t_peak_mjd,
    L_peak_erg_s=None,
    T_peak_k=None,
    t_rise_days=30.0,
    t_start_mjd=None,
    t_end_mjd=None,
    cadence_days=3.0,
    ebv=0.1,
    rv=3.1,
    apply_igm=True,
    filter_dir=None,
    m5_depths=None,
    rng_seed=None,
):
    """
    Simulate a Rubin/LSST WFD multi-band lightcurve for a TDE modelled as a
    time-varying blackbody with IGM Lyα forest and MW dust extinction.

    TDE SED at each epoch:
        F_λ_obs(λ_obs, t) = π B_λ(λ_rest, T) × (R(t) / d_L)² / (1 + z)

    where R(t) = R_peak × sqrt(scale(t)), R_peak = sqrt(L_peak / 4π σ_SB T⁴),
    and scale(t) is a Gaussian rise / t^(−5/3) fallback (see _tde_luminosity_scale).

    Photometric noise model (LSST):
        SNR    = 5 × 10^(−0.4 × (m − m5))
        σ_m    = sqrt(σ_sys² + (1.09 / SNR)²)

    Parameters
    ----------
    redshift     : float  source redshift
    t_peak_mjd   : float  MJD of optical peak
    L_peak_erg_s : float  peak bolometric luminosity [erg/s]
    T_peak_k     : float  blackbody temperature [K] (constant in time)
    t_rise_days  : float  Gaussian rise timescale [days] (default 30)
    t_start_mjd  : float  survey start MJD (default t_peak − 100 days)
    t_end_mjd    : float  survey end   MJD (default t_peak + 500 days)
    cadence_days : float  mean visit interval per band [days] (default 3)
    ebv          : float  Milky Way E(B−V) (default 0.1)
    rv           : float  R_V (default 3.1)
    apply_igm    : bool   apply Lyα forest IGM absorption (default True)
    filter_dir   : str    directory for filter .dat files; downloads if missing
                          (default: simulations/data/filters/)
    m5_depths    : dict   5σ AB mag limits per band; defaults to LSST WFD design values
    rng_seed     : int    random seed for cadence jitter and photon noise

    Returns
    -------
    pd.DataFrame  columns: mjd, band, mag, mag_err, mag_true, detected
    """
    if filter_dir is None:
        filter_dir = os.path.join(os.path.dirname(__file__), 'data', 'filters')
    if m5_depths is None:
        m5_depths = _LSST_M5.copy()
    if t_start_mjd is None:
        t_start_mjd = t_peak_mjd - 100
    if t_end_mjd is None:
        t_end_mjd = t_peak_mjd + 500

    rng = np.random.default_rng(rng_seed)

    # Draw TDE physical parameters from observed distributions if not supplied
    if L_peak_erg_s is None or T_peak_k is None:
        _L, _T = _draw_tde_params(rng)
        if L_peak_erg_s is None:
            L_peak_erg_s = _L
        if T_peak_k is None:
            T_peak_k = _T

    # Physical setup
    R_peak_cm = np.sqrt(L_peak_erg_s / (4 * np.pi * _SIGMA_SB * T_peak_k ** 4))
    d_L_cm    = cosmo.luminosity_distance(redshift).to(u.cm).value
    rubin_filters = load_rubin_filters(filter_dir)

    # Rest-frame and observed wavelength grids [Å]
    wl_rest = np.linspace(500, 20000, 3000)
    wl_obs  = wl_rest * (1 + redshift)

    # Blackbody B_λ on rest-frame grid [erg/s/cm²/Å/sr]
    bb       = BlackBody(temperature=T_peak_k * u.K)
    rad_unit = u.erg / (u.cm ** 2 * u.s * u.AA * u.sr)
    B_lambda = bb(wl_rest * u.AA).to(
        rad_unit, equivalencies=u.spectral_density(wl_rest * u.AA)
    ).value

    # Peak observed SED template [erg/s/cm²/Å]
    F_template = np.pi * B_lambda * (R_peak_cm / d_L_cm) ** 2 / (1 + redshift)

    # MW dust (fixed for all epochs)
    if ebv > 0:
        a_lam = fitzpatrick99(wl_obs.astype(np.float64), rv * ebv, rv)
        F_template *= 10 ** (-0.4 * a_lam)

    # IGM Lyα forest (fixed per source redshift)
    if apply_igm and redshift > 0:
        F_template *= igm_laf_transmission(wl_obs, redshift)

    # Generate realistic WFD visit schedule
    wfd_visits = generate_wfd_visit_mjds(t_start_mjd, t_end_mjd, rng)
    records    = []

    for band in ('u', 'g', 'r', 'i', 'z', 'y'):
        visit_mjds = wfd_visits[band]
        if len(visit_mjds) == 0:
            continue
        filt_wl, filt_thru = rubin_filters[band]
        m5 = m5_depths[band]

        for mjd in visit_mjds:
            scale    = float(_tde_luminosity_scale(np.array([mjd]), t_peak_mjd, t_rise_days * (1 + redshift))[0])
            F_lambda = F_template * scale

            mag_true = _synthetic_ab_mag(F_lambda, wl_obs, filt_wl, filt_thru)
            if not np.isfinite(mag_true):
                continue

            snr         = 5.0 * 10 ** (-0.4 * (mag_true - m5))
            sigma_phot  = 1.09 / max(snr, 1e-4)
            sigma_total = float(np.sqrt(_SIGMA_SYS ** 2 + sigma_phot ** 2))
            mag_obs     = float(mag_true + rng.normal(0.0, sigma_total))

            records.append({
                'mjd':      float(mjd),
                'band':     band,
                'mag':      mag_obs,
                'mag_err':  sigma_total,
                'mag_true': float(mag_true),
                'detected': snr >= 5.0,
            })

    df = pd.DataFrame(records).sort_values('mjd').reset_index(drop=True)
    df['L_peak_erg_s'] = L_peak_erg_s
    df['T_peak_k']     = T_peak_k
    return df


def plot_lsst_tde_lightcurve(lc_df, t_peak_mjd=None, redshift=None,
                              T_k=None, L_peak=None,
                              show_nondetections=True, ax=None,
                              ymin=30, ymax=10, xmin=None, xmax=None):
    """
    Plot a multi-band LSST TDE lightcurve from simulate_lsst_tde_lightcurve.

    Detections: error bars. Non-detections: faint downward triangles.

    Parameters
    ----------
    lc_df              : pd.DataFrame  output of simulate_lsst_tde_lightcurve
    t_peak_mjd         : float         x-axis shows days from peak if provided
    redshift, T_k, L_peak : float      displayed in title if provided
    show_nondetections : bool          show upper-limit arrows (default True)
    ax                 : Axes or None
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(13, 7))

    df = lc_df.copy()
    if t_peak_mjd is not None:
        df['phase'] = df['mjd'] - t_peak_mjd
        t_col  = 'phase'
        xlabel = 'Days from Peak'
    else:
        t_col  = 'mjd'
        xlabel = 'MJD'

    for band in ('u', 'g', 'r', 'i', 'z', 'y'):
        sub    = df[df['band'] == band]
        color  = _BAND_COLORS[band]

        det = sub[sub['detected']]
        if len(det):
            ax.errorbar(det[t_col], det['mag'], yerr=det['mag_err'],
                        fmt='o', color=color, ms=5, lw=1.2, capsize=2,
                        label=band, zorder=3)

        if show_nondetections:
            ndet = sub[~sub['detected']]
            if len(ndet):
                ax.plot(ndet[t_col], ndet['mag'], 'v',
                        color=color, ms=5, alpha=0.25, zorder=2)

    ax.invert_yaxis()
    ax.set_ylim(ymin, ymax)
    if xmin is not None or xmax is not None:
        ax.set_xlim(xmin, xmax)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel('AB mag', fontsize=14)

    # Fall back to drawn values stored in the DataFrame
    if T_k is None and 'T_peak_k' in lc_df.columns:
        T_k = float(lc_df['T_peak_k'].iloc[0])
    if L_peak is None and 'L_peak_erg_s' in lc_df.columns:
        L_peak = float(lc_df['L_peak_erg_s'].iloc[0])

    title_parts = []
    if redshift is not None:
        title_parts.append(f'z = {redshift}')
    if T_k is not None:
        title_parts.append(f'T = {T_k / 1e3:.0f} kK')
    if L_peak is not None:
        title_parts.append(f'log L = {np.log10(L_peak):.1f}')
    ax.set_title('  |  '.join(title_parts), fontsize=12, pad=8)

    ax.legend(fontsize=11, ncol=2, loc='lower right')
    if t_peak_mjd is not None:
        ax.axvline(0, color='k', lw=1, ls='--', alpha=0.4, zorder=1)

    if standalone:
        plt.tight_layout()
        plt.show()


# ── SNIa LSST lightcurve simulation ──────────────────────────────────────────

def _snia_luminosity_scale(t_arr, t_peak_mjd, t_rise_days, tau_decline_days):
    """
    Returns L(t) / L_peak for a simple SNIa model:
      - Pre-peak  : Gaussian rise  exp(-0.5 × (Δt / t_rise)²)
      - Post-peak : exponential decline  exp(-Δt / τ_decline)

    Default τ_decline ≈ 15 days corresponds to Δm15 ≈ 1.1 mag, consistent with
    the width-luminosity relation for normal SNIa (Phillips 1993).
    Both branches equal 1 at t = t_peak.
    """
    dt = np.asarray(t_arr, dtype=float) - t_peak_mjd
    return np.where(
        dt <= 0,
        np.exp(-0.5 * (dt / t_rise_days) ** 2),
        np.exp(-dt / tau_decline_days),
    )


def simulate_lsst_snia_lightcurve(
    redshift,
    t_peak_mjd,
    L_peak_erg_s=None,
    T_peak_k=None,
    t_rise_days=17.0,
    tau_decline_days=15.0,
    t_start_mjd=None,
    t_end_mjd=None,
    cadence_days=3.0,
    ebv=0.1,
    rv=3.1,
    apply_igm=True,
    filter_dir=None,
    m5_depths=None,
    rng_seed=None,
):
    """
    Simulate a Rubin/LSST WFD multi-band lightcurve for a Type Ia supernova
    modelled as a time-varying blackbody with IGM Lyα forest and MW dust.

    The SNIa SED at each epoch is:
        F_λ_obs(λ_obs, t) = π B_λ(λ_rest, T) × (R(t) / d_L)² / (1 + z)

    Luminosity evolution:
      - Pre-peak  : Gaussian rise with σ = t_rise_days
      - Post-peak : exponential decline  exp(-Δt / τ_decline_days)
        Default τ ≈ 15 days → Δm15 ≈ 1.1 mag  (normal SNIa)

    The default L_peak = 2.3e43 erg/s and T = 10000 K give M_r ≈ −19.3 at z ≈ 0,
    consistent with a standard-candle SNIa (Betoule et al. 2014).

    Parameters
    ----------
    redshift          : float  source redshift
    t_peak_mjd        : float  MJD of B-band peak
    L_peak_erg_s      : float  peak bolometric luminosity [erg/s] (default 2.3e43)
    T_peak_k          : float  blackbody temperature [K] (default 10000, held constant)
    t_rise_days       : float  Gaussian rise timescale [days] (default 17)
    tau_decline_days  : float  exponential decline e-folding [days] (default 15)
    t_start_mjd       : float  survey start (default t_peak − 50 days)
    t_end_mjd         : float  survey end   (default t_peak + 200 days)
    cadence_days      : float  mean visit interval per band [days] (default 3)
    ebv               : float  Milky Way E(B−V) (default 0.1)
    rv                : float  R_V (default 3.1)
    apply_igm         : bool   apply Lyα forest IGM absorption (default True)
    filter_dir        : str    directory for filter .dat files; downloads if missing
    m5_depths         : dict   5σ AB mag limits per band; defaults to LSST WFD values
    rng_seed          : int    random seed for cadence jitter and photon noise

    Returns
    -------
    pd.DataFrame  columns: mjd, band, mag, mag_err, mag_true, detected
    """
    if filter_dir is None:
        filter_dir = os.path.join(os.path.dirname(__file__), 'data', 'filters')
    if m5_depths is None:
        m5_depths = _LSST_M5.copy()
    if t_start_mjd is None:
        t_start_mjd = t_peak_mjd - 50
    if t_end_mjd is None:
        t_end_mjd = t_peak_mjd + 200

    rng = np.random.default_rng(rng_seed)

    # Draw SNIa physical parameters from observed distributions if not supplied
    if L_peak_erg_s is None or T_peak_k is None:
        _L, _T = _draw_snia_params(rng)
        if L_peak_erg_s is None:
            L_peak_erg_s = _L
        if T_peak_k is None:
            T_peak_k = _T

    R_peak_cm = np.sqrt(L_peak_erg_s / (4 * np.pi * _SIGMA_SB * T_peak_k ** 4))
    d_L_cm    = cosmo.luminosity_distance(redshift).to(u.cm).value
    rubin_filters = load_rubin_filters(filter_dir)

    wl_rest = np.linspace(500, 20000, 3000)
    wl_obs  = wl_rest * (1 + redshift)

    bb       = BlackBody(temperature=T_peak_k * u.K)
    rad_unit = u.erg / (u.cm ** 2 * u.s * u.AA * u.sr)
    B_lambda = bb(wl_rest * u.AA).to(
        rad_unit, equivalencies=u.spectral_density(wl_rest * u.AA)
    ).value

    F_template = np.pi * B_lambda * (R_peak_cm / d_L_cm) ** 2 / (1 + redshift)

    if ebv > 0:
        a_lam = fitzpatrick99(wl_obs.astype(np.float64), rv * ebv, rv)
        F_template *= 10 ** (-0.4 * a_lam)

    if apply_igm and redshift > 0:
        F_template *= igm_laf_transmission(wl_obs, redshift)

    # Generate realistic WFD visit schedule
    wfd_visits = generate_wfd_visit_mjds(t_start_mjd, t_end_mjd, rng)
    records    = []

    for band in ('u', 'g', 'r', 'i', 'z', 'y'):
        visit_mjds = wfd_visits[band]
        if len(visit_mjds) == 0:
            continue
        filt_wl, filt_thru = rubin_filters[band]
        m5 = m5_depths[band]

        for mjd in visit_mjds:
            scale    = float(_snia_luminosity_scale(
                np.array([mjd]), t_peak_mjd,
                t_rise_days * (1 + redshift), tau_decline_days * (1 + redshift)
            )[0])
            F_lambda = F_template * scale

            mag_true = _synthetic_ab_mag(F_lambda, wl_obs, filt_wl, filt_thru)
            if not np.isfinite(mag_true):
                continue

            snr         = 5.0 * 10 ** (-0.4 * (mag_true - m5))
            sigma_phot  = 1.09 / max(snr, 1e-4)
            sigma_total = float(np.sqrt(_SIGMA_SYS ** 2 + sigma_phot ** 2))
            mag_obs     = float(mag_true + rng.normal(0.0, sigma_total))

            records.append({
                'mjd':      float(mjd),
                'band':     band,
                'mag':      mag_obs,
                'mag_err':  sigma_total,
                'mag_true': float(mag_true),
                'detected': snr >= 5.0,
            })

    df = pd.DataFrame(records).sort_values('mjd').reset_index(drop=True)
    df['L_peak_erg_s'] = L_peak_erg_s
    df['T_peak_k']     = T_peak_k
    return df


def plot_lsst_snia_lightcurve(lc_df, t_peak_mjd=None, redshift=None,
                               T_k=None, L_peak=None,
                               show_nondetections=True, ax=None,
                               ymin=30, ymax=10, xmin=None, xmax=None):
    """
    Plot a multi-band LSST SNIa lightcurve from simulate_lsst_snia_lightcurve.

    Detections: error bars. Non-detections: faint downward triangles.

    Parameters
    ----------
    lc_df              : pd.DataFrame  output of simulate_lsst_snia_lightcurve
    t_peak_mjd         : float         x-axis shows days from peak if provided
    redshift, T_k, L_peak : float      displayed in title if provided
    show_nondetections : bool          show upper-limit arrows (default True)
    ax                 : Axes or None
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(13, 7))

    df = lc_df.copy()
    if t_peak_mjd is not None:
        df['phase'] = df['mjd'] - t_peak_mjd
        t_col  = 'phase'
        xlabel = 'Days from Peak'
    else:
        t_col  = 'mjd'
        xlabel = 'MJD'

    for band in ('u', 'g', 'r', 'i', 'z', 'y'):
        sub    = df[df['band'] == band]
        color  = _BAND_COLORS[band]

        det = sub[sub['detected']]
        if len(det):
            ax.errorbar(det[t_col], det['mag'], yerr=det['mag_err'],
                        fmt='o', color=color, ms=5, lw=1.2, capsize=2,
                        label=band, zorder=3)

        if show_nondetections:
            ndet = sub[~sub['detected']]
            if len(ndet):
                ax.plot(ndet[t_col], ndet['mag'], 'v',
                        color=color, ms=5, alpha=0.25, zorder=2)

    ax.invert_yaxis()
    ax.set_ylim(ymin, ymax)
    if xmin is not None or xmax is not None:
        ax.set_xlim(xmin, xmax)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel('AB mag', fontsize=14)

    # Fall back to drawn values stored in the DataFrame
    if T_k is None and 'T_peak_k' in lc_df.columns:
        T_k = float(lc_df['T_peak_k'].iloc[0])
    if L_peak is None and 'L_peak_erg_s' in lc_df.columns:
        L_peak = float(lc_df['L_peak_erg_s'].iloc[0])

    title_parts = []
    if redshift is not None:
        title_parts.append(f'z = {redshift}')
    if T_k is not None:
        title_parts.append(f'T = {T_k / 1e3:.0f} kK')
    if L_peak is not None:
        title_parts.append(f'log L = {np.log10(L_peak):.1f}')
    ax.set_title('  |  '.join(title_parts), fontsize=12, pad=8)

    ax.legend(fontsize=11, ncol=2, loc='lower right')
    if t_peak_mjd is not None:
        ax.axvline(0, color='k', lw=1, ls='--', alpha=0.4, zorder=1)

    if standalone:
        plt.tight_layout()
        plt.show()


# ── CCSN LSST lightcurve simulation ──────────────────────────────────────────

def _ccsn_luminosity_scale(t_arr, t_peak_mjd, t_rise_days, t_plateau_days, tau_tail_days):
    """
    Returns L(t) / L_peak for a simple CCSN IIP-like model:
      - Pre-peak  : Gaussian rise   exp(-0.5 × (Δt / t_rise)²)
      - Plateau   : flat at 0.3 × L_peak for t_plateau_days after peak
      - Tail      : exponential decay  exp(-Δt_tail / tau_tail)
    """
    dt = np.asarray(t_arr, dtype=float) - t_peak_mjd
    return np.where(
        dt <= 0,
        np.exp(-0.5 * (dt / t_rise_days) ** 2),
        np.where(
            dt <= t_plateau_days,
            0.3,
            0.3 * np.exp(-(dt - t_plateau_days) / tau_tail_days),
        ),
    )


def simulate_lsst_ccsn_lightcurve(
    redshift,
    t_peak_mjd,
    L_peak_erg_s=None,
    T_peak_k=None,
    t_rise_days=10.0,
    t_plateau_days=80.0,
    tau_tail_days=50.0,
    t_start_mjd=None,
    t_end_mjd=None,
    cadence_days=3.0,
    ebv=0.1,
    rv=3.1,
    apply_igm=True,
    filter_dir=None,
    m5_depths=None,
    rng_seed=None,
):
    """
    Simulate a Rubin/LSST WFD multi-band lightcurve for a Type IIP CCSN modelled
    as a time-varying blackbody with IGM Lyα forest and MW dust extinction.

    Luminosity evolution:
      - Pre-peak  : Gaussian rise with σ = t_rise_days
      - Plateau   : 0.3 × L_peak for t_plateau_days (hydrogen recombination)
      - Tail      : exponential decay exp(-Δt / tau_tail_days) (56Co decay)

    The default L_peak = 1e43 erg/s, T = 6000 K give M_r ≈ −17 at z ≈ 0,
    consistent with a typical Type IIP CCSN (Anderson et al. 2014).

    Parameters
    ----------
    redshift         : float  source redshift
    t_peak_mjd       : float  MJD of peak
    L_peak_erg_s     : float  peak bolometric luminosity [erg/s] (default 1e43)
    T_peak_k         : float  blackbody temperature [K] (default 6000)
    t_rise_days      : float  Gaussian rise timescale [days] (default 10)
    t_plateau_days   : float  plateau duration after peak [days] (default 80)
    tau_tail_days    : float  radioactive tail e-folding [days] (default 50)
    t_start_mjd      : float  survey start (default t_peak − 20 days)
    t_end_mjd        : float  survey end   (default t_peak + 250 days)
    cadence_days     : float  mean visit interval per band [days] (default 3)
    ebv              : float  Milky Way E(B−V) (default 0.1)
    rv               : float  R_V (default 3.1)
    apply_igm        : bool   apply Lyα forest IGM absorption (default True)
    filter_dir       : str    directory for filter .dat files
    m5_depths        : dict   5σ AB mag limits per band; defaults to LSST WFD values
    rng_seed         : int    random seed

    Returns
    -------
    pd.DataFrame  columns: mjd, band, mag, mag_err, mag_true, detected
    """
    if filter_dir is None:
        filter_dir = os.path.join(os.path.dirname(__file__), 'data', 'filters')
    if m5_depths is None:
        m5_depths = _LSST_M5.copy()
    if t_start_mjd is None:
        t_start_mjd = t_peak_mjd - 20
    if t_end_mjd is None:
        t_end_mjd = t_peak_mjd + 250

    rng = np.random.default_rng(rng_seed)

    # Draw CCSN physical parameters from observed distributions if not supplied
    if L_peak_erg_s is None or T_peak_k is None:
        _L, _T = _draw_ccsn_params(rng)
        if L_peak_erg_s is None:
            L_peak_erg_s = _L
        if T_peak_k is None:
            T_peak_k = _T

    R_peak_cm = np.sqrt(L_peak_erg_s / (4 * np.pi * _SIGMA_SB * T_peak_k ** 4))
    d_L_cm    = cosmo.luminosity_distance(redshift).to(u.cm).value
    rubin_filters = load_rubin_filters(filter_dir)

    wl_rest = np.linspace(500, 20000, 3000)
    wl_obs  = wl_rest * (1 + redshift)

    bb       = BlackBody(temperature=T_peak_k * u.K)
    rad_unit = u.erg / (u.cm ** 2 * u.s * u.AA * u.sr)
    B_lambda = bb(wl_rest * u.AA).to(
        rad_unit, equivalencies=u.spectral_density(wl_rest * u.AA)
    ).value

    F_template = np.pi * B_lambda * (R_peak_cm / d_L_cm) ** 2 / (1 + redshift)

    if ebv > 0:
        a_lam = fitzpatrick99(wl_obs.astype(np.float64), rv * ebv, rv)
        F_template *= 10 ** (-0.4 * a_lam)

    if apply_igm and redshift > 0:
        F_template *= igm_laf_transmission(wl_obs, redshift)

    wfd_visits = generate_wfd_visit_mjds(t_start_mjd, t_end_mjd, rng)
    records    = []

    for band in ('u', 'g', 'r', 'i', 'z', 'y'):
        visit_mjds = wfd_visits[band]
        if len(visit_mjds) == 0:
            continue
        filt_wl, filt_thru = rubin_filters[band]
        m5 = m5_depths[band]

        for mjd in visit_mjds:
            scale    = float(_ccsn_luminosity_scale(
                np.array([mjd]), t_peak_mjd,
                t_rise_days * (1 + redshift),
                t_plateau_days * (1 + redshift),
                tau_tail_days * (1 + redshift),
            )[0])
            F_lambda = F_template * scale

            mag_true = _synthetic_ab_mag(F_lambda, wl_obs, filt_wl, filt_thru)
            if not np.isfinite(mag_true):
                continue

            snr         = 5.0 * 10 ** (-0.4 * (mag_true - m5))
            sigma_phot  = 1.09 / max(snr, 1e-4)
            sigma_total = float(np.sqrt(_SIGMA_SYS ** 2 + sigma_phot ** 2))
            mag_obs     = float(mag_true + rng.normal(0.0, sigma_total))

            records.append({
                'mjd':      float(mjd),
                'band':     band,
                'mag':      mag_obs,
                'mag_err':  sigma_total,
                'mag_true': float(mag_true),
                'detected': snr >= 5.0,
            })

    df = pd.DataFrame(records).sort_values('mjd').reset_index(drop=True)
    df['L_peak_erg_s'] = L_peak_erg_s
    df['T_peak_k']     = T_peak_k
    return df


def plot_lsst_ccsn_lightcurve(lc_df, t_peak_mjd=None, redshift=None,
                               T_k=None, L_peak=None,
                               show_nondetections=True, ax=None,
                               ymin=30, ymax=10, xmin=None, xmax=None):
    """
    Plot a multi-band LSST CCSN lightcurve from simulate_lsst_ccsn_lightcurve.

    Detections: error bars. Non-detections: faint downward triangles.

    Parameters
    ----------
    lc_df              : pd.DataFrame  output of simulate_lsst_ccsn_lightcurve
    t_peak_mjd         : float         x-axis shows days from peak if provided
    redshift, T_k, L_peak : float      displayed in title if provided
    show_nondetections : bool
    ax                 : Axes or None
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(13, 7))

    df = lc_df.copy()
    if t_peak_mjd is not None:
        df['phase'] = df['mjd'] - t_peak_mjd
        t_col  = 'phase'
        xlabel = 'Days from Peak'
    else:
        t_col  = 'mjd'
        xlabel = 'MJD'

    for band in ('u', 'g', 'r', 'i', 'z', 'y'):
        sub   = df[df['band'] == band]
        color = _BAND_COLORS[band]

        det = sub[sub['detected']]
        if len(det):
            ax.errorbar(det[t_col], det['mag'], yerr=det['mag_err'],
                        fmt='o', color=color, ms=5, lw=1.2, capsize=2,
                        label=band, zorder=3)

        if show_nondetections:
            ndet = sub[~sub['detected']]
            if len(ndet):
                ax.plot(ndet[t_col], ndet['mag'], 'v',
                        color=color, ms=5, alpha=0.25, zorder=2)

    ax.invert_yaxis()
    ax.set_ylim(ymin, ymax)
    if xmin is not None or xmax is not None:
        ax.set_xlim(xmin, xmax)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel('AB mag', fontsize=14)

    # Fall back to drawn values stored in the DataFrame
    if T_k is None and 'T_peak_k' in lc_df.columns:
        T_k = float(lc_df['T_peak_k'].iloc[0])
    if L_peak is None and 'L_peak_erg_s' in lc_df.columns:
        L_peak = float(lc_df['L_peak_erg_s'].iloc[0])

    title_parts = []
    if redshift is not None:
        title_parts.append(f'z = {redshift}')
    if T_k is not None:
        title_parts.append(f'T = {T_k / 1e3:.0f} kK')
    if L_peak is not None:
        title_parts.append(f'log L = {np.log10(L_peak):.1f}')
    ax.set_title('  |  '.join(title_parts), fontsize=12, pad=8)

    ax.legend(fontsize=11, ncol=2, loc='lower right')
    if t_peak_mjd is not None:
        ax.axvline(0, color='k', lw=1, ls='--', alpha=0.4, zorder=1)

    if standalone:
        plt.tight_layout()
        plt.show()


# ── Transient phase space: peak magnitude vs observer timescale ───────────────

# Representative parameters for each transient class.
# L_peak is calibrated so that the r-band absolute magnitude at z≈0 matches:
#   TDE  M_r = −19.0,  SNIa M_r = −19.1,  CCSN M_r = −19.0
# (calibrated numerically via _peak_band_mag at z=0.001, ebv=0, no IGM)
_TRANSIENT_MODELS = {
    'TDE':  {'L_peak': 1.774e44, 'T_k': 30000, 't_rise_days': 25,
             'color': '#e41a1c', 'marker': 'o'},
    'SNIa': {'L_peak': 1.895e43, 'T_k': 10000, 't_rise_days': 10,
             'color': '#377eb8', 'marker': 's'},
    'CCSN': {'L_peak': 1.324e43, 'T_k': 8000,  't_rise_days': 15,
             'color': '#4daf4a', 'marker': '^'},
}


def _peak_band_mag(redshift, L_peak_erg_s, T_peak_k, filt_wl, filt_thru,
                   ebv=0.0, rv=3.1, apply_igm=True):
    """
    Peak apparent AB magnitude of a blackbody transient through one Rubin filter.

    Computes F_λ_obs = π B_λ(λ_rest, T) × (R/d_L)² / (1+z) at the source peak,
    applies MW dust and IGM attenuation, then integrates through the filter.

    Parameters
    ----------
    redshift, L_peak_erg_s, T_peak_k : float  source properties
    filt_wl, filt_thru : arrays  filter wavelength grid [Å] and throughput
    ebv, rv : float  MW dust parameters
    apply_igm : bool  apply Lyα forest absorption

    Returns
    -------
    float  AB magnitude (np.inf if undetectable / outside filter)
    """
    wl_rest = np.linspace(500, 20000, 3000)
    wl_obs  = wl_rest * (1 + redshift)

    R_cm = np.sqrt(L_peak_erg_s / (4 * np.pi * _SIGMA_SB * T_peak_k ** 4))
    d_L  = cosmo.luminosity_distance(redshift).to(u.cm).value

    bb       = BlackBody(temperature=T_peak_k * u.K)
    rad_unit = u.erg / (u.cm ** 2 * u.s * u.AA * u.sr)
    B_lam    = bb(wl_rest * u.AA).to(
        rad_unit, equivalencies=u.spectral_density(wl_rest * u.AA)
    ).value

    F = np.pi * B_lam * (R_cm / d_L) ** 2 / (1 + redshift)

    if ebv > 0:
        F *= 10 ** (-0.4 * fitzpatrick99(wl_obs.astype(np.float64), rv * ebv, rv))
    if apply_igm and redshift > 0:
        F *= igm_laf_transmission(wl_obs, redshift)

    return _synthetic_ab_mag(F, wl_obs, filt_wl, filt_thru)


def plot_transient_magnitude_timescale(
    redshifts=None,
    band='r',
    filter_dir=None,
    ebv=0.0,
    rv=3.1,
    apply_igm=True,
    ax=None,
):
    """
    Peak apparent magnitude vs observer-frame rise timescale for TDE, SNIa, and CCSN
    at a range of redshifts.

    The x-axis shows the observer-frame rise timescale t_rise × (1 + z).
    The y-axis shows peak AB magnitude in `band`.  The LSST single-visit 5σ depth
    is drawn as a reference line — points below it are detectable in a single visit.
    A colorbar encodes redshift; the first and last points on each track are labelled.

    Parameters
    ----------
    redshifts  : list of float  source redshifts to evaluate (default 0.05–3.0)
    band       : str            Rubin band for synthetic photometry (default 'r')
    filter_dir : str            filter .dat directory (downloads if missing)
    ebv        : float          MW E(B−V) (default 0, no dust, for clean comparison)
    rv         : float          R_V (default 3.1)
    apply_igm  : bool           apply IGM Lyα forest (default True)
    ax         : Axes or None
    """
    if redshifts is None:
        redshifts = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
    if filter_dir is None:
        filter_dir = os.path.join(os.path.dirname(__file__), 'data', 'filters')

    z_arr = np.asarray(redshifts, dtype=float)
    rubin_filters = load_rubin_filters(filter_dir)
    filt_wl, filt_thru = rubin_filters[band]

    z_norm = plt.Normalize(vmin=z_arr.min(), vmax=z_arr.max())
    z_cmap = plt.get_cmap('plasma')

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 7))

    for name, params in _TRANSIENT_MODELS.items():
        timescales, mags, valid_z = [], [], []

        for z in z_arr:
            mag = _peak_band_mag(
                z, params['L_peak'], params['T_k'], filt_wl, filt_thru,
                ebv=ebv, rv=rv, apply_igm=apply_igm,
            )
            if np.isfinite(mag):
                timescales.append(params['t_rise_days'] * (1 + z))
                mags.append(mag)
                valid_z.append(z)

        if not timescales:
            continue

        # Connecting line
        ax.plot(timescales, mags, '-', color=params['color'], alpha=0.35, lw=2, zorder=1)

        # Scatter points coloured by redshift
        sc = ax.scatter(timescales, mags,
                        c=valid_z, cmap=z_cmap, norm=z_norm,
                        s=90, marker=params['marker'],
                        edgecolors=params['color'], linewidths=1.5, zorder=3)

        # Label lowest and highest redshift point
        for idx, ha, va in [(0, 'right', 'bottom'), (-1, 'left', 'top')]:
            ax.annotate(f'z={valid_z[idx]:.2g}',
                        xy=(timescales[idx], mags[idx]),
                        xytext=(4 * (-1 if ha == 'right' else 1), 4),
                        textcoords='offset points',
                        fontsize=8, color=params['color'], ha=ha, va=va)

        # Type label near the lowest-z point
        ax.text(timescales[0] * 0.88, mags[0], name,
                color=params['color'], fontsize=11, fontweight='bold',
                ha='right', va='center')

    # LSST single-visit depth
    m5 = _LSST_M5[band]
    ax.axhline(m5, color='k', ls='--', lw=2, alpha=0.75, zorder=2)
    ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] > 1 else 200, m5 - 0.15,
            f'LSST {band}-band 5σ ({m5} mag)', fontsize=9, ha='right', va='top', color='k')

    # Cosmetic
    ax.invert_yaxis()
    ax.set_xscale('log')
    ax.set_xlabel('Observer-frame Rise Timescale (days)', fontsize=13)
    ax.set_ylabel(f'Peak Apparent Magnitude — {band}-band (AB)', fontsize=13)
    igm_str = 'IGM on' if apply_igm else 'IGM off'
    ax.set_title(f'Transient Phase Space  |  {band}-band  |  {igm_str}',
                 fontsize=14, pad=12)

    # Legend: transient types
    handles = [
        Line2D([0], [0], marker=p['marker'], color=p['color'],
               markerfacecolor='white', markeredgewidth=1.5,
               ms=9, lw=2, label=name)
        for name, p in _TRANSIENT_MODELS.items()
    ]
    handles.append(Line2D([0], [0], color='k', ls='--', lw=2,
                          label=f'LSST {band} 5σ ({m5})'))
    ax.legend(handles=handles, fontsize=11, loc='upper left')

    # Colorbar: redshift
    sm = plt.cm.ScalarMappable(cmap=z_cmap, norm=z_norm)
    sm.set_array([])
    cbar = (fig if standalone else ax.get_figure()).colorbar(sm, ax=ax)
    cbar.set_label('Redshift', fontsize=12)

    if standalone:
        plt.tight_layout()
        plt.show()


# ── Redback TDE lightcurve simulation ────────────────────────────────────────

# LSST band central frequencies [Hz] for redback tde_analytical
_LSST_BAND_FREQ = {
    'u': 8.57e14,
    'g': 6.39e14,
    'r': 4.81e14,
    'i': 3.93e14,
    'z': 3.33e14,
    'y': 2.86e14,
}

# LSST band central wavelengths [Å] for dust/IGM corrections
_LSST_BAND_WAVE_AA = {
    'u': 3550.0,
    'g': 4770.0,
    'r': 6231.0,
    'i': 7625.0,
    'z': 9134.0,
    'y': 9894.0,
}

# redback band name mapping for magnitude output_format
_LSST_REDBACK_BAND = {
    'u': 'lsstu',
    'g': 'lsstg',
    'r': 'lsstr',
    'i': 'lssti',
    'z': 'lsstz',
    'y': 'lssty',
}


def simulate_redback_tde_lightcurve(
    redshift,
    t_peak_mjd,
    l0=1e55,
    t_0_turn=50.0,
    mej=1.0,
    vej=1e4,
    kappa=0.1,
    kappa_gamma=10.0,
    temperature_floor=5000.0,
    ebv=0.1,
    rv=3.1,
    apply_igm=True,
    t_start_mjd=None,
    t_end_mjd=None,
    cadence_days=3.0,
    m5_depths=None,
    rng_seed=None,
):
    """
    Simulate a Rubin/LSST WFD multi-band TDE lightcurve using the redback
    tde_analytical model (Diffusion + TemperatureFloor + CutoffBlackbody SED),
    with MW dust and IGM Lya forest corrections applied per band.

    Parameters
    ----------
    redshift           : float  source redshift
    t_peak_mjd         : float  MJD of optical peak
    l0                 : float  bolometric luminosity at 1 second [erg/s] (default 1e55)
    t_0_turn           : float  turn-on time in days; after this L ~ t^-5/3 (default 50)
    mej                : float  ejecta mass [Msun] (default 1.0)
    vej                : float  ejecta velocity [km/s] (default 1e4)
    kappa              : float  opacity [cm^2/g] (default 0.1)
    kappa_gamma        : float  gamma-ray opacity [cm^2/g] (default 10)
    temperature_floor  : float  SED temperature floor [K] (default 5000)
    ebv                : float  Milky Way E(B-V) (default 0.1)
    rv                 : float  R_V (default 3.1)
    apply_igm          : bool   apply Lya forest IGM absorption (default True)
    t_start_mjd        : float  survey start MJD (default t_peak - 100)
    t_end_mjd          : float  survey end MJD (default t_peak + 500)
    cadence_days       : float  mean visit interval per band [days] (default 3)
    m5_depths          : dict   5-sigma AB depth per band; defaults to LSST WFD values
    rng_seed           : int    random seed for cadence jitter and photon noise

    Returns
    -------
    pd.DataFrame  columns: mjd, band, mag, mag_err, mag_true, detected
    """
    if m5_depths is None:
        m5_depths = _LSST_M5.copy()
    if t_start_mjd is None:
        t_start_mjd = t_peak_mjd - 100
    if t_end_mjd is None:
        t_end_mjd = t_peak_mjd + 500

    rng = np.random.default_rng(rng_seed)

    # Pre-compute per-band dust + IGM magnitude offsets (constant per band)
    a_v = rv * ebv
    band_ext_mag = {}
    for band in ('u', 'g', 'r', 'i', 'z', 'y'):
        lam_obs_aa = np.array([_LSST_BAND_WAVE_AA[band]], dtype=np.float64)
        # MW dust extinction [mag] at observed band wavelength
        dust_mag = fitzpatrick99(lam_obs_aa, a_v, rv)[0] if ebv > 0 else 0.0
        # IGM Lya forest: mean transmission -> magnitude offset
        if apply_igm and redshift > 0:
            igm_trans = igm_laf_transmission(lam_obs_aa, redshift)[0]
            igm_mag = -2.5 * np.log10(max(igm_trans, 1e-10))
        else:
            igm_mag = 0.0
        band_ext_mag[band] = dust_mag + igm_mag

    # Dense time grid relative to peak (observer frame) for model evaluation
    t_grid_mjd = np.arange(t_start_mjd, t_end_mjd + 0.5, 0.5)
    t_grid_days = t_grid_mjd - t_peak_mjd + t_0_turn  # shift so peak ~ t_0_turn

    # Positive times only (model requires t > 0)
    valid = t_grid_days > 0
    t_model = t_grid_days[valid]
    t_mjd_valid = t_grid_mjd[valid]

    # Generate realistic WFD visit schedule
    wfd_visits = generate_wfd_visit_mjds(t_start_mjd, t_end_mjd, rng)

    records = []

    for band in ('u', 'g', 'r', 'i', 'z', 'y'):
        rb_band = _LSST_REDBACK_BAND[band]
        ext = band_ext_mag[band]

        # Evaluate model on dense grid
        mags_grid = redback_tde.tde_analytical(
            t_model, redshift=redshift, l0=l0, t_0_turn=t_0_turn,
            mej=mej, vej=vej, kappa=kappa, kappa_gamma=kappa_gamma,
            temperature_floor=temperature_floor,
            bands=rb_band, output_format='magnitude',
        ) + ext  # apply dust + IGM offset

        # Interpolate to visit MJDs
        visit_mjds = wfd_visits[band]
        m5 = m5_depths[band]

        for mjd in visit_mjds:
            t_day = mjd - t_peak_mjd + t_0_turn
            if t_day <= 0 or t_day > t_model[-1]:
                continue

            idx = np.searchsorted(t_mjd_valid, mjd)
            if idx == 0 or idx >= len(t_mjd_valid):
                continue
            # linear interpolation
            t0, t1 = t_mjd_valid[idx - 1], t_mjd_valid[idx]
            m0, m1 = mags_grid[idx - 1], mags_grid[idx]
            if not (np.isfinite(m0) and np.isfinite(m1)):
                continue
            w = (mjd - t0) / (t1 - t0)
            mag_true = float(m0 + w * (m1 - m0))

            snr = 5.0 * 10 ** (-0.4 * (mag_true - m5))
            sigma_phot = 1.09 / max(snr, 1e-4)
            sigma_total = float(np.sqrt(_SIGMA_SYS ** 2 + sigma_phot ** 2))
            mag_obs = float(mag_true + rng.normal(0.0, sigma_total))

            records.append({
                'mjd':      float(mjd),
                'band':     band,
                'mag':      mag_obs,
                'mag_err':  sigma_total,
                'mag_true': mag_true,
                'detected': snr >= 5.0,
            })

    return pd.DataFrame(records).sort_values('mjd').reset_index(drop=True)


def plot_redback_tde_lightcurve(lc_df, t_peak_mjd=None, redshift=None,
                                 l0=None, t_0_turn=None,
                                 show_nondetections=True, ax=None,
                                 ymin=32, ymax=14, xmin=None, xmax=None):
    """
    Plot a multi-band LSST TDE lightcurve from simulate_redback_tde_lightcurve.


    Parameters
    ----------
    lc_df              : pd.DataFrame  output of simulate_redback_tde_lightcurve
    t_peak_mjd         : float         x-axis shows days from peak if provided
    redshift, l0, t_0_turn : float     displayed in title if provided
    show_nondetections : bool          show upper-limit arrows (default True)
    ax                 : Axes or None
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(13, 7))

    df = lc_df.copy()
    if t_peak_mjd is not None:
        df['phase'] = df['mjd'] - t_peak_mjd
        t_col = 'phase'
        xlabel = 'Days from Peak'
    else:
        t_col = 'mjd'
        xlabel = 'MJD'

    for band in ('u', 'g', 'r', 'i', 'z', 'y'):
        sub = df[df['band'] == band]
        color = _BAND_COLORS[band]

        det = sub[sub['detected']]
        if len(det):
            ax.errorbar(det[t_col], det['mag'], yerr=det['mag_err'],
                        fmt='o', color=color, ms=5, lw=1.2, capsize=2,
                        label=band, zorder=3)

        if show_nondetections:
            ndet = sub[~sub['detected']]
            if len(ndet):
                ax.plot(ndet[t_col], ndet['mag'], 'v',
                        color=color, ms=5, alpha=0.25, zorder=2)

    ax.invert_yaxis()
    ax.set_ylim(ymin, ymax)
    if xmin is not None or xmax is not None:
        ax.set_xlim(xmin, xmax)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel('AB mag', fontsize=14)

    title_parts = []
    if redshift is not None:
        title_parts.append(f'z = {redshift}')
    if l0 is not None:
        title_parts.append(f'log l0 = {np.log10(l0):.1f}')
    if t_0_turn is not None:
        title_parts.append(f't_turn = {t_0_turn:.0f} d')
    ax.set_title('  |  '.join(title_parts), fontsize=12, pad=8)

    ax.legend(fontsize=11, ncol=2, loc='lower right')
    if t_peak_mjd is not None:
        ax.axvline(0, color='k', lw=1, ls='--', alpha=0.4, zorder=1)

    if standalone:
        plt.tight_layout()
        plt.show()


# ── Redback SNIa / CCSN lightcurve simulation ────────────────────────────────

def _simulate_redback_sn_lightcurve(
    model_fn, redshift, t_peak_mjd, model_kwargs,
    ebv, rv, apply_igm, t_start_mjd, t_end_mjd, m5_depths, rng_seed,
):
    """Shared cadence + noise simulation for redback supernova models."""
    if m5_depths is None:
        m5_depths = _LSST_M5.copy()
    if t_start_mjd is None:
        t_start_mjd = t_peak_mjd - 30
    if t_end_mjd is None:
        t_end_mjd = t_peak_mjd + 200

    rng = np.random.default_rng(rng_seed)

    a_v = rv * ebv
    band_ext_mag = {}
    for band in ('u', 'g', 'r', 'i', 'z', 'y'):
        lam_obs_aa = np.array([_LSST_BAND_WAVE_AA[band]], dtype=np.float64)
        dust_mag = fitzpatrick99(lam_obs_aa, a_v, rv)[0] if ebv > 0 else 0.0
        if apply_igm and redshift > 0:
            igm_trans = igm_laf_transmission(lam_obs_aa, redshift)[0]
            igm_mag = -2.5 * np.log10(max(igm_trans, 1e-10))
        else:
            igm_mag = 0.0
        band_ext_mag[band] = dust_mag + igm_mag

    # Dense time grid in observer-frame days from peak (model requires t > 0)
    t_duration = t_end_mjd - t_start_mjd + 30
    t_model = np.arange(0.1, t_duration + 0.5, 0.5)
    t_mjd_valid = t_peak_mjd + t_model - t_model[0]

    wfd_visits = generate_wfd_visit_mjds(t_start_mjd, t_end_mjd, rng)
    records = []

    for band in ('u', 'g', 'r', 'i', 'z', 'y'):
        rb_band = _LSST_REDBACK_BAND[band]
        ext = band_ext_mag[band]

        mags_grid = model_fn(
            t_model, redshift=redshift,
            bands=rb_band, output_format='magnitude',
            **model_kwargs,
        ) + ext

        visit_mjds = wfd_visits[band]
        m5 = m5_depths[band]

        for mjd in visit_mjds:
            if mjd < t_mjd_valid[0] or mjd > t_mjd_valid[-1]:
                continue
            idx = np.searchsorted(t_mjd_valid, mjd)
            if idx == 0 or idx >= len(t_mjd_valid):
                continue
            t0, t1 = t_mjd_valid[idx - 1], t_mjd_valid[idx]
            m0, m1 = mags_grid[idx - 1], mags_grid[idx]
            if not (np.isfinite(m0) and np.isfinite(m1)):
                continue
            w = (mjd - t0) / (t1 - t0)
            mag_true = float(m0 + w * (m1 - m0))

            snr = 5.0 * 10 ** (-0.4 * (mag_true - m5))
            sigma_phot = 1.09 / max(snr, 1e-4)
            sigma_total = float(np.sqrt(_SIGMA_SYS ** 2 + sigma_phot ** 2))
            mag_obs = float(mag_true + rng.normal(0.0, sigma_total))

            records.append({
                'mjd':      float(mjd),
                'band':     band,
                'mag':      mag_obs,
                'mag_err':  sigma_total,
                'mag_true': mag_true,
                'detected': snr >= 5.0,
            })

    return pd.DataFrame(records).sort_values('mjd').reset_index(drop=True)


def _plot_redback_sn_lightcurve(lc_df, title, t_peak_mjd=None,
                                 show_nondetections=True, ax=None,
                                 ymin=32, ymax=14, xmin=None, xmax=None):
    """Shared plot logic for redback supernova lightcurves."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(13, 7))

    df = lc_df.copy()
    if t_peak_mjd is not None:
        df['phase'] = df['mjd'] - t_peak_mjd
        t_col  = 'phase'
        xlabel = 'Days from Peak'
    else:
        t_col  = 'mjd'
        xlabel = 'MJD'

    for band in ('u', 'g', 'r', 'i', 'z', 'y'):
        sub   = df[df['band'] == band]
        color = _BAND_COLORS[band]

        det = sub[sub['detected']]
        if len(det):
            ax.errorbar(det[t_col], det['mag'], yerr=det['mag_err'],
                        fmt='o', color=color, ms=5, lw=1.2, capsize=2,
                        label=band, zorder=3)

        if show_nondetections:
            ndet = sub[~sub['detected']]
            if len(ndet):
                ax.plot(ndet[t_col], ndet['mag'], 'v',
                        color=color, ms=5, alpha=0.25, zorder=2)

    ax.invert_yaxis()
    ax.set_ylim(ymin, ymax)
    if xmin is not None or xmax is not None:
        ax.set_xlim(xmin, xmax)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel('AB mag', fontsize=14)
    ax.set_title(title, fontsize=12, pad=8)
    ax.legend(fontsize=11, ncol=2, loc='lower right')
    if t_peak_mjd is not None:
        ax.axvline(0, color='k', lw=1, ls='--', alpha=0.4, zorder=1)

    if standalone:
        plt.tight_layout()
        plt.show()


def simulate_redback_snia_lightcurve(
    redshift,
    t_peak_mjd,
    f_nickel=0.6,
    mej=1.4,
    kappa=0.1,
    kappa_gamma=10.0,
    vej=1e4,
    temperature_floor=3000.0,
    ebv=0.1,
    rv=3.1,
    apply_igm=True,
    t_start_mjd=None,
    t_end_mjd=None,
    m5_depths=None,
    rng_seed=None,
):
    """
    Simulate a Rubin/LSST WFD multi-band SNIa lightcurve using the redback
    type_1a model (nickel-powered, CutoffBlackbody SED + line absorption).

    Parameters
    ----------
    redshift          : float  source redshift
    t_peak_mjd        : float  MJD of optical peak
    f_nickel          : float  nickel mass fraction (default 0.6)
    mej               : float  ejecta mass [Msun] (default 1.4)
    kappa             : float  opacity [cm^2/g] (default 0.1)
    kappa_gamma       : float  gamma-ray opacity [cm^2/g] (default 10)
    vej               : float  ejecta velocity [km/s] (default 1e4)
    temperature_floor : float  SED temperature floor [K] (default 3000)
    ebv               : float  Milky Way E(B-V) (default 0.1)
    rv                : float  R_V (default 3.1)
    apply_igm         : bool   apply Lya forest IGM absorption (default True)
    t_start_mjd       : float  survey start MJD (default t_peak - 30)
    t_end_mjd         : float  survey end MJD (default t_peak + 200)
    m5_depths         : dict   5-sigma AB depth per band
    rng_seed          : int    random seed

    Returns
    -------
    pd.DataFrame  columns: mjd, band, mag, mag_err, mag_true, detected
    """
    model_kwargs = dict(
        f_nickel=f_nickel, mej=mej, kappa=kappa, kappa_gamma=kappa_gamma,
        vej=vej, temperature_floor=temperature_floor,
    )
    return _simulate_redback_sn_lightcurve(
        redback_sn.type_1a, redshift, t_peak_mjd, model_kwargs,
        ebv, rv, apply_igm, t_start_mjd, t_end_mjd, m5_depths, rng_seed,
    )


def plot_redback_snia_lightcurve(lc_df, t_peak_mjd=None, redshift=None,
                                  show_nondetections=True, ax=None,
                                  ymin=32, ymax=14, xmin=None, xmax=None):
    """Plot a multi-band LSST SNIa lightcurve from simulate_redback_snia_lightcurve."""
    parts = ['SNIa']
    if redshift is not None:
        parts.append(f'z = {redshift}')
    _plot_redback_sn_lightcurve(
        lc_df, '  |  '.join(parts), t_peak_mjd=t_peak_mjd,
        show_nondetections=show_nondetections, ax=ax,
        ymin=ymin, ymax=ymax, xmin=xmin, xmax=xmax,
    )


def simulate_redback_ccsn_lightcurve(
    redshift,
    t_peak_mjd,
    f_nickel=0.07,
    mej=8.0,
    kappa=0.1,
    kappa_gamma=10.0,
    vej=8e3,
    temperature_floor=3000.0,
    ebv=0.1,
    rv=3.1,
    apply_igm=True,
    t_start_mjd=None,
    t_end_mjd=None,
    m5_depths=None,
    rng_seed=None,
):
    """
    Simulate a Rubin/LSST WFD multi-band CCSN lightcurve using the redback
    Arnett model (nickel-powered, TemperatureFloor + Blackbody SED).

    Parameters
    ----------
    redshift          : float  source redshift
    t_peak_mjd        : float  MJD of optical peak
    f_nickel          : float  nickel mass fraction (default 0.07)
    mej               : float  ejecta mass [Msun] (default 8.0)
    kappa             : float  opacity [cm^2/g] (default 0.1)
    kappa_gamma       : float  gamma-ray opacity [cm^2/g] (default 10)
    vej               : float  ejecta velocity [km/s] (default 8000)
    temperature_floor : float  SED temperature floor [K] (default 3000)
    ebv               : float  Milky Way E(B-V) (default 0.1)
    rv                : float  R_V (default 3.1)
    apply_igm         : bool   apply Lya forest IGM absorption (default True)
    t_start_mjd       : float  survey start MJD (default t_peak - 30)
    t_end_mjd         : float  survey end MJD (default t_peak + 200)
    m5_depths         : dict   5-sigma AB depth per band
    rng_seed          : int    random seed

    Returns
    -------
    pd.DataFrame  columns: mjd, band, mag, mag_err, mag_true, detected
    """
    model_kwargs = dict(
        f_nickel=f_nickel, mej=mej, kappa=kappa, kappa_gamma=kappa_gamma,
        vej=vej, temperature_floor=temperature_floor,
    )
    return _simulate_redback_sn_lightcurve(
        redback_sn.arnett, redshift, t_peak_mjd, model_kwargs,
        ebv, rv, apply_igm, t_start_mjd, t_end_mjd, m5_depths, rng_seed,
    )


def plot_redback_ccsn_lightcurve(lc_df, t_peak_mjd=None, redshift=None,
                                  show_nondetections=True, ax=None,
                                  ymin=32, ymax=14, xmin=None, xmax=None):
    """Plot a multi-band LSST CCSN lightcurve from simulate_redback_ccsn_lightcurve."""
    parts = ['CCSN']
    if redshift is not None:
        parts.append(f'z = {redshift}')
    _plot_redback_sn_lightcurve(
        lc_df, '  |  '.join(parts), t_peak_mjd=t_peak_mjd,
        show_nondetections=show_nondetections, ax=ax,
        ymin=ymin, ymax=ymax, xmin=xmin, xmax=xmax,
    )