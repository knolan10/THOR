import os
import astropy.units as u
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
from matplotlib.ticker import ScalarFormatter
rcParams["font.family"] = "Liberation Serif"
rcParams["text.usetex"] = False


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


# ── Plotting ─────────────────────────────────────────────────────────────
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
