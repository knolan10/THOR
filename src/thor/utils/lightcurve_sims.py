import os
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from astropy.cosmology import Planck18 as cosmo
from astropy.modeling.models import BlackBody
from extinction import fitzpatrick99
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter


# ── Transient characteristics and SED simulation ────────────────────────────────────────────────


def plot_blackbody_sed_with_filters(temperature_k, redshifts, solid_angle_sr=np.pi,
                              ebv=0.1, rv=3.1, apply_igm=True):
    """
    Plot the SED as spectral flux density F_λ = B_λ * solid_angle_sr,
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
    ax.set_title(f'Blackbody SED ({temperature_k} K) Redshifted through Rubin Bands{igm_label}', fontsize=16, pad=25)
    
    # Force Scientific Notation
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    ax.legend(loc='upper right', frameon=True, fontsize=12)
    ax.set_xlim(1000, 12000)
    ax.set_ylim(0, None)

    plt.tight_layout()
    plt.show()


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

# ── Observational pathway (dust and telescopes) ────────────────────────────────────────────────

# FIXME: better to use a Rubin pointing database
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


# ── Per-class defaults ────────────────────────────────────────────────────────

_TRANSIENT_LC_DEFAULTS = {
    'TDE':  {'t_start_offset': -100, 't_end_offset': 500,  't_rise_days': 30.0,
             'draw_fn': '_draw_tde_params'},
    'SNIa': {'t_start_offset': -50,  't_end_offset': 200,  't_rise_days': 17.0,
             'draw_fn': '_draw_snia_params'},
    'CCSN': {'t_start_offset': -20,  't_end_offset': 250,  't_rise_days': 10.0,
             'draw_fn': '_draw_ccsn_params'},
}

# ── Unified LSST lightcurve simulation ───────────────────────────────────────

def simulate_lsst_lightcurve(
    transient_class,
    redshift,
    t_peak_mjd,
    L_peak_erg_s=None,
    T_peak_k=None,
    t_rise_days=None,
    tau_decline_days=15.0,
    t_plateau_days=80.0,
    tau_tail_days=50.0,
    t_start_mjd=None,
    t_end_mjd=None,
    ebv=0.1,
    rv=3.1,
    apply_igm=True,
    filter_dir=None,
    m5_depths=None,
    rng_seed=None,
):
    """
    Simulate a Rubin/LSST WFD multi-band lightcurve for a blackbody transient
    with IGM Lyα forest and MW dust extinction.

    SED at each epoch:
        F_λ_obs(λ_obs, t) = π B_λ(λ_rest, T) × (R(t) / d_L)² / (1 + z)

    Photometric noise model (LSST):
        SNR  = 5 × 10^(−0.4 × (m − m5))
        σ_m  = sqrt(σ_sys² + (1.09 / SNR)²)

    Parameters
    ----------
    transient_class  : str   'TDE', 'SNIa', or 'CCSN'
    redshift         : float source redshift
    t_peak_mjd       : float MJD of optical peak
    L_peak_erg_s     : float peak bolometric luminosity [erg/s]
    T_peak_k         : float blackbody temperature [K] (constant in time)
    t_rise_days      : float Gaussian rise timescale [days]
                             (defaults: TDE 30, SNIa 17, CCSN 10)
    tau_decline_days : float SNIa only — exponential decline e-folding [days] (default 15)
    t_plateau_days   : float CCSN only — plateau duration after peak [days] (default 80)
    tau_tail_days    : float CCSN only — radioactive tail e-folding [days] (default 50)
    t_start_mjd      : float survey start MJD (defaults: TDE t_peak−100, SNIa −50, CCSN −20)
    t_end_mjd        : float survey end   MJD (defaults: TDE t_peak+500, SNIa +200, CCSN +250)
    ebv              : float Milky Way E(B−V) (default 0.1)
    rv               : float R_V (default 3.1)
    apply_igm        : bool  apply Lyα forest IGM absorption (default True)
    filter_dir       : str   directory for filter .dat files; downloads if missing
    m5_depths        : dict  5σ AB mag limits per band; defaults to LSST WFD design values
    rng_seed         : int   random seed for cadence jitter and photon noise

    Returns
    -------
    pd.DataFrame  columns: mjd, band, mag, mag_err, mag_true, detected
    """
    if transient_class not in _TRANSIENT_LC_DEFAULTS:
        raise ValueError(f"transient_class must be one of {list(_TRANSIENT_LC_DEFAULTS)}")

    defaults = _TRANSIENT_LC_DEFAULTS[transient_class]

    if t_rise_days is None:
        t_rise_days = defaults['t_rise_days']
    if filter_dir is None:
        filter_dir = os.path.join(os.path.dirname(__file__), 'data', 'filters')
    if m5_depths is None:
        m5_depths = _LSST_M5.copy()
    if t_start_mjd is None:
        t_start_mjd = t_peak_mjd + defaults['t_start_offset']
    if t_end_mjd is None:
        t_end_mjd = t_peak_mjd + defaults['t_end_offset']

    rng = np.random.default_rng(rng_seed)

    # Draw physical parameters from observed distributions if not supplied
    if L_peak_erg_s is None or T_peak_k is None:
        draw_fn = {'TDE': _draw_tde_params, 'SNIa': _draw_snia_params,
                   'CCSN': _draw_ccsn_params}[transient_class]
        _L, _T = draw_fn(rng)
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

    # MW dust and IGM (fixed for all epochs)
    if ebv > 0:
        F_template *= 10 ** (-0.4 * fitzpatrick99(wl_obs.astype(np.float64), rv * ebv, rv))
    if apply_igm and redshift > 0:
        F_template *= igm_laf_transmission(wl_obs, redshift)

    # Resolve scale function once before the loop
    z1 = 1 + redshift
    if transient_class == 'TDE':
        def _scale(mjd):
            return float(_tde_luminosity_scale(np.array([mjd]), t_peak_mjd, t_rise_days * z1)[0])
    elif transient_class == 'SNIa':
        def _scale(mjd):
            return float(_snia_luminosity_scale(np.array([mjd]), t_peak_mjd, t_rise_days * z1, tau_decline_days * z1)[0])
    else:
        def _scale(mjd):
            return float(_ccsn_luminosity_scale(np.array([mjd]), t_peak_mjd, t_rise_days * z1, t_plateau_days * z1, tau_tail_days * z1)[0])

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
            F_lambda = F_template * _scale(mjd)

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


def plot_lsst_lightcurve(lc_df, transient_class=None, t_peak_mjd=None, redshift=None,
                         T_k=None, L_peak=None, show_nondetections=True, ax=None,
                         ymin=30, ymax=10, xmin=None, xmax=None):
    """
    Plot a multi-band LSST lightcurve from simulate_lsst_lightcurve.

    Detections: error bars. Non-detections: faint downward triangles.

    Parameters
    ----------
    lc_df              : pd.DataFrame  output of simulate_lsst_lightcurve
    transient_class    : str           'TDE', 'SNIa', or 'CCSN' — shown in title
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

    if T_k is None and 'T_peak_k' in lc_df.columns:
        T_k = float(lc_df['T_peak_k'].iloc[0])
    if L_peak is None and 'L_peak_erg_s' in lc_df.columns:
        L_peak = float(lc_df['L_peak_erg_s'].iloc[0])

    title_parts = []
    if transient_class is not None:
        title_parts.append(transient_class)
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


# ── Transient phase space plotting: peak magnitude vs observer timescale ───────────────

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


def plot_rubin_abs_mag_limit(z_max=5, ax=None):
    """Plot Rubin LSST r-band detection limit as absolute magnitude vs redshift.

    Parameters
    ----------
    z_max : float
        Maximum redshift for the x-axis. Default 5.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure is created.
    """
    from astropy.cosmology import Planck18

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 5))

    z = np.linspace(0.01, z_max, 500)

    # Rubin LSST r-band 5-sigma depths (Ivezic+2019)
    rubin_limits = {
        'Single visit (r~24.5)': 24.5,
        '10-yr coadd (r~27.5)': 27.5,
    }

    # Luminosity distance -> distance modulus
    dl_pc = Planck18.luminosity_distance(z).to(u.pc).value
    DM = 5 * np.log10(dl_pc / 10)

    for label, m_lim in rubin_limits.items():
        # M = m - DM  (no K-correction)
        M_lim = m_lim - DM
        ax.plot(z, M_lim, label=label)

    ax.invert_yaxis()
    ax.set_xlabel('Redshift')
    ax.set_ylabel(r'Absolute magnitude $M_r$ [AB]')
    ax.set_xlim(0, z_max)
    ax.set_title('Rubin LSST detection limit vs redshift (r-band, no K-correction)')
    ax.legend()
    ax.grid(alpha=0.3)

    if standalone:
        plt.tight_layout()
        plt.show()
