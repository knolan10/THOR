import redback
import os
import matplotlib.pyplot as plt

VALID_SOURCES = ('otter', 'oac_tde', 'oac_sn', 'fink_tde', 'fink_sn')

# Band colors physically motivated by each filter's central wavelength.
# Used by all plotting functions; extend as needed.
BAND_COLORS = {
    # LSST
    'lsstu': 'violet',
    'lsstg': 'forestgreen',
    'lsstr': 'tomato',
    'lssti': 'darkorange',
    'lsstz': 'firebrick',
    'lssty': 'saddlebrown',
    # ZTF
    'ztfg': 'forestgreen',
    'ztfr': 'tomato',
    'ztfi': 'darkorange',
    # Swift/UVOT
    'W2': 'darkviolet',
    'M2': 'blueviolet',
    'W1': 'mediumslateblue',
    'U':  'slateblue',
    'B':  'royalblue',
    'V':  'yellowgreen',
    # Johnson-Cousins
    'Uj': 'violet',
    'Bj': 'royalblue',
    'Vj': 'yellowgreen',
    'Rj': 'tomato',
    'Ij': 'saddlebrown',
}


def load_data_redback_object(object_name, data_source, outdir=None):
    """
    Load photometry for a transient via redback and return the data table.

    Parameters
    ----------
    object_name : str
        Transient name (e.g. 'AT2019cmw', 'ZTF25acfyeke').
    data_source : str
        One of:
          'otter'    – TDE from the OTTER catalog
          'oac_tde'  – TDE from the Open Access Catalog
          'oac_sn'   – supernova from the Open Access Catalog
          'fink_tde' – TDE from Fink (ZTF stream)
          'fink_sn'  – supernova from Fink (ZTF stream)
    outdir : str or None
        Root directory under which redback will write its data subdirectories
        (e.g. 'supernova/<name>/').  Defaults to the current working directory.

    Returns
    -------
    data : pd.DataFrame
        Photometry table returned by redback.
    """
    if data_source not in VALID_SOURCES:
        raise ValueError(f"data_source must be one of {VALID_SOURCES}, got {data_source!r}")

    # redback bug fix: otter requires this directory to exist before fetching
    # Redback hardcodes paths relative to CWD; temporarily switch to outdir.
    _original_cwd = os.getcwd()
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        os.chdir(outdir)
    try:
        if data_source == 'otter':
            uvoir_dir = f'tidal_disruption_event/{object_name}/uvoir'
            os.makedirs(uvoir_dir, exist_ok=True)
            # if a previous run failed after writing the raw file but before writing
            # metadata, delete the raw file so both get re-fetched cleanly
            raw = f'{uvoir_dir}/{object_name}_rawdata.csv'
            meta = f'{uvoir_dir}/{object_name}_metadata.csv'
            if os.path.exists(raw) and not os.path.exists(meta):
                os.remove(raw)

        if data_source == 'otter':
            return redback.get_data.get_tidal_disruption_event_data_from_otter(transient=object_name)
        elif data_source == 'oac_tde':
            return redback.get_data.get_tidal_disruption_event_data_from_open_transient_catalog_data(transient=object_name)
        elif data_source == 'oac_sn':
            return redback.get_data.get_supernova_data_from_open_transient_catalog_data(transient=object_name)
        elif data_source == 'fink_tde':
            return redback.get_data.get_fink_data(transient=object_name, transient_type='tidal_disruption_event')
        elif data_source == 'fink_sn':
            return redback.get_data.get_fink_data(transient=object_name, transient_type='supernova')
    finally:
        os.chdir(_original_cwd)


def create_redback_transient(object_name, data_source, data,
                              data_mode='flux_density', exclude_bands=None, outdir=None):
    """
    Create a redback transient object from previously loaded photometry data.

    Parameters
    ----------
    object_name : str
        Transient name (e.g. 'AT2019cmw').
    data_source : str
        Same source used in load_data_redback_object. Controls which redback
        class and constructor are used.
    data : pd.DataFrame
        Output of load_data_redback_object — used to determine available bands.
    data_mode : str
        Redback data mode. Default: 'flux_density'.
    exclude_bands : list or None
        Bands to exclude from active_bands (e.g. ['W2', 'M2', 'W1', 'U']).
        If None, all bands in the data are used.
    outdir : str or None
        Must match the outdir passed to load_data_redback_object so redback
        can find the saved data files. Defaults to current working directory.

    Returns
    -------
    transient_object : redback transient
        Redback TDE or Supernova object ready for fitting/plotting.
    """
    if data_source not in VALID_SOURCES:
        raise ValueError(f"data_source must be one of {VALID_SOURCES}, got {data_source!r}")

    bands = data['band'].unique()
    if exclude_bands is not None:
        bands = [b for b in bands if b not in exclude_bands]

    _original_cwd = os.getcwd()
    if outdir is not None:
        os.chdir(outdir)
    try:
        if data_source == 'otter':
            return redback.tde.TDE.from_otter(
                name=object_name, data_mode=data_mode, active_bands=bands)
        elif data_source in ('oac_tde', 'fink_tde'):
            return redback.tde.TDE.from_open_access_catalogue(
                name=object_name, data_mode=data_mode, active_bands=bands)
        elif data_source in ('oac_sn', 'fink_sn'):
            return redback.supernova.Supernova.from_open_access_catalogue(
                name=object_name, data_mode=data_mode, active_bands=bands)
    finally:
        os.chdir(_original_cwd)


def _sort_transient_by_time(transient_object):
    """Sort all per-observation arrays on a redback transient in-place by time.

    Workaround for a redback bug in Diffusion.convert_input_luminosity where
    dense_times is built from time[-1] rather than time.max(), causing an
    IndexError when the data are not time-ordered.
    """
    import numpy as np
    idx = np.argsort(transient_object.x)
    for attr in ('x', 'y', 'y_err', 'x_err', 'bands', 'frequency'):
        arr = getattr(transient_object, attr, None)
        if arr is not None and hasattr(arr, '__len__') and len(arr) == len(idx):
            setattr(transient_object, attr, arr[idx])


def fit_redback_model(
    transient_object,
    object_name,
    model='tde_analytical',
    sampler='dynesty',
    nlive=500,
    npool=4,
    resume=True,
    prior_overrides=None,
    multiband_filters=None,
    random_models=100,
    outdir_root='redback_inference',
    model_kwargs_overrides=None,
):
    """
    Run redback inference on a transient object and plot results.

    Parameters
    ----------
    transient_object : redback transient
        Output of create_redback_transient.
    object_name : str
        Transient name — used for outdir and result label.
    model : str
        Redback model name. Default: 'tde_analytical'.
    sampler : str
        Bilby sampler. Default: 'dynesty'.
    nlive : int
        Number of live points. Default: 500.
    npool : int
        Number of parallel processes. Default: 4.
    resume : bool
        Resume from existing checkpoint if available. Default: True.
    prior_overrides : dict or None
        Parameter overrides applied on top of the default priors.
        Values can be floats (fixed) or bilby Prior objects.
        Example::

            import bilby
            prior_overrides = {
                'redshift': 0.403,
                'temperature_0': bilby.core.prior.Uniform(10000, 60000,
                                     name='temperature_0',
                                     latex_label='$T_0$ (K)'),
            }

    multiband_filters : list of str or None
        Filters for plot_multiband_lightcurve (e.g. ['ztfg', 'ztfr']).
        If None, the multiband plot is skipped.
    random_models : int
        Number of posterior draws to overplot. Default: 100.
    outdir_root : str
        Root directory for inference outputs. Default: 'redback_inference'.
    model_kwargs_overrides : dict or None
        Extra kwargs merged into model_kwargs before fitting. Useful for
        sncosmo-based models, e.g. {'sncosmo_model': 'salt2-extended'}.

    Returns
    -------
    result : redback result
        Bilby/redback result object.
    """
    # Workaround for redback bug in Diffusion.convert_input_luminosity:
    # dense_times uses time[-1] as the max, which is only correct if times are
    # sorted. Unsorted data causes searchsorted to return an out-of-bounds index.
    _sort_transient_by_time(transient_object)

    outdir = f'{outdir_root}/{object_name}/{model}'
    os.makedirs(outdir, exist_ok=True)

    priors = redback.priors.get_priors(model=model)
    if prior_overrides:
        for key, val in prior_overrides.items():
            priors[key] = val

    model_kwargs = dict(
        frequency=transient_object.filtered_frequencies,
        output_format='flux_density',
    )
    if model_kwargs_overrides:
        model_kwargs.update(model_kwargs_overrides)

    result = redback.fit_model(
        transient=transient_object,
        model=model,
        sampler=sampler,
        model_kwargs=model_kwargs,
        prior=priors,
        sample='rslice',
        nlive=nlive,
        npool=npool,
        resume=resume,
        outdir=outdir,
        label=f'{object_name}_nlive{nlive}',
        plot=False,
    )

    # result.transient reconstructs from metadata with wrong path;
    # use transient_object directly for plots
    model_func = redback.model_library.all_models_dict[model]

    result.plot_corner(save=False, show=True)
    plt.show()
    plt.close('all')

    transient_object.plot_lightcurve(
        model=model_func,
        posterior=result.posterior,
        model_kwargs=result.model_kwargs,
        random_models=random_models,
        band_colors=BAND_COLORS,
        save=False,
        show=True,
    )

    if multiband_filters is not None:
        transient_object.plot_multiband_lightcurve(
            model=model_func,
            posterior=result.posterior,
            model_kwargs=result.model_kwargs,
            random_models=random_models,
            filters=multiband_filters,
            band_colors=BAND_COLORS,
            save=False,
            show=True,
        )

    # delete dynesty checkpoint plots and resume pickle — not useful after a
    # completed run; the result.json retains everything needed for analysis
    import glob
    for pat in ('*_checkpoint_*.png', '*_resume.pickle'):
        for f in glob.glob(os.path.join(outdir, pat)):
            os.remove(f)

    return result


def load_redback_result(
    name,
    data_source,
    model,
    nlive=500,
    exclude_bands=None,
    base_dir='redback_inference',
    random_models=100,
    plot=False,
):
    """
    Load a saved redback result from disk and optionally plot the lightcurve.

    Parameters
    ----------
    name : str
        Transient name (e.g. 'ZTF18acecugr').
    data_source : str
        Same source used when fitting. One of VALID_SOURCES.
    model : str
        Redback model name (e.g. 'tde_analytical').
    nlive : int
        Number of live points used during fitting — used to reconstruct the result label.
    exclude_bands : list or None
        Bands to exclude when recreating the transient object.
    base_dir : str
        Root directory containing the redback_inference subdirectory.
        Default: 'redback_inference'.
    random_models : int
        Number of posterior draws to overplot. Default: 100.
    plot : bool
        If True, plot the lightcurve with posterior draws. Default: False.

    Returns
    -------
    result : redback result
        The loaded result object.
    transient : redback transient
        The recreated transient object.
    """
    result = redback.result.read_in_result(
        outdir=f'{base_dir}/{name}/{model}',
        label=f'{name}_nlive{nlive}',
    )

    data = load_data_redback_object(object_name=name, data_source=data_source)
    transient = create_redback_transient(
        object_name=name,
        data_source=data_source,
        data=data,
        data_mode='flux_density',
        exclude_bands=exclude_bands,
    )

    if plot:
        model_func = redback.model_library.all_models_dict[model]
        transient.plot_lightcurve(
            model=model_func,
            posterior=result.posterior,
            model_kwargs=result.model_kwargs,
            random_models=random_models,
            band_colors=BAND_COLORS,
            save=False,
            show=True,
        )

    return result, transient


# backwards-compatible alias
plot_redback_result = load_redback_result


def plot_lsst_projection(
    result,
    transient,
    name,
    model,
    model_func=None,
    redshift=None,
    n_draws=200,
    n_times=400,
    figsize=(10, 5),
):
    """
    Plot posterior-predictive lightcurves projected onto all LSST ugrizy bands.

    Parameters
    ----------
    result : redback result
        Loaded result object (e.g. from plot_redback_result).
    transient : redback transient
        Transient object (used to set the time axis range).
    name : str
        Transient name — used in the plot title.
    model : str
        Redback model name — used in the plot title.
    model_func : callable or None
        The model function. If None, looked up from redback.model_library.
    redshift : float or None
        Redshift to use for all draws. If None, the fitted redshift from each
        posterior sample is used.
    n_draws : int
        Number of posterior draws. Default: 200.
    n_times : int
        Number of time grid points. Default: 400.
    figsize : tuple
        Figure size. Default: (10, 5).

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    import numpy as np

    lsst_bands  = ['lsstu', 'lsstg', 'lsstr', 'lssti', 'lsstz', 'lssty']
    lsst_labels = ['u', 'g', 'r', 'i', 'z', 'y']
    lsst_colors = [BAND_COLORS[b] for b in lsst_bands]
    lsst_freqs  = dict(zip(lsst_bands, redback.utils.bands_to_frequency(lsst_bands)))

    if model_func is None:
        model_func = redback.model_library.all_models_dict[model]

    times = np.linspace(transient.x.min(), transient.x.max(), n_times)

    param_cols = [c for c in result.posterior.columns if c not in ('log_likelihood', 'log_prior')]
    samples = result.posterior[param_cols].sample(n_draws, replace=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=figsize)

    for band, label, color in zip(lsst_bands, lsst_labels, lsst_colors):
        freqs = np.full(len(times), lsst_freqs[band])

        draws = []
        for _, row in samples.iterrows():
            params = row.to_dict()
            if redshift is not None:
                params['redshift'] = redshift
            try:
                draws.append(model_func(times, **params, frequency=freqs, output_format='flux_density'))
            except ValueError:
                break
        else:
            draws = np.array(draws)
            median = np.median(draws, axis=0)
            lo, hi = np.percentile(draws, [16, 84], axis=0)
            ax.fill_between(times, lo, hi, alpha=0.2, color=color)
            ax.plot(times, median, color=color, lw=1.5, label=f'LSST {label}')
            continue

        print(f'Skipping LSST {label} ({band}): outside model wavelength range')

    z_label = f'z = {redshift}' if redshift is not None else 'fitted z'
    ax.set_xlabel('Observer-frame time (days)', fontsize=13)
    ax.set_ylabel('Flux density (mJy)', fontsize=13)
    ax.set_title(f'{name} — {model} projected onto LSST ugrizy ({z_label})', fontsize=14)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    return fig, ax


def plot_redshift_evolution(
    result,
    name,
    model,
    model_func=None,
    model_kwargs=None,
    transient=None,
    times=None,
    redshifts=None,
    figsize=(15, 8),
):
    """
    Plot LSST ugrizy absolute-magnitude lightcurves across a range of redshifts
    using the median posterior parameters from a fitted result.

    Parameters
    ----------
    result : redback result
        Loaded result object.
    name : str
        Transient name — used in the plot title.
    model : str
        Redback model name — used in the plot title.
    model_func : callable or None
        The model function. If None, looked up from redback.model_library.
    model_kwargs : dict or None
        Extra kwargs forwarded to the model (e.g. {'sncosmo_model': 'salt2'}).
        If None, uses result.model_kwargs with output_format and frequency removed.
    transient : redback transient or None
        If provided, the time axis is set from transient.x.min()/max().
        Takes precedence over the `times` default but not an explicit `times` argument.
    times : array-like or None
        Observer-frame time grid in days. If None and transient is provided, uses
        transient data range. If both are None, defaults to np.linspace(0, 300, 500).
    redshifts : list of (float, linestyle) or None
        Redshift/linestyle pairs to overplot per band. Default:
        [(0.1, '-'), (0.5, '--'), (1.0, '-.'), (2.0, ':'), (3.0, (0, (3, 1, 1, 1)))].
    figsize : tuple
        Figure size. Default: (15, 8).

    Returns
    -------
    fig, axes : matplotlib Figure and axes array (2x3)
    """
    import numpy as np
    from astropy.cosmology import Planck18 as cosmo

    if model_func is None:
        model_func = redback.model_library.all_models_dict[model]

    if times is None:
        if transient is not None:
            times = np.linspace(transient.x.min(), transient.x.max(), 500)
        else:
            times = np.linspace(0, 300, 500)

    if redshifts is None:
        redshifts = [(0.1, '-'), (0.5, '--'), (1.0, '-.'), (2.0, ':'), (3.0, (0, (3, 1, 1, 1)))]

    # Build extra model kwargs: strip frequency/output_format (we supply those per call)
    if model_kwargs is None:
        extra_kwargs = {k: v for k, v in (result.model_kwargs or {}).items()
                        if k not in ('frequency', 'output_format')}
    else:
        extra_kwargs = {k: v for k, v in model_kwargs.items()
                        if k not in ('frequency', 'output_format')}

    band_names  = {'u': 'lsstu', 'g': 'lsstg', 'r': 'lsstr', 'i': 'lssti', 'z': 'lsstz', 'y': 'lssty'}
    band_colors = {label: BAND_COLORS[reg] for label, reg in band_names.items()}
    lsst_freqs  = dict(zip(band_names.values(),
                           redback.utils.bands_to_frequency(list(band_names.values()))))

    param_cols = [c for c in result.posterior.columns
                  if c not in ('log_likelihood', 'log_prior', 'redshift')]
    params = result.posterior[param_cols].median().to_dict()

    # No sharex/sharey — hidden empty axes with shared limits cause NaN tick errors
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    finite_vals = []
    for ax, (band, band_reg) in zip(axes.flat, band_names.items()):
        band_plotted = False
        for z, ls in redshifts:
            # Try magnitude output first; fall back to flux_density → AB mag conversion
            # (some models, e.g. sncosmo-based, don't support output_format='magnitude')
            try:
                mag = model_func(
                    times, redshift=z, **params,
                    bands=np.full(len(times), band_reg),
                    output_format='magnitude',
                    **extra_kwargs,
                )
            except Exception:
                try:
                    flux = model_func(
                        times, redshift=z, **params,
                        frequency=np.full(len(times), lsst_freqs[band_reg]),
                        output_format='flux_density',
                        **extra_kwargs,
                    )
                    # flux in mJy → AB magnitude
                    with np.errstate(divide='ignore', invalid='ignore'):
                        mag = -2.5 * np.log10(flux * 1e-3 / 3631.0)
                except Exception as e:
                    print(f'Skipping LSST {band} z={z}: {e}')
                    continue
            abs_mag = mag - cosmo.distmod(z).value
            # Replace inf with nan so matplotlib draws clean gaps instead of artifacts
            abs_mag = np.where(np.isfinite(abs_mag), abs_mag, np.nan)
            finite = abs_mag[np.isfinite(abs_mag)]
            if len(finite) == 0:
                print(f'Skipping LSST {band} z={z}: all-NaN output')
                continue
            ax.plot(times, abs_mag, color=band_colors[band], linestyle=ls, label=f'z={z}')
            finite_vals.extend(finite.tolist())
            band_plotted = True
        if not band_plotted:
            ax.set_visible(False)
        else:
            ax.set_title(band, fontsize=14, color=band_colors[band])
            ax.legend(fontsize=9)
            ax.set_xlim(times[0], times[-1])

    # Apply consistent limits and labels only to visible axes
    if finite_vals:
        ymin = float(np.min(finite_vals))
        ymax = float(np.max(finite_vals))
        pad = (ymax - ymin) * 0.05
        for ax in axes.flat:
            if ax.get_visible():
                ax.set_ylim(ymax + pad, ymin - pad)  # inverted: brighter on top

    for ax in axes[1]:
        if ax.get_visible():
            ax.set_xlabel('Time (days)', size=13)
    for ax in axes[:, 0]:
        if ax.get_visible():
            ax.set_ylabel('Absolute magnitude', size=13)

    fig.suptitle(f'{name} ({model} median posterior) — LSST bands by redshift', size=15)
    plt.tight_layout()
    plt.show()

    return fig, axes


class RubinSimulator:
    """
    Simulate Rubin/LSST photometry for a fitted redback model using lightcurvelynx.

    Parameters
    ----------
    result : redback result
        Fitted result object. Median posterior is used as the model parameters.
    model : str
        Redback model name (e.g. 'tde_analytical', 'salt2').
    survey : str
        'baseline' or 'ddf'. Selects the correct OpSim database automatically
        when used with data_dir.
    data_dir : str
        Directory containing the OpSim .db files. The correct file is chosen
        based on survey. Default: current working directory.
    opsim_path : str or None
        Explicit path to an OpSim database, overriding data_dir + survey lookup.
    redshift : float or None
        Override redshift. None uses the median posterior redshift.
    filters : list of str or None
        LSST filters to simulate. Default: ['g', 'r', 'i', 'z'].
    phase_bounds : tuple of (float, float)
        (min, max) phase in days passed to RedbackWrapperModel. Default: (0.1, 300.0).
    """

    OPSIM_FILENAMES = {
        'baseline': 'p13_baseline_v5.3.0_10yrs.db',
        'ddf':      'ddf_sd_v5.3.0_10yrs.db',
    }

    # Models that must go through SncosmoWrapperModel rather than RedbackWrapperModel.
    # Maps redback model name → sncosmo source name.
    SNCOSMO_MODELS = {
        'salt2': 'salt2',
        'salt3': 'salt3',
    }

    BAND_COLORS = {
        'u': BAND_COLORS['lsstu'],
        'g': BAND_COLORS['lsstg'],
        'r': BAND_COLORS['lsstr'],
        'i': BAND_COLORS['lssti'],
        'z': BAND_COLORS['lsstz'],
        'y': BAND_COLORS['lssty'],
    }

    def __init__(
        self,
        result,
        model,
        survey,
        data_dir='.',
        opsim_path=None,
        redshift=None,
        filters=None,
        phase_bounds=(0.1, 300.0),
    ):
        if survey not in ('baseline', 'ddf'):
            raise ValueError(f"survey must be 'baseline' or 'ddf', got {survey!r}")

        self.result = result
        self.model = model
        self.survey = survey
        self.opsim_path = opsim_path or os.path.join(data_dir, self.OPSIM_FILENAMES[survey])
        self.redshift = redshift
        self.filters = filters or ['g', 'r', 'i', 'z']
        self.phase_bounds = phase_bounds
        self.lightcurves = None
        self._params = None

        self._setup()

    @classmethod
    def from_params(
        cls,
        params,
        model,
        survey,
        data_dir='.',
        opsim_path=None,
        filters=None,
        phase_bounds=(0.1, 300.0),
    ):
        """
        Create a RubinSimulator from a manually-specified parameter dict,
        without requiring a fitted redback result object.

        Parameters
        ----------
        params : dict
            Model parameters (e.g. ``{'mej': 1.0, 'vej': 1e4, 'kappa': 0.1,
            'redshift': 0.5, ...}``).  A ``'redshift'`` key is required.
            ``'t0'`` will be set automatically to the survey start time.
        model : str
            Redback model name (e.g. 'tde_analytical').
        survey : str
            'baseline' or 'ddf'.
        data_dir : str
            Directory containing the OpSim .db files. Default: current directory.
        opsim_path : str or None
            Explicit path to an OpSim database, overriding data_dir + survey lookup.
        filters : list of str or None
            LSST filters to simulate. Default: ['g', 'r', 'i', 'z'].
        phase_bounds : tuple of (float, float)
            (min, max) phase in days. Default: (0.1, 300.0).

        Returns
        -------
        sim : RubinSimulator

        Examples
        --------
        >>> sim = RubinSimulator.from_params(
        ...     params={
        ...         'mej': 1.0, 'vej': 1e4, 'kappa': 0.1,
        ...         'kappa_gamma': 10.0, 'temperature_floor': 5000.0,
        ...         'l0': 1e55, 't_0_turn': 50.0, 'redshift': 0.5,
        ...     },
        ...     model='tde_analytical',
        ...     survey='baseline',
        ...     data_dir='../../data',
        ...     filters=['g', 'r', 'i', 'z'],
        ... )
        >>> sim.simulate()
        >>> sim.plot()
        """
        if 'redshift' not in params:
            raise ValueError("params must contain a 'redshift' key.")

        # Build a minimal mock object that _setup() can use in place of a real result.
        # We store _manual_params so _setup() can detect this path.
        instance = cls.__new__(cls)
        instance.result = None
        instance._manual_params = dict(params)
        instance.model = model
        instance.survey = survey
        if survey not in ('baseline', 'ddf'):
            raise ValueError(f"survey must be 'baseline' or 'ddf', got {survey!r}")
        instance.opsim_path = opsim_path or os.path.join(data_dir, cls.OPSIM_FILENAMES[survey])
        instance.redshift = params['redshift']
        instance.filters = filters or ['g', 'r', 'i', 'z']
        instance.phase_bounds = phase_bounds
        instance.lightcurves = None
        instance._params = None
        instance._setup()
        return instance

    def _setup(self):
        """Load OpSim db, build survey_info and the appropriate source model."""
        from lightcurvelynx.obstable.opsim import OpSim
        from lightcurvelynx.astro_utils.passbands import PassbandGroup
        from lightcurvelynx.survey_info import SurveyInfo
        from lightcurvelynx.math_nodes.ra_dec_sampler import ObsTableRADECSampler
        from lightcurvelynx.utils.extrapolate import ConstantPadding
        import numpy as np

        opsim_db = OpSim.from_db(self.opsim_path)
        opsim_db = opsim_db.filter_rows(np.isin(opsim_db['filter'], self.filters))
        t_min, _ = opsim_db.time_bounds()

        self._survey_info = SurveyInfo(
            obstable=opsim_db,
            passbands=PassbandGroup.from_preset(preset='LSST', filters=self.filters, units='nm'),
        )

        if hasattr(self, '_manual_params'):
            # from_params() path: use the dict directly
            params = dict(self._manual_params)
        else:
            params = (
                self.result.posterior
                .drop(columns=['log_likelihood', 'log_prior'], errors='ignore')
                .median()
                .to_dict()
            )
            if self.redshift is not None:
                params['redshift'] = self.redshift
        self._params = params

        ra_dec_radius = 1.5 if self.survey == 'ddf' else 3.0
        ra_dec_sampler = ObsTableRADECSampler(opsim_db, radius=ra_dec_radius)

        if self.model in self.SNCOSMO_MODELS:
            # sncosmo-native models: use SncosmoWrapperModel to avoid the
            # sncosmo.Model vs sncosmo.Source mismatch in RedbackWrapperModel.
            from lightcurvelynx.models.sncosmo_models import SncosmoWrapperModel
            self._t0 = t_min  # place SN peak at survey start so it falls in the LSST window
            self._source = SncosmoWrapperModel(
                self.SNCOSMO_MODELS[self.model],
                x0=params['x0'],
                x1=params['x1'],
                c=params['c'],
                redshift=params['redshift'],
                t0=self._t0,
                ra=ra_dec_sampler.ra,
                dec=ra_dec_sampler.dec,
                time_extrapolation=(ConstantPadding(0.0), ConstantPadding(0.0)),
            )
        else:
            from lightcurvelynx.models.redback_models import RedbackWrapperModel
            params['t0'] = t_min  # redback t0 = survey start MJD
            self._t0 = t_min
            self._source = RedbackWrapperModel(
                redback.model_library.all_models_dict[self.model],
                parameters=params,
                ra=ra_dec_sampler.ra,
                dec=ra_dec_sampler.dec,
                phase_bounds=self.phase_bounds,
                time_extrapolation=(ConstantPadding(0.0), ConstantPadding(0.0)),
            )

    def simulate(self, n_sims=1):
        """
        Run the simulation.

        Parameters
        ----------
        n_sims : int
            Number of lightcurves to simulate. Default: 1.

        Returns
        -------
        lightcurves : nested_pandas.NestedFrame
        """
        from lightcurvelynx.simulate import simulate_lightcurves
        self.lightcurves = simulate_lightcurves(self._source, n_sims, self._survey_info)
        return self.lightcurves

    def plot(self, idx=0, tmax_days=None, plot_magnitudes=True, name=None, ax=None):
        """
        Plot a simulated lightcurve.

        Parameters
        ----------
        idx : int
            Index into lightcurves to plot. Default: 0.
        tmax_days : float or None
            Clip lightcurve to this many days after t0. None shows all.
        plot_magnitudes : bool
            Plot AB magnitude instead of flux. Default: True.
        name : str or None
            Object name for the plot title. None omits it.
        ax : matplotlib Axes or None
            Axes to plot into. None creates a new figure.

        Returns
        -------
        ax : matplotlib Axes
        """
        from lightcurvelynx.utils.plotting import plot_lightcurves as _plot_lc

        if self.lightcurves is None:
            raise RuntimeError("No lightcurves yet — call simulate() first.")

        lc = self.lightcurves.iloc[idx]['lightcurve'].copy()
        lc['days'] = lc['mjd'] - self._t0
        if tmax_days is not None:
            lc = lc[lc['days'] <= tmax_days]

        z = self._params['redshift']
        title = (
            f'{name} ({self.model}, z={z:.3f}, survey={self.survey}) simulated LSST photometry'
            if name else
            f'{self.model} z={z:.3f} [{self.survey}] simulated LSST photometry'
        )

        ax = _plot_lc(
            fluxes=lc['flux'].values,
            times=lc['days'].values,
            fluxerrs=lc['fluxerr'].values,
            filters=lc['filter'].values,
            colormap=self.BAND_COLORS,
            plot_magnitudes=plot_magnitudes,
            title=title,
            ax=ax,
        )

        # Set y limits from photometry values only, ignoring error bars
        import numpy as np
        fluxes = lc['flux'].values
        valid = np.isfinite(fluxes) & (fluxes > 0)
        if valid.any():
            if plot_magnitudes:
                from lightcurvelynx.astro_utils.mag_flux import flux2mag
                mags = flux2mag(fluxes[valid])
                finite_mags = mags[np.isfinite(mags)]
                if len(finite_mags) > 0:
                    pad = (finite_mags.max() - finite_mags.min()) * 0.1
                    ax.set_ylim(finite_mags.max() + pad, finite_mags.min() - pad)
            else:
                f = fluxes[valid]
                pad = (f.max() - f.min()) * 0.1
                ax.set_ylim(f.min() - pad, f.max() + pad)

        ax.set_xlabel('Days since t0')
        plt.show()
        return ax
