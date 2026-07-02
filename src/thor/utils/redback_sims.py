import redback
import os
import matplotlib.pyplot as plt

VALID_SOURCES = ('otter', 'oac_tde', 'oac_sn', 'fink_tde', 'fink_sn')


def load_data_redback_object(object_name, data_source):
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

    Returns
    -------
    data : pd.DataFrame
        Photometry table returned by redback.
    """
    if data_source not in VALID_SOURCES:
        raise ValueError(f"data_source must be one of {VALID_SOURCES}, got {data_source!r}")

    # redback bug fix: otter requires this directory to exist before fetching
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


def create_redback_transient(object_name, data_source, data,
                              data_mode='flux_density', exclude_bands=None):
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

    if data_source == 'otter':
        return redback.tde.TDE.from_otter(
            name=object_name, data_mode=data_mode, active_bands=bands)
    elif data_source in ('oac_tde', 'fink_tde'):
        return redback.tde.TDE.from_open_access_catalogue(
            name=object_name, data_mode=data_mode, active_bands=bands)
    elif data_source in ('oac_sn', 'fink_sn'):
        return redback.supernova.Supernova.from_open_access_catalogue(
            name=object_name, data_mode=data_mode, active_bands=bands)


def _sort_transient_by_time(transient_object):
    """Sort all per-observation arrays on a redback transient in-place by time.

    Workaround for a redback bug in Diffusion.convert_input_luminosity where
    dense_times is built from time[-1] rather than time.max(), causing an
    IndexError when the data are not time-ordered.
    """
    import numpy as np
    idx = np.argsort(transient_object.x)
    for attr in ('x', 'y', 'y_err', 'x_err', 'bands', 'frequencies'):
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
            save=False,
            show=True,
        )

    return result
