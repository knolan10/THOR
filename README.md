## THOR (Transients at High Observed Redshifts) identifies distant Tidal Disruption Events (TDEs) in [LSST data](https://rubinobservatory.org/).

This work simulates expected rates, SEDs, and lightcurves for TDEs and other transients across redshifts. It filters the LSST alert stream for high redshift objects, through catalog crossmatching and photometric selection techniques.


### Getting started

Create an environement and install packages. For example:
  ```bash
  git clone https://github.com/knolan10/THOR.git
  cd THOR
  uv venv --python 3.12 && source .venv/bin/activate
  uv pip install -r requirementstxt
  uv pip install -e .
  ```
  
Some [example notebooks](docs/notebooks/) demonstrate fetching and filtering LSST alerts, and visualizing the LSST alert stream.

To fetch and visualise LSST alerts for a given night, run:

```bash
 `python src/thor/summarize_rubin_alerts.py 07-01-2026 07-02-2026`
 ```

This uses the Babamul alert broker to fetch alerts, which requires user credentials stored in a .env file locally. Omit dates to default to the previous night. The generated skymap is saved to `data/plots/`

To fetch alerts, crossmatch against available catalogs, and save candidates, run:

```bash
python src/thor/crossmatch_alerts.py --start 06-28-2026 --end 06-30-2026 --additional_filtering tde_filter --scan
```

If there are any matches, the crossmatch result is saved to `data/lsst_alert_download/crossmatch_candidates_{timestamp}.json`. To apply additional TDE-specific filtering, pass `--additional_filtering tde_filter`. Add flag `--save` to save raw alerts to `data/lsst_alert_download/raw_files/`, and flag `--save_results` in order to save the crossmatch details locally to `data/lsst_alert_download/`. A summary of results will be printed in command line, but the --scan flag can also be included to open a temp jupyter notebook in browser and use Babamul's scanning tool.


### Data

Large data files (catalogs, simulation results, alert stream outputs) are not tracked in this repository. They are all saved locally in `THOR/data`.

In `data/catalogs` we keep all catalogs used for crossmatching. These catalogs are recorded in [`catalogs_catalog`](src/thor/catalogs_catalog.py). Most catalogs used have had basic quality cuts to select only galaxies, and have been reduced to selected columns.

In `data/lsst_alert_download` we save LSST alerts fetched with the [Babamul alert broker](https://babamul.caltech.edu/).

`data/` also contains the [Elassticc data](https://portal.nersc.gov/cfs/lsst/DESC_TD_PUBLIC/ELASTICC/ELASTICC2_TRAINING_SAMPLE_2/
) used for contaminant transients for filter development. It contains the [rubin pointing database](https://s3df.slac.stanford.edu/data/rubin/sim-data/sims_featureScheduler_runs5.3/baseline/) used to simulate LSST lightcurves.


[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/THOR?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/THOR/)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/knolan10/THOR/smoke-test.yml)](https://github.com/knolan10/THOR/actions/workflows/smoke-test.yml)
[![Codecov](https://codecov.io/gh/knolan10/THOR/branch/main/graph/badge.svg)](https://codecov.io/gh/knolan10/THOR)
[![Benchmarks](https://img.shields.io/github/actions/workflow/status/knolan10/THOR/asv-main.yml?label=benchmarks)](https://knolan10.github.io/THOR/)

For questions [Report an issue](https://github.com/knolan10/THOR/issues) or reach out at kinolan@unc.edu.

This project was automatically generated using the LINCC-Frameworks [python-project-template](https://github.com/lincc-frameworks/python-project-template).


