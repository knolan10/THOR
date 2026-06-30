## THOR (Transients at High Observed Redshifts) identifies distant Tidal Disruption Events (TDEs) in [LSST data](https://rubinobservatory.org/).

In [`simulations`](src/thor/simulations/) we simulate expectations of transients observed by LSST at high (z>1) redshifts. We develop filters to select for these distant objects.

In [`alert_stream`](src/thor/alert_stream/) we develop a pipeline to filter real LSST alerts, using the alert broker [Babamul/Boom](https://www.ztf.caltech.edu/ztf-boom-babamul.html#:~:text=A%20multi%2Dsurvey%20data%20archive%20and%20alert%20broker%20(Babamul)%20combining%20data%20from&text=The%20source%20code%20and%20documentation%20are%20open%20source%20and%20hosted%20on%20Github.). This includes crossmatches with high-redshift galaxies from catalogs, as well as searches for hostless sources. 

### Data

Large data files (catalogs, simulation results, alert stream outputs) are not tracked in this repository. 

In 'data/catalogs' we keep all catalogs used for crossmatching. These catalogs are recorded in [`catalogs_catalog`](src/thor/alert_stream/catalogs_catalog.py). Most catalogs used have had basic quality cuts to select only galaxies, and have been reduced to selected columns.

In 'data/lsst_alert_download' we save LSST alerts fetched with the [Babamul alert broker](https://babamul.caltech.edu/).

'data/' also contains the [Elassticc data](https://portal.nersc.gov/cfs/lsst/DESC_TD_PUBLIC/ELASTICC/ELASTICC2_TRAINING_SAMPLE_2/
) used for contaminant transients for filter development. It contains the [rubin pointing database](https://s3df.slac.stanford.edu/data/rubin/sim-data/sims_featureScheduler_runs5.3/baseline/) used to simulate LSST lightcurves.


[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/THOR?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/THOR/)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/knolan10/THOR/smoke-test.yml)](https://github.com/knolan10/THOR/actions/workflows/smoke-test.yml)
[![Codecov](https://codecov.io/gh/knolan10/THOR/branch/main/graph/badge.svg)](https://codecov.io/gh/knolan10/THOR)
[![Benchmarks](https://img.shields.io/github/actions/workflow/status/knolan10/THOR/asv-main.yml?label=benchmarks)](https://knolan10.github.io/THOR/)

For questions [Report an issue](https://github.com/knolan10/THOR/issues) or reach out at kinolan@unc.edu.

This project was automatically generated using the LINCC-Frameworks [python-project-template](https://github.com/lincc-frameworks/python-project-template).
