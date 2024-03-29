"""Simulate a PZFlow-generated catalog of 1e6 galaxies."""
import pandas as pd
from photerr import LsstErrorModel
from pzflow import Flow
from showyourwork.paths import user as Paths

# instantiate the paths
paths = Paths()

# generate true photometry (+ redshift)
flow = Flow(file=paths.data / "main_galaxy_flow" / "flow.pzflow.pkl")
truth_sample = flow.sample(1_000_000, seed=0)

# generate true ellipticity and size
cond_flow = Flow(file=paths.data / "conditional_galaxy_flow" / "flow.pzflow.pkl")
ellip_and_size = cond_flow.sample(
    nsamples=1, conditions=truth_sample, save_conditions=False, seed=1
)

# simulate observations with the fiducial LSST error model
err_model = LsstErrorModel(sigLim=1)
obs_sample = err_model(truth_sample[list("ugrizy")], random_state=2)

# relabel the true and observed values
truth_sample = truth_sample.rename(
    columns={col: f"true_{col}" for col in truth_sample.columns}
)
ellip_and_size = ellip_and_size.rename(
    columns={col: f"true_{col}" for col in ellip_and_size.columns}
)
obs_sample = obs_sample.rename(
    columns={col: f"obs_{col}" for col in obs_sample.columns}
)

# combine the two catalogs into a single DataFrame
pzflow_catalog = pd.concat((truth_sample, obs_sample, ellip_and_size), axis=1)
pzflow_catalog = pzflow_catalog[
    ["true_redshift"]
    + [
        col
        for band in list("ugrizy")
        for col in [f"true_{band}", f"obs_{band}", f"obs_{band}_err"]
    ]
    + ["true_ellipticity", "true_size"]
]

# save the catalog!
pzflow_catalog.to_pickle(paths.data / "pzflow_catalog.pkl")
