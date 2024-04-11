"""Simulate a PZFlow-generated catalog of 1e6 galaxies."""
import pandas as pd
from photerr import LsstErrorModel
from pzflow import Flow
from showyourwork.paths import user as Paths

# instantiate the paths
paths = Paths()

# generate redshift and observed photometry
flow = Flow(file=paths.data / "main_galaxy_flow" / "flow.pzflow.pkl")
z_and_photo = flow.sample(100_000, seed=0)

# generate true ellipticity and size
cond_flow = Flow(file=paths.data / "conditional_galaxy_flow" / "flow.pzflow.pkl")
ellip_and_size = cond_flow.sample(
    nsamples=1, conditions=z_and_photo, save_conditions=False, seed=1
)

# add minor sizes
q = (1 - ellip_and_size["ellipticity"]) / (1 + ellip_and_size["ellipticity"])
ellip_and_size["size_minor"] = q * ellip_and_size["size"]

# join photometric and size catalogs
catalog = pd.concat([z_and_photo, ellip_and_size], axis=1)

# calculate errors for observed photometry
err_model = LsstErrorModel(
    extendedType="auto",
    majorCol="size",
    minorCol="size_minor",
    decorrelate=False,
    errLoc="alone",
)
errs = err_model(catalog, random_state=0)

# add errors to catalog
catalog = pd.concat([catalog, errs], axis=1)

# re-order columns
catalog = catalog[
    ["redshift"]
    + [col for band in "ugrizy" for col in [band, band + "_err"]]
    + ["size", "ellipticity"]
]

# save the catalog!
catalog.to_pickle(paths.data / "pzflow_catalog.pkl")
