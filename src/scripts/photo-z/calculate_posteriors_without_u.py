"""Caculate posteriors but marginalize over the u band."""
import numpy as np
from load_pzflow_catalog import load_pzflow_catalog
from pzflow import FlowEnsemble
from showyourwork.paths import user as Paths

# instantiate the paths
paths = Paths()

# load the flow ensemble
flowEns = FlowEnsemble(file=paths.data / "pz_ensemble" / "pz_ensemble.pzflow.pkl")

# load the test set
test_set = load_pzflow_catalog(subset="test")
test_set = test_set[:20]

# delete the u band magnitudes
test_set["u"] *= np.nan

# setup the redshift grid
grid = np.linspace(0, 3.5, 351)

# calculate posteriors
pdfs = flowEns.posterior(
    test_set,
    "redshift",
    grid,
    err_samples=10,
    batch_size=100,
    marg_rules={
        "flag": np.nan, 
        "u": lambda row: np.linspace(row["g"]-2, row["g"]+1, 10),
    }
)

# save the posteriors
np.savez(paths.data / "redshift_posteriors_without_u.npz", grid=grid, pdfs=pdfs)