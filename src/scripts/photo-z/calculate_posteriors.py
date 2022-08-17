"""Use the pz flow ensemble to estimate redshift posteriors for test set galaxies."""
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

# setup the redshift grid
grid = np.linspace(0, 3.5, 351)

# calculate posteriors
pdfs = flowEns.posterior(
    test_set[:100_000],
    "redshift",
    grid,
    err_samples=10,
    batch_size=100,
)

# save the posteriors
np.savez(paths.data / "redshift_posteriors.npz", grid=grid, pdfs=pdfs)
