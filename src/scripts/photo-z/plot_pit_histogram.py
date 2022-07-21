"""Plot the PIT histogram."""
import matplotlib.pyplot as plt
import numpy as np
from load_pzflow_catalog import load_pzflow_catalog
from showyourwork.paths import user as Paths

# instantiate the paths
paths = Paths()

# load the estimated posteriors
data = np.load(paths.data / "redshift_posteriors.npz")

grid = data["grid"]
pdfs = data["pdfs"]

# calculate cdfs
cdfs = pdfs.cumsum(axis=1) * (grid[1] - grid[0])

# get the true redshifts
test_set = load_pzflow_catalog(subset="test")
z_spec = test_set.redshift.to_numpy()[: len(cdfs)]

# get PIT values
idx = np.abs(grid - z_spec[:, None]).argmin(axis=1)
pits = cdfs[np.arange(len(cdfs)), idx]

# plot the PIT histogram
fig, ax = plt.subplots(figsize=(3, 2.5), constrained_layout=True)
ax.hist(pits, bins=20, density=True, histtype="step")
ax.plot([0, 1], [1, 1], c="k", ls="--", alpha=0.75)
ax.set(
    xlim=(0, 1),
    xlabel="PIT Value",
    ylabel="Density",
)

fig.savefig(paths.figures / "pit_histogram.pdf")
