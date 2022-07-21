"""Plot the photo-z point estimates."""
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

# get point estimates
z_phot = grid[pdfs.argmax(axis=1)]

# load the true redshifts
test_set = load_pzflow_catalog(subset="test")
z_spec = test_set.redshift[: len(z_phot)]

# create the figure
fig, ax = plt.subplots(figsize=(3, 3), constrained_layout=True)
ax.scatter(z_spec, z_phot, s=1, rasterized=True)
ax.plot(grid, grid, c="k", lw=1)
ax.set(
    xlabel="redshift",
    ylabel="photo-z",
    xticks=np.arange(0, 4),
    yticks=np.arange(0, 4),
    xlim=(grid.min(), grid.max()),
    ylim=(grid.min(), grid.max()),
)

fig.savefig(paths.figures / "pz_point_estimates.pdf", dpi=600)
