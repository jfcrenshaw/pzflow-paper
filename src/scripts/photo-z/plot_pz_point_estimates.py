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
z_spec = test_set.redshift.iloc[: len(z_phot)].to_numpy()

# select the non-zero z_phots
idx = np.where(z_phot > 0)

# create the figure
fig, ax = plt.subplots(figsize=(3.3, 2.7), constrained_layout=True, dpi=200)
hb = ax.hexbin(z_spec[idx], z_phot[idx], extent=(0, 3, 0, 3), gridsize=205, bins="log")
cb = fig.colorbar(hb, ax=ax)
cb.set_label("Galaxies")
ax.set(
    xlabel="true redshift",
    ylabel="photo-z",
    xticks=np.arange(0, 4),
    yticks=np.arange(0, 4),
    xlim=(0, 3),
    ylim=(0, 3),
)
ax.plot([0, 3], [0, 3], c="k", ls="--", lw=1)

fig.savefig(paths.figures / "pz_point_estimates.pdf", dpi=600)
