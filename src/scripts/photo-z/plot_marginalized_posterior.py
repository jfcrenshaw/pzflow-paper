"""Plot the posterior with u band marginalized."""
import matplotlib.pyplot as plt
import numpy as np
from load_pzflow_catalog import load_pzflow_catalog
from showyourwork.paths import user as Paths

# instantiate the paths
paths = Paths()

# load the test set
test_set = load_pzflow_catalog(subset="test")
test_set = test_set[:20]

# load the posteriors
grid = np.load(paths.data / "redshift_posteriors.npz")["grid"]
pdfs0 = np.load(paths.data / "redshift_posteriors.npz")["pdfs"][:20]
pdfs1 = np.load(paths.data / "redshift_posteriors_without_u.npz")["pdfs"]

# create the figure
fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(2.5, 3.4),
    dpi=120,
    constrained_layout=True,
    gridspec_kw={
        "height_ratios": [4, 1],
    },
)

# plot the true redshift
idx = 18
z = test_set["redshift"].iloc[idx]
ax1.axvline(z, c="C3", ls="--", label="True redshift")

# plot the posteriors
ax1.plot(grid, pdfs0[idx], c="k", label="Posterior with all bands")
ax1.plot(grid, pdfs1[idx], c="C1", label="Posterior with $u$ marginalized")

# set the x label and limits, and remove the yticks
ax1.set(xlabel="redshift", xlim=(0, 3), yticks=[])

# put the legend in axis 2
handles, labels = ax1.get_legend_handles_labels()
ax2.legend(handles, labels, ncol=1, fontsize=9, handlelength=1.5, loc=10, frameon=False)
ax2.axis(False)  # hide the axes

fig.savefig(paths.figures / "posterior_marginalized.pdf")
