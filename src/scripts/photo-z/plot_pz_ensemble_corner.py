"""Make the corner plot for the pz estimating ensemble."""
import corner
import matplotlib.pyplot as plt
import numpy as np
from load_pzflow_catalog import load_pzflow_catalog
from matplotlib.ticker import MaxNLocator
from pzflow import FlowEnsemble
from showyourwork.paths import user as Paths

# instantiate the paths
paths = Paths()

# load the flow ensemble
flowEns = FlowEnsemble(file=paths.data / "pz_ensemble" / "pz_ensemble.pzflow.pkl")

# load the test set
test_set = load_pzflow_catalog(subset="test").iloc[:10_000]

# sample from the ensemble
samples = flowEns.sample(test_set.shape[0], seed=0)

# create the corner plot
fig = plt.figure(figsize=(7.1, 7.1))

# some global corner settings
corner_settings = {
    "fig": fig,
    "bins": 20,
    "range": [
        (-0.1, 3),
        (20, 29.5),
        (20, 27.9),
        (19, 27.5),
        (18, 27),
        (18, 27),
        (18, 27),
    ],
    "hist_bin_factor": 1,
    "labels": test_set.columns,
}

# plot the test set in red
corner.corner(test_set.to_numpy(), color="C3", **corner_settings)

# plot the pz estimator samples in blue
corner.corner(
    samples.to_numpy(), color="C0", data_kwargs={"ms": 1.5}, **corner_settings
)

# set border ticks to integers
axes = np.array(fig.axes).reshape((7, 7))
for ax in axes[-1, :]:
    ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
for ax in axes[1:, 0]:
    ax.yaxis.set_major_locator(MaxNLocator(4, integer=True))

# remove interior ticks
for ax in axes[:-1, :].flatten():
    ax.set(xticks=[])
for ax in axes[:, 1:].flatten():
    ax.set(yticks=[])

# add a legend
axes[2, 5].plot([], c="C0", label="PZ Estimator Model")
axes[2, 5].plot([], c="C3", label="PZFlow Catalog")
axes[2, 5].legend()

# save the figure
fig.savefig(paths.figures / "pz_estimator_corner.pdf", dpi=600, bbox_inches="tight")
