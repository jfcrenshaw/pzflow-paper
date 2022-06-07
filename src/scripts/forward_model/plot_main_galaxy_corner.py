"""Make the corner plot for the main galaxy flow."""
import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from pzflow import Flow
from showyourwork.paths import user as Paths

# instantiate the paths
paths = Paths()

# set the number of samples to plot
n_samples = 10_000

# load the test set
data = pd.read_pickle(paths.data / "cosmoDC2_subset.pkl")
test_set = data.loc[int(0.2 * len(data)) :, ["redshift"] + list("ugrizy")]
test_set = test_set[:n_samples]

# draw samples from the saved flow
flow = Flow(file=paths.data / "main_galaxy_flow" / "flow.pzflow.pkl")
samples = flow.sample(n_samples, seed=0)

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

# plot the PZFlow samples in blue
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
axes[2, 5].plot([], c="C0", label="PZFlow")
axes[2, 5].plot([], c="C3", label="CosmoDC2")
axes[2, 5].legend()

# save the figure
fig.savefig(paths.figures / "main_galaxy_corner.pdf", dpi=600, bbox_inches="tight")
