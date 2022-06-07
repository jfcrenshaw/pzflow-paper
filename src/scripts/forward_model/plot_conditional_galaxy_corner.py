"""Make the corner plot for the conditional galaxy flow."""
import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import paths
from matplotlib.ticker import MaxNLocator
from pzflow import Flow

# set the number of samples to plot
n_samples = 10_000

# load the test set
data = pd.read_pickle(paths.data / "cosmoDC2_subset.pkl")
data = data.drop(["ra", "dec"], axis=1)
test_set = data.loc[int(0.2 * len(data)) :]
test_set = test_set[:n_samples]

# draw samples from the saved flow
flow = Flow(file=paths.data / "conditional_galaxy_flow" / "flow.pzflow.pkl")
samples = flow.sample(1, conditions=test_set, seed=0)[test_set.columns]

# create the corner plot
fig = plt.figure(figsize=(7, 7))

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
        (-0.1, 1.07),
        (0, 1.07),
    ],
    "hist_bin_factor": 1,
    "labels": test_set.columns,
    "labelpad": 0.2,
}

# plot the test set in red
corner.corner(test_set.to_numpy(), color="C3", **corner_settings)

# plot the PZFlow samples in blue
corner.corner(
    samples.to_numpy(), color="C0", data_kwargs={"ms": 1.5}, **corner_settings
)

# pull out axes
axes = np.array(fig.axes).reshape((9, 9))

# hide the unnecessary panels
for ax in axes[:-2].flatten():
    ax.set_visible(False)

# set redshift and magnitude ticks to integers
for row in axes[-2:]:
    for j in range(7):
        row[j].xaxis.set_major_locator(MaxNLocator(4, integer=True))

# set ellipticity and size ticks
for ax in [axes[-2, 0], axes[-1, 0]]:
    ax.set(yticks=[0, 0.5, 1.0])
for ax in [axes[-1, -2], axes[-1, -1]]:
    ax.set(xticks=[0, 0.5, 1.0])

# set ylim on first row
for ax in axes[-2, :-2]:
    ax.set(ylim=(0, 1.03))

# add a legend
axes[-2, -1].plot([], c="C0", label="PZFlow")
axes[-2, -1].plot([], c="C3", label="DC2")
axes[-2, -1].legend(handlelength=1, fontsize=8, borderaxespad=0)

# save the figure
fig.savefig(
    paths.figures / "conditional_galaxy_corner.pdf", dpi=600, bbox_inches="tight"
)
