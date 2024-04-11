"""Make the corner plot for the conditional galaxy flow."""
import corner
import matplotlib.pyplot as plt
import numpy as np
from load_training_data import load_data, split_data
from matplotlib.ticker import MaxNLocator
from pzflow import Flow
from showyourwork.paths import user as Paths

# instantiate the paths
paths = Paths()

# set the number of samples to plot
n_samples = 10_000

# load data with noisy photometry
data = load_data(sizes=True)

# split training and validation sets
train_set, val_set = split_data(data)
val_set = val_set[:n_samples]

# draw samples from the saved flow
flow = Flow(file=paths.data / "conditional_galaxy_flow" / "flow.pzflow.pkl")
samples = flow.sample(1, conditions=val_set, seed=0)[val_set.columns]

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
        (0, 1.07),
        (0, 1.07),
    ],
    "hist_bin_factor": 1,
    "labels": val_set.columns,
    "labelpad": 0.2,
}

# plot the test set in red
corner.corner(val_set.to_numpy(), color="C3", **corner_settings)

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
for ax in axes[-2:, :-2].flatten():
    ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))

# set ellipticity and size ticks
for ax in [axes[-2, 0], axes[-1, 0]]:
    ax.set(yticks=[0, 0.5, 1.0])
for ax in [axes[-1, -2], axes[-1, -1]]:
    ax.set(xticks=[0, 0.5, 1.0])

# remove interior ticks
for ax in axes[-2, :]:
    ax.set(xticks=[])
for ax in axes[-2:, 1:].flatten():
    ax.set(yticks=[])

# set ylim
for ax in axes[-2, :-2]:
    ax.set(ylim=(0, 1.05))
for ax in axes[-1, :-1]:
    ax.set(ylim=(0, 1.05))

# set yticks
axes[-2, 0].set(yticks=[0, 0.4, 0.8])
axes[-1, 0].set(yticks=[0, 0.4, 0.8])

# add a legend
axes[-2, -1].plot([], c="C0", label="PZFlow")
axes[-2, -1].plot([], c="C3", label="DC2")
axes[-2, -1].legend(handlelength=1, fontsize=8, borderaxespad=0)

# save the figure
fig.savefig(
    paths.figures / "conditional_galaxy_corner.pdf", dpi=600, bbox_inches="tight"
)
