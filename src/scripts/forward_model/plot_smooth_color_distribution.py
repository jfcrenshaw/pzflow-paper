"""Plot distribution of r-i vs redshift for CosmoDC2 and PZFlow."""
import matplotlib.pyplot as plt
import pandas as pd
from load_training_data import load_data, split_data
from pzflow import Flow
from showyourwork.paths import user as Paths

# instantiate the paths
paths = Paths()

# load the cosmoDC2 data
data = pd.read_pickle(paths.data / "cosmoDC2_subset.pkl")

# load the flow
flow = Flow(file=paths.data / "main_galaxy_flow" / "flow.pzflow.pkl")

# load data without noise
truth = load_data(noisy=False)
_, truth = split_data(truth)

# load data with noisy photometry
data = load_data()

# split training and validation sets
train_set, val_set = split_data(data)

# draw a same-sized sample from the flow
samples = flow.sample(val_set.shape[0], seed=0)

# create the figure
fig, (ax1, ax2, ax3) = plt.subplots(
    1, 3, figsize=(7, 2.5), constrained_layout=True, dpi=120
)

# set global plot settings
plot_settings = {
    "bins": 512,
    "cmap": "ocean_r",
    "rasterized": True,
}

ax1.hist2d(truth["redshift"], truth["r"] - truth["i"], **plot_settings)
ax2.hist2d(val_set["redshift"], val_set["r"] - val_set["i"], **plot_settings)
ax3.hist2d(samples["redshift"], samples["r"] - samples["i"], **plot_settings)

# set global axis settings
ax_settings = {
    "xlim": (0, 2.5),
    "ylim": (-0.25, 1.3),
    "xlabel": "redshift",
}

# set axis settings
ax1.set(ylabel="$r - i$", **ax_settings)
ax2.set(ylabel=" ", yticklabels=[], **ax_settings)
ax3.set(ylabel=" ", yticklabels=[], **ax_settings)


# set the labels
text_settings = {
    "x": 0.95,
    "y": 0.95,
    "va": "top",
    "ha": "right",
}
ax1.text(s="CosmoDC2", transform=ax1.transAxes, **text_settings)
ax2.text(s="CosmoDC2\nw/ noise", transform=ax2.transAxes, **text_settings)
ax3.text(s="PZFlow", transform=ax3.transAxes, **text_settings)

# save the figure!
fig.savefig(paths.figures / "smooth_color_distribution.pdf", dpi=300)
