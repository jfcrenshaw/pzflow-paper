"""Plot distribution of r-i vs redshift for CosmoDC2 and PZFlow."""
import matplotlib.pyplot as plt
import pandas as pd
from pzflow import Flow
from showyourwork.paths import user as Paths

# instantiate the paths
paths = Paths()

# load the cosmoDC2 data
data = pd.read_pickle(paths.data / "cosmoDC2_subset.pkl")

# load the flow
flow = Flow(file=paths.data / "main_galaxy_flow" / "flow.pzflow.pkl")

# pull out a subset of cosmoDC2 for plotting
data = data.iloc[:100_000]

# draw a same-sized sample from the flow
samples = flow.sample(data.shape[0], seed=0)

# create the figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.7), constrained_layout=True, dpi=120)

# set global plot settings
plot_settings = {
    "bins": 512,
    "cmap": "ocean_r",
    "rasterized": True,
}

ax1.hist2d(data["redshift"], data["r"] - data["i"], **plot_settings)
ax2.hist2d(samples["redshift"], samples["r"] - samples["i"], **plot_settings)

# set global axis settings
ax_settings = {
    "xlim": (0, 2.5),
    "ylim": (-0.25, 1.3),
    "xlabel": "redshift",
}

# set axis settings
ax1.set(ylabel="$r - i$", **ax_settings)
ax2.set(ylabel=" ", **ax_settings)


# set the labels
text_settings = {
    "x": 0.95,
    "y": 0.95,
    "va": "top",
    "ha": "right",
}
ax1.text(s="CosmoDC2", transform=ax1.transAxes, **text_settings)
ax2.text(s="PZFlow", transform=ax2.transAxes, **text_settings)

# save the figure!
fig.savefig(paths.figures / "smooth_color_distribution.pdf", dpi=300)
