"""Create the two moons example figure."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from pzflow import Flow
from pzflow.examples import get_twomoons_data
from showyourwork.paths import user as Paths

# instantiate the paths
paths = Paths()

# load the two moons example data
data = get_twomoons_data()

# load the trained flow
flow = Flow(file=paths.data / "two_moons_flow.pzflow.pkl")

# calculate the latent variables for a subset of the data
samples = data[:5_000].to_numpy()
latents = flow._forward(
    flow._params[1], samples, conditions=np.zeros((samples.shape[0], 1))
)[0]


# set the colors of the samples
def thread(x: np.ndarray) -> np.ndarray:
    """Calculate function that threads between the two moons."""
    return 0.5 * np.sin(-3 * (x - 0.5)) + 0.25


colors = []
for row in samples:
    # upper moon
    if thread(row[0]) < row[1]:
        # left half
        if row[0] < 0:
            colors.append("C0")
        # right half
        else:
            colors.append("C1")

    # lower moon
    else:
        # left half
        if row[0] > 1:
            colors.append("C2")
        # right half
        else:
            colors.append("C3")


# create the figure
fig = plt.figure(figsize=(6, 2.6), constrained_layout=True)
gs = GridSpec(1, 5, figure=fig)

# make the scatter plots
scatter_settings = {
    "c": colors,
    "marker": ".",
    "s": 3,
    "rasterized": True,
}

# scatter plot of data (left panel)
ax1 = fig.add_subplot(gs[:2])
ax1.scatter(samples[:, 0], samples[:, 1], **scatter_settings)
ax1.set(
    title="Data Space",
    xlim=(-1.5, 2.5),
    ylim=(-1.5, 2),
    xticks=[],
    yticks=[],
    aspect=4 / 3.5,
)

# scatter plot of latents (right panel)
ax2 = fig.add_subplot(gs[3:])
ax2.scatter(latents[:, 0], latents[:, 1], **scatter_settings)
ax2.set(
    title="Latent Space",
    xlim=(-5.5, 5.5),
    ylim=(-5.5, 5.5),
    xticks=[],
    yticks=[],
    aspect=1,
)

# add the text in between
center = fig.add_subplot(gs[2])
center.axis("off")  # hide the axes

# right arrow and text
center.annotate(
    "",
    xy=(0.9, 0.6),
    xytext=(0.1, 0.6),
    xycoords="axes fraction",
    arrowprops={"arrowstyle": "->"},
)
center.text(
    0.5,
    0.6,
    "$x \sim p_X$\n$u = f(x)$\n",
    transform=center.transAxes,
    ha="center",
    va="bottom",
)

# left arrow and text
center.annotate(
    "",
    xy=(0.9, 0.4),
    xytext=(0.1, 0.4),
    xycoords="axes fraction",
    arrowprops={"arrowstyle": "<-"},
)
center.text(
    0.5,
    0.4,
    "\n$u \sim p_U$\n$x = f^{\,-1}(u)$",
    transform=center.transAxes,
    ha="center",
    va="top",
)

# save the figure!
fig.savefig(paths.figures / "twomoons_example.pdf", dpi=600)
