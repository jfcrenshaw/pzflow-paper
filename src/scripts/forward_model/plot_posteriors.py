import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pzflow import Flow
from showyourwork.paths import user as Paths

# instantiate the paths
paths = Paths()

# load the PZFlow generated catalog
pzflow_catalog = pd.read_pickle(paths.data / "pzflow_catalog.pkl")

# split into truth and observed catalogs
truth_catalog = pzflow_catalog.filter(like="true_")
truth_catalog.columns = truth_catalog.columns.str.removeprefix("true_")

obs_catalog = pzflow_catalog.filter(like="obs_")
obs_catalog.columns = obs_catalog.columns.str.removeprefix("obs_")

no_u_catalog = obs_catalog.copy()
no_u_catalog["u"] = np.nan
no_u_catalog = no_u_catalog.drop("u_err", axis=1)

# for nice visualization, let's pull out number 77
idx = [77]
truth_catalog = truth_catalog.iloc[idx]
obs_catalog = obs_catalog.iloc[idx]
no_u_catalog = no_u_catalog.iloc[idx]

# and let's decrease the u error by factor of 10
obs_catalog["u_err"] /= 10

# load the flow for posterior estimation
flow = Flow(file=paths.data / "main_galaxy_flow" / "flow.pzflow.pkl")

# set the redshift grid
grid = np.linspace(0.1, 0.6, 100)

# calculate posteriors using the true magnitudes
pdfs_true_mags = flow.posterior(truth_catalog, column="redshift", grid=grid)

# calculate posteriors using the observed magnitudes
pdfs_obs_mags = flow.posterior(obs_catalog, column="redshift", grid=grid)

# calculate posteriors using error convolution
pdfs_conv = flow.posterior(
    obs_catalog, column="redshift", grid=grid, err_samples=10_000, seed=0
)

# calculate posteriors with u band marginalization
marg_rules = {
    "flag": np.nan,
    "u": lambda row: np.linspace(24, 28, 20),
}
pdfs_no_u = flow.posterior(
    no_u_catalog,
    column="redshift",
    grid=grid,
    err_samples=10_000,
    seed=0,
    marg_rules=marg_rules,
)

# now create the figure
# the second row is for putting the legend below the plot
fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(3, 4),
    dpi=120,
    constrained_layout=True,
    gridspec_kw={
        "height_ratios": [4, 1],
    },
)

# pull out the true redshift and plot it
z = truth_catalog["redshift"].iloc[0]
ax1.axvline(z, c="C3", ls="--", label="True redshift")

# plot the true posterior
ax1.plot(grid, pdfs_true_mags[0], c="k", label="True posterior")

# place holder to get label spacing in the legend correct
# basically I want "Not convolved" in the second column of the legend
ax1.plot([], label=" ", c="w")

# plot the posterior without errors convolved
ax1.plot(grid, pdfs_obs_mags[0], label="Not convolved")

# plot the posterior with the errors convolved
ax1.plot(grid, pdfs_conv[0], label="Convolved")

# plot the posterior with errors convolved and u band marginalized
ax1.plot(grid, pdfs_no_u[0], label="$u$ marginalized")

# set the x label and limits, and remove the yticks
ax1.set(xlabel="redshift", xlim=(0.1, 0.6), yticks=[])

# put the legend in axis 2
h, l = ax1.get_legend_handles_labels()
ax2.legend(h, l, ncol=2, fontsize=9, handlelength=1.5, loc=10, frameon=False)
ax2.axis(False)  # hide the axes

fig.savefig(paths.figures / "posteriors.pdf")
