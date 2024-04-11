"""Plot the true, convolved, etc posteriors for a galaxy."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pzflow import Flow
from showyourwork.paths import user as Paths

# instantiate the paths
paths = Paths()

# load the PZFlow generated catalog
catalog = pd.read_pickle(paths.data / "pzflow_catalog.pkl")
catalog = catalog[:100]

# create a catalog with no u band
no_u_catalog = catalog.copy()
no_u_catalog["u"] = np.nan
no_u_catalog = no_u_catalog.drop("u_err", axis=1)

# select a single galaxy
idx = 6
catalog = catalog.iloc[[idx]]
no_u_catalog = no_u_catalog.iloc[[idx]]

# load the flow for posterior estimation
flow = Flow(file=paths.data / "main_galaxy_flow" / "flow.pzflow.pkl")

# set the redshift grid
grid = np.linspace(0, 3, 100)

# calculate posteriors using the observed magnitudes
pdfs = flow.posterior(catalog, column="redshift", grid=grid)

# calculate posteriors with u band marginalization
marg_rules = {
    "flag": np.nan,
    "u": lambda row: np.linspace(20, 29, 100),
}
pdfs_no_u = flow.posterior(
    no_u_catalog,
    column="redshift",
    grid=grid,
    seed=0,
    marg_rules=marg_rules,
)

# now create the figure
fig, ax = plt.subplots(
    figsize=(3, 3.5),
    dpi=120,
    constrained_layout=True,
)

# plot posteriors and true redshift
ax.axvline(catalog["redshift"].iloc[0], c="C3", ls="--", label="True redshift")
ax.plot(grid, pdfs[0], c="k", label="True posterior")
ax.plot(grid, pdfs_no_u[0], label="$u$ marginalized")
ax.legend(loc="upper left", fontsize=9, handlelength=1.5)

ax.set(xlabel="redshift", yticks=[], xlim=(0, 1))

fig.savefig(paths.figures / "posteriors.pdf")
