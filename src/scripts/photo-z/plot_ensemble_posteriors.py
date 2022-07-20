"""Plot posterior ensembles for a few galaxies."""
import matplotlib.pyplot as plt
import numpy as np
from load_pzflow_catalog import load_pzflow_catalog
from pzflow import FlowEnsemble
from showyourwork.paths import user as Paths

# instantiate the paths
paths = Paths()

# load the flow ensemble
flowEns = FlowEnsemble(file=paths.data / "pz_ensemble" / "pz_ensemble.pzflow.pkl")

# load the test set
test_set = load_pzflow_catalog(subset="test")

# select galaxies for plotting
idx = [21, 54, 99]
test_set = test_set.iloc[idx]

# calculate posteriors
grid = np.linspace(0, 3.5, 351)
pdf_ens = flowEns.posterior(
    test_set, "redshift", grid, returnEnsemble=True, err_samples=100, seed=0
)
pdf_mean = flowEns.posterior(
    test_set, "redshift", grid, returnEnsemble=False, err_samples=100, seed=0
)

# create the figure
fig, axes = plt.subplots(1, 3, figsize=(7, 2.3), constrained_layout=True, dpi=120)

# loop through the panels
for i, ax in enumerate(axes):
    # plot the ensemble of pdfs
    for j, pdf in enumerate(pdf_ens[i]):
        ax.plot(grid, pdf, label=f"Flow {j + 1}")

    # plot the ensemble mean
    ax.plot(grid, pdf_mean[i], c="k", ls="--", label="Mean")

    # plot the true redshift
    ax.axvline(test_set.iloc[i]["redshift"], c="gray", label="Truth")

    # set labels and limits
    ax.set(yticks=[], xlabel="redshift", xlim=(grid.min(), grid.max()))

# put a legend in the center
axes[1].legend(loc="upper left")

fig.savefig(paths.figures / "ensemble_posteriors.pdf")
