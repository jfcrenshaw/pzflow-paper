"""Plot the binned metrics for photo-z point estimates."""
# %%
import matplotlib.pyplot as plt
import numpy as np
from load_pzflow_catalog import load_pzflow_catalog
from showyourwork.paths import user as Paths

# instantiate the paths
paths = Paths()

# load the estimated posteriors
data = np.load(paths.data / "redshift_posteriors.npz")

grid = data["grid"]
pdfs = data["pdfs"]

# get point estimates
z_phot = grid[pdfs.argmax(axis=1)]

# load the true redshifts
test_set = load_pzflow_catalog(subset="test")
z_spec = test_set.redshift[: len(z_phot)]

# calculate the weighted error values
ez = (z_phot - z_spec) / (1 + z_spec)

# %%
# sort the galaxies by their spectroscopic bins
dz = 0.25
bins = np.arange(0, 3.0 + dz, dz)
bin_idx = np.digitize(z_spec, bins)

# iterate over bins and calculate the metrics
bin_z = []
bias = []
sigma_iqr = []
mad = []
outlier_frac = []
for i in range(len(bins)):
    # calculate the mean redshift of the bin
    bin_zs = z_spec[bin_idx == i]

    # if this bin is empty, move on to next bin
    if len(bin_zs) == 0:
        continue

    # get the weighted errors for the bin
    bin_ez = ez[bin_idx == i]
    bin_z.append(bin_zs.mean())

    # calculate the bias
    bias.append(np.median(bin_ez))

    # calculate the sigma_iqr
    q75, q25 = np.percentile(bin_ez, [75.0, 25.0])
    iqr = q75 - q25
    sigma_iqr.append(iqr / 1.349)

    # calculate the outlier fraction
    cut = max(0.06, 3 * sigma_iqr[-1])
    outlier_frac.append(np.mean(bin_ez > cut))

# %%
fig, (ax1, ax2, ax3) = plt.subplots(
    1, 3, figsize=(7, 2), constrained_layout=True, dpi=200
)

ax1.set(xlabel="redshift", ylabel="bias", xlim=(bins.min(), bins.max()))
ax1.plot(bin_z, bias)
ax1.plot(bins, +0.003 * np.ones(len(bins)), c="k", ls="--", alpha=0.75)
ax1.plot(bins, -0.003 * np.ones(len(bins)), c="k", ls="--", alpha=0.75)

ax2.set(
    xlabel="redshift", ylabel="$\sigma_{\mathrm{IQR}}$", xlim=(bins.min(), bins.max())
)
ax2.plot(bin_z, sigma_iqr)
ax2.plot(bins, 0.02 * np.ones(len(bins)), c="k", ls="--", alpha=0.75)

ax3.set(xlabel="redshift", ylabel="outlier fraction", xlim=(bins.min(), bins.max()))
ax3.plot(bin_z, outlier_frac)
ax3.plot(bins, 0.10 * np.ones(len(bins)), c="k", ls="--", alpha=0.75)

fig.savefig(paths.figures / "binned_metrics.pdf")
