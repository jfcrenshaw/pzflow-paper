"""Train the flow ensemble that estimates photometric redshifts."""
import pickle
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import optax
from jax import random
from load_pzflow_catalog import load_pzflow_catalog
from pzflow import FlowEnsemble
from pzflow.bijectors import Chain, ColorTransform, RollingSplineCoupling, ShiftBounds
from showyourwork.paths import user as Paths

# instantiate the paths
paths = Paths()

# load the training set of galaxies from the PZFlow catalog
train_set = load_pzflow_catalog(subset="train")

# set up the bijector
# we will use the same one used for the forward model

# the first bijector is the color transform
# we need to tell it which column to use as the reference magnitude
ref_idx = train_set.columns.get_loc("i")
# and which columns correspond to the magnitudes we want colors for
mag_idx = [train_set.columns.get_loc(band) for band in "ugrizy"]

# the next bijector is shift bounds
# we need to set the mins and maxes
# I am setting strict limits on redshift, but am adding some padding to
# the magnitudes and colors so that the flow can sample a little
colors = -np.diff(train_set[list("ugrizy")].to_numpy())
mins = np.concatenate(([0, train_set["i"].min() - 0.1], colors.min(axis=0)))
maxs = np.concatenate(([4, train_set["i"].max() + 0.1], colors.max(axis=0)))

# I will add buffers to the mins and maxs in case that the train set
# doesn't cover the full range of the test set
ranges = maxs - mins
buffer = ranges
buffer[0] = 0  # except no buffer for redshift!
mins -= buffer
maxs += buffer

# finally, the settings for the RQ-RSC
nlayers = train_set.shape[1]  # layers = number of dimensions
K = 16  # number of spline knots
transformed_dim = 1  # only transform one dimension at a time

# chain all the bijectors together
bijector = Chain(
    ColorTransform(ref_idx, mag_idx),
    ShiftBounds(mins, maxs),
    RollingSplineCoupling(nlayers, K=K, transformed_dim=transformed_dim),
)


# we will also build the photometric error model which is Gaussian in flux space
def photometric_error_model(
    key: jnp.ndarray,
    X: jnp.ndarray,
    Xerr: jnp.ndarray,
    nsamples: int,
) -> jnp.ndarray:
    """Sample from the photometric error distribution.

    Parameters
    ----------
    key : jnp.ndarray
        A jax.random PRNGKey.
    X : jnp.ndarray
        An array of photometric means. Shape (n_galaxies x n_bands)
    Xerr : jnp.ndarray
        An array of photometric errors, corresponding to the entries in X.
    nsamples : int
        The number of samples to draw.

    Returns
    -------
    jnp.ndarray
        The array of samples, with shape (n_galaxies x nsamples x n_bands).
    """
    # calculate fluxes
    F = 10 ** (X / -2.5)
    # calculate flux errors
    dF = jnp.log(10) / 2.5 * F * Xerr

    # add Gaussian errors
    eps = random.normal(key, shape=(F.shape[0], nsamples, F.shape[1]))
    F = F[:, None, :] + eps * dF[:, None, :]

    # add a flux floor to avoid infinite magnitudes
    # this flux corresponds to a max magnitude of 30
    F = jnp.clip(F, 1e-12, None)

    # calculate magnitudes
    M = -2.5 * jnp.log10(F)

    return M


# build the flow ensemble
flowEns = FlowEnsemble(
    ["redshift"] + list("ugrizy"),
    bijector,
    data_error_model=photometric_error_model,
    N=4,
)

# train for three rounds of 50 epochs
# after each round, decrease the learning rate by a factor of 10

# we will also convolve the photometric errors during training

opt = optax.adam(1e-6)
losses1 = flowEns.train(
    train_set, convolve_errs=True, epochs=50, optimizer=opt, seed=0, verbose=True
)

opt = optax.adam(1e-7)
losses2 = flowEns.train(
    train_set, convolve_errs=True, epochs=50, optimizer=opt, seed=1, verbose=True
)

opt = optax.adam(1e-8)
losses3 = flowEns.train(
    train_set, convolve_errs=True, epochs=50, optimizer=opt, seed=2, verbose=True
)

losses = {
    flow: [
        loss for loss_dict in [losses1, losses2, losses3] for loss in loss_dict[flow]
    ]
    for flow in losses1
}

# save some info with the model
flowEns.info = (
    "This is an ensemble of normalizing flows trained on true redshifts and noisy "
    "photometry simulated by another normalized flow that was trained on CosmoDC2 "
    "(arXiv:1907.06530)."
)

# create the directory the outputs will be saved in
output_dir = paths.data / "pz_ensemble"
Path.mkdir(output_dir, exist_ok=True)

# save the ensemble
flowEns.save(output_dir / "pz_ensemble.pzflow.pkl")

# save the losses
with open(output_dir / "losses.pkl", "wb") as file:
    pickle.dump(losses, file)
