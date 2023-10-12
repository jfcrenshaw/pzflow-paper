"""Train the flow that models galaxy redshift and photometry."""
from pathlib import Path

import numpy as np
import optax
from load_training_data import load_data, split_data
from pzflow import Flow
from pzflow.bijectors import Chain, ColorTransform, RollingSplineCoupling, ShiftBounds
from pzflow.distributions import CentBeta13
from showyourwork.paths import user as Paths

# instantiate the paths
paths = Paths()

# load data with noisy photometry
data = load_data()

# split training and validation sets
train_set, val_set = split_data(data)

# set up the bijector
# the first bijector is the color transform
# we need to tell it which column to use as the reference magnitude
ref_idx = train_set.columns.get_loc("i")
# and which columns correspond to the magnitudes we want colors for
mag_idx = [train_set.columns.get_loc(band) for band in "ugrizy"]

# the next bijector is shift bounds
# we need to set the mins and maxes
# I am setting strict limits on redshift, but am adding some padding to
# the magnitudes and colors so that the flow can sample a little
colors = -np.diff(data[list("ugrizy")].to_numpy())
mins = np.concatenate(([0, data["i"].min() - 0.25], colors.min(axis=0) - 0.5))
maxs = np.concatenate(([3, data["i"].max() + 0.25], colors.max(axis=0) + 0.5))

# finally, the settings for the RQ-RSC
ndim = train_set.shape[1]  # layers = number of dimensions
K = 8  # number of spline knots
transformed_dim = 1  # only transform one dimension at a time

# chain all the bijectors together
bijector = Chain(
    ColorTransform(ref_idx, mag_idx),
    ShiftBounds(mins, maxs),
    RollingSplineCoupling(nlayers=ndim, K=K, transformed_dim=transformed_dim),
)

# build the flow
flow = Flow(train_set.columns, bijector=bijector, latent=CentBeta13(ndim))

# train for three rounds of 50 epochs
# after each round, decrease the learning rate by a factor of 10
opt = optax.adam(1e-5)
losses1 = flow.train(
    train_set,
    val_set,
    epochs=100,
    optimizer=opt,
    seed=0,
    verbose=True,
)
losses1 = np.array(losses1)

opt = optax.adam(1e-6)
losses2 = flow.train(
    train_set,
    val_set,
    epochs=50,
    optimizer=opt,
    seed=1,
    verbose=True,
)
losses2 = np.array(losses2)

opt = optax.adam(1e-7)
losses3 = flow.train(
    train_set,
    val_set,
    epochs=50,
    optimizer=opt,
    seed=2,
    verbose=True,
)
losses3 = np.array(losses3)

# stack all the losses together
losses = np.hstack((losses1, losses2, losses3))

# save some info with the model
flow.info = (
    "This is a normalizing flow trained on true redshifts and noisy photometry "
    "for 1 million galaxies from CosmoDC2 (arXiv:1907.06530). LSST Y10 "
    "photometric errors were added using PhotErr, with sigLim=5."
)

# create the directory the outputs will be saved in
output_dir = paths.data / "main_galaxy_flow"
Path.mkdir(output_dir, exist_ok=True)

# save the flow
flow.save(output_dir / "flow.pzflow.pkl")

# save the losses
np.save(output_dir / "losses.npy", losses)
