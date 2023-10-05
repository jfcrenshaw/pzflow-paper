"""Train conditional flow that models p(ellipticity, size | redshift, photometry)."""
from pathlib import Path

import numpy as np
import optax
from load_training_data import load_data, split_data
from pzflow import Flow
from pzflow.bijectors import Chain, RollingSplineCoupling, ShiftBounds
from showyourwork.paths import user as Paths

# instantiate the paths
paths = Paths()

# load data with noisy photometry
data = load_data(sizes=True)

# split training and validation sets
train_set, val_set = split_data(data)

# set up the bijector
# the first bijector is shift bounds
# we need to set the mins and maxes
mins = np.array([0.0, 0.0])
maxs = np.array([1.0, 1.0])

# now, the settings for the RQ-RSC
nlayers = 2  # layers = number of dimensions
n_conditions = 7  # number of conditional dimensions
K = 16  # number of spline knots
transformed_dim = 1  # only transform one dimension at a time

# chain all the bijectors together
bijector = Chain(
    ShiftBounds(mins, maxs),
    RollingSplineCoupling(
        nlayers, n_conditions=n_conditions, K=K, transformed_dim=transformed_dim
    ),
)

# build the flow
flow = Flow(
    ["ellipticity", "size"],
    conditional_columns=["redshift"] + list("ugrizy"),
    bijector=bijector,
)

# train for three rounds of 150 epochs
# after each round, decrease the learning rate by a factor of 10
opt = optax.adam(1e-5)
losses1 = flow.train(
    train_set,
    val_set,
    epochs=150,
    optimizer=opt,
    seed=0,
    verbose=True,
)
losses1 = np.array(losses1)

opt = optax.adam(1e-6)
losses2 = flow.train(
    train_set,
    val_set,
    epochs=150,
    optimizer=opt,
    seed=1,
    verbose=True,
)
losses2 = np.array(losses2)

opt = optax.adam(1e-7)
losses3 = flow.train(
    train_set,
    val_set,
    epochs=150,
    optimizer=opt,
    seed=2,
    verbose=True,
)
losses3 = np.array(losses3)

# stack all the losses together
losses = np.hstack((losses1, losses2, losses3))

# save some info with the model
flow.info = (
    "This is a conditional normalizing flow trained to model "
    "p(ellipticity, size | redshift, photometry) "
    "where ellipticity, size, and redshift are their true values "
    "from CosmoDC2 (arXiv:1907.06530). The photometry has had LSST Y10 "
    "photometric errors added using PhotErr."
)

# create the directory the outputs will be saved in
output_dir = paths.data / "conditional_galaxy_flow"
Path.mkdir(output_dir, exist_ok=True)

# save the flow
flow.save(output_dir / "flow.pzflow.pkl")

# save the losses
np.save(output_dir / "losses.npy", np.array(losses))
