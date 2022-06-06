"""Train conditional flow that models p(ellipticity, size | redshift, photometry)."""
from pathlib import Path

import numpy as np
import optax
import pandas as pd
import paths
from pzflow import Flow
from pzflow.bijectors import Chain, RollingSplineCoupling, ShiftBounds

# load the training data
data = pd.read_pickle(paths.data / "cosmoDC2_subset.pkl")
train_set = data.loc[: int(0.8 * len(data))]

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
opt = optax.adam(1e-3)
losses = flow.train(train_set, epochs=150, optimizer=opt, seed=0)

opt = optax.adam(1e-4)
losses += flow.train(train_set, epochs=150, optimizer=opt, seed=1)

opt = optax.adam(1e-5)
losses += flow.train(train_set, epochs=150, optimizer=opt, seed=2)

# save some info with the model
flow.info = (
    "This is a conditional normalizing flow trained to model "
    "p(ellipticity, size | redshift, photometry) "
    "where all quantities are their true values from CosmoDC2 (arXiv:1907.06530). "
)

# create the directory the outputs will be saved in
output_dir = paths.data / "conditional_galaxy_flow"
Path.mkdir(output_dir, exist_ok=True)

# save the flow
flow.save(output_dir / "flow.pzflow.pkl")

# save the losses
np.save(output_dir / "losses.npy", np.array(losses))
