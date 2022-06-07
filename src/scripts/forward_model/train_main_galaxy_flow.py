"""Train the flow that models galaxy redshift and photometry."""
from pathlib import Path

import numpy as np
import optax
import pandas as pd
import paths
from pzflow import Flow
from pzflow.bijectors import Chain, ColorTransform, RollingSplineCoupling, ShiftBounds

# load the training data
data = pd.read_pickle(paths.data / "cosmoDC2_subset.pkl")
train_set = data.loc[: int(0.8 * len(data)), ["redshift"] + list("ugrizy")]

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
colors = -np.diff(train_set[list("ugrizy")].to_numpy())
mins = np.concatenate(([0, train_set["i"].min() - 0.1], colors.min(axis=0) - 0.1))
maxs = np.concatenate(([3, train_set["i"].max() + 0.1], colors.max(axis=0) - 0.1))

# I will add 10% buffers to the mins and maxs in case that the train set
# doesn't cover the full range of the test set
ranges = maxs - mins
buffer = ranges / 10 / 2
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

# build the flow
flow = Flow(train_set.columns, bijector=bijector)

# train for three rounds of 50 epochs
# after each round, decrease the learning rate by a factor of 10
opt = optax.adam(1e-3)
losses = flow.train(data, epochs=50, optimizer=opt, seed=0)

opt = optax.adam(1e-4)
losses += flow.train(data, epochs=50, optimizer=opt, seed=1)

opt = optax.adam(1e-5)
losses += flow.train(data, epochs=50, optimizer=opt, seed=2)

# save some info with the model
flow.info = (
    "This is a normalizing flow trained on true redshifts and photometry "
    "for 1 million galaxies from CosmoDC2 (arXiv:1907.06530)."
)

# create the directory the outputs will be saved in
output_dir = paths.data / "main_galaxy_flow"
Path.mkdir(output_dir, exist_ok=True)

# save the flow
flow.save(output_dir / "flow.pzflow.pkl")

# save the losses
np.save(output_dir / "losses.npy", np.array(losses))
