"""Plot the training losses for the pz ensemble."""
# %%
import pickle
from pathlib import Path

import numpy as np
from showyourwork.paths import user as Paths

"""
This currently doesn't work because I accidentally saved the losses as jax floats.
I need to re-run training and save the losses as regular floats.
"""

# instantiate the paths
paths = Paths()

# %%
# load the losses
with open(paths.data / "pz_ensemble/losses.pkl", "rb") as file:
    losses = pickle.load(file)

# %%
