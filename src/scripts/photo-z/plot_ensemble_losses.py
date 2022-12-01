"""Plot the training losses for the pz ensemble."""
import pickle

import matplotlib.pyplot as plt
from showyourwork.paths import user as Paths

# instantiate the paths
paths = Paths()

# load the losses
with open(paths.data / "pz_ensemble/losses.pkl", "rb") as file:
    losses = pickle.load(file)

# plot the losses
fig, ax = plt.subplots(figsize=(3, 3), constrained_layout=True)
for flow, loss in losses.items():
    ax.plot(loss, alpha=0.5, label=flow)
ax.legend()
ax.set(xlabel="Epochs", ylabel="Loss", ylim=(-1.8, -1))

# save the figure
fig.savefig(paths.figures / "ensemble_losses.pdf")
