"""Plot the training losses for the galaxy flows."""
import matplotlib.pyplot as plt
import numpy as np
import paths

# load the losses
main_losses = np.load(paths.data / "main_galaxy_flow" / "losses.npy")
conditional_losses = np.load(paths.data / "conditional_galaxy_flow" / "losses.npy")

# plot the losses
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3), constrained_layout=True)

main_epochs = np.arange(len(main_losses))
ax1.plot(main_epochs, main_losses)
ax1.set(xlabel="Epochs", ylabel="Loss", title="Main flow")

conditional_epochs = np.arange(len(conditional_losses))
ax2.plot(conditional_epochs[10:], conditional_losses[10:])
ax2.set(xlabel="Epochs", title="Conditional flow")

# save the figure
fig.savefig(paths.figures / "galaxy_flow_losses.pdf")
