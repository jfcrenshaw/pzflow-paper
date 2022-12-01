"""Load the PZFlow catalog created by the forward model."""
import numpy as np
import pandas as pd
from showyourwork.paths import user as Paths


def load_pzflow_catalog(
    truth: bool = False,
    subset: str = None,
    train_frac: float = 0.8,
    finite: bool = True,
) -> pd.DataFrame:
    """Load the catalog created by the PZFlow forward model.

    Parameters
    ----------
    truth: bool, default=False
        Whether to return the truth catalog. If False, the observed catalog is
        returned. Note that the true redshifts are returned in both cases.
    subset: str, default=None
        If `train` or `test`, this function returns the corresponding sets. If
        None, the full set is returned.
    train_frac: float, default=0.8
        The fraction of the catalog to include in the train set.
        1 - train_frac is the fraction included in the test set.
    finite: bool, default=True
        Whether to only include galaxies whose properties are all finite.
    """
    # instantiate the paths
    paths = Paths()

    # load the training data
    data = pd.read_pickle(paths.data / "pzflow_catalog.pkl")

    # rename the columns
    col_names = {"true_redshift": "redshift"}
    if truth:
        col_names = col_names | {f"true_{band}": band for band in "ugrizy"}
    else:
        col_names = (
            col_names
            | {f"obs_{band}": band for band in "ugrizy"}
            | {f"obs_{band}_err": f"{band}_err" for band in "ugrizy"}
        )

    data = data.rename(columns=col_names)[list(col_names.values())]

    # prune the columns
    data = data[list(col_names.values())]

    # select the subset
    split_idx = int(train_frac * len(data))
    if subset == "train":
        data = data.iloc[:split_idx]
    elif subset == "test":
        data = data.iloc[split_idx:]

    if finite:
        # get the rows without NaNs
        data = data[np.isfinite(data).all(axis=1)]

    return data
