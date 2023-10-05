"""Train the flow that models galaxy redshift and photometry."""
from typing import Tuple

import numpy as np
import pandas as pd
from photerr import LsstErrorModel
from showyourwork.paths import user as Paths

# instantiate the paths
paths = Paths()


def load_data(
        noisy: bool = True,
        finite: bool = True,
        errs: bool = False,
        sizes: bool = False,
) -> pd.DataFrame:
    """Return the training data.
  
    Parameters
    ----------
    noisy: bool, default=True
        Whether to return with photometric noise.
    finite: bool, default=True
        Whether to return only galaxies for which all values are finite.
    errs: bool, default=True
        Whether to return the photometric error columns.
    sizes: bool, default=False
        Whether to return the ellipticity, and size.

    Returns
    -------
    pd.DataFrame
        The data.
    """
    # load the datadata
    data = pd.read_pickle(paths.data / "cosmoDC2_subset.pkl")

    if noisy:
        # add minor sizes
        q = (1 - data["ellipticity"]) / (1 + data["ellipticity"])
        data["size_minor"] = q * data["size"]

        # add LSST Y10 photometric errors
        err_model = LsstErrorModel(
            sigLim=5,
            ndMode="sigLim",
            extendedType="auto",
            majorCol="size",
            minorCol="size_minor",
        )
        data = err_model(data, random_state=0)

    if finite:
        # remove galaxies with non-detections
        data = data[np.isfinite(data).all(axis=1)]

    # determine which columns to return
    columns = ["redshift"] + list("ugrizy")
    if errs:
        columns += [f"{band}_err" for band in "ugrizy"]
    if sizes:
        columns += ["size", "ellipticity"]
    data = data[columns]

    return data


def split_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the data into training and validation sets.
    
    Parameters
    ----------
    data: pd.DataFrame
        DataFrame to split

    Returns
    -------
    pd.DataFrame
        The training set.
    pd.DataFrame
        The validation set.
    """
    fval = 0.2  # fraction reserved for validation

    n_val = int(fval * len(data))

    train_set = data.loc[n_val:]
    val_set = data.loc[:n_val]

    return train_set, val_set