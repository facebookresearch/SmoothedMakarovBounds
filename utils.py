"""
Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import io
import os
import random
from typing import Dict, Optional

import numpy as np
import pandas as pd


def set_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)  # deterministic hashing of dicts / sets
    random.seed(seed)  # std-lib RNG
    np.random.seed(seed)


def create_data_collection(
    d_train: dict,
    d_val: dict,
    d_test: dict,
    deltas: Optional[list] = None,
    outcome_type: Optional[str] = None,
    y_grid_size: int = 100,
    filter_dict: Optional[dict] = None,
):
    # First get the support of the distribution using train and validation data
    y_data = d_train["y"]
    if d_val is not None:
        y_data = np.concatenate([y_data, d_val["y"]], 0)
    y_unique = np.unique(y_data)
    n_y_unique = len(y_unique)

    # Select default delta grid by heuristic if none is provided
    if deltas is None:
        if n_y_unique <= 5:
            deltas = list(
                np.sort(
                    np.concatenate(
                        [np.squeeze(y_unique), -np.squeeze(y_unique)], axis=0
                    )
                )
            )
        else:
            q = np.linspace(0.1, 0.9, 5)
            y1d = np.asarray(y_data).ravel()
            deltas_pos = np.quantile(y1d, q)
            deltas_neg = -np.quantile(y1d, q)
            deltas = list(np.sort(np.concatenate([deltas_pos, deltas_neg], axis=0)))
    if 0 not in deltas or 0.0 not in deltas:
        deltas = [0] + deltas
    # Select default outcome type by heuristic if none is provided
    if outcome_type is None:
        # Compute number of different y values
        if n_y_unique <= 14:
            outcome_type = "discrete"
        else:
            # Check if y is count data
            if np.all(np.mod(d_train["y"], 1) == 0) and n_y_unique <= 150:
                outcome_type = "poisson"
            else:
                outcome_type = "continuous"

    y_grid_dict = {}
    y_grid_dict[0] = {}

    if outcome_type == "continuous":
        y_support = list(
            np.linspace(
                np.min(y_data),
                np.max(y_data),
                y_grid_size,
            )
        )
    elif outcome_type in "discrete":
        y_support = list(np.sort(y_unique))
    elif outcome_type == "poisson":
        if y_grid_size < n_y_unique:
            y1d = np.asarray(y_data).ravel()
            # k equally spaced quantiles, inclusive

            q = np.linspace(0, 1, y_grid_size)

            # data values at those quantiles
            y_support = list(np.quantile(y1d, q))
        else:
            y_support = list(np.sort(y_unique))
    else:
        raise ValueError("outcome_type not defined")

    # Treatment cdf support
    # Add a small offset to the first element to include the leftmost point (where cdf is zero)
    offset = y_support[1] - y_support[0]
    y_grid_dict[1] = np.array([y_support[0] - offset] + y_support)

    if outcome_type in ["discrete", "poisson"]:
        # Small epsilon to take left limit
        eps = float(np.min(np.diff(y_grid_dict[1])) / 10)
        # Add small right shift points to the grid
        y_grid_dict[1] = np.sort(
            np.concatenate([y_grid_dict[1], 2 * eps + y_grid_dict[1]])
        )
        # For control arm shift with delta and small epsilon
        for delta in deltas:
            y_grid_dict[0][delta] = y_grid_dict[1] - delta - eps
    else:
        for delta in deltas:
            y_grid_dict[0][delta] = y_grid_dict[1] - delta

    if filter_dict is None:
        filter_dict = {
            "d_train": {"full": np.ones(len(d_train["y"]), dtype=bool)},
            "d_val": {"full": np.ones(len(d_val["y"]), dtype=bool)},
            "d_test": {"full": np.ones(len(d_test["y"]), dtype=bool)},
        }

    return {
        "d_train": d_train,
        "d_val": d_val,
        "d_test": d_test,
        "y_grids": y_grid_dict,
        "filter_dict": filter_dict,
        "outcome_type": outcome_type,
        "deltas": deltas,
    }
