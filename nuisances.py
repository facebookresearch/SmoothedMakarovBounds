"""
Copyright (c) Meta Platforms, Inc. and affiliates.
"""

from typing import List, NamedTuple, Union

import matplotlib.pyplot as plt

import numpy as np

from models import (
    ConditionalCDFModel,
    ConditionalGaussianMixture,
    ConditionalZeroInflatedPoisson,
    MultiLabelClassifier,
)


## ECDF estimation --------------------------
# CDF estimation without ML
def empirical_cdf_ipw(
    A: np.ndarray, Y: np.ndarray, pi: float, y_grid: np.ndarray, a: int
):
    idx_a = np.where(A == a, 1, 0)

    # Reshape y_grid to (1, k) to allow broadcasting with Y (n, 1)
    y_grid = y_grid.reshape(1, -1)
    Y_col = Y.reshape(-1, 1)
    idx_y = (Y_col <= y_grid).astype(int)

    # Compute IPW estimator for A=1
    weights = idx_a / pi
    numerator = np.sum(weights * idx_y, axis=0)
    denominator = np.sum(weights)
    return np.expand_dims(numerator / denominator, axis=0)


def estimate_ecdfs(
    data_collection: dict,
    deltas: list,
) -> dict:
    test_preds = {}
    filter_conditions = data_collection["filter_dict"]["d_train"]
    for filter_key in filter_conditions.keys():
        d_train_filtered = {
            key: data_collection["d_train"][key][filter_conditions[filter_key]]
            for key in data_collection["d_train"].keys()
            if key != "x"
        }
        ecdf1 = empirical_cdf_ipw(
            d_train_filtered["a"],
            d_train_filtered["y"],
            0.5,
            data_collection["y_grids"][1],
            1,
        )
        ecdfs0 = {}
        for delta in deltas:
            ecdfs0[delta] = empirical_cdf_ipw(
                d_train_filtered["a"],
                d_train_filtered["y"],
                0.5,
                data_collection["y_grids"][0][delta],
                0,
            )
        test_preds[filter_key] = {1: ecdf1, 0: ecdfs0}

    return test_preds


# Nuisance model training --------------------------
def train_conditional_cdf(
    treatment: int,
    data_collection: dict,
    model_params: dict,
    outcome_type: str,
    plotting: bool = True,
) -> ConditionalCDFModel:
    # Split data into treatment and control ("T-learner style")
    idx_train = data_collection["d_train"]["a"][:, 0] == treatment
    d_train_nuisance = {
        "x": data_collection["d_train"]["x"][idx_train],
        "y": data_collection["d_train"]["y"][idx_train],
    }
    idx_val = data_collection["d_val"]["a"][:, 0] == treatment
    d_val_nuisance = {
        "x": data_collection["d_val"]["x"][idx_val],
        "y": data_collection["d_val"]["y"][idx_val],
    }

    # Choose model based on outcome type
    if outcome_type == "continuous":
        model = ConditionalGaussianMixture(lgb_params=model_params)
        loss_name = "avg_log_likelihood"
    elif outcome_type == "discrete":
        model_params["num_class"] = len(
            np.unique(
                np.concatenate(
                    [d_train_nuisance["y"], d_val_nuisance["y"]],
                    axis=0,
                )
            )
        )
        model = MultiLabelClassifier(lgb_params=model_params)
        loss_name = "multi_logloss"
    elif outcome_type == "poisson":
        model = ConditionalZeroInflatedPoisson(lgb_params=model_params)
        loss_name = "avg_log_likelihood"
    else:
        raise ValueError("outcome_type not defined")

    # Train model
    train_results = model.fit(d_train_nuisance, d_val_nuisance)

    return model


def get_nuisance_model_predictions(
    model: ConditionalCDFModel,
    data_collection: dict,
    treatment: int,
    delta: Union[float, int],
    train_preds: bool = False,
):
    if treatment == 1:
        y_grid = data_collection["y_grids"][1]
    else:
        y_grid = data_collection["y_grids"][0][delta]

    preds = {"d_test": model.predict_cdf(data_collection["d_test"]["x"], y_grid)}
    if train_preds:
        preds["d_train"] = model.predict_cdf(data_collection["d_train"]["x"], y_grid)
    return preds
