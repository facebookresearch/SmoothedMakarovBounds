"""
Copyright (c) Meta Platforms, Inc. and affiliates.
"""

# The following default experiments
FNA_METHODS = ["envelope_marginal", "smoothing_marginal"]
FNA_LGB_PARAMS = {
    "boosting_type": "gbdt",
    "min_gain_to_split": 0,
    "num_leaves": 31,
    "max_depth": -1,
    "min_data_in_leaf": 25,
    "learning_rate": 0.05,
    "verbose": -1,
    "num_boost_round": 2000,
    "min_child_weight": 1,
    "early_stopping_nb": 100,
    "feature_frac": 0.8,
    "bagging_frac": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0,
    "lambda_l2": 0.2,
    "n_components": 2,
    "y_transform": "standard",
}
FNA_T1 = [1, 10, 100, 200, 500]
FNA_T2 = [100, 3000]
FNA_MAX_Y_GRID_SIZE = 200
FNA_MAX_TRAINING_ROWS = 30000000
