"""
Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import itertools
from typing import List, Optional, Union

import numpy as np
from scipy import stats
from scipy.special import expit, logsumexp


# Normal confidence intervals
def compute_normal_ci(n: int, estimator: float, std: float, alpha=0.05):
    z = stats.norm.ppf(1 - alpha / 2)
    return estimator - (z * std / np.sqrt(n)), estimator + (z * std / np.sqrt(n))


def compute_makarov_bounds_envelope_scores(
    bound_type: str,
    y_grid1: np.ndarray,
    y_grid0: np.ndarray,
    cdf1: np.ndarray,
    cdf0: np.ndarray,
    A: np.ndarray,
    Y: np.ndarray,
    pi: float = 0.5,
    method: str = "dr",
):
    if bound_type not in ["lower", "upper"]:
        raise ValueError("type must be either 'lower' or 'upper'")
    # Compute envelopes
    cdf_diff = cdf1 - cdf0
    if bound_type == "lower":
        y_star_idx = np.argmax(cdf_diff, 1)
    else:
        y_star_idx = np.argmin(cdf_diff, 1)
    y_star1 = y_grid1[y_star_idx, None]
    y_star0 = y_grid0[y_star_idx, None]
    # Plug argmins/max into cdfs
    cdf1_star = np.expand_dims(cdf1[np.arange(cdf1.shape[0]), y_star_idx.flatten()], 1)
    cdf0_star = np.expand_dims(cdf0[np.arange(cdf0.shape[0]), y_star_idx.flatten()], 1)
    # Indicator usign argmins/ argmaxs
    ind_y_star1 = (Y <= y_star1).astype(int)
    ind_y_star0 = (Y <= y_star0).astype(int)
    if bound_type == "lower":
        idx_trunc = (cdf1_star - cdf0_star > 0).astype(int)
    else:
        idx_trunc = (cdf1_star - cdf0_star < 0).astype(int)
    if method == "dm":
        envelopes = idx_trunc * (cdf1_star - cdf0_star)
    elif method == "ipw":
        envelopes = idx_trunc * (
            ((A / pi) * ind_y_star1) - (((1 - A) / (1 - pi)) * ind_y_star0)
        )
    elif method == "dr":
        envelopes = idx_trunc * (
            ((A / pi) * (ind_y_star1 - cdf1_star))
            - (((1 - A) / (1 - pi)) * (ind_y_star0 - cdf0_star))
            + cdf1_star
            - cdf0_star
        )
    else:
        raise ValueError("method must be either 'dm', 'ipw', or 'dr'")
    return (
        envelopes,
        np.mean(y_star1),
        np.mean(y_star0),
    )


def compute_makarov_bounds_envelope_from_scores(
    scores: np.ndarray,
    bound_type: str,
    n: int,
    alpha: float = 0.05,
    y_star1: Optional[np.ndarray] = None,
    y_star0: Optional[np.ndarray] = None,
):
    # Point estimators
    if bound_type == "lower":
        point_est = np.mean(scores, 0)
    else:
        point_est = 1 + np.mean(scores, 0)
    std = np.std(scores, 0, ddof=1)
    # Compute normal CI
    z = stats.norm.ppf(1 - alpha / 2)
    if bound_type == "lower":
        ci = point_est - (z * std / np.sqrt(n))
    else:
        ci = point_est + (z * std / np.sqrt(n))
    results_dict = {
        "point_est": np.clip(point_est, 0, 1),
        "ci": np.clip(ci, 0, 1),
        "point_est_var": std**2,
    }
    if y_star1 is not None:
        results_dict["y_star1"] = y_star1
    if y_star0 is not None:
        results_dict["y_star0"] = y_star0
    return results_dict


# Baseline method: Makarov bound envelope estimator
def compute_envelope_estimator(
    bound_type: str,
    y_grid1: np.ndarray,
    y_grid0: np.ndarray,
    cdf1: np.ndarray,
    cdf0: np.ndarray,
    A: np.ndarray,
    Y: np.ndarray,
    pi: float = 0.5,
    alpha: float = 0.05,
    method: str = "dr",
):
    scores, y_star1, y_star0 = compute_makarov_bounds_envelope_scores(
        bound_type,
        y_grid1,
        y_grid0,
        cdf1,
        cdf0,
        A=A,
        Y=Y,
        pi=0.5,
        method=method,
    )
    return compute_makarov_bounds_envelope_from_scores(
        scores,
        bound_type,
        A.shape[0],
        alpha=alpha,
        y_star1=y_star1,
        y_star0=y_star0,
    )


def compute_makarov_bounds_approx_scores(
    bound_type: str,
    y_grid1: np.ndarray,
    y_grid0: np.ndarray,
    t1: int,
    t2: int,
    cdf1: np.ndarray,
    cdf0: np.ndarray,
    A: np.ndarray,
    Y: np.ndarray,
    pi: float = 0.5,
    outcome_type: str = "continuous",
):
    # Numerical integration
    def _trapz_weights(x):
        w = np.empty_like(x, dtype=float)
        w[1:-1] = 0.5 * (x[2:] - x[:-2])
        w[0] = 0.5 * (x[1] - x[0])
        w[-1] = 0.5 * (x[-1] - x[-2])
        return w

    def _left_rectangle_weights(x: np.ndarray) -> np.ndarray:
        """
        Left-rectangle weights on a sorted grid x:
        w[j<G-1] = x[j+1] - x[j]
        w[-1]    = 0
        """
        w = np.empty_like(x, dtype=float)
        w[:-1] = np.diff(x)
        w[-1] = 0.0
        return w

    if outcome_type in ["continuous"]:
        w = _trapz_weights(y_grid1)  # (G,)
    elif outcome_type in ["discrete", "poisson"]:
        w = _left_rectangle_weights(y_grid1)  # (G,)
    else:
        raise ValueError("outcome_type not defined")
    log_w = np.log(w + 1e-300)  # avoid -inf

    if bound_type not in ["lower", "upper"]:
        raise ValueError("type must be either 'lower' or 'upper'")
    # Check if t1, t2 are integers and convert to np arrays
    if not isinstance(t1, int):
        # Reshape to (n, n_y_grid, n_t_grid) for hyperparameter tuning
        t1 = np.array(t1).reshape(1, 1, -1)
        t2 = np.array(t2).reshape(1, 1, -1)
        y_grid1 = y_grid1.reshape(1, -1, 1)
        y_grid0 = y_grid0.reshape(1, -1, 1)
        cdf1 = np.expand_dims(cdf1, 2)
        cdf0 = np.expand_dims(cdf0, 2)
        A = np.expand_dims(A, 2)
        Y = np.expand_dims(Y, 2)
        log_w = np.expand_dims(np.expand_dims(log_w, 0), 2)

    t1_neg = t1
    if bound_type == "upper":
        t1_neg = -t1

    lse = logsumexp((t1_neg * (cdf1 - cdf0)) + log_w, axis=1, keepdims=True)
    # Computations in log space for numerical stability

    plugin = (
        1
        / t2
        * np.logaddexp(
            0,
            (t2 / t1) * lse,
        )
    )
    if bound_type == "upper":
        plugin = 1 - plugin

    bias_correction = expit((t2 / t1) * lse) * (
        ((A - pi) / (pi * (1 - pi)))
        * (
            A
            * np.expand_dims(
                np.sum(
                    np.exp((t1_neg * (cdf1 - cdf0)) - lse + log_w)
                    * ((Y <= y_grid1).astype(int) - cdf1),
                    axis=1,
                ),
                1,
            )
            + (1 - A)
            * np.expand_dims(
                np.sum(
                    np.exp((t1_neg * (cdf1 - cdf0)) - lse + log_w)
                    * ((Y <= y_grid0).astype(int) - cdf0),
                    axis=1,
                ),
                1,
            )
        )
    )

    return plugin + bias_correction


# Our method: Makarov bound approximation estimator
def compute_makarov_bounds_approx_from_scores(
    scores: np.ndarray,
    bound_type: str,
    y_grid: np.ndarray,
    t1: int,
    t2: int,
    n: int,
    alpha: float = 0.05,
):
    # AIPTW estimator
    point_est = np.mean(scores, 0)
    # Standard deviations
    std = np.std(scores, 0, ddof=1)
    # Corrections for approximations
    log_y = np.maximum(np.log(y_grid.max() - y_grid.min()), 0)
    correction = (np.log(2) / t2) + (log_y / t1)

    # Compute normal CI
    z = stats.norm.ppf(1 - alpha / 2)

    if bound_type == "lower":
        ci = point_est - (z * std / np.sqrt(n)) - correction
    else:
        ci = point_est + (z * std / np.sqrt(n)) + correction

    return {
        "point_est": np.clip(point_est, 0, 1),
        "ci": np.clip(ci, 0, 1),
        "point_est_var": std**2,
        "correction": correction,
        "t1": t1,
        "t2": t2,
    }


def compute_makarov_bounds_approx(
    bound_type: str,
    y_grid1: np.ndarray,
    y_grid0: np.ndarray,
    t1: int,
    t2: int,
    cdf1: np.ndarray,
    cdf0: np.ndarray,
    A: np.ndarray,
    Y: np.ndarray,
    pi: float = 0.5,
    outcome_type: str = "continuous",
    alpha: float = 0.05,
):
    scores = compute_makarov_bounds_approx_scores(
        bound_type,
        y_grid1,
        y_grid0,
        t1,
        t2,
        cdf1,
        cdf0,
        pi=pi,
        A=A,
        Y=Y,
        outcome_type=outcome_type,
    )
    return compute_makarov_bounds_approx_from_scores(
        scores,
        bound_type,
        y_grid1,
        t1,
        t2,
        A.shape[0],
        alpha=alpha,
    )


def tune_smoothing_params(
    bound_type: str,
    y_grid1: np.ndarray,
    y_grid0: np.ndarray,
    t1_grid: Union[int, List[int]],
    t2_grid: Union[int, List[int]],
    cdf1: np.ndarray,
    cdf0: np.ndarray,
    A: np.ndarray,
    Y: np.ndarray,
    pi: float = 0.5,
    n_test: Optional[int] = None,
    outcome_type: str = "continuous",
    alpha: float = 0.05,
    tune_method: str = "lepski",
    max_neighbors: int = 3,  # how many rougher neighbors (by smaller B) to check for stability
):
    # normalize grids to lists
    t1_list = [t1_grid] if isinstance(t1_grid, int) else list(t1_grid)
    t2_list = [t2_grid] if isinstance(t2_grid, int) else list(t2_grid)
    # all (t1, t2) pairs (flattened index k = 0..P-1)
    pairs = list(itertools.product(t1_list, t2_list))
    # t1_new, t2_new = map(list, zip(*pairs))  # lists of length P
    t1_new, t2_new = list(zip(*pairs))
    P = len(pairs)

    # scores: shape (n, P) after squeeze; each column are per-observation IF-contribs
    scores = compute_makarov_bounds_approx_scores(
        bound_type,
        y_grid1,
        y_grid0,
        t1_new,
        t2_new,
        cdf1,
        cdf0,
        pi=pi,
        A=A,
        Y=Y,
        outcome_type=outcome_type,
    )
    scores = np.squeeze(scores)  # expected shape (n, P)
    if scores.ndim == 1:
        scores = scores[:, None]
    n_sel = n_test if (n_test is not None) else scores.shape[0]

    # summary for each pair: point_est, var(phi), correction B(t1,t2)
    result = compute_makarov_bounds_approx_from_scores(
        scores,
        bound_type,
        y_grid1,
        t1_new,
        t2_new,
        A.shape[0],
        alpha=alpha,
    )
    point_est = np.asarray(result["point_est"]).reshape(-1)  # (P,)
    var_phi = np.asarray(result["point_est_var"]).reshape(-1)  # (P,) = Var(scores)
    B = np.asarray(result["correction"]).reshape(-1)  # (P,)
    SE = np.sqrt(np.maximum(var_phi, 0.0) / float(n_sel))  # (P,)
    z = stats.norm.ppf(1 - alpha / 2)
    if tune_method == "lepski":
        """
        Select (t1, t2) by a Lepski-style stability rule ordered by the full bias bound
        B(t1,t2) = log(2)/t2 + (log|Y|)_+/t1, then freeze and return the pair.
        """

        # order indices by decreasing B (more smoothing first)
        order = np.argsort(-B)  # large -> small
        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(P)

        # helper: SE of difference between two pairs from EIF differences
        def se_diff(k, r):
            d = scores[:, k] - scores[:, r]  # per-observation diff
            return np.std(d, ddof=1) / np.sqrt(n_sel)

        # neighbor set: next few rougher pairs (smaller B) closest in B
        def neighbors_by_B(idx_k, K=max_neighbors):
            # indices with smaller B (rougher), sorted by how close their B is
            rough = [idx for idx in order[inv_order[idx_k] + 1 :]]  # all with smaller B
            if not rough:
                return []
            # pick up to K nearest by |B_k - B_r|
            diffs = np.abs(B[rough] - B[idx_k])
            keep = np.argsort(diffs)[: min(K, len(rough))]
            return [rough[i] for i in keep]

        # Lepski scan: pick the first (largest-B) pair that is stable vs its rougher neighbors
        chosen = None
        for k in order:
            nbrs = neighbors_by_B(k, max_neighbors)
            if not nbrs:
                chosen = k  # nothing rougher; accept
                break
            stable = True
            for r in nbrs:
                sed = se_diff(k, r)
                if sed == 0.0:
                    continue
                if abs(point_est[k] - point_est[r]) > z * sed:
                    stable = False
                    break
            if stable:
                chosen = k
                break

        # fallback if nothing passed: minimize CI half-length over all pairs
        if chosen is None:
            widths = z * SE + B  # half-lengths for the one-sided bias-corrected CI
            chosen = int(np.argmin(widths))
    elif tune_method == "mse":
        objective = B**2 + var_phi / n_sel
        chosen = np.argmin(np.squeeze(objective))
    elif tune_method == "ci_halfwidth":
        objective = B + SE
        chosen = np.argmin(np.squeeze(objective))
    elif tune_method == "ci_endpoint":
        if bound_type == "lower":
            # Find the t1, t2 pair that MSE for lower bound
            objective = -result["ci"]
        else:
            # Find the t1, t2 pair that MSE for upper bound
            objective = result["ci"]
        # Find the minimum value
        min_val = np.min(objective)
        # Find all indices where the value equals the minimum
        min_indices = np.where(objective == min_val)[0]
        # Choose the last index
        chosen = min_indices[-1]
    else:
        raise ValueError(f"Unknown tune_method: {tune_method}")

    # selected pair
    t1 = t1_new[chosen]
    t2 = t2_new[chosen]
    return t1, t2


# Batched versions -------------------------------------


# Split into batches
def batch_generator(
    cdf1: np.ndarray,
    cdf0: np.ndarray,
    a: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    num_batches: int,
    cdf_batched: bool,
):
    for i in range(num_batches):
        if cdf_batched:
            yield {
                "cdf1": cdf1[i * batch_size : (i + 1) * batch_size],
                "cdf0": cdf0[i * batch_size : (i + 1) * batch_size],
                "a": a[i * batch_size : (i + 1) * batch_size],
                "y": y[i * batch_size : (i + 1) * batch_size],
            }
        else:
            yield {
                "cdf1": cdf1,
                "cdf0": cdf0,
                "a": a[i * batch_size : (i + 1) * batch_size],
                "y": y[i * batch_size : (i + 1) * batch_size],
            }


def compute_envelope_estimator_batchwise(
    bound_type: str,
    y_grid1: np.ndarray,
    y_grid0: np.ndarray,
    cdf1: np.ndarray,
    cdf0: np.ndarray,
    cdf_batched: bool,
    num_batches: int,
    A: np.ndarray,
    Y: np.ndarray,
    pi: float = 0.5,
    alpha: float = 0.05,
):
    n_test = A.shape[0]
    batch_size = n_test // num_batches
    scores_list = []
    for batch in batch_generator(
        cdf1, cdf0, A, Y, batch_size, num_batches, cdf_batched
    ):
        scores_list.append(
            compute_makarov_bounds_envelope_scores(
                bound_type,
                y_grid1,
                y_grid0,
                batch["cdf1"],
                batch["cdf0"],
                pi=pi,
                A=batch["a"],
                Y=batch["y"],
            )[0]
        )

    return compute_makarov_bounds_envelope_from_scores(
        np.concatenate(scores_list, axis=0),
        bound_type,
        batch_size * num_batches,
        alpha=alpha,
    )


def compute_makarov_bounds_approx_batchwise(
    bound_type: str,
    y_grid1: np.ndarray,
    y_grid0: np.ndarray,
    t1: int,
    t2: int,
    cdf1: np.ndarray,
    cdf0: np.ndarray,
    cdf_batched: bool,
    num_batches: int,
    A: np.ndarray,
    Y: np.ndarray,
    pi: float = 0.5,
    outcome_type: str = "continuous",
    alpha: float = 0.05,
):
    n_test = A.shape[0]
    batch_size = n_test // num_batches
    scores_list = []
    for batch in batch_generator(
        cdf1, cdf0, A, Y, batch_size, num_batches, cdf_batched
    ):
        scores_list.append(
            compute_makarov_bounds_approx_scores(
                bound_type,
                y_grid1,
                y_grid0,
                t1,
                t2,
                batch["cdf1"],
                batch["cdf0"],
                pi=pi,
                A=batch["a"],
                Y=batch["y"],
                outcome_type=outcome_type,
            )
        )

    return compute_makarov_bounds_approx_from_scores(
        np.concatenate(scores_list, axis=0),
        bound_type,
        y_grid1,
        t1,
        t2,
        batch_size * num_batches,
        alpha=alpha,
    )


def tune_smoothing_params_batchwise(
    bound_type: str,
    y_grid1: np.ndarray,
    y_grid0: np.ndarray,
    t1_grid: Union[int, List[int]],
    t2_grid: Union[int, List[int]],
    cdf1: np.ndarray,
    cdf0: np.ndarray,
    cdf_batched: bool,
    num_batches: int,
    A: np.ndarray,
    Y: np.ndarray,
    pi: float = 0.5,
    n_test: Optional[int] = None,
    outcome_type: str = "continuous",
    alpha: float = 0.05,
):
    # Convert integers
    if isinstance(t1_grid, int):
        t1_grid_list = [t1_grid]
    else:
        t1_grid_list = t1_grid
    if isinstance(t2_grid, int):
        t2_grid_list = [t2_grid]
    else:
        t2_grid_list = t2_grid

    # Create new lists containing all combinations of elements in t1_grid and t2_grid
    # Get all pairs in the Cartesian product
    pairs = list(itertools.product(t1_grid_list, t2_grid_list))
    # Unzip the pairs into two lists
    t1_new, t2_new = zip(*pairs)
    # Convert to lists if needed
    t1_new = list(t1_new)
    t2_new = list(t2_new)
    # Compute MSE for each t1, t2 pair, both upper and lower bounds
    mses = []
    for i in range(len(t1_new)):
        result = compute_makarov_bounds_approx_batchwise(
            bound_type,
            y_grid1,
            y_grid0,
            t1_new[i],
            t2_new[i],
            cdf1,
            cdf0,
            cdf_batched,
            num_batches,
            pi=pi,
            A=A,
            Y=Y,
            outcome_type=outcome_type,
            alpha=alpha,
        )
        # if bound_type == "lower":
        #     # Find the t1, t2 pair that MSE for lower bound
        #     mses.append(-result["ci"])
        # else:
        #     # Find the t1, t2 pair that MSE for upper bound
        #     mses.append(result["ci"])
        mses.append(result["point_est_var"] / n_test + result["correction"] ** 2)

    # Find the t1, t2 pair that MSE for lower bound
    t1_t2_idx = np.argmin(np.squeeze(mses))
    t1 = t1_new[t1_t2_idx]
    t2 = t2_new[t1_t2_idx]
    return t1, t2
