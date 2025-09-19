"""
Copyright (c) Meta Platforms, Inc. and affiliates.
"""

from abc import ABC, abstractmethod

import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from statsmodels.distributions.copula.api import GumbelCopula


class TwoUniformMixture:
    """
    Equal-weight mixture of TWO frozen scipy.stats.uniform distributions.

    Accepts u1,u2 with vectorized loc/scale, e.g. (n,1). Methods:
      - pdf(x) = 0.5*u1.pdf(x) + 0.5*u2.pdf(x)
      - cdf(x): (k,) -> (n,k); (n,1)/(n,) -> (n,1)
      - ppf(q): closed-form, returns (n,1) for q (n,1)/(n,)
      - rvs(size): returns (n, size)
    """

    def __init__(self, u1, u2):
        if (
            getattr(getattr(u1, "dist", None), "name", "") != "uniform"
            or getattr(getattr(u2, "dist", None), "name", "") != "uniform"
        ):
            raise ValueError("u1 and u2 must be frozen scipy.stats.uniform objects.")
        self.u1, self.u2 = u1, u2

        # Endpoints from the frozen objects (works with array params)
        a1, b1 = self._endpoints(u1)
        a2, b2 = self._endpoints(u2)

        # Broadcast to common parameter shape
        a1, b1, a2, b2 = np.broadcast_arrays(a1, b1, a2, b2)

        # Record "row shape" (drop trailing singleton if present, e.g. (n,1)->(n,))
        self.param_shape = a1.shape
        if a1.ndim >= 1 and self.param_shape[-1] == 1:
            self.row_shape = self.param_shape[:-1]  # e.g. (n,)
            self.col_shape = self.row_shape + (1,)  # e.g. (n,1)
        else:
            self.row_shape = self.param_shape  # e.g. () or (n,)
            self.col_shape = self.row_shape + (1,)

        # Order intervals per row: left (L) has the smaller start
        swap = a1 > a2
        self.aL = np.where(swap, a2, a1)
        self.bL = np.where(swap, b2, b1)
        self.aR = np.where(swap, a1, a2)
        self.bR = np.where(swap, b1, b2)

        self.lenL = self.bL - self.aL
        self.lenR = self.bR - self.aR
        if np.any(self.lenL <= 0) or np.any(self.lenR <= 0):
            raise ValueError("Uniform scales must be strictly positive.")

        # Mixture slopes (weight 1/2 per component)
        self.sL = 0.5 / self.lenL
        self.sR = 0.5 / self.lenR

    @staticmethod
    def _endpoints(u):
        # Prefer kwds (exact loc/scale); fallback to ppf if needed
        kw = getattr(u, "kwds", {}) or {}
        if ("loc" in kw) and ("scale" in kw):
            loc = np.asarray(kw["loc"])
            scale = np.asarray(kw["scale"])
            return loc, loc + scale
        a = np.asarray(u.ppf(0.0))
        b = np.asarray(u.ppf(1.0 - 1e-12))
        return a, b

    # ----------------------- pdf / cdf -----------------------
    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        return 0.5 * (self.u1.pdf(x) + self.u2.pdf(x))

    def cdf(self, x):
        x = np.asarray(x, dtype=float)

        # Broadcast-grid mode: x is (k,) → squeeze the singleton column dim
        if x.ndim == 1:
            F = 0.5 * (self.u1.cdf(x) + self.u2.cdf(x))  # shape ~ row + (1,) + (k,)
            if F.ndim >= 2 and F.shape[-2] == 1:
                F = np.squeeze(F, axis=-2)  # (n,k)
            return F

        # Per-row mode: x is (n,1) or (n,)
        x_vec = np.squeeze(x)
        if x_vec.shape != self.row_shape:
            raise ValueError(
                f"Expected x shape {self.row_shape} or {self.col_shape}, got {x.shape}."
            )
        x_col = x_vec.reshape(self.col_shape)  # ensure (n,1)
        F = 0.5 * (self.u1.cdf(x_col) + self.u2.cdf(x_col))  # (n,1)
        return F

    # ----------------------- closed-form PPF -----------------------
    def ppf(self, q):
        """
        Closed-form inverse for equal-weight 2-uniform mixture.
        q: (n,1) or (n,) (or scalar if params scalar). Returns (n,1).
        """
        qv = np.squeeze(np.asarray(q, dtype=float))
        if np.any((qv <= 0.0) | (qv >= 1.0)):
            raise ValueError("q must be strictly in (0,1).")
        if qv.shape != self.row_shape:
            raise ValueError(f"q must have shape {self.row_shape} or {self.col_shape}.")
        qv = qv.reshape(self.col_shape)  # (n,1)

        # Segment masses:
        # 1) left-only: [aL, min(bL, aR)]
        left_end = np.minimum(self.bL, self.aR)
        mass1 = self.sL * np.clip(left_end - self.aL, 0.0, None)

        # 2) overlap: [aR, min(bL, bR)]
        ov_start = self.aR
        ov_end = np.minimum(self.bL, self.bR)
        mass2 = (self.sL + self.sR) * np.clip(ov_end - ov_start, 0.0, None)

        # 3) right-only: [max(bL, aR), bR]
        right_start = np.maximum(self.bL, self.aR)
        # mass3 = 1 - (mass1 + mass2) implicitly

        m1 = mass1
        m2 = mass1 + mass2

        in_left = qv <= m1
        in_mid = (~in_left) & (qv <= m2)

        # Affine inverses on each segment
        x_left = self.aL + qv / self.sL
        denom = self.sL + self.sR
        x_mid = ov_start + np.where(denom > 0, (qv - m1) / denom, 0.0)
        x_right = right_start + (qv - m2) / self.sR

        x = np.where(in_left, x_left, np.where(in_mid, x_mid, x_right))
        return x  # (n,1)

    # ----------------------- sampling -----------------------
    def rvs(self, size=1, random_state=None):
        if isinstance(size, int):
            size = (size,)
        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )

        target = self.col_shape + tuple(size)  # (n,1,...) so we can squeeze
        J = rng.integers(0, 2, size=target)
        U = rng.random(size=target)
        X1 = self.u1.ppf(U)
        X2 = self.u2.ppf(U)
        X = np.where(J == 0, X1, X2)  # (n,1, size)
        if X.ndim >= 2 and X.shape[-2] == 1:
            X = np.squeeze(X, axis=-2)  # -> (n, size)
        return X



class SCM(ABC):
    def __init__(self, data_config):
        self.data_config = data_config
        # Real data
        self.X = None
        self.A = None
        self.Y = None
        self.outcome_type = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sample(self, n):
        # Sample confounders standard normal
        if self.X is None:
            X = np.random.uniform(low=0, high=1, size=(n, self.data_config["p"]))
        else:
            if n == -1:
                n = len(self.X)
            n_x = len(self.X)
            n = min(n, n_x)
            # Sample n random indices
            idx = np.random.choice(n_x, n, replace=False)
            X = self.X[idx]
        # Get propensity scores and sample treatments
        if self.A is None:
            pi = self.propensity(X)
            A = np.random.binomial(n=1, p=pi)
        else:
            A = self.A[idx]
        if self.Y is None:
            # Outcome
            # noise = np.random.uniform(low=0, high=1, size=(n, 2))
            noise_coupled = self.response_copula(X).rvs(n)
            Y1 = self.response1_marginal(X).ppf(noise_coupled[:, 0:1])
            Y0 = self.response0_marginal(X).ppf(noise_coupled[:, 1:2])
            Y = A * Y1 + (1 - A) * Y0
        else:
            Y = self.Y[idx]
        return X, A, Y

    def get_y_grids(self, Y, deltas, y_grid_size=200):
        y_grid_dict = {}
        y_grid_dict[0] = {}

        # Get outcome support depending on the outcome type
        if self.outcome_type == "continuous":
            y_support = list(
                np.linspace(
                    np.min(Y),
                    np.max(Y),
                    y_grid_size,
                )
            )
        elif self.outcome_type == "discrete":
            y_support = list(np.sort(np.unique(Y)))
        elif self.outcome_type == "poisson":
            if y_grid_size < len(np.unique(Y)):
                y1d = np.asarray(Y).ravel()
                # k equally spaced quantiles, inclusive
                q = np.linspace(0, 1, y_grid_size)
                # data values at those quantiles
                y_support = list(np.quantile(y1d, q))
            else:
                y_support = list(np.sort(np.unique(Y)))
        else:
            raise ValueError("outcome_type not defined")

        # Treatment cdf support
        # Add a small offset to the first element to include the leftmost point (where cdf is zero)
        offset = y_support[1] - y_support[0]
        y_grid_dict[1] = np.array([y_support[0] - offset] + y_support)

        if self.outcome_type in ["discrete", "poisson"]:
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
        return y_grid_dict

    @abstractmethod
    def get_ground_truth(self, y_grid, deltas):
        pass


# SCM class for synthetic outcomes
class SCM_synthetic(SCM, ABC):
    def __init__(self, data_config):
        super().__init__(data_config)

    @abstractmethod
    def propensity(self, x):
        pass

    @abstractmethod
    def response1_marginal(self, x):
        pass

    @abstractmethod
    def response0_marginal(self, x):
        pass

    @abstractmethod
    def response_copula(self, x):
        pass

    def sample_counterfactuals(self, X):
        noise_coupled = self.response_copula(X).rvs(X.shape[0])
        Y1 = self.response1_marginal(X).ppf(noise_coupled[:, 0:1])
        Y0 = self.response0_marginal(X).ppf(noise_coupled[:, 1:2])
        return Y1, Y0

    # Get ground-truth nuisances
    def get_gt_nuisances(self, nuisance_keys, X):
        gt_nuisances = {}
        A1 = np.ones_like(X[:, :1])
        A0 = np.zeros_like(X[:, :1])

        if "F1" in nuisance_keys:
            gt_nuisances["F1"] = self.response_marginal(X, A1)

        if "F0" in nuisance_keys:
            gt_nuisances["F0"] = self.response_marginal(X, A0)

        if "propensity" in nuisance_keys:
            gt_nuisances["propensity"] = self.propensity(X)
        return gt_nuisances

    def get_gt_makarov_bounds(self, n, y_grid1, y_grid0, delta=0):
        # Get FNA
        X, _, _ = self.sample(n)
        Y1, Y0 = self.sample_counterfactuals(X)
        fna = np.mean((Y1 - Y0) <= delta)
        # Makarov bounds
        F1 = self.response1_marginal(X).cdf(y_grid1)
        F0 = self.response0_marginal(X).cdf(y_grid0)
        lower = np.mean(np.maximum(np.max(F1 - F0, 1), 0), 0)
        upper = 1 + np.mean(np.minimum(np.min(F1 - F0, 1), 0), 0)
        return fna, lower, upper

    def get_gt_makarov_bounds_marginal(self, n, y_grid1, y_grid0, delta=0):
        # Get FNA
        X, _, _ = self.sample(n)
        Y1, Y0 = self.sample_counterfactuals(X)
        fna = np.mean((Y1 - Y0) <= delta)
        # Makarov bounds
        F1 = np.mean(self.response1_marginal(X).cdf(y_grid1), 0)
        F0 = np.mean(self.response0_marginal(X).cdf(y_grid0), 0)
        lower = np.maximum(np.max(F1 - F0), 0)
        upper = 1 + np.minimum(np.min(F1 - F0), 0)
        return fna, lower, upper

    def get_ground_truth(self, y_grids_dict, deltas):
        ground_truth = {
            "makarov_bounds_covariate": {},
            "makarov_bounds_marginal": {},
            "fna": {},
        }
        for delta in deltas:
            fna, makarov_lower, makarov_upper = self.get_gt_makarov_bounds(
                n=100000,
                y_grid1=y_grids_dict[1],
                y_grid0=y_grids_dict[0][delta],
                delta=delta,
            )
            fna, makarov_lower_marginal, makarov_upper_marginal = (
                self.get_gt_makarov_bounds_marginal(
                    n=100000,
                    y_grid1=y_grids_dict[1],
                    y_grid0=y_grids_dict[0][delta],
                    delta=delta,
                )
            )
            ground_truth["makarov_bounds_marginal"][delta] = {
                "lower": makarov_lower_marginal,
                "upper": makarov_upper_marginal,
            }
            ground_truth["makarov_bounds_covariate"][delta] = {
                "lower": makarov_lower,
                "upper": makarov_upper,
            }
            ground_truth["fna"][delta] = fna
        return ground_truth


class SCM_continuous_uniform2(SCM_synthetic):
    def __init__(self, data_config):
        super().__init__(data_config)
        self.outcome_type = "continuous"
        self.gamma = float(self.data_config.get("gamma", 0.3))
        self.rho = float(self.data_config.get("rho", 0.5))

    # randomized
    def propensity(self, x):
        return 0.5 * np.ones((x.shape[0], 1), dtype=float)

    def response1_marginal(self, x):
        # Y1 ~ Unif[0,0.5] everywhere
        rho = 0.5 * x**2 + 0.5
        n = x.shape[0]
        loc = np.zeros((n, 1), dtype=float)
        # scale = np.full((n, 1), 0.5, dtype=float)  # length of interval
        return stats.uniform(loc=loc, scale=rho)

    def response0_marginal(self, x):
        n = x.shape[0]
        gamma = self.gamma * x
        rho = 0.5 * x**2 + 0.5
        loc1 = np.full((n, 1), -gamma / 4.0, dtype=float)
        # scale1 = np.full((n, 1), 0.25, dtype=float)
        scale1 = 0.5 * rho
        loc2 = np.full((n, 1), 0.5 * rho + gamma / 4.0, dtype=float)
        # scale2 = np.full((n, 1), 0.25, dtype=float)
        scale2 = 0.5 * rho
        u1 = stats.uniform(loc=loc1, scale=scale1)
        u2 = stats.uniform(loc=loc2, scale=scale2)
        return TwoUniformMixture(u1, u2)

    def response_copula(self, x):
        return GumbelCopula(theta=5, k_dim=2


def get_scm(data_config):
    SCM_REGISTRY = {
        "continuous_uniform2": SCM_continuous_uniform2,
    }

    if data_config["scm_name"] in SCM_REGISTRY.keys():
        scm = SCM_REGISTRY[data_config["scm_name"]](data_config)
    else:
        raise ValueError("Unknown dataset name")
    return scm


def load_data(scm, data_config, deltas, y_grid_size=None) -> dict:
    # Train test split
    X, A, Y = scm.sample(n=data_config["n"])
    X_trainval, X_test, A_trainval, A_test, Y_trainval, Y_test = train_test_split(
        X, A, Y, test_size=data_config["test_frac"]
    )
    X_train, X_val, A_train, A_val, Y_train, Y_val = train_test_split(
        X_trainval, A_trainval, Y_trainval, test_size=data_config["val_frac"]
    )

    d_train = {
        "x": X_train,
        "a": A_train,
        "y": Y_train,
    }
    d_val = {
        "x": X_val,
        "a": A_val,
        "y": Y_val,
    }
    d_test = {
        "x": X_test,
        "a": A_test,
        "y": Y_test,
    }

    y_grids_dict = scm.get_y_grids(Y_trainval, deltas, y_grid_size)
    ground_truth = scm.get_ground_truth(y_grids_dict, deltas)

    filter_dict = {
        "d_train": {"full": np.ones(Y_train.shape[0], dtype=bool)},
        "d_val": {"full": np.ones(Y_val.shape[0], dtype=bool)},
        "d_test": {"full": np.ones(Y_test.shape[0], dtype=bool)},
    }

    data_collection = {
        "d_train": d_train,
        "d_val": d_val,
        "d_test": d_test,
        "y_grids": y_grids_dict,
        "ground_truth": ground_truth,
        "filter_dict": filter_dict,
    }
    return data_collection
