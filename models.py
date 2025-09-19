"""
Copyright (c) Meta Platforms, Inc. and affiliates.
"""

# Imports
import datetime
import os
from abc import ABC, abstractmethod
from typing import Optional

import lightgbm as lgb
import numpy as np
import scipy.stats as stats
from scipy.special import gammaln, logsumexp, softmax
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    LabelEncoder,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)


class ConditionalCDFModel(ABC):
    def __init__(self, lgb_params: dict):
        self.lgb_params = lgb_params.copy()
        self.model = None

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _stabilize_derivative(
        self, input_der: np.ndarray, method: str = "MAD", eps: float = 1e-4
    ) -> np.ndarray:
        """
        Rescales a gradient or Hessian array so that its magnitude is
        comparable across distributional parameters.
        """
        # ---------- replace NaNs by the (finite) mean -------------------------
        finite_mask = ~np.isnan(input_der)
        mean_val = input_der[finite_mask].mean() if finite_mask.any() else 0.0
        x = np.nan_to_num(input_der, nan=mean_val, copy=True)

        if method == "MAD":
            med = np.median(x, axis=0, keepdims=True)
            div = np.median(np.abs(x - med), axis=0, keepdims=True)
            div = np.maximum(div, eps)  # floor at eps
            return x / div

        if method == "L2":
            div = np.sqrt(np.mean(x**2, axis=0, keepdims=True))
            div = np.clip(div, eps, 1e4)  # floor and cap
            return x / div

        if method == "None":
            return x

        raise ValueError("method must be 'MAD', 'L2', or 'None'")

    @abstractmethod
    def fit(self, d_train: dict, d_val: Optional[dict] = None):
        pass

    @abstractmethod
    def predict_cdf(
        self, X: np.ndarray, y_grid: np.ndarray, raw_iteration: str = "best"
    ):
        pass


class ConditionalGaussianMixture(ConditionalCDFModel):
    def __init__(self, lgb_params: dict):
        """
        Conditional Gaussian Mixture for CDF prediction
        """
        super().__init__(lgb_params)
        self.n_components = self.lgb_params["n_components"]
        self.model = None
        self.n_outputs = 3 * self.n_components
        if self.n_components == 1:
            self.n_outputs = 2
        self.lgb_params["objective"] = self.train_objective
        self.lgb_params["num_class"] = self.n_outputs
        self.lgb_params["boost_from_average"] = False
        self.lgb_params["device_type"] = ["cpu"]
        self.lgb_params["num_threads"] = os.cpu_count()
        if "y_transform" in lgb_params.keys():
            if self.lgb_params["y_transform"] == "standard":
                self.y_transformer = StandardScaler()
            elif self.lgb_params["y_transform"] == "robust":
                self.y_transformer = RobustScaler()
            elif self.lgb_params["y_transform"] == "log":
                self.y_transformer = FunctionTransformer(np.log1p)
            elif self.lgb_params["y_transform"] == "quantile":
                self.y_transformer = QuantileTransformer()
            elif self.lgb_params["y_transform"] == "log_standard":
                self.y_transformer = Pipeline(
                    [
                        ("log", FunctionTransformer(np.log1p, validate=False)),
                        ("scale", StandardScaler()),
                    ]
                )
            else:
                raise ValueError("y_transform not defined")
        else:
            self.y_transformer = None
        self.means_init = None
        self.log_vars_init = None

    def get_init_scores(self, n: int):
        mu = np.broadcast_to(self.means_init, (n, self.n_components))
        log_var = np.broadcast_to(self.log_vars_init, (n, self.n_components))
        if self.n_components > 1:
            pi = np.zeros((n, self.n_components), dtype=mu.dtype)
            return np.concatenate([pi, mu, log_var], axis=1)
        else:
            return np.concatenate([mu, log_var], axis=1)

    def trainsform_preds_to_params(self, preds: np.ndarray):
        K = self.n_components
        if K > 1:
            z_pi = preds[:, :K]  # logits
            mu = preds[:, K : 2 * K]  # means
            s = preds[:, 2 * K : 3 * K]  # log-variances
            sigma2 = np.exp(s)  # variances
            # ---------- mixture weights --------------------------------------------
            pi = softmax(z_pi, axis=1)  # (n, K)
            return pi, mu, sigma2
        else:
            mu = preds[:, :1]  # means
            s = preds[:, 1:2]  # log variances
            sigma2 = np.exp(s)  # variances
            return np.ones_like(mu), mu, sigma2

    def train_objective(self, y_pred: np.ndarray, dtrain: lgb.Dataset):
        hess_floor = 1e-6
        K = self.n_components
        y_true = dtrain.get_label()

        # ---------- unpack raw scores ------------------------------------------
        if K > 1:
            z_pi = y_pred[:, :K]  # logits
            mu = y_pred[:, K : 2 * K]  # means
            s = y_pred[:, 2 * K : 3 * K]  # log-variances
            # ---------- mixture weights --------------------------------------------
            log_pi = z_pi - logsumexp(z_pi, axis=1, keepdims=True)
            pi = softmax(z_pi, axis=1)  # (n, K)
        else:
            mu = y_pred[:, :1]  # logits
            s = y_pred[:, 1:2]  # means
            log_pi = np.zeros_like(s)
            pi = np.exp(log_pi)  # (n, 1)

        sigma2 = np.exp(s)  # variances

        # ---------- component log-densities ------------------------------------
        y = np.expand_dims(np.asarray(y_true), 1)  # broadcast to (n, 1)
        sq = (y - mu) ** 2
        u = sq * np.exp(-s)  # (y−μ)² / σ²
        log_phi = -0.5 * np.log(2 * np.pi) - 0.5 * s - 0.5 * u

        # total log-likelihood per sample
        log_p = logsumexp(log_pi + log_phi, axis=1, keepdims=True)

        # ---------- responsibilities r_{ik} ------------------------------------
        r = np.exp(log_pi + log_phi - log_p)  # (n, K)

        # ===================== GRADIENTS =========================
        g_pi = pi - r  # mixture logits
        g_mu = r * (mu - y) / sigma2  # means
        g_s = 0.5 * r * (1 - u)  # log-variances

        # ===================== DIAGONAL HESSIANS ===============================
        # 1) mixture logits
        h_pi = pi * (1.0 - pi) - r * (1.0 - r)

        # 2) means
        h_mu = r * (1.0 / sigma2 - (1.0 - r) * (y - mu) ** 2 / sigma2**2)

        # 3) log-variances
        delta = 0.5 * (u - 1.0)  # (u-1)/2
        h_s = -r * (1.0 - r) * delta**2 + 0.5 * r * u

        # ---------- numerical safety -------------------------------------------
        h_pi = np.clip(h_pi, hess_floor, None)
        h_mu = np.clip(h_mu, hess_floor, None)
        h_s = np.clip(h_s, hess_floor, None)

        # Stack together
        grad_matrix = np.empty_like(y_pred)
        hess_matrix = np.empty_like(y_pred)
        if K > 1:
            grad_matrix[:, :K] = g_pi
            grad_matrix[:, K : 2 * K] = g_mu
            grad_matrix[:, 2 * K :] = g_s
            hess_matrix[:, :K] = h_pi
            hess_matrix[:, K : 2 * K] = h_mu
            hess_matrix[:, 2 * K :] = h_s
        else:
            grad_matrix[:, :1] = g_mu
            grad_matrix[:, 1:2] = g_s
            hess_matrix[:, :1] = h_mu
            hess_matrix[:, 1:2] = h_s
        # Stabilize and flatten (order='F' for LightGBM)
        grad = self._stabilize_derivative(grad_matrix).ravel(order="F")
        hess = self._stabilize_derivative(hess_matrix).ravel(order="F")
        return grad, hess

    def average_log_likelihood(self, y_pred: np.ndarray, dval: lgb.Dataset):
        y_true = np.asarray(dval.get_label())
        pi, mu, sigma2 = self.trainsform_preds_to_params(y_pred)
        # ---------- component log-densities ------------------------------------
        y = y_true.reshape(-1, 1)  # broadcast to (n, 1)
        sq = (y - mu) ** 2
        u = sq / sigma2  # (y−μ)² / σ²

        log_phi = -0.5 * np.log(2 * np.pi) - 0.5 * np.log(sigma2) - 0.5 * u
        log_pi = np.log(pi)  # (n, K)
        # total log-likelihood per sample
        log_p = logsumexp(log_pi + log_phi, axis=1, keepdims=True)
        # Returns name, result, is_higher_better
        return "avg_log_likelihood", np.mean(log_p), True

    def fit(self, d_train: dict, d_val: Optional[dict] = None):
        """Fit the model"""
        y_train = np.asarray(d_train["y"], dtype=np.float32).reshape(-1, 1)
        if self.y_transformer is not None:
            y_train_transformed = self.y_transformer.fit_transform(y_train)
        else:
            y_train_transformed = y_train

        # Compute initial scores on training set
        K = self.n_components
        n_init_max = np.minimum(100000, y_train_transformed.shape[0])
        if K > 1:
            kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
            labels = kmeans.fit_predict(y_train_transformed[:n_init_max, :])
            centers = kmeans.cluster_centers_.ravel()
        else:  # K == 1
            labels = np.zeros_like(y_train_transformed[:n_init_max, :], dtype=int)
            centers = np.mean(y_train_transformed[:n_init_max, :])
        global_var = np.var(
            y_train_transformed[:n_init_max, :]
        )  # single scalar for shrinkage / floor

        # Cluster level variances
        cluster_var = np.array(
            [
                (
                    np.var(y_train_transformed[:n_init_max, :][labels == k])
                    if (labels == k).any()
                    else global_var
                )
                for k in range(K)
            ]
        )
        # Numerical safeguards (avoid log(0) and huge Hessians)
        var_floor = 1e-6 * global_var
        var_cap = 1e2 * global_var
        cluster_var = np.clip(cluster_var, var_floor, var_cap)
        self.means_init = centers.astype(np.float32)
        self.log_vars_init = np.log(cluster_var).astype(np.float32)

        train_data = lgb.Dataset(
            d_train["x"].astype(np.float32),
            label=y_train_transformed.astype(np.float32),
            init_score=self.get_init_scores(y_train_transformed.shape[0]),
        )

        if d_val is None:
            valid_sets = [train_data]
        else:
            y_val = np.asarray(d_val["y"], dtype=np.float32).reshape(-1, 1)
            if self.y_transformer is not None:
                y_val_transformed = self.y_transformer.transform(y_val)
            else:
                y_val_transformed = y_val
            valid_data = lgb.Dataset(
                d_val["x"].astype(np.float32),
                label=y_val_transformed.astype(np.float32),
                init_score=self.get_init_scores(y_val_transformed.shape[0]),
                reference=train_data,
            )
            valid_sets = [train_data, valid_data]
        # Print time
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Start model fitting at {current_time}")
        # Train
        logging_dict = {}
        self.model = lgb.train(
            self.lgb_params,
            train_data,
            num_boost_round=self.lgb_params["num_boost_round"],
            valid_sets=valid_sets,
            feval=self.average_log_likelihood,
            callbacks=[
                lgb.early_stopping(self.lgb_params["early_stopping_nb"]),
                lgb.log_evaluation(period=100),
                lgb.record_evaluation(logging_dict),
            ],
            keep_training_booster=True,
        )
        return logging_dict

    def predict_parameters(self, X: np.ndarray, raw_iteration: str = "best"):
        """
        Returns model output

        Input:
        -----------
        X : array-like, shape (n_samples, n_features)
            Covariates
        """

        if raw_iteration == "best":
            n_iter = self.model.best_iteration
        elif raw_iteration == "last":
            n_iter = self.model.current_iteration()
        else:
            n_iter = raw_iteration
        if self.model is None:
            raise ValueError("Model not fitted")
        X = np.array(X)
        n_samples = X.shape[0]
        # Get predictions
        raw_pred = self.model.predict(X, num_iteration=n_iter, raw_score=True)
        preds = raw_pred.reshape(n_samples, self.n_outputs)
        # Add initial scores
        preds += self.get_init_scores(n_samples)
        return self.trainsform_preds_to_params(preds)

    def predict_cdf(
        self, X: np.ndarray, y_grid: np.ndarray, raw_iteration: str = "best"
    ):
        """
        Predict CDF values P(Y <= y | X)

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Covariates
        y_grid : array-like, shape (n_grid,)
            Values at which to evaluate CDF

        Returns:
        --------
        array : CDF values P(Y <= y | X)
        """
        if self.y_transformer is not None:
            y_grid_transformed = np.squeeze(
                self.y_transformer.transform(y_grid.reshape(-1, 1))
            )
        else:
            y_grid_transformed = y_grid
        n_samples = len(X)
        n_grid = len(y_grid)
        pi, mu, sigma2 = self.predict_parameters(X, raw_iteration=raw_iteration)
        K = self.n_components

        # Compute gaussian mixture CDF
        cdf_values = np.zeros((n_samples, n_grid))
        for k in range(K):
            # CDF of k-th component at point y
            component_cdf = stats.norm(
                loc=mu[:, k : k + 1], scale=np.sqrt(sigma2)[:, k : k + 1]
            ).cdf(y_grid_transformed)
            cdf_values += pi[:, k : k + 1] * component_cdf

        return cdf_values


class MultiLabelClassifier(ConditionalCDFModel):
    def __init__(self, lgb_params: dict):
        """
        Multilabel classifier for discrete CDF prediction
        """
        super().__init__(lgb_params)
        self.lgb_params["objective"] = "multiclass"
        self.lgb_params["metric"] = "multi_logloss"
        self.lgb_params["device_type"] = ["cpu"]
        self.y_transformer = LabelEncoder()

    def fit(self, d_train: dict, d_val: Optional[dict] = None):
        """Fit the model"""
        if d_val is None:
            self.y_transformer.fit(np.concatenate([d_train["y"]], axis=0))
        else:
            self.y_transformer.fit(np.concatenate([d_train["y"], d_val["y"]], axis=0))
        y_train_transformed = self.y_transformer.transform(d_train["y"])

        train_data = lgb.Dataset(
            d_train["x"].astype(np.float32),
            label=y_train_transformed.astype(np.int32),
        )

        if d_val is None:
            valid_sets = [train_data]
        else:
            y_val_transformed = self.y_transformer.transform(d_val["y"])
            valid_data = lgb.Dataset(
                d_val["x"].astype(np.float32),
                label=y_val_transformed.astype(np.int32),
                reference=train_data,
            )
            valid_sets = [train_data, valid_data]
        # Print time
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Start model fitting at {current_time}")
        # Train
        logging_dict = {}
        self.model = lgb.train(
            self.lgb_params,
            train_data,
            num_boost_round=self.lgb_params["num_boost_round"],
            valid_sets=valid_sets,
            callbacks=[
                lgb.early_stopping(self.lgb_params["early_stopping_nb"]),
                lgb.log_evaluation(period=100),
                lgb.record_evaluation(logging_dict),
            ],
            keep_training_booster=True,
        )
        return logging_dict

    def predict_cdf(
        self, X: np.ndarray, y_grid: np.ndarray, raw_iteration: str = "best"
    ):
        if raw_iteration == "best":
            n_iter = self.model.best_iteration
        elif raw_iteration == "last":
            n_iter = self.model.current_iteration()
        else:
            n_iter = raw_iteration
        if self.model is None:
            raise ValueError("Model not fitted")

        # Get predictions
        proba = self.model.predict(X, num_iteration=n_iter, raw_score=False)
        # Get the class values in order (e.g., [0.3, 1.7, 1.9])
        if not hasattr(self.y_transformer, "classes_"):
            raise ValueError("Model must be fitted before prediction")
        # pyre-ignore[16]: `classes_` exists on fitted LabelEncoder
        class_values = np.array(self.y_transformer.classes_)  # shape: (n_classes,)

        # For each y in y_grid, sum probabilities for class_values <= y
        cdf = np.zeros((proba.shape[0], len(y_grid)))
        for i, y in enumerate(y_grid):
            mask = class_values <= y  # shape: (n_classes,)
            cdf[:, i] = proba[:, mask].sum(axis=1)
        return cdf


class ConditionalZeroInflatedPoisson(ConditionalCDFModel):
    """
    Conditional Zero-Inflated Poisson:
        P(Y = 0 | X)   = ψ(X) + (1-ψ(X))·e^{-λ(X)}
        P(Y = y>0 | X) = (1-ψ(X))·e^{-λ(X)} λ(X)^y / y!
    Model outputs two raw scores per sample:
        z₁ = log λ ,  z₂ = logit ψ
    """

    # ------------------------------------------------------------------ #
    # INITIALISATION
    # ------------------------------------------------------------------ #
    def __init__(self, lgb_params: dict):
        super().__init__(lgb_params)
        self.lgb_params = lgb_params.copy()
        self.n_outputs = 2  # log-λ , ψ-logit
        # Configure LightGBM
        self.lgb_params["objective"] = self.train_objective
        self.lgb_params["num_class"] = self.n_outputs
        self.lgb_params["boost_from_average"] = False
        self.lgb_params["device_type"] = ["cpu"]
        # Initial scores (set during fit)
        self.lam_init = None  # log-λ initial value
        self.psi_init = None  # ψ-logit initial value
        self.y_transformer = LabelEncoder()

    # ------------------------------------------------------------------ #
    # RAW SCORES  →  PARAMETERS
    # ------------------------------------------------------------------ #
    def _raw_to_params(self, preds: np.ndarray):
        z_loglam = preds[:, :1]  # log λ
        z_psi = preds[:, 1:2]  # logit ψ
        lam = np.exp(z_loglam)  # λ ≥ 0
        psi = self._sigmoid(z_psi)  # 0 < ψ < 1
        return lam, psi  # shape (n,1), (n,1)

    # ------------------------------------------------------------------ #
    # OBJECTIVE (grad & Hessian of −log-lik)
    # ------------------------------------------------------------------ #
    def train_objective(self, y_pred: np.ndarray, dtrain: lgb.Dataset):
        """
        Custom LightGBM objective for ZIP (single component).
        Returns flattened gradient & Hessian (Fortran order).
        """
        hess_floor = 1e-6
        y_true = np.asarray(dtrain.get_label()).astype(np.int32).reshape(-1, 1)
        lam, psi = self._raw_to_params(y_pred)
        y_zero = y_true == 0

        # ---------- per-sample log-lik derivatives --------------------
        # Common helpers
        p0 = np.exp(-lam)
        A = psi + (1 - psi) * p0  # P(Y=0)
        # dL/dλ
        dL_dlam = np.where(y_zero, -(1 - psi) * p0 / A, -1.0 + y_true / lam)
        # dL/dψ
        dL_dpsi = np.where(y_zero, (1 - p0) / A, -1.0 / (1 - psi))

        # Chain rule to raw scores
        g_loglam = -lam * dL_dlam  # −∂L/∂z₁
        g_psilog = -psi * (1 - psi) * dL_dpsi  # −∂L/∂z₂

        # Positive diagonal Hessian approximations
        h_loglam = np.clip(lam, hess_floor, None)  # Fisher info of λ
        h_psilog = np.clip(psi * (1 - psi), hess_floor, None)

        # ---------- stack & flatten for LightGBM ----------------------
        grad = self._stabilize_derivative(np.hstack([g_loglam, g_psilog])).ravel(
            order="F"
        )
        hess = self._stabilize_derivative(np.hstack([h_loglam, h_psilog])).ravel(
            order="F"
        )
        return grad, hess

    # ------------------------------------------------------------------ #
    # METRIC  (average log-likelihood, higher = better)
    # ------------------------------------------------------------------ #
    def average_log_likelihood(self, y_pred: np.ndarray, dval: lgb.Dataset):
        y_true = np.asarray(dval.get_label()).astype(np.int32).reshape(-1, 1)
        lam, psi = self._raw_to_params(y_pred)
        y_zero = y_true == 0
        log_pmf = np.where(
            y_zero,
            np.log(psi + (1 - psi) * np.exp(-lam)),
            np.log1p(-psi) - lam + y_true * np.log(lam) - gammaln(y_true + 1),
        )
        return "avg_log_likelihood", np.mean(log_pmf), True

    # ------------------------------------------------------------------ #
    # INITIAL SCORES
    # ------------------------------------------------------------------ #
    def get_init_scores(self, n: int):
        z_lam = np.full((n, 1), np.log(self.lam_init + 1e-12), dtype=np.float32)
        z_psi = np.full((n, 1), self.psi_init, dtype=np.float32)
        return np.hstack([z_lam, z_psi])

    # ------------------------------------------------------------------ #
    # FIT
    # ------------------------------------------------------------------ #
    def fit(self, d_train: dict, d_val: Optional[dict] = None):
        if d_val is None:
            self.y_transformer.fit(np.concatenate([d_train["y"]], axis=0))
        else:
            self.y_transformer.fit(np.concatenate([d_train["y"], d_val["y"]], axis=0))
        y_train = self.y_transformer.transform(d_train["y"])

        # Simple empirical initialisation
        y_int = y_train.astype(int).ravel()
        self.lam_init = y_int.mean() + 1e-3
        p_zero = (y_int == 0).mean()
        self.psi_init = np.log(p_zero / (1 - p_zero + 1e-12))  # logit

        train_data = lgb.Dataset(
            d_train["x"].astype(np.float32),
            label=y_train.astype(np.int32),
            init_score=self.get_init_scores(len(y_train)),
        )

        valid_sets = [train_data]
        if d_val is not None:
            y_val = self.y_transformer.transform(d_val["y"])
            valid_sets.append(
                lgb.Dataset(
                    d_val["x"].astype(np.float32),
                    label=y_val.astype(np.int32),
                    init_score=self.get_init_scores(len(y_val)),
                    reference=train_data,
                )
            )

        print(
            "Start model fitting at",
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        logging_dict = {}
        self.model = lgb.train(
            self.lgb_params,
            train_data,
            num_boost_round=self.lgb_params["num_boost_round"],
            valid_sets=valid_sets,
            feval=self.average_log_likelihood,
            callbacks=[
                lgb.early_stopping(self.lgb_params["early_stopping_nb"]),
                lgb.log_evaluation(period=100),
                lgb.record_evaluation(logging_dict),
            ],
            keep_training_booster=True,
        )
        return logging_dict

    # ------------------------------------------------------------------ #
    # PREDICTION
    # ------------------------------------------------------------------ #
    def predict_parameters(self, X: np.ndarray, raw_iteration: str = "best"):
        if self.model is None:
            raise ValueError("Model not fitted")
        n_iter = (
            self.model.best_iteration
            if raw_iteration == "best"
            else (
                self.model.current_iteration()
                if raw_iteration == "last"
                else raw_iteration
            )
        )
        raw = self.model.predict(np.asarray(X), num_iteration=n_iter, raw_score=True)
        raw = raw.reshape(len(X), self.n_outputs) + self.get_init_scores(len(X))
        return self._raw_to_params(raw)

    def predict_cdf(
        self, X: np.ndarray, y_grid: np.ndarray, raw_iteration: str = "best"
    ):
        """
        Return F_Y(y | X) on an arbitrary grid y_grid (float or int).

        For a discrete distribution the CDF is right-continuous:
            F(y) = P(Y ≤ ⌊y⌋)           for y ≥ 0
                = 0                    for y < 0
        and for a ZIP:
            P(Y ≤ m) = ψ + (1-ψ)·PoisCDF(m ; λ)   whenever m ≥ 0
        """
        # ------------------------------------------------------------------
        # 1. parameters
        lam, psi = self.predict_parameters(X, raw_iteration=raw_iteration)  # (n,1)
        # ------------------------------------------------------------------
        # 2. prepare grid  (vectorised)
        if not hasattr(self.y_transformer, "classes_"):
            raise ValueError("Model must be fitted before prediction")
        # pyre-ignore[16]: `classes_` exists on fitted LabelEncoder
        classes = self.y_transformer.classes_.astype(float)  # (k,)
        # right-continuous → searchsorted(side="right") − 1
        m_grid = np.searchsorted(classes, y_grid, side="right") - 1  # (g,)
        neg_mask = y_grid < 0  # bool mask (g,)

        # ------------------------------------------------------------------
        # 3. Poisson CDF for each (sample, grid-point)
        pois_cdf = stats.poisson(mu=lam).cdf(m_grid[None, :])  # (n,g)

        # ------------------------------------------------------------------
        # 4. ZIP CDF
        cdf = psi + (1.0 - psi) * pois_cdf  # (n,g)
        # correct for y < 0  →  F=0
        if neg_mask.any():
            cdf[:, neg_mask] = 0.0
        np.clip(cdf, 0.0, 1.0, out=cdf)
        return cdf
