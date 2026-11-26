"""Data transformations used in Section 5 of the paper.

Adapted from the authors' src/data_transforms.py
(https://github.com/LeoGrin/tabular-benchmark):
- remove_features_rf / select_features_rf  -> rf_importance_order, keep_top_k, remove_least_important
- add_uninformative_features              -> add_uninformative_features
- apply_random_rotation                   -> random_rotation
- remove_high_frequency_from_train        -> smooth_labels (vectorized rewrite, same formula)
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular
from scipy.stats import special_ortho_group
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.ensemble import RandomForestClassifier


# ---------------------------------------------------------------- feature importance
def rf_importance_order(X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Feature indices sorted from LEAST to MOST important (Random Forest importance)."""
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed)
    rf.fit(X, y)
    return np.argsort(rf.feature_importances_)


def keep_top_k(X_train, X_test, order, k):
    cols = order[-k:]
    return X_train[:, cols], X_test[:, cols], cols


def remove_least_important(X_train, X_test, order, frac):
    """Drop the `frac` least important features. Also returns the removed part (Figure 4)."""
    d = X_train.shape[1]
    n_remove = min(int(frac * d), d - 1)  # keep at least one feature
    removed, kept = order[:n_remove], order[n_remove:]
    return (X_train[:, kept], X_test[:, kept]), (X_train[:, removed], X_test[:, removed])


# ---------------------------------------------------------------- uninformative features
def add_uninformative_features(X_train, X_test, frac, rng):
    """Add int(frac * d) Gaussian columns, independent of the target.

    Like the authors' code, each new column imitates the mean and interquartile range
    of a randomly chosen existing column (for a Gaussian, IQR = 1.349 sigma).
    """
    d = X_train.shape[1]
    n_new = int(frac * d)
    if n_new == 0:
        return X_train, X_test
    cols = rng.choice(d, n_new)
    mean = X_train.mean(axis=0)[cols]
    iqr = np.quantile(X_train, 0.75, axis=0) - np.quantile(X_train, 0.25, axis=0)
    std = iqr[cols] / 1.349
    new_tr = rng.normal(mean, std, size=(X_train.shape[0], n_new))
    new_te = rng.normal(mean, std, size=(X_test.shape[0], n_new))
    return np.hstack([X_train, new_tr]), np.hstack([X_test, new_te])


# ---------------------------------------------------------------- rotation
def random_rotation(X_train, X_test, rng):
    """Multiply train and test by the SAME random rotation matrix (special orthogonal group)."""
    d = X_train.shape[1]
    if d < 2:
        return X_train, X_test
    R = special_ortho_group.rvs(d, random_state=rng)
    return X_train @ R, X_test @ R


# ---------------------------------------------------------------- smoothing (Finding 1)
def smooth_labels(X, y, cov_mult, robust=False, chunk=1024):
    """Gaussian kernel smoothing of the TRAIN labels (Appendix A.4.1).

    y_new(x_i) = sum_j K(x_i, x_j) y_j / sum_j K(x_i, x_j),
    K(x, x') = exp(-1/2 (x - x')^T Sigma^-1 (x - x')),  Sigma = cov_mult * Cov(X).
    cov_mult plays the role of the "(squared) lengthscale" on the x-axis of Figure 3.
    For classification the smoothed value is thresholded at 0.5, as in the authors' code.
    Returns (new_labels, fraction_of_labels_changed).
    """
    y = np.asarray(y)
    if cov_mult == 0:
        return y.copy(), 0.0

    est = MinCovDet(random_state=0) if robust else EmpiricalCovariance()
    try:
        cov = est.fit(X).covariance_
    except Exception:  # robust estimation can fail on degenerate data
        cov = EmpiricalCovariance().fit(X).covariance_
    cov = cov_mult * cov + 1e-9 * np.eye(X.shape[1])

    # Mahalanobis distance = Euclidean distance after whitening with Cholesky factor.
    L = np.linalg.cholesky(cov)
    Z = solve_triangular(L, X.T, lower=True).T
    sq = np.einsum("ij,ij->i", Z, Z)

    y_float = y.astype(float)
    smoothed = np.empty(len(y), dtype=float)
    for start in range(0, len(y), chunk):
        stop = min(start + chunk, len(y))
        d2 = sq[start:stop, None] + sq[None, :] - 2.0 * Z[start:stop] @ Z.T
        np.maximum(d2, 0.0, out=d2)
        d2 -= d2.min(axis=1, keepdims=True)  # numerical stability, cancels in the ratio
        w = np.exp(-0.5 * d2)
        smoothed[start:stop] = (w @ y_float) / w.sum(axis=1)

    new_y = (smoothed > 0.5).astype(int)
    return new_y, float(np.mean(new_y != y))
