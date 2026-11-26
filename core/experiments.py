"""The experiments of Section 5, on one dataset at a time.

Each function returns a tidy DataFrame (one row per model x setting x seed)
so the app can plot it directly. Scores are raw test accuracies (not normalized),
because only one dataset is used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Prepared
from .models import fit_score
from .transforms import (
    add_uninformative_features,
    keep_top_k,
    random_rotation,
    remove_least_important,
    rf_importance_order,
    smooth_labels,
)


# ---------------------------------------------------------------- decision boundary (Figure 20)
def fit_two_feature_models(data: Prepared, f1: int, f2: int, seed: int):
    """Fit RandomForest and MLP on two features only."""
    cols = [f1, f2]
    Xtr, Xte = data.X_train[:, cols], data.X_test[:, cols]
    out = {}
    for name in ("RandomForest", "MLP"):
        out[name] = fit_score(name, Xtr, data.y_train, Xte, data.y_test, seed)
    return out


def decision_grid(model, xlim, ylim, resolution=200):
    gx = np.linspace(xlim[0], xlim[1], resolution)
    gy = np.linspace(ylim[0], ylim[1], resolution)
    xx, yy = np.meshgrid(gx, gy)
    proba = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    return gx, gy, proba.reshape(xx.shape)


# ---------------------------------------------------------------- Finding 1: smoothing (Figure 3)
def run_smoothing(data: Prepared, cov_mults, k_features, robust, models, seeds):
    rows, label_changes = [], []
    for seed in seeds:
        order = rf_importance_order(data.X_train, data.y_train, seed)
        k = min(k_features, data.X_train.shape[1])
        Xtr, Xte, _ = keep_top_k(data.X_train, data.X_test, order, k)
        for cm in cov_mults:
            y_smooth, changed = smooth_labels(Xtr, data.y_train, cm, robust=robust)
            label_changes.append({"lengthscale2": cm, "seed": seed, "labels_changed": changed})
            for name in models:
                r = fit_score(name, Xtr, y_smooth, Xte, data.y_test, seed)  # test labels untouched
                rows.append({"model": name, "lengthscale2": cm, "seed": seed,
                             "test_acc": r["test_acc"], "seconds": r["seconds"]})
    return pd.DataFrame(rows), pd.DataFrame(label_changes)


# ---------------------------------------------------------------- Finding 2: features (Figures 4, 5)
def run_add_features(data: Prepared, fracs, models, seeds):
    rows = []
    for seed in seeds:
        for frac in fracs:
            rng = np.random.default_rng(seed)
            Xtr, Xte = add_uninformative_features(data.X_train, data.X_test, frac, rng)
            for name in models:
                r = fit_score(name, Xtr, data.y_train, Xte, data.y_test, seed)
                rows.append({"model": name, "fraction": frac, "seed": seed,
                             "n_features": Xtr.shape[1], "test_acc": r["test_acc"]})
    return pd.DataFrame(rows)


def run_remove_features(data: Prepared, fracs, models, seeds, fit_on_removed=True):
    rows = []
    for seed in seeds:
        order = rf_importance_order(data.X_train, data.y_train, seed)
        for frac in fracs:
            (Xtr, Xte), (Rtr, Rte) = remove_least_important(data.X_train, data.X_test, order, frac)
            for name in models:
                r = fit_score(name, Xtr, data.y_train, Xte, data.y_test, seed)
                rows.append({"model": name, "fraction": frac, "seed": seed, "trained_on": "kept",
                             "n_features": Xtr.shape[1], "test_acc": r["test_acc"]})
            if fit_on_removed and Rtr.shape[1] > 0:
                # Figure 4: a GBT trained ONLY on the removed features
                r = fit_score("GradientBoostingTree", Rtr, data.y_train, Rte, data.y_test, seed)
                rows.append({"model": "GradientBoostingTree", "fraction": frac, "seed": seed,
                             "trained_on": "removed", "n_features": Rtr.shape[1],
                             "test_acc": r["test_acc"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Finding 3: rotation (Figure 6)
def run_rotation(data: Prepared, remove_half, models, seeds):
    rows = []
    for seed in seeds:
        Xtr, Xte = data.X_train, data.X_test
        if remove_half:  # Figure 6b: drop the least important half BEFORE rotating
            order = rf_importance_order(Xtr, data.y_train, seed)
            (Xtr, Xte), _ = remove_least_important(Xtr, Xte, order, 0.5)
        rng = np.random.default_rng(seed)
        Rtr, Rte = random_rotation(Xtr, Xte, rng)
        for condition, (a, b) in (("Không xoay", (Xtr, Xte)), ("Xoay ngẫu nhiên", (Rtr, Rte))):
            for name in models:
                r = fit_score(name, a, data.y_train, b, data.y_test, seed)
                rows.append({"model": name, "condition": condition, "seed": seed,
                             "test_acc": r["test_acc"]})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Mean and std of test accuracy over seeds."""
    g = df.groupby(by, as_index=False)["test_acc"]
    out = g.mean().rename(columns={"test_acc": "mean_acc"})
    out["std_acc"] = g.std()["test_acc"].fillna(0.0).to_numpy()
    return out
