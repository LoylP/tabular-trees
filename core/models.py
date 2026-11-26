"""Models used in the demo (scikit-learn only, CPU, default-ish hyperparameters).

- GradientBoostingTree / RandomForest: scikit-learn defaults, as the paper's default configs.
- MLP: a plain MLP close to the paper's default MLP (Appendix A.3, Table 3: 4 layers of 256,
  lr 1e-3, batch 512), without dropout because scikit-learn's MLP has none.
  Note: trained with Adam, so it is only *approximately* rotation invariant.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

MODEL_NAMES = ["GradientBoostingTree", "RandomForest", "MLP"]

MODEL_COLORS = {
    "GradientBoostingTree": "#E4572E",
    "RandomForest": "#C9A227",
    "HistGradientBoostingTree": "#76B041",
    "XGBoost": "#2E933C",
    "FT Transformer": "#17BEBB",
    "Resnet": "#2F6FDB",
    "MLP": "#8E44AD",
    "SAINT": "#E83F6F",
}


def make_model(name: str, seed: int):
    if name == "GradientBoostingTree":
        return GradientBoostingClassifier(random_state=seed)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed)
    if name == "MLP":
        return MLPClassifier(
            hidden_layer_sizes=(256, 256, 256, 256),
            learning_rate_init=1e-3,
            batch_size=512,
            alpha=1e-5,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=seed,
        )
    raise ValueError(f"Unknown model {name}")


def fit_score(name, X_train, y_train, X_test, y_test, seed):
    """Fit one model, return it with train/test accuracy and fit time."""
    model = make_model(name, seed)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model.fit(X_train, y_train)
    seconds = time.perf_counter() - t0
    return {
        "model": model,
        "train_acc": float(np.mean(model.predict(X_train) == y_train)),
        "test_acc": float(np.mean(model.predict(X_test) == y_test)),
        "seconds": seconds,
    }
