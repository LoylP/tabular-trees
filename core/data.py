"""Load the benchmark datasets (already curated by the authors) and prepare them.

Datasets come from the Hugging Face mirror of the benchmark:
https://huggingface.co/datasets/inria-soda/tabular-benchmark
They are cached in data/clf_num/ after the first download.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

HF_BASE = "https://huggingface.co/datasets/inria-soda/tabular-benchmark/resolve/main"

# Classification benchmark on numerical features (the one used in Section 5 of the paper).
CLF_NUM_DATASETS = [
    "electricity",
    "MagicTelescope",
    "bank-marketing",
    "california",
    "credit",
    "eye_movements",
    "house_16H",
    "pol",
    "heloc",
    "jannis",
    "MiniBooNE",
    "covertype",
    "Higgs",
    "default-of-credit-card-clients",
    "Diabetes130US",
    "Bioresponse",
]

# Large files: slow first download (the app still only uses a subsample).
LARGE_DATASETS = {"Higgs", "covertype", "MiniBooNE", "jannis", "Diabetes130US", "Bioresponse"}

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "clf_num"


def local_path(name: str) -> Path:
    return DATA_DIR / f"{name}.csv"


def hf_url(name: str) -> str:
    return f"{HF_BASE}/clf_num/{name}.csv"


def load_raw(name: str) -> pd.DataFrame:
    """Read data/clf_num/<name>.csv, downloading it from Hugging Face the first time."""
    path = local_path(name)
    if path.exists():
        return pd.read_csv(path)
    df = pd.read_csv(hf_url(name))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def guess_target(columns: list[str]) -> str:
    """Target column: a column named class/target/label/y if any, else the last column."""
    lower = {c.lower(): c for c in columns}
    for hint in ("class", "target", "label", "y"):
        if hint in lower:
            return lower[hint]
    return columns[-1]


@dataclass
class Prepared:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    class_names: list[str]
    n_rows_available: int


def prepare(df: pd.DataFrame, target: str, n_train: int, n_test: int, seed: int) -> Prepared:
    """Binary target, numeric features, stratified subsample, Gaussianized features.

    Features are Gaussianized with QuantileTransformer (Section 3.5 / Section 5 of the paper).
    Trees are invariant to this monotone transform, so all models share the same inputs.
    """
    y_raw = df[target]
    X = df.drop(columns=[target]).select_dtypes(include="number")
    if X.shape[1] < 2:
        raise ValueError("Cần ít nhất 2 cột số làm feature.")

    mask = X.notna().all(axis=1) & y_raw.notna()
    X, y_raw = X[mask], y_raw[mask]

    if y_raw.nunique() > 20:
        raise ValueError(
            f"Cột target '{target}' có {y_raw.nunique()} giá trị khác nhau — có vẻ đây không phải "
            "cột nhãn phân loại. Hãy chọn lại cột target ở sidebar."
        )

    # Binary classification on the two most frequent classes (Section 3.2).
    top2 = list(y_raw.value_counts().index[:2])
    if len(top2) < 2:
        raise ValueError("Cột target chỉ có 1 giá trị.")
    classes = sorted(top2, key=str)
    keep = y_raw.isin(classes)
    X, y_raw = X[keep], y_raw[keep]
    y = (y_raw == classes[1]).astype(int).to_numpy()

    n_rows = len(X)
    n_train = int(min(n_train, 0.7 * n_rows))
    n_test = int(min(n_test, n_rows - n_train))
    X_tr, X_te, y_tr, y_te = train_test_split(
        X.to_numpy(dtype=float), y,
        train_size=n_train, test_size=n_test,
        stratify=y, random_state=seed,
    )

    qt = QuantileTransformer(
        output_distribution="normal",
        n_quantiles=min(1000, n_train),
        random_state=seed,
    )
    X_tr = qt.fit_transform(X_tr)
    X_te = qt.transform(X_te)

    return Prepared(
        X_train=X_tr, X_test=X_te, y_train=y_tr, y_test=y_te,
        feature_names=list(X.columns), class_names=[str(c) for c in classes],
        n_rows_available=n_rows,
    )
