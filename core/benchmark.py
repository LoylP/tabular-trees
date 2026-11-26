"""Benchmark curves of Figures 1-2 from a random-search results CSV.

Python port of the authors' R code (analyses/random_search_utils.R and
analyses/plot_utils.R::plot_aggregated_results):
1. Normalize scores per dataset: (score - quantile_q) / (max - quantile_q),
   q = 0.1 for classification, 0.5 for regression; negative test scores clipped to 0
   for regression.
2. Repeat n_shuffles times: shuffle the order of the random search iterations
   (default hyperparameters always first), keep for every iteration the TEST score of
   the best model so far on VALIDATION, carry it forward.
   For fairness, every model on a dataset is cut to the smallest number of iterations
   available on that dataset.
3. Average over datasets; report mean / min / max over the shuffles.

Small simplification: the authors give the random iterations ranks in 1..n with a gap
where the default row was; here they simply get ranks 1..k after the default (rank 0).

Expected columns (format of the authors' benchmark_total.csv):
model_name, data__keyword, mean_val_score, mean_test_score
optional: hp ("default" for default hyperparameters), mean_time, benchmark
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

REQUIRED = ["model_name", "data__keyword", "mean_val_score", "mean_test_score"]
OPTIONAL = ["hp", "mean_time", "benchmark"]

PAPER_RESULTS_URL = "https://figshare.com/ndownloader/files/40081681"


def read_results(source) -> pd.DataFrame:
    """source: path or file-like. Only the needed columns are read (the file is large)."""
    wanted = set(REQUIRED + OPTIONAL)
    df = pd.read_csv(source, usecols=lambda c: c in wanted, low_memory=False)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"File kết quả thiếu cột: {missing}")
    return df


def normalize_scores(df: pd.DataFrame, quantile: float, truncate: bool) -> pd.DataFrame:
    df = df.copy()
    if "mean_time" in df.columns:
        df = df[df["mean_time"].notna()]
    for col in ("mean_val_score", "mean_test_score"):
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        grouped = df.groupby("data__keyword")[col]
        q = grouped.transform(lambda s: s.quantile(quantile))
        m = grouped.transform("max")
        df[col] = ((df[col] - q) / (m - q)).replace([np.inf, -np.inf], np.nan)
    if truncate:
        df["mean_test_score"] = df["mean_test_score"].clip(lower=0)
    return df


def _shuffled_order(n_rows, is_default, rng):
    perm = rng.permutation(n_rows)
    defaults = [i for i in perm if is_default[i]][:1]  # one default run, at rank 0
    randoms = [i for i in perm if not is_default[i]]
    order = np.array(defaults + randoms, dtype=int)
    start = 0 if defaults else 1
    ranks = np.arange(start, start + len(order))
    return order, ranks


def random_search_curves(df: pd.DataFrame, n_shuffles=15, max_iter=400, seed=0):
    """Return (curves, ranks): curves[model] has shape (n_shuffles, n_ranks), rank 0 = default."""
    rng = np.random.default_rng(seed)
    has_hp = "hp" in df.columns

    groups = {}
    for (ds, model), g in df.groupby(["data__keyword", "model_name"], sort=True):
        val = g["mean_val_score"].to_numpy(float)
        test = g["mean_test_score"].to_numpy(float)
        is_def = (g["hp"].astype(str) == "default").to_numpy() if has_hp else np.zeros(len(g), bool)
        groups[(ds, model)] = (val, test, is_def)

    models = sorted({m for _, m in groups})
    datasets = sorted({d for d, _ in groups})
    # equalize_n_iteration: smallest number of valid runs among the models of each dataset
    limit = {
        ds: min(int(np.sum(~np.isnan(groups[(ds, m)][0]))) for m in models if (ds, m) in groups)
        for ds in datasets
    }

    G = max_iter + 1
    curves = {m: np.full((n_shuffles, G), np.nan) for m in models}
    global_max = 0
    for s in range(n_shuffles):
        sums = {m: np.zeros(G) for m in models}
        counts = {m: np.zeros(G) for m in models}
        for (ds, m), (val, test, is_def) in groups.items():
            order, ranks = _shuffled_order(len(val), is_def, rng)
            keep = ranks <= min(limit[ds], max_iter)
            order, ranks = order[keep], ranks[keep]
            if len(order) == 0:
                continue
            global_max = max(global_max, int(ranks[-1]))
            v = np.where(np.isnan(val[order]), -np.inf, val[order])
            t = test[order]
            running_max = np.maximum.accumulate(v)
            improved = np.r_[True, running_max[1:] > running_max[:-1]]
            best_idx = np.maximum.accumulate(np.where(improved, np.arange(len(v)), 0))
            curve = np.full(G, np.nan)
            curve[ranks] = t[best_idx]
            curve = pd.Series(curve).ffill().to_numpy()  # carry the best value forward
            ok = ~np.isnan(curve)
            sums[m][ok] += curve[ok]
            counts[m][ok] += 1
        for m in models:
            with np.errstate(invalid="ignore", divide="ignore"):
                curves[m][s] = sums[m] / np.where(counts[m] > 0, counts[m], np.nan)

    n_ranks = global_max + 1
    curves = {m: c[:, :n_ranks] for m, c in curves.items()}
    return curves, np.arange(n_ranks), len(datasets)


def summarize_curves(curves):
    """mean / min / max over shuffles for each model."""
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for m, c in curves.items():
            out[m] = {"mean": np.nanmean(c, axis=0), "min": np.nanmin(c, axis=0),
                      "max": np.nanmax(c, axis=0)}
    return out
