"""Discrimination, stability and calibration metrics."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def auc(y, score):
    """AUC where a higher `score` means more likely to be bad."""
    return float(roc_auc_score(y, score))


def gini(y, score):
    return 2 * auc(y, score) - 1


def ks(y, score):
    """Kolmogorov Smirnov: the largest gap between the cumulative
    distributions of bads and goods."""
    y = np.asarray(y)
    score = np.asarray(score, dtype=float)
    order = np.argsort(score)
    y, score = y[order], score[order]

    cum_bad = np.cumsum(y) / y.sum()
    cum_good = np.cumsum(1 - y) / (1 - y).sum()

    # only compare at the last row of each distinct score so ties are not split
    last = np.r_[score[1:] != score[:-1], True]
    return float(np.max(np.abs(cum_bad[last] - cum_good[last])))


def psi(expected, actual, bins=10):
    """Population stability index between two score samples.

    Bin edges come from deciles of `expected` (the training scores).
    Rule of thumb: below 0.1 stable, 0.1 to 0.25 watch, above 0.25 shifted.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf

    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e = np.clip(e, 1e-6, None)
    a = np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_table(expected, actual, bins=10):
    """Same as psi() but returns the per bin breakdown."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf

    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    t = pd.DataFrame({
        "bin": [f"{lo:.0f} to {hi:.0f}" for lo, hi in zip(edges[:-1], edges[1:])],
        "expected_pct": e,
        "actual_pct": a,
    })
    ec, ac = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    t["psi"] = (ac - ec) * np.log(ac / ec)
    return t


def calibration_table(y, pd_hat, score, n=10):
    """Predicted vs actual bad rate by score decile (decile 1 = lowest score)."""
    df = pd.DataFrame({"y": y, "pd": pd_hat, "score": score})
    df["decile"] = pd.qcut(df["score"].rank(method="first"), n, labels=range(1, n + 1))
    t = df.groupby("decile", observed=True).agg(
        min_score=("score", "min"),
        max_score=("score", "max"),
        count=("y", "size"),
        predicted_bad_rate=("pd", "mean"),
        actual_bad_rate=("y", "mean"),
    )
    return t.reset_index()


def summary(y, pd_hat):
    return {"AUC": auc(y, pd_hat), "Gini": gini(y, pd_hat), "KS": ks(y, pd_hat)}
