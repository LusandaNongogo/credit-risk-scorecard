"""Charts used in the notebook and report."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

TRAIN_C = "#1f4e79"
TEST_C = "#c55a11"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)


def roc_plot(y_train, p_train, y_test, p_test, ax=None):
    ax = ax or plt.subplots(figsize=(5, 5))[1]
    for y, p, c, name in [(y_train, p_train, TRAIN_C, "Train"),
                          (y_test, p_test, TEST_C, "Test")]:
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=c, label=name)
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    ax.legend(loc="lower right")
    _style(ax)
    return ax


def ks_plot(y, score, ax=None, title="KS (test)"):
    """Cumulative share of bads and goods by score (low score first)."""
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    y = np.asarray(y)
    order = np.argsort(score)
    s, y = np.asarray(score)[order], y[order]
    cb = np.cumsum(y) / y.sum()
    cg = np.cumsum(1 - y) / (1 - y).sum()
    i = np.argmax(np.abs(cb - cg))
    ax.plot(s, cb, color=TEST_C, label="Bads")
    ax.plot(s, cg, color=TRAIN_C, label="Goods")
    ax.vlines(s[i], cg[i], cb[i], color="black", ls=":",
              label=f"KS = {cb[i] - cg[i]:.2f} at {s[i]:.0f}")
    ax.set(xlabel="Score", ylabel="Cumulative share", title=title)
    ax.legend()
    _style(ax)
    return ax


def score_dist_plot(score, y, ax=None, title="Score distribution (test)"):
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    score, y = np.asarray(score), np.asarray(y)
    bins = np.linspace(score.min(), score.max(), 25)
    ax.hist(score[y == 0], bins=bins, alpha=0.6, color=TRAIN_C, label="Good")
    ax.hist(score[y == 1], bins=bins, alpha=0.6, color=TEST_C, label="Bad")
    ax.set(xlabel="Score", ylabel="Count", title=title)
    ax.legend()
    _style(ax)
    return ax


def calibration_plot(cal, ax=None, title="Calibration by score decile (test)"):
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    x = cal["decile"].astype(int)
    ax.bar(x, cal["actual_bad_rate"], color=TRAIN_C, alpha=0.7, label="Actual")
    ax.plot(x, cal["predicted_bad_rate"], color=TEST_C, marker="o", label="Predicted")
    ax.set(xlabel="Score decile (1 = riskiest)", ylabel="Bad rate", title=title,
           xticks=list(x))
    ax.legend()
    _style(ax)
    return ax
