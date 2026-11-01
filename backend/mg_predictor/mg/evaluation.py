"""Metrics, threshold selection and all charts (confusion matrix, bar graph, pie chart)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)

PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2"]


# ----------------------------------------------------------------- thresholds
def choose_threshold(y_true, scores, strategy: str = "youden") -> float:
    """Pick the decision threshold from out-of-fold scores.

    'youden' maximises sensitivity + specificity - 1, which avoids both the
    "flag everyone" and "flag no-one" extremes that plain accuracy / F-scores
    drift to when only ~1% of rows are positive.
    """
    y_true = np.asarray(y_true)
    fpr, tpr, thr = roc_curve(y_true, scores)
    if strategy == "youden":
        j = tpr - fpr
        j[~np.isfinite(thr)] = -np.inf
        return float(thr[int(np.argmax(j))])
    raise ValueError(f"Unknown threshold strategy: {strategy}")


# -------------------------------------------------------------------- metrics
def compute_metrics(y_true, y_pred, y_score) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "avg_precision": average_precision_score(y_true, y_score),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


# --------------------------------------------------------------------- charts
def plot_confusion_matrices(results: dict, path):
    """One confusion matrix per algorithm (counts + row-wise %)."""
    n = len(results)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, r) in zip(axes, results.items()):
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
        pct = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n({pct[i, j]:.0%})", ha="center", va="center",
                        color="white" if pct[i, j] > 0.5 else "black", fontsize=12)
        ax.set_xticks([0, 1], ["No MG", "MG"])
        ax.set_yticks([0, 1], ["No MG", "MG"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(name, fontweight="bold")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Confusion matrices (hold-out test set, % = share of each actual class)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_metric_bars(results: dict, path):
    """Grouped bar graph: accuracy, precision, recall, F1 and ROC-AUC for every model."""
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    names = list(results)
    x = np.arange(len(names))
    w = 0.16
    fig, ax = plt.subplots(figsize=(13, 6))
    for i, m in enumerate(metrics):
        vals = [results[n][m] for n in names]
        bars = ax.bar(x + (i - 2) * w, vals, w, label=m.replace("_", " ").title(), color=PALETTE[i])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center",
                    fontsize=8, rotation=90)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison on the hold-out test set", fontweight="bold")
    ax.legend(ncol=5, loc="upper center")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_accuracy_precision_bars(results: dict, path):
    """Simple two-panel bar graph focused on the two headline metrics."""
    names = list(results)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, m in zip(axes, ["accuracy", "precision"]):
        vals = [results[n][m] for n in names]
        bars = ax.bar(names, vals, color=PALETTE[: len(names)])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center")
        ax.set_ylim(0, 1.1)
        ax.set_title(m.title(), fontweight="bold")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_class_pie(breakdown: dict, path):
    """Pie chart of the dataset composition."""
    labels, sizes = list(breakdown), list(breakdown.values())
    total = sum(sizes)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    wedges, _ = ax.pie(sizes, colors=["#E45756", "#54A24B", "#4C78A8"], startangle=90,
                       explode=[0.12, 0, 0], wedgeprops=dict(edgecolor="white"))
    ax.legend(wedges, [f"{l}: {s} ({s / total:.1%})" for l, s in zip(labels, sizes)],
              loc="lower center", bbox_to_anchor=(0.5, -0.12))
    ax.set_title("Dataset composition", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_outcome_pie(name: str, r: dict, path):
    """Pie chart of the best model's test-set outcomes (TN / FP / FN / TP)."""
    labels = ["True negative", "False positive", "False negative", "True positive"]
    sizes = [r["tn"], r["fp"], r["fn"], r["tp"]]
    colors = ["#54A24B", "#F58518", "#E45756", "#4C78A8"]
    total = sum(sizes)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    wedges, _ = ax.pie(sizes, colors=colors, startangle=90, wedgeprops=dict(edgecolor="white"))
    ax.legend(wedges, [f"{l}: {s} ({s / total:.1%})" for l, s in zip(labels, sizes)],
              loc="lower center", bbox_to_anchor=(0.5, -0.12))
    ax.set_title(f"Prediction outcomes - {name} (test set)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(curves: dict, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for (name, (fpr, tpr, auc)), c in zip(curves.items(), PALETTE):
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{name} (AUC {auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (hold-out test set)", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feature_importance(names, values, title, path):
    order = np.argsort(values)
    fig, ax = plt.subplots(figsize=(9, 11))
    ax.barh(np.array(names)[order], np.array(values)[order], color="#4C78A8")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Importance")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
