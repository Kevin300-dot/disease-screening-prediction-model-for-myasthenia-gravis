"""
mg_evaluation.py
Statistical rigour add-ons for the MG screening pipeline:

  1. Bootstrap confidence intervals on every test-set metric.
     With ~30 test cases, a point estimate like "sensitivity = 90%" is close to
     meaningless on its own. Resampling the test set with replacement and
     recomputing the metric each time gives an honest interval instead.

  2. Calibration assessment for the hybrid model.
     Brier score is already computed, but it collapses calibration and
     discrimination into one number. A calibration curve shows whether a
     predicted probability of 0.7 actually corresponds to ~70% of those
     patients having MG, which is what matters if a clinician reads the
     number as a probability rather than just a ranking.

Imported by train_pipeline.py. No side effects on import.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve


# ---------------------------------------------------------------------------
# 1. Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def _metrics_from_arrays(y_true, y_pred, y_prob):
    """Compute the metric set on one (possibly resampled) set of arrays."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    # ROC-AUC / PR-AUC are undefined if a bootstrap resample happens to contain
    # only one class. Return NaN for those draws and drop them later rather
    # than crashing or silently substituting a wrong value.
    if len(np.unique(y_true)) < 2:
        roc, pr = np.nan, np.nan
    else:
        roc = roc_auc_score(y_true, y_prob)
        pr = average_precision_score(y_true, y_prob)

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": specificity,
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "ROC-AUC": roc,
        "PR-AUC": pr,
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Brier Score": brier_score_loss(y_true, y_prob),
    }


def bootstrap_metric_cis(
    y_true,
    y_prob,
    threshold,
    model_name="Model",
    n_boot=1000,
    alpha=0.05,
    random_state=42,
    stratified=True,
):
    """
    Percentile bootstrap CIs for every metric on the held-out test set.

    stratified=True resamples the positive and negative classes separately,
    keeping the number of MG cases fixed across draws. With very few positives
    this avoids a large share of resamples containing zero MG cases (which
    would make sensitivity and ROC-AUC undefined) and keeps the interval
    reflecting sampling noise rather than degenerate draws.

    Returns (summary_df, raw_draws_df). summary_df has one row per metric with
    the point estimate, CI bounds, and how many valid draws it rests on.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    rng = np.random.RandomState(random_state)

    point = _metrics_from_arrays(y_true, (y_prob >= threshold).astype(int), y_prob)

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n_total = len(y_true)

    draws = []
    for _ in range(n_boot):
        if stratified:
            idx = np.concatenate([
                rng.choice(pos_idx, size=len(pos_idx), replace=True),
                rng.choice(neg_idx, size=len(neg_idx), replace=True),
            ])
        else:
            idx = rng.choice(np.arange(n_total), size=n_total, replace=True)

        yt = y_true[idx]
        yp = y_prob[idx]
        draws.append(_metrics_from_arrays(yt, (yp >= threshold).astype(int), yp))

    draws_df = pd.DataFrame(draws)

    rows = []
    lo_q, hi_q = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    for metric in draws_df.columns:
        vals = draws_df[metric].dropna().values
        if len(vals) == 0:
            rows.append({
                "Model": model_name, "Metric": metric,
                "Estimate": point[metric], "CI_Lower": np.nan,
                "CI_Upper": np.nan, "Valid_Draws": 0,
            })
            continue
        rows.append({
            "Model": model_name,
            "Metric": metric,
            "Estimate": point[metric],
            "CI_Lower": np.percentile(vals, lo_q),
            "CI_Upper": np.percentile(vals, hi_q),
            "Valid_Draws": int(len(vals)),
        })

    return pd.DataFrame(rows), draws_df


def format_ci_table(ci_df, decimals=3):
    """Render the CI table as 'estimate (95% CI: lo-hi)' strings for slides."""
    out = ci_df.copy()
    out["Reported"] = out.apply(
        lambda r: (
            f"{r['Estimate']:.{decimals}f} "
            f"(95% CI: {r['CI_Lower']:.{decimals}f}-{r['CI_Upper']:.{decimals}f})"
            if pd.notna(r["CI_Lower"]) else f"{r['Estimate']:.{decimals}f} (CI undefined)"
        ),
        axis=1,
    )
    return out[["Model", "Metric", "Reported", "Valid_Draws"]]


def bootstrap_metric_difference(
    y_true, prob_a, thr_a, prob_b, thr_b,
    metric="ROC-AUC", n_boot=1000, alpha=0.05, random_state=42,
):
    """
    Paired bootstrap CI on the difference (model A - model B) for one metric.

    Both models are scored on the SAME resampled rows each draw, which is what
    makes the comparison paired. This is the honest way to answer "is the
    hybrid actually better than the classical twin, or is the gap noise?" --
    if the interval spans zero, the gap is not distinguishable from noise.
    """
    y_true = np.asarray(y_true).astype(int)
    prob_a = np.asarray(prob_a).astype(float)
    prob_b = np.asarray(prob_b).astype(float)
    rng = np.random.RandomState(random_state)

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    obs = (
        _metrics_from_arrays(y_true, (prob_a >= thr_a).astype(int), prob_a)[metric]
        - _metrics_from_arrays(y_true, (prob_b >= thr_b).astype(int), prob_b)[metric]
    )

    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([
            rng.choice(pos_idx, size=len(pos_idx), replace=True),
            rng.choice(neg_idx, size=len(neg_idx), replace=True),
        ])
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        a = _metrics_from_arrays(yt, (prob_a[idx] >= thr_a).astype(int), prob_a[idx])[metric]
        b = _metrics_from_arrays(yt, (prob_b[idx] >= thr_b).astype(int), prob_b[idx])[metric]
        if pd.isna(a) or pd.isna(b):
            continue
        diffs.append(a - b)

    diffs = np.asarray(diffs)
    if len(diffs) == 0:
        return {"metric": metric, "observed_difference": float(obs),
                "ci_lower": np.nan, "ci_upper": np.nan,
                "valid_draws": 0, "crosses_zero": True, "interpretation": "Too few valid draws."}

    lo = float(np.percentile(diffs, 100 * (alpha / 2)))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    crosses_zero = lo <= 0.0 <= hi

    if crosses_zero:
        interp = (
            f"The {metric} difference of {obs:+.3f} has a 95% CI of [{lo:+.3f}, {hi:+.3f}], "
            "which includes zero. The two models are not distinguishable on this test set."
        )
    elif obs > 0:
        interp = (
            f"Model A leads on {metric} by {obs:+.3f} (95% CI [{lo:+.3f}, {hi:+.3f}]), "
            "with the interval entirely above zero."
        )
    else:
        interp = (
            f"Model B leads on {metric} by {-obs:.3f} (95% CI [{lo:+.3f}, {hi:+.3f}]), "
            "with the interval entirely below zero."
        )

    return {
        "metric": metric, "observed_difference": float(obs),
        "ci_lower": lo, "ci_upper": hi,
        "valid_draws": int(len(diffs)), "crosses_zero": bool(crosses_zero),
        "interpretation": interp,
    }


# ---------------------------------------------------------------------------
# 2. Calibration
# ---------------------------------------------------------------------------

def calibration_analysis(y_true, y_prob, n_bins=5, strategy="quantile"):
    """
    Bin predictions and compare mean predicted probability against the observed
    positive rate in each bin.

    n_bins defaults to 5, not the usual 10: with ~30 test cases, 10 bins leaves
    ~3 patients per bin and the curve becomes pure noise. 'quantile' strategy
    puts an equal number of patients in each bin rather than equal-width
    probability ranges, which is more stable when predictions cluster.

    Also returns ECE (Expected Calibration Error): the average gap between
    predicted and observed rates, weighted by bin size. Lower is better.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    n_bins = max(2, min(n_bins, len(np.unique(y_prob))))
    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy=strategy
    )

    # Bin membership counts for ECE weighting
    if strategy == "quantile":
        edges = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
        edges = np.unique(edges)
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    bin_ids = np.clip(np.digitize(y_prob, edges[1:-1], right=True), 0, len(edges) - 2)
    counts, gaps = [], []
    for b in range(len(edges) - 1):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        counts.append(mask.sum())
        gaps.append(abs(y_prob[mask].mean() - y_true[mask].mean()))

    counts = np.asarray(counts, dtype=float)
    ece = float(np.sum(counts / counts.sum() * np.asarray(gaps))) if len(counts) else np.nan

    curve_df = pd.DataFrame({
        "Mean_Predicted_Probability": prob_pred,
        "Observed_Positive_Rate": prob_true,
    })

    return {
        "curve": curve_df,
        "ece": ece,
        "brier": float(brier_score_loss(y_true, y_prob)),
        "n_bins_used": int(n_bins),
        "strategy": strategy,
    }


def plot_calibration(calib_results, model_name="Hybrid QML", ax=None, save_path=None):
    """Reliability diagram: perfect calibration is the diagonal."""
    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        created = True

    curve = calib_results["curve"]
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfect calibration")
    ax.plot(
        curve["Mean_Predicted_Probability"],
        curve["Observed_Positive_Rate"],
        marker="o", linewidth=2, label=model_name,
    )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive rate")
    ax.set_title(
        f"Calibration -- {model_name}\n"
        f"ECE = {calib_results['ece']:.3f} | Brier = {calib_results['brier']:.3f} "
        f"({calib_results['n_bins_used']} bins)"
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    if save_path and created:
        plt.tight_layout()
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
    return ax


def interpret_calibration(calib_results):
    """One honest sentence about what the calibration numbers mean."""
    ece = calib_results["ece"]
    if np.isnan(ece):
        return "Calibration could not be assessed -- too few distinct predicted probabilities."
    if ece < 0.05:
        quality = "well calibrated: predicted probabilities track observed rates closely"
    elif ece < 0.15:
        quality = "moderately calibrated: probabilities are usable as rough risk estimates but not exact"
    else:
        quality = (
            "poorly calibrated: the predicted numbers should be treated as a ranking score, "
            "not as literal probabilities"
        )
    return (
        f"Expected Calibration Error = {ece:.3f}, so the model is {quality}. "
        "Note this is measured on a small test set, so the calibration curve itself is uncertain."
    )
