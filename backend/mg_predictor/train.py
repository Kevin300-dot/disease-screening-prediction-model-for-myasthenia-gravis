"""Train and evaluate all five models, save charts, metrics and the model bundle.

Usage:  python train.py
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, learning_curve, train_test_split

from mg import config as C
from mg import evaluation as E
from mg.data import build_xy, dataset_breakdown, load_dataset
from mg.models import PlattCalibrator, get_models, positive_score


def out_of_fold_scores(model, X, y):
    """Cross-validated scores on the development set (no row scored by a model that saw it)."""
    skf = StratifiedKFold(C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)
    oof = np.zeros(len(y))
    for tr, va in skf.split(X, y):
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        oof[va] = positive_score(m, X.iloc[va])
    return oof


def compute_learning_curves(X_dev, y_dev) -> dict:
    """
    Accuracy-vs-training-set-size for all five models (a LEARNING curve, not
    an "epoch" curve -- epochs are specific to iterative models like neural
    networks re-passing over the same data; four of these five algorithms
    don't train in epochs at all). Scored on held-out CV folds using ROC-AUC
    rather than raw accuracy, since accuracy is misleading at ~1% prevalence
    (a model that always predicts "no MG" already scores ~99% accuracy).
    This directly answers "does this model generalize, or does it need more
    data" -- the actual question a learning curve is for.
    """
    train_sizes = np.linspace(0.1, 1.0, 8)
    skf = StratifiedKFold(C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)
    curves = {}
    for name, model in get_models().items():
        sizes, train_scores, val_scores = learning_curve(
            clone(model), X_dev, y_dev,
            train_sizes=train_sizes, cv=skf, scoring="roc_auc",
            shuffle=True, random_state=C.RANDOM_STATE, n_jobs=-1,
        )
        curves[name] = {
            "train_sizes": sizes.tolist(),
            "train_roc_auc_mean": train_scores.mean(axis=1).tolist(),
            "val_roc_auc_mean": val_scores.mean(axis=1).tolist(),
            "val_roc_auc_std": val_scores.std(axis=1).tolist(),
        }
    return curves


def main():
    C.MODEL_DIR.mkdir(exist_ok=True)
    C.OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_dataset()
    X, y = build_xy(df)
    print(f"Rows: {len(df)} | MG cases: {int(y.sum())} ({y.mean():.2%}) | features: {len(C.FEATURES)}")

    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=C.TEST_SIZE, stratify=y, random_state=C.RANDOM_STATE
    )
    print(f"Development set: {len(y_dev)} ({int(y_dev.sum())} MG) | Test set: {len(y_test)} ({int(y_test.sum())} MG)\n")

    fitted, thresholds, calibrators, results, oof_auc, roc_curves = {}, {}, {}, {}, {}, {}

    for name, model in get_models().items():
        # 1) honest out-of-fold scores -> decision threshold + probability calibration
        oof = out_of_fold_scores(model, X_dev, y_dev)
        oof_auc[name] = roc_auc_score(y_dev, oof)
        calibrators[name] = PlattCalibrator().fit(oof, y_dev)
        thr = calibrators[name].threshold_at(0.5)   # decision rule: calibrated P(MG) >= 50%
        thresholds[name] = thr

        # 2) final fit on the whole development set, evaluate once on the untouched test set
        final = clone(model).fit(X_dev, y_dev)
        fitted[name] = final
        scores = positive_score(final, X_test)
        pred = (scores >= thr).astype(int)
        results[name] = E.compute_metrics(y_test, pred, scores)
        fpr, tpr, _ = roc_curve(y_test, scores)
        roc_curves[name] = (fpr, tpr, results[name]["roc_auc"])

    best = max(oof_auc, key=oof_auc.get)   # selected on CV only - never on the test set

    print("Computing learning curves (accuracy vs. training-set size) for all 5 models...")
    curves = compute_learning_curves(X_dev, y_dev)
    with open(C.OUTPUT_DIR / "learning_curve.json", "w") as f:
        json.dump({"metric": "roc_auc", "n_dev": int(len(y_dev)), "curves": curves}, f, indent=2)

    # ------------------------------------------------------------- report
    table = pd.DataFrame(results).T[
        ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "tn", "fp", "fn", "tp"]
    ]
    table.insert(0, "cv_roc_auc", pd.Series(oof_auc))
    table.to_csv(C.OUTPUT_DIR / "metrics.csv", float_format="%.4f")
    print(table.round(3).to_string())
    print(f"\nBest model by cross-validated ROC-AUC: {best}\n")

    # ------------------------------------------------------------- charts
    E.plot_confusion_matrices(results, C.OUTPUT_DIR / "confusion_matrices.png")
    E.plot_metric_bars(results, C.OUTPUT_DIR / "model_comparison_bar.png")
    E.plot_accuracy_precision_bars(results, C.OUTPUT_DIR / "accuracy_precision_bar.png")
    E.plot_class_pie(dataset_breakdown(df), C.OUTPUT_DIR / "dataset_pie.png")
    E.plot_outcome_pie(best, results[best], C.OUTPUT_DIR / "outcome_pie.png")
    E.plot_roc(roc_curves, C.OUTPUT_DIR / "roc_curves.png")

    rf = fitted["Random Forest"]
    pretty = {**{k: v[0] for k, v in C.MG_SIGNS.items()}, **C.SYMPTOMS,
              "age": "Age", "gender_male": "Male sex", "duration_months": "Symptom duration"}
    E.plot_feature_importance(
        [pretty.get(c, c) for c in C.FEATURES], rf.feature_importances_,
        "Random Forest feature importance", C.OUTPUT_DIR / "feature_importance_bar.png",
    )

    # -------------------------------------------------------------- save
    bundle = {
        "features": C.FEATURES,
        "models": fitted,
        "thresholds": thresholds,
        "calibrators": calibrators,
        "best_model": best,
        "prevalence": float(y.mean()),
        "cv_roc_auc": oof_auc,
        "test_metrics": results,
        "n_train": int(len(y_dev)),
    }
    joblib.dump(bundle, C.BUNDLE_PATH)
    with open(C.OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump({"best_model": best, "prevalence": bundle["prevalence"],
                   "thresholds": thresholds, "cv_roc_auc": oof_auc, "test": results}, f, indent=2)
    print(f"Saved bundle -> {C.BUNDLE_PATH}\nSaved charts/metrics/learning_curve.json -> {C.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
