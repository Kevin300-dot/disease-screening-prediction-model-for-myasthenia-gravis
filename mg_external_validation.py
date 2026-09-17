"""
mg_external_validation.py
Leave-Diseases-Out (LDO) validation.

THE QUESTION THIS ANSWERS
-------------------------
Standard random train/test splitting shows every disease category to the model
during training. So a model can score well by learning "these 115 specific
diseases look like X, and MG is the leftover" -- pattern-matching the exact
differential set rather than learning MG-like features.

A real deployment never has that luxury: a new hospital's patient mix contains
conditions the model never trained on. Real external validation needs another
institution's data, which is not available here. LDO simulates the spirit of it
from a single dataset:

  1. Hold out a random subset of NON-MG disease categories entirely.
     Not a single patient from those diagnoses enters training.
  2. Train on MG + the remaining diseases.
  3. Test on held-out MG patients + patients from the unseen diseases.

If performance holds, the model is separating MG from "not MG" generally.
If it collapses, the earlier score was partly memorisation of the differential.

This is repeated over several random disease splits, because which diseases get
held out matters a lot -- holding out conditions that look nothing like MG is an
easy test; holding out neuromuscular mimics is a hard one.

IMPORTANT: uses the classical twin architecture, not the quantum hybrid, for
speed. Each repeat trains a model from scratch, and the quantum layer's
per-sample tf.map_fn makes that far too slow to run many times. The twin was
already shown to be architecturally equivalent minus the quantum layer, so it
is a fair stand-in for the generalisation question. State this openly if asked.
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    roc_auc_score, average_precision_score, recall_score,
    confusion_matrix, balanced_accuracy_score,
)


def _early_stopping(patience=8):
    """
    Imported lazily so this module can be imported (and unit-tested) without
    TensorFlow installed. Any model object exposing .fit/.predict works here.
    """
    try:
        from tensorflow.keras.callbacks import EarlyStopping
        return [EarlyStopping(monitor="loss", patience=patience, restore_best_weights=True)]
    except ImportError:
        return []


def leave_diseases_out_validation(
    X_raw,
    y,
    diagnosis_series,
    build_model_fn,
    target_disease="Myasthenia gravis",
    holdout_fraction=0.20,
    n_repeats=5,
    n_selected_features=20,
    epochs=40,
    batch_size=64,
    min_sensitivity=0.80,
    random_state=42,
    verbose=True,
):
    """
    Run LDO validation over n_repeats random disease holdouts.

    X_raw            : raw feature DataFrame (pre-impute/scale/select)
    y                : binary MG labels aligned to X_raw
    diagnosis_series : the Diagnosis text column, aligned to X_raw
    build_model_fn   : callable(n_features) -> compiled Keras model

    Every repeat refits imputer, scaler AND feature selector on that repeat's
    training rows only, so the unseen-disease patients never influence
    preprocessing -- the same leakage discipline as the main pipeline.

    Returns (per_repeat_df, summary_dict).
    """
    X_raw = X_raw.reset_index(drop=True)
    y = pd.Series(np.asarray(y).astype(int)).reset_index(drop=True)
    diagnosis_series = pd.Series(diagnosis_series).reset_index(drop=True)

    non_mg_diseases = sorted(
        d for d in diagnosis_series.unique()
        if str(d).strip().lower() != target_disease.lower()
    )
    n_holdout = max(1, int(round(len(non_mg_diseases) * holdout_fraction)))

    if verbose:
        print("=" * 60)
        print("LEAVE-DISEASES-OUT (EXTERNAL-STYLE) VALIDATION")
        print("=" * 60)
        print(f"Non-MG disease categories available : {len(non_mg_diseases)}")
        print(f"Held out entirely per repeat        : {n_holdout}")
        print(f"Repeats                             : {n_repeats}")
        print()
        print("Diseases held out are invisible during training -- this tests whether")
        print("MG detection survives contact with conditions the model never saw.")
        print()

    rng = np.random.RandomState(random_state)
    rows = []

    for repeat in range(1, n_repeats + 1):
        held_out = list(rng.choice(non_mg_diseases, size=n_holdout, replace=False))
        is_held_out_disease = diagnosis_series.isin(held_out)

        mg_mask = y == 1
        mg_idx = np.where(mg_mask)[0]
        rng.shuffle(mg_idx)
        n_mg_test = max(1, int(round(len(mg_idx) * 0.20)))
        mg_test_idx = set(mg_idx[:n_mg_test])

        test_mask = np.array([
            (i in mg_test_idx) or (is_held_out_disease.iloc[i] and y.iloc[i] == 0)
            for i in range(len(y))
        ])
        train_mask = ~test_mask & ~is_held_out_disease.values

        X_tr_raw, y_tr = X_raw[train_mask], y[train_mask]
        X_te_raw, y_te = X_raw[test_mask], y[test_mask]

        if y_tr.sum() < 5 or y_te.sum() < 2 or len(np.unique(y_te)) < 2:
            if verbose:
                print(f"Repeat {repeat}: skipped (insufficient MG cases after split).")
            continue

        # Preprocessing refit inside the repeat -- no leakage from unseen diseases
        imp = SimpleImputer(strategy="median")
        X_tr = imp.fit_transform(X_tr_raw)
        X_te = imp.transform(X_te_raw)

        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)

        k = min(n_selected_features, X_tr.shape[1])
        sel = SelectKBest(score_func=mutual_info_classif, k=k)
        X_tr = sel.fit_transform(X_tr, y_tr)
        X_te = sel.transform(X_te)

        classes = np.unique(y_tr)
        weights = compute_class_weight("balanced", classes=classes, y=y_tr)
        class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

        model = build_model_fn(X_tr.shape[1])
        model.fit(
            X_tr, y_tr.values,
            epochs=epochs, batch_size=batch_size,
            class_weight=class_weight,
            callbacks=_early_stopping(patience=8),
            verbose=0,
        )

        prob = model.predict(X_te, verbose=0).ravel()

        # Threshold picked on TRAINING predictions -- the test set stays unseen.
        train_prob = model.predict(X_tr, verbose=0).ravel()
        thr = _pick_threshold(train_prob, y_tr.values, min_sensitivity)
        pred = (prob >= thr).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
        rows.append({
            "Repeat": repeat,
            "Held_Out_Diseases": "; ".join(map(str, held_out)),
            "N_Held_Out": len(held_out),
            "Train_N": int(train_mask.sum()),
            "Test_N": int(test_mask.sum()),
            "Test_MG": int(y_te.sum()),
            "Test_NonMG_Unseen": int((y_te == 0).sum()),
            "Threshold": float(thr),
            "ROC-AUC": float(roc_auc_score(y_te, prob)),
            "PR-AUC": float(average_precision_score(y_te, prob)),
            "Sensitivity": float(recall_score(y_te, pred, zero_division=0)),
            "Specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan,
            "Balanced Accuracy": float(balanced_accuracy_score(y_te, pred)),
        })

        if verbose:
            r = rows[-1]
            print(f"Repeat {repeat}: ROC-AUC {r['ROC-AUC']:.3f} | "
                  f"Sens {r['Sensitivity']:.3f} | Spec {r['Specificity']:.3f} | "
                  f"test = {r['Test_MG']} MG vs {r['Test_NonMG_Unseen']} unseen-disease")

    per_repeat = pd.DataFrame(rows)
    if per_repeat.empty:
        return per_repeat, {"status": "no valid repeats"}

    metric_cols = ["ROC-AUC", "PR-AUC", "Sensitivity", "Specificity", "Balanced Accuracy"]
    summary = {
        "n_valid_repeats": int(len(per_repeat)),
        "mean": per_repeat[metric_cols].mean().to_dict(),
        "std": per_repeat[metric_cols].std().to_dict(),
        "min": per_repeat[metric_cols].min().to_dict(),
        "max": per_repeat[metric_cols].max().to_dict(),
    }
    return per_repeat, summary


def _pick_threshold(probabilities, y_true, min_sensitivity=0.80):
    """Highest-specificity threshold that still meets the sensitivity floor."""
    grid = np.arange(0.10, 0.91, 0.01)
    best_thr, best_spec = 0.50, -1.0
    for t in grid:
        pred = (probabilities >= t).astype(int)
        sens = recall_score(y_true, pred, zero_division=0)
        if sens < min_sensitivity:
            continue
        tn, fp, _, _ = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        if spec > best_spec:
            best_spec, best_thr = spec, float(t)
    if best_spec < 0:  # sensitivity floor unreachable -> fall back to balanced accuracy
        scores = [balanced_accuracy_score(y_true, (probabilities >= t).astype(int)) for t in grid]
        best_thr = float(grid[int(np.argmax(scores))])
    return best_thr


def compare_to_internal(ldo_summary, internal_metrics, metric="ROC-AUC", drop_threshold=0.05):
    """
    Compare LDO performance against the standard random-split result and return
    an honest verdict. A large drop is a finding worth reporting, not a failure
    to hide -- it quantifies how much of the internal score came from having
    seen the full differential during training.
    """
    if "mean" not in ldo_summary:
        return {"status": "unavailable", "interpretation": "LDO validation produced no valid repeats."}

    ldo_val = ldo_summary["mean"].get(metric, np.nan)
    ldo_sd = ldo_summary["std"].get(metric, np.nan)
    internal_val = internal_metrics.get(metric, np.nan)
    drop = internal_val - ldo_val

    if np.isnan(drop):
        interp = "Comparison unavailable."
    elif drop <= 0:
        interp = (
            f"{metric} on unseen diseases ({ldo_val:.3f}) is at least as high as on the "
            f"standard split ({internal_val:.3f}). The model is not relying on having seen "
            "the full set of competing diagnoses."
        )
    elif drop < drop_threshold:
        interp = (
            f"{metric} falls only {drop:.3f} on unseen diseases ({ldo_val:.3f} vs "
            f"{internal_val:.3f}). Performance largely holds, which supports the claim that "
            "the model learned MG-like patterns rather than memorising the differential set."
        )
    else:
        interp = (
            f"{metric} falls {drop:.3f} on unseen diseases ({ldo_val:.3f} vs {internal_val:.3f}). "
            "A meaningful part of the internal score came from having seen every competing "
            "diagnosis during training. The standard-split number therefore overstates how "
            "this model would perform on a new patient population."
        )

    return {
        "metric": metric,
        "internal": float(internal_val) if not np.isnan(internal_val) else None,
        "leave_diseases_out_mean": float(ldo_val) if not np.isnan(ldo_val) else None,
        "leave_diseases_out_sd": float(ldo_sd) if not np.isnan(ldo_sd) else None,
        "drop": float(drop) if not np.isnan(drop) else None,
        "interpretation": interp,
    }
