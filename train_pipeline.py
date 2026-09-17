# Cell 1: Imports and basic setup
import os
import json
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pennylane as qml
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
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
    roc_curve,
    precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# --- Rigour / explainability add-ons (local modules) ---
from mg_evaluation import (
    bootstrap_metric_cis,
    format_ci_table,
    bootstrap_metric_difference,
    calibration_analysis,
    plot_calibration,
    interpret_calibration,
)
from mg_explain import (
    local_feature_contributions,
    explain_patient_text,
    compare_local_vs_global,
    generate_local_recommendations,
    recommendations_to_dataframe,
)
from mg_external_validation import (
    leave_diseases_out_validation,
    compare_to_internal,
)

warnings.filterwarnings("ignore")
# Reproducibility
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
print("Libraries loaded.")
print("Random seed set to 42.")

# Cell 2: Load and inspect the dataset
DATA_FILE = "disease_dataset.csv"
# NOTE: if you keep the file as .xlsx instead of converting to .csv,
# replace the line below with: df = pd.read_excel(DATA_FILE)
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"Could not find '{DATA_FILE}'. "
        "Make sure the file is in the same folder as this notebook."
    )
df = pd.read_csv(DATA_FILE)
print("=" * 60)
print("DATASET CHECK")
print("=" * 60)
print(f"Rows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")
print()
print("Columns:")
print(", ".join(df.columns))
print()
print(f"Duplicate rows: {df.duplicated().sum()}")
missing = df.isna().sum()
missing = missing[missing > 0]
print()
if len(missing) == 0:
    print("Missing values: none")
else:
    print("Columns with missing values:")
    print(missing.sort_values(ascending=False))
print()
print("Diagnosis counts:")
print(df["Diagnosis"].value_counts().head(15))
print()
print("Dataset loaded successfully.")

# Cell 3: Create the Myasthenia Gravis target
# This dataset ships a precomputed target (MG_Label) plus explicit
# metadata columns telling you which columns are safe to use as predictors.
# We use MG_Label directly, but sanity-check it against the Diagnosis text
# in case a different copy of the file is loaded later.
TARGET_DISEASE = "Myasthenia gravis"
REQUIRED_COLS = ["MG_Label", "Diagnosis", "Patient_ID"]
for col in REQUIRED_COLS:
    if col not in df.columns:
        raise ValueError(f"The dataset is missing the expected column '{col}'.")

diagnosis_clean = df["Diagnosis"].astype(str).str.strip().str.lower()
diagnosis_derived_target = (diagnosis_clean == TARGET_DISEASE.lower()).astype(int)
if not (df["MG_Label"] == diagnosis_derived_target).all():
    raise ValueError(
        "MG_Label does not match Diagnosis=='Myasthenia gravis' for all rows. "
        "Check for a corrupted or mismatched dataset before continuing."
    )
y = df["MG_Label"].astype(int)

# --- Feature exclusion, driven by the dataset's own metadata flags ---
# Diagnosis / Patient_ID: explicitly flagged as "exclude from predictors".
# MG_Label / Use_for_MG_Model / Leakage_or_Availability_Flag /
#   Sickness_Duration_Months_Model_Status: target or documentation columns,
#   not real per-patient features -- must never enter X.
# Sickness_Duration_Months: the dataset's own flag says exclude it from
#   screening predictors UNLESS your study defines it as information
#   available at first workup. Default here is to exclude it (safer,
#   consistent with an "early/first-presentation screening" framing).
INCLUDE_SICKNESS_DURATION = False  # set True only if you can justify it's known at intake

ALWAYS_EXCLUDE = [
    "Diagnosis",
    "Patient_ID",
    "MG_Label",
    "Use_for_MG_Model",
    "Leakage_or_Availability_Flag",
    "Sickness_Duration_Months_Model_Status",
]
if not INCLUDE_SICKNESS_DURATION:
    ALWAYS_EXCLUDE.append("Sickness_Duration_Months")

X = df.drop(columns=ALWAYS_EXCLUDE, errors="ignore").copy()

print("=" * 60)
print("TARGET SETUP")
print("=" * 60)
print(f"Target disease: {TARGET_DISEASE}")
print(f"MG cases: {int(y.sum()):,}")
print(f"Other cases: {int((y == 0).sum()):,}")
print(f"MG prevalence: {y.mean() * 100:.2f}%")
print()
print(f"Columns excluded from features: {ALWAYS_EXCLUDE}")
print(f"Features before preprocessing: {X.shape[1]}")
print()
print("Target coding:")
print("  1 = Myasthenia gravis")
print("  0 = Other / non-MG")
if y.sum() < 10:
    raise ValueError("Too few Myasthenia gravis cases were found in the dataset.")
print()
print("Target created and verified against Diagnosis text.")

# Cell 4: Clean features and create train/test sets
# Encode Gender explicitly
if "Gender" in X.columns:
    gender_map = {
        "Male": 0,
        "Female": 1,
        "male": 0,
        "female": 1
    }
    X["Gender"] = X["Gender"].map(gender_map)
# Convert all remaining columns to numeric where possible
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")
# Remove columns that contain no useful information
constant_columns = [
    col for col in X.columns
    if X[col].nunique(dropna=True) <= 1
]
if constant_columns:
    X = X.drop(columns=constant_columns)
print("=" * 60)
print("PREPARING THE DATA")
print("=" * 60)
print(f"Removed constant columns: {len(constant_columns)}")
if constant_columns:
    print("Removed:")
    print(", ".join(constant_columns))
# Keep the original row indices so X and y stay aligned
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)
print()
print(f"Training records: {len(X_train):,}")
print(f"Test records: {len(X_test):,}")
print()
print("Training class counts:")
print(f"  MG: {int(y_train.sum()):,}")
print(f"  Non-MG: {int((y_train == 0).sum()):,}")
print()
print("Test class counts:")
print(f"  MG: {int(y_test.sum()):,}")
print(f"  Non-MG: {int((y_test == 0).sum()):,}")

# Cell 5: Imputation, scaling and feature selection
# Fill missing values using only information from the training set
imputer = SimpleImputer(strategy="median")
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)
# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)
# Select the most informative features.
# Mutual information works well for mixed and non-linear relationships.
N_SELECTED_FEATURES = min(20, X_train_scaled.shape[1])
selector = SelectKBest(
    score_func=mutual_info_classif,
    k=N_SELECTED_FEATURES
)
X_train_selected = selector.fit_transform(
    X_train_scaled,
    y_train
)
X_test_selected = selector.transform(
    X_test_scaled
)
selected_features = X_train.columns[selector.get_support()].tolist()
print("=" * 60)
print("FEATURE PREPARATION")
print("=" * 60)
print(f"Features before selection: {X_train_scaled.shape[1]}")
print(f"Features selected: {X_train_selected.shape[1]}")
print()
print("Selected features:")
for i, feature in enumerate(selected_features, 1):
    print(f"  {i:2}. {feature}")
print()
print("Missing values were filled using training-set medians.")
print("Features were then scaled and reduced to the most informative variables.")
print()
print("NOTE: This imputer/scaler/selector (fit on the FULL training set) is used")
print("for the final model (Cells 10-16) and for the held-out test set only.")
print("Cross-validation in Cell 9 refits its own preprocessing per fold to avoid leakage.")

# Cell 6: Classical baseline models
print("=" * 60)
print("CLASSICAL BASELINES")
print("=" * 60)
# Class weights are important because MG is the minority class.
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=y_train
)
class_weight_dict = {
    0: class_weights[0],
    1: class_weights[1]
}
print("Training Logistic Regression...")
lr = LogisticRegression(
    random_state=RANDOM_STATE,
    max_iter=1500,
    class_weight="balanced"
)
lr.fit(X_train_selected, y_train)
lr_pred = lr.predict(X_test_selected)
lr_prob = lr.predict_proba(X_test_selected)[:, 1]
print("Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=RANDOM_STATE,
    class_weight="balanced",
    n_jobs=-1
)
rf.fit(X_train_selected, y_train)
rf_pred = rf.predict(X_test_selected)
rf_prob = rf.predict_proba(X_test_selected)[:, 1]
print("Training SVM...")
svm = SVC(
    kernel="rbf",
    probability=True,
    class_weight="balanced",
    random_state=RANDOM_STATE
)
svm.fit(X_train_selected, y_train)
svm_pred = svm.predict(X_test_selected)
svm_prob = svm.predict_proba(X_test_selected)[:, 1]
def calculate_metrics(y_true, predictions, probabilities, model_name):
    cm = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Sensitivity": recall_score(y_true, predictions, zero_division=0),
        "Specificity": specificity,
        "F1-score": f1_score(y_true, predictions, zero_division=0),
        "Balanced Accuracy": balanced_accuracy_score(y_true, predictions),
        "ROC-AUC": roc_auc_score(y_true, probabilities),
        "PR-AUC": average_precision_score(y_true, probabilities),
        "MCC": matthews_corrcoef(y_true, predictions),
        "Brier Score": brier_score_loss(y_true, probabilities)
    }
baseline_metrics = [
    calculate_metrics(y_test, lr_pred, lr_prob, "Logistic Regression"),
    calculate_metrics(y_test, rf_pred, rf_prob, "Random Forest"),
    calculate_metrics(y_test, svm_pred, svm_prob, "SVM")
]
baseline_df = pd.DataFrame(baseline_metrics)
print()
print("Baseline results:")
print(baseline_df.round(4).to_string(index=False))
# Random Forest feature importance
feature_importance = pd.DataFrame({
    "Feature": selected_features,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False)
print()
print("Most important selected features:")
print(feature_importance.head(10).round(4).to_string(index=False))

# Cell 7-8-8B: Model architectures now live in mg_model.py
# (shared with app.py so the website rebuilds the exact same architecture
# before loading saved weights -- avoids fragile custom-layer deserialization).
from mg_model import (
    RANDOM_STATE as _MG_RANDOM_STATE,
    N_QUBITS,
    N_LAYERS,
    quantum_circuit,
    weight_shapes,
    build_hybrid_model,
    build_classical_twin_model
)
assert _MG_RANDOM_STATE == RANDOM_STATE, "RANDOM_STATE mismatch between script and mg_model.py"
print("=" * 60)
print("QUANTUM LAYER")
print("=" * 60)
print(f"Qubits: {N_QUBITS}")
print(f"Variational layers: {N_LAYERS}")
print(f"Trainable quantum parameters: {N_LAYERS * N_QUBITS}")
print("Simulator: PennyLane default.qubit")
print()
model = build_hybrid_model(X_train_selected.shape[1])
print("=" * 60)
print("HYBRID MODEL")
print("=" * 60)
print("Classical layers -> 2-dimensional representation -> quantum circuit -> MG probability")
print()
model.summary()
print()
print("=" * 60)
print("CLASSICAL TWIN MODEL (ablation control)")
print("=" * 60)
print("Same architecture as the Hybrid QML model, with the quantum circuit")
print("replaced by Dense(N_QUBITS, activation='relu'). Used to check whether")
print("the quantum layer itself adds anything over an equivalent classical layer.")
print()
classical_twin_model_preview = build_classical_twin_model(X_train_selected.shape[1])
classical_twin_model_preview.summary()

# Cell 9: Cross-validation for the hybrid model (LEAKAGE-CORRECTED)
# Preprocessing (imputer/scaler/selector) is refit INSIDE each fold, using
# only that fold's training rows, so validation-fold data never influences
# imputation medians, scaling statistics, or which features get selected.
print("=" * 60)
print("CROSS-VALIDATION")
print("=" * 60)
kf = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=RANDOM_STATE
)
cv_results = []
for fold_number, (train_idx, val_idx) in enumerate(
    kf.split(X_train, y_train),
    start=1
):
    print(f"\nRunning fold {fold_number}/3...")

    fold_imputer = SimpleImputer(strategy="median")
    X_fold_train_imp = fold_imputer.fit_transform(X_train.iloc[train_idx])
    X_fold_val_imp = fold_imputer.transform(X_train.iloc[val_idx])

    fold_scaler = StandardScaler()
    X_fold_train_scaled = fold_scaler.fit_transform(X_fold_train_imp)
    X_fold_val_scaled = fold_scaler.transform(X_fold_val_imp)

    fold_selector = SelectKBest(
        score_func=mutual_info_classif,
        k=N_SELECTED_FEATURES
    )
    X_fold_train_sel = fold_selector.fit_transform(
        X_fold_train_scaled,
        y_train.iloc[train_idx]
    )
    X_fold_val_sel = fold_selector.transform(
        X_fold_val_scaled
    )

    fold_model = build_hybrid_model(
        X_fold_train_sel.shape[1]
    )
    fold_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=y_train.iloc[train_idx]
    )
    fold_class_weights = {
        0: fold_weights[0],
        1: fold_weights[1]
    }
    fold_stop = EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True
    )
    fold_model.fit(
        X_fold_train_sel,
        y_train.iloc[train_idx].values,
        validation_data=(
            X_fold_val_sel,
            y_train.iloc[val_idx].values
        ),
        epochs=40,
        batch_size=64,
        class_weight=fold_class_weights,
        callbacks=[fold_stop],
        verbose=0
    )
    val_prob = fold_model.predict(
        X_fold_val_sel,
        verbose=0
    ).ravel()
    val_pred = (val_prob >= 0.5).astype(int)
    fold_metrics = calculate_metrics(
        y_train.iloc[val_idx],
        val_pred,
        val_prob,
        f"Hybrid QML - Fold {fold_number}"
    )
    cv_results.append(fold_metrics)
    print(
        f"  Accuracy: {fold_metrics['Accuracy']:.3f} | "
        f"Sensitivity: {fold_metrics['Sensitivity']:.3f} | "
        f"Specificity: {fold_metrics['Specificity']:.3f} | "
        f"ROC-AUC: {fold_metrics['ROC-AUC']:.3f}"
    )
cv_results_df = pd.DataFrame(cv_results)
print()
print("Average cross-validation performance:")
print(
    cv_results_df[
        [
            "Accuracy",
            "Sensitivity",
            "Specificity",
            "F1-score",
            "ROC-AUC",
            "PR-AUC"
        ]
    ].mean().round(4)
)
print()
print("Cross-validation completed (preprocessing refit per fold -- no leakage).")

# Cell 10: Final training with a separate validation set
print("=" * 60)
print("FINAL HYBRID MODEL")
print("=" * 60)
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train_selected,
    y_train,
    test_size=0.15,
    stratify=y_train,
    random_state=RANDOM_STATE
)
final_class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=y_fit
)
final_class_weight_dict = {
    0: final_class_weights[0],
    1: final_class_weights[1]
}
model = build_hybrid_model(
    X_train_selected.shape[1]
)
final_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)
history = model.fit(
    X_fit,
    y_fit.values,
    validation_data=(X_val, y_val.values),
    epochs=60,
    batch_size=64,
    class_weight=final_class_weight_dict,
    callbacks=[final_stop],
    verbose=1
)
print()
print("Final hybrid model training is complete.")

# Cell 10B: Final training of the classical twin (same split, same recipe)
print("=" * 60)
print("FINAL CLASSICAL TWIN MODEL")
print("=" * 60)
classical_twin_model = build_classical_twin_model(
    X_train_selected.shape[1]
)
twin_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)
twin_history = classical_twin_model.fit(
    X_fit,
    y_fit.values,
    validation_data=(X_val, y_val.values),
    epochs=60,
    batch_size=64,
    class_weight=final_class_weight_dict,
    callbacks=[twin_stop],
    verbose=1
)
print()
print("Final classical twin training is complete.")
print("Trained on the exact same X_fit/y_fit/X_val/y_val split as the hybrid model.")

# Cell 11: Choose the screening threshold using validation data
def choose_threshold(probabilities, y_true, min_sensitivity=0.80):
    """
    Shared threshold-selection logic used for both the hybrid model and the
    classical twin, so the two are compared on equal footing rather than
    one being tuned more carefully than the other.
    """
    threshold_values = np.arange(0.10, 0.91, 0.01)
    threshold_results = []
    for threshold in threshold_values:
        pred = (probabilities >= threshold).astype(int)
        sensitivity = recall_score(y_true, pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        threshold_results.append({
            "Threshold": threshold,
            "Sensitivity": sensitivity,
            "Specificity": specificity
        })
    threshold_df = pd.DataFrame(threshold_results)
    acceptable = threshold_df[threshold_df["Sensitivity"] >= min_sensitivity]
    if len(acceptable) > 0:
        best_row = acceptable.sort_values(
            ["Specificity", "Threshold"], ascending=[False, True]
        ).iloc[0]
    else:
        balanced_scores = [
            balanced_accuracy_score(y_true, (probabilities >= t).astype(int))
            for t in threshold_values
        ]
        best_row = threshold_df.iloc[int(np.argmax(balanced_scores))]
    return float(best_row["Threshold"]), best_row

print("=" * 60)
print("SCREENING THRESHOLD")
print("=" * 60)
val_prob = model.predict(X_val, verbose=0).ravel()
BEST_THRESHOLD, best_row = choose_threshold(val_prob, y_val)
print(f"Hybrid QML chosen threshold: {BEST_THRESHOLD:.2f}")
print(f"  Validation sensitivity: {best_row['Sensitivity']:.3f}")
print(f"  Validation specificity: {best_row['Specificity']:.3f}")
print()
print("The threshold was selected using validation data.")
print("The test set is kept untouched for the final evaluation.")

# Cell 11B: Threshold selection for the classical twin (same procedure)
print()
print("=" * 60)
print("SCREENING THRESHOLD - CLASSICAL TWIN")
print("=" * 60)
val_prob_twin = classical_twin_model.predict(X_val, verbose=0).ravel()
TWIN_THRESHOLD, twin_best_row = choose_threshold(val_prob_twin, y_val)
print(f"Classical twin chosen threshold: {TWIN_THRESHOLD:.2f}")
print(f"  Validation sensitivity: {twin_best_row['Sensitivity']:.3f}")
print(f"  Validation specificity: {twin_best_row['Specificity']:.3f}")

# Cell 12: Final test evaluation
print("=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)
hybrid_prob = model.predict(
    X_test_selected,
    verbose=0
).ravel()
hybrid_pred = (
    hybrid_prob >= BEST_THRESHOLD
).astype(int)
hybrid_metrics = calculate_metrics(
    y_test,
    hybrid_pred,
    hybrid_prob,
    "Hybrid QML"
)

# Cell 12B: Final test evaluation of the classical twin + ablation verdict
twin_prob = classical_twin_model.predict(X_test_selected, verbose=0).ravel()
twin_pred = (twin_prob >= TWIN_THRESHOLD).astype(int)
twin_metrics = calculate_metrics(
    y_test,
    twin_pred,
    twin_prob,
    "Classical Twin (no quantum layer)"
)

all_results = baseline_metrics + [hybrid_metrics, twin_metrics]
comparison_df = pd.DataFrame(all_results)
print()
print(comparison_df.round(4).to_string(index=False))
tn, fp, fn, tp = confusion_matrix(
    y_test,
    hybrid_pred,
    labels=[0, 1]
).ravel()
print()
print("Hybrid QML confusion matrix:")
print(f"  True negatives : {tn}")
print(f"  False positives: {fp}")
print(f"  False negatives: {fn}")
print(f"  True positives : {tp}")
print()
print("The most important screening metrics are sensitivity and specificity.")

# --- Explicit ablation verdict: did the quantum layer buy anything? ---
print()
print("=" * 60)
print("ABLATION: HYBRID QML vs. CLASSICAL TWIN")
print("=" * 60)
auc_gap = hybrid_metrics["ROC-AUC"] - twin_metrics["ROC-AUC"]
pr_gap = hybrid_metrics["PR-AUC"] - twin_metrics["PR-AUC"]
print(f"Hybrid QML ROC-AUC:      {hybrid_metrics['ROC-AUC']:.4f}")
print(f"Classical twin ROC-AUC: {twin_metrics['ROC-AUC']:.4f}")
print(f"Difference:              {auc_gap:+.4f}")
print(f"Hybrid QML PR-AUC:       {hybrid_metrics['PR-AUC']:.4f}")
print(f"Classical twin PR-AUC:  {twin_metrics['PR-AUC']:.4f}")
print(f"Difference:              {pr_gap:+.4f}")
print()
if auc_gap > 0.01:
    verdict = (
        f"The Hybrid QML model outperforms its classical twin by {auc_gap:.3f} ROC-AUC. "
        "This is evidence the quantum layer contributes beyond an equivalent classical bottleneck."
    )
elif auc_gap < -0.01:
    verdict = (
        f"The classical twin outperforms the Hybrid QML model by {-auc_gap:.3f} ROC-AUC. "
        "There is no evidence the quantum layer helps here -- the classical twin is a stronger choice."
    )
else:
    verdict = (
        f"Hybrid QML and its classical twin are within {abs(auc_gap):.3f} ROC-AUC of each other. "
        "There is no meaningful evidence the quantum layer adds value over an equivalent classical layer "
        "at this scale (2 qubits, 6 trainable quantum parameters)."
    )
print(verdict)
print()
print("This verdict is the direct answer to 'what did the quantum part buy you'.")

# Cell 12C: Bootstrap confidence intervals on the test metrics
# With ~30 test patients, a bare "sensitivity = 0.90" is not a defensible claim.
# Resampling the test set with replacement and recomputing each metric gives an
# interval that reflects how little data the estimate actually rests on.
print()
print("=" * 60)
print("BOOTSTRAP CONFIDENCE INTERVALS (test set)")
print("=" * 60)
print(f"Test set: {int(y_test.sum())} MG vs {int((y_test == 0).sum())} non-MG")
print("Resampling 1000x, stratified by class, percentile 95% CIs.")
print()

hybrid_ci_df, hybrid_draws = bootstrap_metric_cis(
    y_test, hybrid_prob, BEST_THRESHOLD,
    model_name="Hybrid QML", n_boot=1000, random_state=RANDOM_STATE,
)
print("Hybrid QML:")
print(format_ci_table(hybrid_ci_df).to_string(index=False))

twin_ci_df, _ = bootstrap_metric_cis(
    y_test, twin_prob, TWIN_THRESHOLD,
    model_name="Classical Twin", n_boot=1000, random_state=RANDOM_STATE,
)
print()
print("Classical twin:")
print(format_ci_table(twin_ci_df).to_string(index=False))

print()
print("Report metrics WITH these intervals. The width is the honest signal --")
print("it shows how much the small test set limits what can be concluded.")

# Paired bootstrap on the hybrid-vs-twin gap. The unpaired verdict above says
# which number is larger; this says whether that gap survives resampling noise.
print()
print("=" * 60)
print("ABLATION GAP -- PAIRED BOOTSTRAP CI")
print("=" * 60)
ablation_ci = bootstrap_metric_difference(
    y_test, hybrid_prob, BEST_THRESHOLD, twin_prob, TWIN_THRESHOLD,
    metric="ROC-AUC", n_boot=1000, random_state=RANDOM_STATE,
)
print(f"Observed ROC-AUC difference (hybrid - twin): {ablation_ci['observed_difference']:+.4f}")
print(f"95% CI: [{ablation_ci['ci_lower']:+.4f}, {ablation_ci['ci_upper']:+.4f}]")
print(f"Interval includes zero: {ablation_ci['crosses_zero']}")
print()
print(ablation_ci["interpretation"])

# Cell 12D: Calibration of the hybrid model
# Brier score mixes calibration and discrimination into one number. The curve
# separates them: does a predicted 0.7 actually mean ~70% of such patients have MG?
print()
print("=" * 60)
print("CALIBRATION (Hybrid QML)")
print("=" * 60)
calib = calibration_analysis(y_test, hybrid_prob, n_bins=5, strategy="quantile")
print(f"Bins used: {calib['n_bins_used']} (quantile-based, sized for a small test set)")
print(f"Brier score: {calib['brier']:.4f}")
print(f"Expected Calibration Error: {calib['ece']:.4f}")
print()
print(calib["curve"].round(3).to_string(index=False))
print()
print(interpret_calibration(calib))

plot_calibration(calib, "Hybrid QML", save_path="SIH_MG_calibration.png")
plt.close()
print()
print("Saved calibration curve to SIH_MG_calibration.png")

# Cell 13: Visual results
print("=" * 60)
print("CREATING RESULTS FIGURES")
print("=" * 60)
fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 10)
)
models_for_curves = [
    ("Logistic Regression", lr_prob),
    ("Random Forest", rf_prob),
    ("SVM", svm_prob),
    ("Hybrid QML", hybrid_prob),
    ("Classical Twin", twin_prob)
]
for name, probabilities in models_for_curves:
    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities
    )
    auc = roc_auc_score(
        y_test,
        probabilities
    )
    axes[0, 0].plot(
        fpr,
        tpr,
        label=f"{name} (AUC={auc:.3f})"
    )
axes[0, 0].plot(
    [0, 1],
    [0, 1],
    "k--",
    alpha=0.5
)
axes[0, 0].set_title("ROC Curves")
axes[0, 0].set_xlabel("False Positive Rate")
axes[0, 0].set_ylabel("True Positive Rate")
axes[0, 0].legend(fontsize=8)
for name, probabilities in models_for_curves:
    precision, recall, _ = precision_recall_curve(
        y_test,
        probabilities
    )
    axes[0, 1].plot(
        recall,
        precision,
        label=name
    )
axes[0, 1].set_title("Precision-Recall Curves")
axes[0, 1].set_xlabel("Recall / Sensitivity")
axes[0, 1].set_ylabel("Precision")
axes[0, 1].legend(fontsize=8)
sns.heatmap(
    confusion_matrix(
        y_test,
        hybrid_pred,
        labels=[0, 1]
    ),
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=axes[1, 0]
)
axes[1, 0].set_title(
    f"Hybrid QML Confusion Matrix\nThreshold = {BEST_THRESHOLD:.2f}"
)
axes[1, 0].set_xlabel("Predicted")
axes[1, 0].set_ylabel("Actual")
axes[1, 0].set_xticklabels(
    ["Non-MG", "MG"]
)
axes[1, 0].set_yticklabels(
    ["Non-MG", "MG"],
    rotation=0
)
metrics_to_plot = [
    "Accuracy",
    "Sensitivity",
    "Specificity",
    "F1-score",
    "ROC-AUC"
]
comparison_df.set_index("Model")[
    metrics_to_plot
].plot(
    kind="bar",
    ax=axes[1, 1]
)
axes[1, 1].set_title("Model Comparison")
axes[1, 1].set_ylim(0, 1.05)
axes[1, 1].set_ylabel("Score")
axes[1, 1].tick_params(
    axis="x",
    rotation=20
)
plt.tight_layout()
plt.savefig(
    "SIH_MG_results.png",
    dpi=180,
    bbox_inches="tight"
)
plt.show()
print()
print("Figure saved as SIH_MG_results.png")

# Cell 14: Explainability (fixed to explain the DEPLOYED model, not a proxy)
# Previously this cell printed Random Forest feature importance and called it
# the model's explainability, even though the RF is only a baseline -- the
# Hybrid QML model is what actually gets deployed. Two views are shown now:
#   1) RF importance, explicitly labeled as a baseline proxy, kept because
#      it's a familiar reference point for the presentation.
#   2) Permutation importance computed directly on the deployed Hybrid QML
#      model using the held-out test set -- this is what should be quoted
#      as "the model's explainability" going forward.
print("=" * 60)
print("EXPLAINABILITY")
print("=" * 60)
print()
print("(1) BASELINE PROXY -- Random Forest feature importance.")
print("    NOTE: the Random Forest is not the deployed model. This is shown")
print("    only as a familiar, easy-to-compute reference point.")
print()
for rank, (_, row) in enumerate(
    feature_importance.head(10).iterrows(),
    start=1
):
    print(
        f"{rank:2}. {row['Feature']:<35} "
        f"{row['Importance']:.4f}"
    )

def hybrid_predict_fn(X_data):
    return model.predict(X_data, verbose=0).ravel()

def permutation_importance_hybrid(
    predict_fn, X_data, y_data, feature_names, n_repeats=5, random_state=RANDOM_STATE
):
    """
    Model-agnostic permutation importance computed directly on the deployed
    Hybrid QML model. For each feature: shuffle it, measure the drop in
    ROC-AUC relative to the unshuffled baseline, repeat and average.
    A larger drop means the model relies on that feature more.
    """
    rng = np.random.RandomState(random_state)
    baseline_prob = predict_fn(X_data)
    baseline_score = roc_auc_score(y_data, baseline_prob)
    rows = []
    for i, feat in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            X_permuted = X_data.copy()
            rng.shuffle(X_permuted[:, i])
            permuted_score = roc_auc_score(y_data, predict_fn(X_permuted))
            drops.append(baseline_score - permuted_score)
        rows.append({
            "Feature": feat,
            "Mean_ROC_AUC_Drop": np.mean(drops),
            "Std_ROC_AUC_Drop": np.std(drops)
        })
    return pd.DataFrame(rows).sort_values("Mean_ROC_AUC_Drop", ascending=False)

print()
print("(2) DEPLOYED MODEL -- permutation importance on the actual Hybrid QML model.")
print("    Computed on the held-out test set. This is what the presentation")
print("    should cite as the model's explainability.")
print()
hybrid_importance = permutation_importance_hybrid(
    hybrid_predict_fn,
    X_test_selected,
    y_test.values,
    selected_features
)
print(hybrid_importance.head(10).round(4).to_string(index=False))
print()
print("A larger 'Mean_ROC_AUC_Drop' means the Hybrid QML model relies more")
print("heavily on that feature -- unlike RF importance, this reflects the")
print("model that actually produces the screening output.")

# Cell 14B: LOCAL (per-patient) explainability
# Permutation importance is a population-level statement. A tool claiming
# "explainable individual risk prediction" has to answer the per-patient
# question: what drove THIS person's score? Method is counterfactual
# occlusion -- neutralise one input, re-score, attribute the change.
print()
print("=" * 60)
print("LOCAL EXPLAINABILITY -- PER-PATIENT ATTRIBUTION")
print("=" * 60)
print("Global importance says what the model uses overall.")
print("Local attribution says what moved THIS patient's probability.")
print()

demo_idx = int(np.argmax(hybrid_prob))  # highest-risk test patient, most informative to show
demo_row = X_test_selected[demo_idx:demo_idx + 1]
demo_prob = float(model.predict(demo_row, verbose=0)[0, 0])

demo_contrib = local_feature_contributions(
    hybrid_predict_fn, demo_row, selected_features
)
print(f"Demo patient: test index {demo_idx} "
      f"(true label = {'MG' if y_test.iloc[demo_idx] == 1 else 'non-MG'})")
print()
print(explain_patient_text(demo_contrib, demo_prob, BEST_THRESHOLD, n=5))

print()
print("Local rank vs global rank (shows the two genuinely differ):")
lvg = compare_local_vs_global(demo_contrib, hybrid_importance, n=8)
print(lvg[["Feature", "Local_Rank", "Global_Rank", "Rank_Shift", "Contribution"]]
      .round(4).to_string(index=False))
print()
print("Non-zero 'Rank_Shift' means a feature matters differently for this")
print("patient than it does on average -- the point of local explainability.")

# Recommendations ordered by this patient's own attributions, not a fixed list
demo_raw_values = X_test.iloc[demo_idx].to_dict()
demo_recs = generate_local_recommendations(
    demo_contrib, demo_raw_values, demo_prob, BEST_THRESHOLD
)
print()
print("Recommendations, ordered by contribution to THIS patient's risk:")
print(recommendations_to_dataframe(demo_recs).to_string(index=False))

# Cell 14C: Leave-Diseases-Out (external-style) validation
# Does MG detection survive contact with diagnoses never seen in training?
print()
ldo_per_repeat, ldo_summary = leave_diseases_out_validation(
    X_raw=X,
    y=y,
    diagnosis_series=df["Diagnosis"],
    build_model_fn=build_classical_twin_model,
    target_disease=TARGET_DISEASE,
    holdout_fraction=0.20,
    n_repeats=5,
    n_selected_features=N_SELECTED_FEATURES,
    random_state=RANDOM_STATE,
    verbose=True,
)

if not ldo_per_repeat.empty:
    print()
    print("Per-repeat results:")
    print(ldo_per_repeat[[
        "Repeat", "N_Held_Out", "Test_MG", "Test_NonMG_Unseen",
        "ROC-AUC", "Sensitivity", "Specificity"
    ]].round(4).to_string(index=False))
    print()
    print("Summary across repeats:")
    for m in ["ROC-AUC", "PR-AUC", "Sensitivity", "Specificity"]:
        print(f"  {m:<14} mean {ldo_summary['mean'][m]:.3f} "
              f"(sd {ldo_summary['std'][m]:.3f}, "
              f"range {ldo_summary['min'][m]:.3f}-{ldo_summary['max'][m]:.3f})")

    ldo_verdict = compare_to_internal(ldo_summary, hybrid_metrics, metric="ROC-AUC")
    print()
    print("=" * 60)
    print("LDO vs INTERNAL SPLIT")
    print("=" * 60)
    print(ldo_verdict["interpretation"])
    print()
    print("Note: LDO uses the classical twin, not the quantum hybrid -- each repeat")
    print("retrains from scratch and the quantum layer is too slow for that. State")
    print("this openly rather than implying the hybrid itself was LDO-validated.")
else:
    ldo_verdict = {"status": "unavailable",
                   "interpretation": "LDO produced no valid repeats."}
    print("LDO validation could not run (insufficient cases after splitting).")

sample_number = 0
sample_features = X_test_selected[
    sample_number:sample_number + 1
]
sample_probability = float(
    model.predict(
        sample_features,
        verbose=0
    )[0, 0]
)
sample_prediction = int(
    sample_probability >= BEST_THRESHOLD
)
if sample_probability >= 0.70:
    risk_level = "High"
elif sample_probability >= BEST_THRESHOLD:
    risk_level = "Moderate"
else:
    risk_level = "Low"
print()
print("Example screening result:")
print(f"  Estimated MG probability: {sample_probability:.3f}")
print(f"  Screening category: {risk_level}")
print(
    f"  Model output: "
    f"{'MG flagged' if sample_prediction == 1 else 'MG not flagged'}"
)
print()
print("Feature importance is used to make the screening pipeline easier to interpret.")

# Cell 15: Patient screening function (+ minimal recommendation add-on)
def predict_mg_risk(patient_data):
    """
    Estimate the model's Myasthenia Gravis screening risk for one patient.
    patient_data can be a dict of original feature names, or a one-row DataFrame.
    """
    if isinstance(patient_data, dict):
        patient_df = pd.DataFrame([patient_data])
    elif isinstance(patient_data, pd.DataFrame):
        patient_df = patient_data.copy()
    else:
        raise TypeError(
            "patient_data must be a dictionary or pandas DataFrame."
        )
    # Drop the same non-feature / documentation columns as in Cell 3
    patient_df = patient_df.drop(columns=ALWAYS_EXCLUDE, errors="ignore")
    if "Gender" in patient_df.columns:
        gender_map = {
            "Male": 0,
            "Female": 1,
            "male": 0,
            "female": 1
        }
        patient_df["Gender"] = patient_df["Gender"].map(gender_map)
    patient_df = patient_df.reindex(columns=X.columns)
    for col in patient_df.columns:
        patient_df[col] = pd.to_numeric(patient_df[col], errors="coerce")
    patient_imputed = imputer.transform(patient_df)
    patient_scaled = scaler.transform(patient_imputed)
    patient_selected = selector.transform(patient_scaled)
    probability = float(
        model.predict(patient_selected, verbose=0)[0, 0]
    )
    if probability >= 0.70:
        category = "High"
        message = "Further clinical evaluation is recommended."
    elif probability >= BEST_THRESHOLD:
        category = "Moderate"
        message = "Further assessment may be appropriate."
    else:
        category = "Low"
        message = "The model did not flag this case for MG screening."
    return {
        "MG_probability": probability,
        "risk_category": category,
        "screening_threshold": BEST_THRESHOLD,
        "message": message
    }

# --- Minimal rule-based recommendation layer ---
# Reuses the Random Forest feature_importance already computed in Cell 6.
RECOMMENDATION_RULES = {
    "Fatigue or chronic tiredness": "Track fatigue pattern (worse with exertion, improves with rest) -- a hallmark MG feature.",
    "Joint pain": "Note joint symptom pattern; consider rheumatology input to rule out overlapping conditions.",
    "Dry eyes and/or mouth": "Consider Sjogren's overlap screening alongside MG workup.",
    "Weight loss": "Flag for nutritional review and thyroid function check.",
    "Low-grade fever": "Monitor for infection or systemic inflammatory cause.",
    "Stiffness in the joints": "Consider rheumatology referral if persistent.",
    "ESR": "Elevated ESR warrants follow-up inflammatory workup.",
    "CRP": "Elevated CRP warrants follow-up inflammatory workup.",
}

def generate_recommendations(patient_row, top_features):
    recs = []
    for feat in top_features:
        value = patient_row.get(feat, None)
        if value not in (0, None) and feat in RECOMMENDATION_RULES:
            recs.append(RECOMMENDATION_RULES[feat])
    if not recs:
        recs.append("No specific flags from top contributing features; maintain routine follow-up.")
    recs.append(
        "Neurology referral and AChR/MuSK antibody testing recommended "
        "for confirmatory diagnosis if this case is flagged Moderate or High."
    )
    return recs

example_patient = X_test.iloc[0].to_dict()
screening_result = predict_mg_risk(example_patient)
print("=" * 60)
print("PATIENT SCREENING DEMO")
print("=" * 60)
print(f"Estimated MG probability: {screening_result['MG_probability']:.3f}")
print(f"Screening category: {screening_result['risk_category']}")
print(f"Threshold used: {screening_result['screening_threshold']:.2f}")
print(f"Message: {screening_result['message']}")
example_recommendations = generate_recommendations(
    example_patient,
    hybrid_importance["Feature"].head(10).tolist()  # uses the DEPLOYED model's importance, not RF's
)
print()
print("Preventive / follow-up recommendations:")
for rec in example_recommendations:
    print(f"  - {rec}")
print()
print("This is a research screening model, not a medical diagnosis.")

# Cell 16: Save results and model
print("=" * 60)
print("SAVING PROJECT RESULTS")
print("=" * 60)
# NOTE: model.save(...) for the full .keras file was removed here. Our
# custom QuantumLayer holds a reference to the qnode function itself, which
# Keras cannot serialize into a config -- and we don't need it anyway, since
# app.py rebuilds the architecture from mg_model.py and loads just the
# weights (deployment_artifacts/hybrid_model.weights.h5, saved below).
with open("selected_features.json", "w") as f:
    json.dump(selected_features, f, indent=2)
results = {
    "project": "Hybrid Quantum Machine Learning for Myasthenia Gravis Screening",
    "dataset": {
        "file": DATA_FILE,
        "total_records": int(len(df)),
        "total_columns": int(df.shape[1]),
        "mg_cases": int(y.sum()),
        "non_mg_cases": int((y == 0).sum()),
        "mg_prevalence_percent": float(y.mean() * 100),
        "sickness_duration_included": INCLUDE_SICKNESS_DURATION,
        "excluded_columns": ALWAYS_EXCLUDE,
    },
    "preprocessing": {
        "features_before_selection": int(X_train_scaled.shape[1]),
        "features_selected": int(X_train_selected.shape[1]),
        "selected_features": selected_features
    },
    "quantum_model": {
        "qubits": N_QUBITS,
        "variational_layers": N_LAYERS,
        "simulator": "PennyLane default.qubit"
    },
    "screening": {
        "threshold": BEST_THRESHOLD
    },
    "classical_baselines": baseline_metrics,
    "hybrid_qml": hybrid_metrics,
    "classical_twin_ablation": {
        "metrics": twin_metrics,
        "twin_threshold": TWIN_THRESHOLD,
        "roc_auc_gap_hybrid_minus_twin": float(hybrid_metrics["ROC-AUC"] - twin_metrics["ROC-AUC"]),
        "verdict": verdict
    },
    "hybrid_model_permutation_importance": hybrid_importance.to_dict(orient="records"),
    "cross_validation": {
        "folds": 3,
        "mean": cv_results_df.mean(numeric_only=True).to_dict(),
        "std": cv_results_df.std(numeric_only=True).to_dict()
    },
    # --- Added rigour sections ---
    "bootstrap_confidence_intervals": {
        "n_bootstrap": 1000,
        "method": "stratified percentile bootstrap, 95%",
        "test_set_size": int(len(y_test)),
        "test_mg_cases": int(y_test.sum()),
        "hybrid_qml": hybrid_ci_df.to_dict(orient="records"),
        "classical_twin": twin_ci_df.to_dict(orient="records"),
    },
    "ablation_paired_bootstrap": ablation_ci,
    "calibration": {
        "brier": calib["brier"],
        "expected_calibration_error": calib["ece"],
        "n_bins_used": calib["n_bins_used"],
        "binning_strategy": calib["strategy"],
        "curve": calib["curve"].to_dict(orient="records"),
        "interpretation": interpret_calibration(calib),
    },
    "local_explainability_demo": {
        "test_index": int(demo_idx),
        "probability": float(demo_prob),
        "true_label": int(y_test.iloc[demo_idx]),
        "method": "counterfactual occlusion (neutralise one feature, re-score)",
        "contributions": demo_contrib.to_dict(orient="records"),
        "recommendations": demo_recs,
    },
    "leave_diseases_out_validation": {
        "method": (
            "Whole non-MG disease categories withheld from training entirely; "
            "tested on held-out MG cases vs patients with unseen diagnoses. "
            "Uses the classical twin architecture for runtime reasons."
        ),
        "per_repeat": ldo_per_repeat.to_dict(orient="records") if not ldo_per_repeat.empty else [],
        "summary": ldo_summary,
        "vs_internal": ldo_verdict,
    },
}
with open("SIH_MG_results.json", "w") as f:
    json.dump(results, f, indent=2)

# --- Deployment artifacts for the website (app.py) ---
# Everything the web app needs to score a new patient WITHOUT retraining.
import joblib

os.makedirs("deployment_artifacts", exist_ok=True)

joblib.dump(imputer, "deployment_artifacts/imputer.joblib")
joblib.dump(scaler, "deployment_artifacts/scaler.joblib")
joblib.dump(selector, "deployment_artifacts/selector.joblib")

# Save hybrid model WEIGHTS (not the full .keras file) -- rebuilding the
# architecture from mg_model.py and loading weights is the robust way to
# reload a model containing a custom PennyLane quantum layer.
model.save_weights("deployment_artifacts/hybrid_model.weights.h5")

# Raw feature columns in the exact order the imputer/scaler expect them
raw_feature_columns = X_train.columns.tolist()
with open("deployment_artifacts/raw_feature_columns.json", "w") as f:
    json.dump(raw_feature_columns, f, indent=2)

# Which raw columns are binary (0/1) vs continuous, so the web form can
# render checkboxes vs number inputs appropriately
binary_columns = [c for c in raw_feature_columns if X_train[c].dropna().isin([0, 1]).all()]
continuous_columns = [c for c in raw_feature_columns if c not in binary_columns]
with open("deployment_artifacts/column_types.json", "w") as f:
    json.dump({"binary": binary_columns, "continuous": continuous_columns}, f, indent=2)

# Sensible default values for every raw feature (median for continuous,
# mode for binary) so the web form can be pre-filled instead of requiring
# every field to be typed in from scratch
default_values = {}
for c in raw_feature_columns:
    if c in binary_columns:
        default_values[c] = int(X_train[c].mode(dropna=True).iloc[0])
    else:
        default_values[c] = float(X_train[c].median())
with open("deployment_artifacts/default_values.json", "w") as f:
    json.dump(default_values, f, indent=2)

# Thresholds
with open("deployment_artifacts/thresholds.json", "w") as f:
    json.dump({"hybrid": BEST_THRESHOLD, "classical_twin": TWIN_THRESHOLD}, f, indent=2)

# Selected features + importances, for the "why this result" panel
with open("deployment_artifacts/selected_features.json", "w") as f:
    json.dump(selected_features, f, indent=2)
hybrid_importance.to_json("deployment_artifacts/hybrid_importance.json", orient="records", indent=2)
feature_importance.to_json("deployment_artifacts/rf_importance_baseline_proxy.json", orient="records", indent=2)

# Model performance summary for the web app's "about this model" panel, so the
# site can show metrics WITH their intervals rather than bare point estimates.
model_card = {
    "model": "Hybrid QML (2-qubit variational circuit + classical layers)",
    "test_set_size": int(len(y_test)),
    "test_mg_cases": int(y_test.sum()),
    "threshold": float(BEST_THRESHOLD),
    "metrics_with_ci": format_ci_table(hybrid_ci_df).to_dict(orient="records"),
    "calibration": {
        "brier": calib["brier"],
        "ece": calib["ece"],
        "interpretation": interpret_calibration(calib),
    },
    "ablation": {
        "observed_roc_auc_difference": ablation_ci["observed_difference"],
        "ci_lower": ablation_ci["ci_lower"],
        "ci_upper": ablation_ci["ci_upper"],
        "crosses_zero": ablation_ci["crosses_zero"],
        "interpretation": ablation_ci["interpretation"],
    },
    "external_style_validation": ldo_verdict,
}
with open("deployment_artifacts/model_card.json", "w") as f:
    json.dump(model_card, f, indent=2, default=float)

# Neutral reference row for local occlusion attribution in the app.
# Zeros = the training mean in StandardScaler space.
np.save("deployment_artifacts/reference_row.npy",
        np.zeros(X_train_selected.shape[1], dtype=float))

print("Deployment artifacts saved to ./deployment_artifacts/:")
print("  - imputer.joblib, scaler.joblib, selector.joblib")
print("  - hybrid_model.weights.h5")
print("  - raw_feature_columns.json, column_types.json, default_values.json")
print("  - thresholds.json, selected_features.json")
print("  - hybrid_importance.json, rf_importance_baseline_proxy.json")
print("  - model_card.json (metrics WITH confidence intervals, calibration, ablation)")
print("  - reference_row.npy (neutral baseline for per-patient attribution)")
print("Run 'streamlit run app.py' next to launch the website using these artifacts.")
print("Saved:")
print("  - selected_features.json")
print("  - SIH_MG_results.json")
print("  - SIH_MG_results.png")
print()
print("=" * 60)
print("PROJECT RUN COMPLETE")
print("=" * 60)
print()
print("Next:")
print("  1. Check the final model comparison table.")
print("  2. Check sensitivity and specificity.")
print("  3. Check the ROC and precision-recall curves.")
print("  4. Use the selected features for the explainability slide.")
print("  5. Use the model architecture and actual results in the SIH presentation.")
print()
print("Note: This system is a research screening prototype, not a medical diagnostic device.")
print("Note: Sickness_Duration_Months was excluded from features per the dataset's own")
print("Sickness_Duration_Months_Model_Status flag, consistent with an early-screening framing.")
print(f"Note: Classical-twin ablation verdict -- {verdict}")
print("Note: Explainability now reports permutation importance on the deployed Hybrid QML")
print("model (test set) as the primary result; RF importance is kept only as a labeled baseline proxy.")
print()
print("-" * 60)
print("HOW TO REPORT THESE RESULTS")
print("-" * 60)
print("1. Quote every metric WITH its bootstrap CI, never the point estimate alone.")
print("2. For the quantum claim, quote the PAIRED bootstrap CI on the ROC-AUC gap.")
print("   If it crosses zero, say the models are indistinguishable -- that is the finding.")
print("3. Quote the calibration ECE when describing the probability as a probability.")
print("4. Quote the leave-diseases-out result when asked about generalisation.")
print("5. Read LIMITATIONS.md before the presentation. Raising these yourself reads as")
print("   rigour; having a judge find them first does not.")
