"""
app.py -- MG Screening website (Streamlit)

Run AFTER train_pipeline.py has produced ./deployment_artifacts/.
    streamlit run app.py

Loads trained artifacts only -- never retrains, never touches the raw dataset.

Tabs:
  1. Single patient  -- entry form, score, PER-PATIENT explanation, PDF download
  2. Batch mode      -- upload a CSV, score everyone, download results
  3. Model card      -- metrics with confidence intervals, calibration, limitations
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from mg_model import build_hybrid_model
from mg_explain import (
    local_feature_contributions,
    generate_local_recommendations,
    recommendations_to_dataframe,
    compare_local_vs_global,
)

ARTIFACT_DIR = "deployment_artifacts"

st.set_page_config(page_title="MG Screening", layout="wide", page_icon="🩺")


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------

@st.cache_resource
def load_artifacts():
    def _json(name, default=None):
        path = f"{ARTIFACT_DIR}/{name}"
        if not os.path.exists(path):
            return default
        with open(path) as f:
            return json.load(f)

    imputer = joblib.load(f"{ARTIFACT_DIR}/imputer.joblib")
    scaler = joblib.load(f"{ARTIFACT_DIR}/scaler.joblib")
    selector = joblib.load(f"{ARTIFACT_DIR}/selector.joblib")

    raw_feature_columns = _json("raw_feature_columns.json", [])
    column_types = _json("column_types.json", {"binary": [], "continuous": []})
    default_values = _json("default_values.json", {})
    thresholds = _json("thresholds.json", {"hybrid": 0.5})
    selected_features = _json("selected_features.json", [])
    model_card = _json("model_card.json", None)

    hybrid_importance = pd.read_json(f"{ARTIFACT_DIR}/hybrid_importance.json")

    ref_path = f"{ARTIFACT_DIR}/reference_row.npy"
    reference_row = (np.load(ref_path) if os.path.exists(ref_path)
                     else np.zeros(len(selected_features)))

    model = build_hybrid_model(len(selected_features))
    model.predict(np.zeros((1, len(selected_features))), verbose=0)  # init shapes
    model.load_weights(f"{ARTIFACT_DIR}/hybrid_model.weights.h5")

    return {
        "imputer": imputer, "scaler": scaler, "selector": selector,
        "raw_feature_columns": raw_feature_columns,
        "column_types": column_types, "default_values": default_values,
        "thresholds": thresholds, "selected_features": selected_features,
        "hybrid_importance": hybrid_importance, "reference_row": reference_row,
        "model_card": model_card, "model": model,
    }


def _to_selected_space(artifacts, raw_df):
    """Raw feature frame -> impute -> scale -> select. Same order as training."""
    row = raw_df.reindex(columns=artifacts["raw_feature_columns"])
    row = row.apply(pd.to_numeric, errors="coerce")
    imputed = artifacts["imputer"].transform(row)
    scaled = artifacts["scaler"].transform(imputed)
    return artifacts["selector"].transform(scaled)


def _categorise(probability, threshold):
    if probability >= 0.70:
        return "High", "Further clinical evaluation is recommended."
    if probability >= threshold:
        return "Moderate", "Further assessment may be appropriate."
    return "Low", "The model did not flag this case for MG screening."


def predict_single(artifacts, patient_values: dict):
    selected = _to_selected_space(artifacts, pd.DataFrame([patient_values]))
    probability = float(artifacts["model"].predict(selected, verbose=0)[0, 0])
    threshold = artifacts["thresholds"]["hybrid"]
    category, message = _categorise(probability, threshold)
    return probability, category, message, threshold, selected


def predict_batch(artifacts, raw_df):
    selected = _to_selected_space(artifacts, raw_df)
    probs = artifacts["model"].predict(selected, verbose=0).ravel()
    threshold = artifacts["thresholds"]["hybrid"]
    cats = [_categorise(p, threshold)[0] for p in probs]
    return probs, cats, threshold


artifacts = load_artifacts()
THRESHOLD = artifacts["thresholds"]["hybrid"]


# ---------------------------------------------------------------------------
# Header + sidebar
# ---------------------------------------------------------------------------

st.title("Myasthenia Gravis Screening")
st.caption(
    "Research screening prototype -- NOT a medical diagnosis. "
    "Differential-diagnosis support for symptomatic patients already under workup."
)

with st.sidebar:
    st.header("Model at a glance")
    st.write(f"Features used: **{len(artifacts['selected_features'])}**")
    st.write(f"Screening threshold: **{THRESHOLD:.2f}**")

    card = artifacts["model_card"]
    if card:
        st.caption(
            f"Evaluated on {card.get('test_set_size', '?')} test patients "
            f"({card.get('test_mg_cases', '?')} MG cases)."
        )
        rows = card.get("metrics_with_ci", [])
        headline = {r["Metric"]: r["Reported"] for r in rows
                    if r["Metric"] in ("Sensitivity", "Specificity", "ROC-AUC")}
        if headline:
            st.write("**Performance (95% CI):**")
            for k, v in headline.items():
                st.caption(f"{k}: {v}")
            st.caption(
                "Intervals are wide because the test set is small. "
                "They are shown deliberately rather than hidden."
            )
    else:
        st.info("model_card.json not found -- re-run train_pipeline.py to populate this.")

    st.divider()
    st.write("**Globally most-relied-on features**")
    st.caption("Population-level. The per-patient view is in the result panel.")
    st.dataframe(
        artifacts["hybrid_importance"].head(8),
        hide_index=True, use_container_width=True,
    )


tab_single, tab_batch, tab_card = st.tabs(
    ["Single patient", "Batch mode (CSV)", "Model card & limitations"]
)


# ---------------------------------------------------------------------------
# TAB 1 -- single patient
# ---------------------------------------------------------------------------

with tab_single:
    st.subheader("Enter patient values")
    st.caption(
        "Pre-filled with training-set typical values -- change only what applies."
    )

    patient_values = dict(artifacts["default_values"])
    binary_cols = artifacts["column_types"]["binary"]
    continuous_cols = artifacts["column_types"]["continuous"]

    patient_label = st.text_input(
        "Patient reference (optional, appears on the PDF)", value="Unnamed patient"
    )

    sub1, sub2 = st.tabs(["Key inputs (model features)", "All inputs (advanced)"])

    with sub1:
        st.write("Only the features the model actually uses for screening.")
        cols = st.columns(3)
        for i, feat in enumerate(artifacts["selected_features"]):
            col = cols[i % 3]
            if feat in binary_cols:
                patient_values[feat] = int(
                    col.checkbox(feat, value=bool(patient_values.get(feat, 0)))
                )
            elif feat == "Gender":
                choice = col.selectbox(
                    "Gender", ["Female", "Male"],
                    index=int(patient_values.get(feat, 0)),
                )
                patient_values[feat] = 1 if choice == "Female" else 0
            else:
                patient_values[feat] = col.number_input(
                    feat, value=float(patient_values.get(feat, 0.0))
                )

    with sub2:
        st.write("Every raw feature the training pipeline used.")
        with st.expander("Binary / symptom flags"):
            cols = st.columns(3)
            for i, feat in enumerate(binary_cols):
                col = cols[i % 3]
                patient_values[feat] = int(col.checkbox(
                    feat, value=bool(patient_values.get(feat, 0)), key=f"adv_{feat}"
                ))
        with st.expander("Lab values / continuous measurements"):
            cols = st.columns(3)
            for i, feat in enumerate(continuous_cols):
                if feat == "Gender":
                    continue
                col = cols[i % 3]
                patient_values[feat] = col.number_input(
                    feat, value=float(patient_values.get(feat, 0.0)), key=f"adv_{feat}"
                )
        if "Gender" in artifacts["raw_feature_columns"]:
            choice = st.selectbox(
                "Gender", ["Female", "Male"],
                index=int(patient_values.get("Gender", 0)), key="adv_gender",
            )
            patient_values["Gender"] = 1 if choice == "Female" else 0

    if st.button("Run MG screening", type="primary"):
        probability, category, message, threshold, selected_row = predict_single(
            artifacts, patient_values
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Estimated MG probability", f"{probability:.1%}")
        c2.metric("Screening category", category)
        c3.metric("Threshold used", f"{threshold:.2f}")
        st.info(message)

        # --- Per-patient explanation ---
        st.subheader("Why this patient got this score")
        st.caption(
            "Each input is neutralised one at a time and the patient re-scored. "
            "The change is that input's contribution for THIS patient -- "
            "not a population average."
        )

        contrib = local_feature_contributions(
            lambda X: artifacts["model"].predict(X, verbose=0).ravel(),
            selected_row,
            artifacts["selected_features"],
            reference_row=artifacts["reference_row"],
        )

        up = contrib[contrib["Contribution"] > 1e-4].head(8).copy()
        down = contrib[contrib["Contribution"] < -1e-4].tail(5).copy()

        ec1, ec2 = st.columns(2)
        with ec1:
            st.markdown("**Raising this patient's risk**")
            if up.empty:
                st.caption("No input materially raised the estimate.")
            else:
                show = up[["Feature", "Contribution", "Probability_If_Neutralised"]].copy()
                show["Contribution"] = (show["Contribution"] * 100).map(lambda v: f"{v:+.1f} pts")
                show["Probability_If_Neutralised"] = show["Probability_If_Neutralised"].map(
                    lambda v: f"{v:.1%}"
                )
                st.dataframe(show.rename(columns={
                    "Probability_If_Neutralised": "If removed"
                }), hide_index=True, use_container_width=True)
        with ec2:
            st.markdown("**Lowering this patient's risk**")
            if down.empty:
                st.caption("No input materially lowered the estimate.")
            else:
                show = down.iloc[::-1][["Feature", "Contribution"]].copy()
                show["Contribution"] = (show["Contribution"] * 100).map(lambda v: f"{v:+.1f} pts")
                st.dataframe(show, hide_index=True, use_container_width=True)

        with st.expander("How this patient differs from the population average"):
            lvg = compare_local_vs_global(contrib, artifacts["hybrid_importance"], n=10)
            st.caption(
                "Rank_Shift != 0 means the feature matters differently for this "
                "patient than it does globally. That gap is the reason local "
                "explainability is worth computing at all."
            )
            st.dataframe(
                lvg[["Feature", "Local_Rank", "Global_Rank", "Rank_Shift", "Contribution"]].round(4),
                hide_index=True, use_container_width=True,
            )

        # --- Recommendations ordered by this patient's own attributions ---
        st.subheader("Suggested follow-up, ordered by contribution")
        recs = generate_local_recommendations(
            contrib, patient_values, probability, threshold
        )
        st.dataframe(
            recommendations_to_dataframe(recs),
            hide_index=True, use_container_width=True,
        )
        st.caption(
            "Ordering comes from this patient's own attributions, so the first item "
            "addresses whatever most raised their estimate. Rule text is generic and "
            "has not been clinically validated."
        )

        # --- PDF download ---
        try:
            from mg_report import build_screening_pdf

            card = artifacts["model_card"] or {}
            model_info = {"Model": card.get("model", "Hybrid QML"),
                          "Threshold": f"{threshold:.2f}"}
            for r in card.get("metrics_with_ci", []):
                if r["Metric"] in ("Sensitivity", "Specificity"):
                    model_info[r["Metric"]] = r["Reported"]

            pdf_bytes = build_screening_pdf(
                probability=probability, category=category, message=message,
                threshold=threshold, local_contributions=contrib,
                recommendations=recs, patient_values=patient_values,
                model_info=model_info, patient_label=patient_label or "Unnamed patient",
            )
            st.download_button(
                "Download PDF summary", data=pdf_bytes,
                file_name=f"MG_screening_{(patient_label or 'patient').replace(' ', '_')}.pdf",
                mime="application/pdf", type="secondary",
            )
        except ImportError:
            st.warning("PDF export needs reportlab:  pip install reportlab")
        except Exception as e:
            st.warning(f"PDF generation failed: {e}")

        st.caption("This is a research screening model, not a medical diagnosis.")


# ---------------------------------------------------------------------------
# TAB 2 -- batch mode
# ---------------------------------------------------------------------------

with tab_batch:
    st.subheader("Score a whole cohort at once")
    st.caption(
        "Upload a CSV with one row per patient. Missing columns are imputed with "
        "the same training-set medians used during training; extra columns are ignored."
    )

    template = pd.DataFrame([artifacts["default_values"]])
    st.download_button(
        "Download blank CSV template",
        data=template.to_csv(index=False).encode(),
        file_name="mg_batch_template.csv", mime="text/csv",
    )

    uploaded = st.file_uploader("Upload patient CSV", type=["csv"])
    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            st.write(f"Loaded **{len(batch_df)}** rows, **{batch_df.shape[1]}** columns.")

            expected = set(artifacts["raw_feature_columns"])
            present = expected.intersection(batch_df.columns)
            missing = expected - set(batch_df.columns)
            st.caption(
                f"Matched {len(present)} of {len(expected)} model input columns."
                + (f" {len(missing)} imputed from training medians." if missing else "")
            )
            if missing and len(missing) > len(expected) * 0.5:
                st.warning(
                    "More than half the model's input columns are missing. Results will be "
                    "dominated by imputed values and should not be trusted."
                )

            if st.button("Run batch screening", type="primary"):
                probs, cats, thr = predict_batch(artifacts, batch_df)

                results = batch_df.copy()
                results["MG_Probability"] = np.round(probs, 4)
                results["Risk_Category"] = cats
                results["Flagged"] = (probs >= thr).astype(int)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Patients scored", len(results))
                m2.metric("Flagged (>= threshold)", int(results["Flagged"].sum()))
                m3.metric("High risk", int((results["Risk_Category"] == "High").sum()))
                m4.metric("Mean probability", f"{probs.mean():.1%}")

                st.write("**Results, highest risk first**")
                display_cols = ["MG_Probability", "Risk_Category", "Flagged"]
                id_col = next(
                    (c for c in ("Patient_ID", "ID", "Patient") if c in results.columns), None
                )
                if id_col:
                    display_cols = [id_col] + display_cols
                st.dataframe(
                    results[display_cols].sort_values("MG_Probability", ascending=False),
                    hide_index=True, use_container_width=True, height=380,
                )

                st.download_button(
                    "Download scored CSV",
                    data=results.to_csv(index=False).encode(),
                    file_name="mg_batch_results.csv", mime="text/csv", type="secondary",
                )
                st.caption(
                    "Batch output gives risk ranking only. Per-patient explanations are "
                    "computed individually in the Single patient tab."
                )
        except Exception as e:
            st.error(f"Could not process that file: {e}")


# ---------------------------------------------------------------------------
# TAB 3 -- model card & limitations
# ---------------------------------------------------------------------------

with tab_card:
    st.subheader("Model card")
    card = artifacts["model_card"]

    if not card:
        st.info("Re-run train_pipeline.py to generate deployment_artifacts/model_card.json.")
    else:
        st.write(f"**Architecture:** {card.get('model', 'Hybrid QML')}")
        st.write(
            f"**Evaluated on:** {card.get('test_set_size', '?')} held-out patients, "
            f"of which {card.get('test_mg_cases', '?')} are MG cases."
        )

        st.markdown("#### Performance, with 95% bootstrap confidence intervals")
        rows = card.get("metrics_with_ci", [])
        if rows:
            st.dataframe(
                pd.DataFrame(rows)[["Metric", "Reported"]],
                hide_index=True, use_container_width=True,
            )
        st.caption(
            "Intervals come from 1000 stratified bootstrap resamples of the test set. "
            "They are wide because the test set is small -- that width is the honest "
            "measure of how much these numbers can be trusted."
        )

        calib = card.get("calibration", {})
        if calib:
            st.markdown("#### Calibration")
            cc1, cc2 = st.columns(2)
            cc1.metric("Brier score", f"{calib.get('brier', float('nan')):.4f}")
            cc2.metric("Expected Calibration Error", f"{calib.get('ece', float('nan')):.4f}")
            st.caption(calib.get("interpretation", ""))
            if os.path.exists("SIH_MG_calibration.png"):
                st.image("SIH_MG_calibration.png", width=480)

        abl = card.get("ablation", {})
        if abl:
            st.markdown("#### Does the quantum layer help?")
            st.write(
                f"ROC-AUC difference (hybrid - classical twin): "
                f"**{abl.get('observed_roc_auc_difference', 0):+.4f}** "
                f"(95% CI [{abl.get('ci_lower', 0):+.4f}, {abl.get('ci_upper', 0):+.4f}])"
            )
            if abl.get("crosses_zero", True):
                st.warning(abl.get("interpretation", ""))
            else:
                st.success(abl.get("interpretation", ""))

        ext = card.get("external_style_validation", {})
        if ext and ext.get("interpretation"):
            st.markdown("#### Generalisation to unseen diagnoses")
            st.caption(
                "Whole disease categories were withheld from training, then used as the "
                "negative class at test time."
            )
            st.write(ext["interpretation"])

    st.divider()
    st.subheader("Limitations")
    if os.path.exists("LIMITATIONS.md"):
        with open("LIMITATIONS.md") as f:
            st.markdown(f.read())
    else:
        st.markdown(
            "- Dataset is not prospectively-collected clinical data.\n"
            "- Test set is very small; all intervals are wide.\n"
            "- Low specificity is a deliberate screening trade-off, not a bug.\n"
            "- The quantum layer has not been shown to outperform its classical twin.\n"
            "- Recommendation rules are illustrative and not clinically validated.\n\n"
            "_Place LIMITATIONS.md beside app.py to show the full version here._"
        )
