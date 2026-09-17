"""
mg_explain.py
Per-patient (local) explainability + recommendations driven by it.

WHY THIS EXISTS
---------------
Permutation importance answers "what does the model rely on across all
patients?". That is a global statement. A screening tool that claims to do
"explainable individual risk prediction" needs to answer a different question:
"what drove THIS person's score?"

Those two answers genuinely differ. A feature can be globally important yet
irrelevant for a patient who is negative on it, and a globally mid-ranked
feature can be the single reason one particular patient got flagged.

METHOD: counterfactual occlusion
--------------------------------
For each feature, replace that patient's value with a neutral reference value
(the training-set median in scaled space, which is ~0 after standardisation)
and re-score. The change in predicted probability is that feature's local
contribution:

    contribution_j = p(patient) - p(patient with feature j neutralised)

Positive contribution = that feature pushed this patient's risk UP.
Negative contribution = it pulled risk DOWN.

This is model-agnostic (it only needs predict), needs no extra dependency,
and works fine through the custom PennyLane quantum layer -- which matters,
because KernelSHAP/DeepSHAP would either be very slow or not support it.

HONEST LIMITATION: occlusion is a one-feature-at-a-time method, so it does not
decompose interaction effects the way exact Shapley values would. For a model
with a 2-qubit bottleneck and 20 features it is a reasonable approximation, but
it is an approximation -- say so if asked rather than calling it "SHAP values".
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Local attribution
# ---------------------------------------------------------------------------

def local_feature_contributions(
    predict_fn,
    patient_selected_row,
    feature_names,
    reference_row=None,
):
    """
    Occlusion-based local attributions for a single patient.

    predict_fn            : callable, (n, d) array -> (n,) probabilities
    patient_selected_row  : (d,) or (1, d) array, AFTER impute/scale/select
    feature_names         : the selected feature names, in column order
    reference_row         : (d,) neutral baseline. Defaults to all zeros, which
                            is the training mean in StandardScaler space.

    Returns a DataFrame sorted by how strongly each feature pushed risk upward.
    """
    x = np.asarray(patient_selected_row, dtype=float).reshape(1, -1)
    d = x.shape[1]

    if reference_row is None:
        reference_row = np.zeros(d, dtype=float)
    reference_row = np.asarray(reference_row, dtype=float).reshape(-1)

    baseline_prob = float(predict_fn(x)[0])

    # Build all occluded variants in one batch -- one predict call instead of d
    variants = np.repeat(x, d, axis=0)
    for j in range(d):
        variants[j, j] = reference_row[j]
    occluded_probs = np.asarray(predict_fn(variants)).ravel()

    contributions = baseline_prob - occluded_probs

    out = pd.DataFrame({
        "Feature": list(feature_names),
        "Patient_Value_Scaled": x.ravel(),
        "Probability_If_Neutralised": occluded_probs,
        "Contribution": contributions,
        "Abs_Contribution": np.abs(contributions),
        "Direction": np.where(contributions > 0, "increases risk",
                     np.where(contributions < 0, "decreases risk", "no effect")),
    })
    out.attrs["baseline_probability"] = baseline_prob
    return out.sort_values("Contribution", ascending=False).reset_index(drop=True)


def top_local_drivers(contrib_df, n=5, only_increasing=True, min_abs=1e-4):
    """The n features that most raised this patient's risk (noise filtered)."""
    df = contrib_df[contrib_df["Abs_Contribution"] >= min_abs]
    if only_increasing:
        df = df[df["Contribution"] > 0]
    return df.head(n).reset_index(drop=True)


def explain_patient_text(contrib_df, probability, threshold, n=5):
    """Plain-language explanation of one patient's score, for the UI/report."""
    drivers = top_local_drivers(contrib_df, n=n)
    protective = contrib_df[contrib_df["Contribution"] < -1e-4].tail(3)

    lines = [
        f"Estimated MG probability: {probability:.1%} "
        f"(screening threshold {threshold:.2f})."
    ]

    if len(drivers) == 0:
        lines.append(
            "No single input stands out as driving this score -- the estimate comes from "
            "the overall combination of values rather than one dominant factor."
        )
    else:
        lines.append("Largest contributors to THIS patient's score:")
        for _, r in drivers.iterrows():
            lines.append(
                f"  - {r['Feature']}: +{r['Contribution']:.3f} probability "
                f"(without it, the estimate would fall to {r['Probability_If_Neutralised']:.1%})"
            )

    if len(protective) > 0:
        lines.append("Values pulling the estimate downward:")
        for _, r in protective.iloc[::-1].iterrows():
            lines.append(f"  - {r['Feature']}: {r['Contribution']:.3f} probability")

    lines.append(
        "Contributions are measured by neutralising one input at a time and re-scoring, "
        "so they are specific to this patient, not population averages."
    )
    return "\n".join(lines)


def compare_local_vs_global(contrib_df, global_importance_df, n=10):
    """
    Side-by-side local rank vs global permutation-importance rank.

    Useful in the presentation: it shows concretely that the per-patient
    explanation is not just a re-print of the global chart.
    """
    local = contrib_df[["Feature", "Contribution", "Abs_Contribution"]].copy()
    local["Local_Rank"] = local["Abs_Contribution"].rank(ascending=False, method="min").astype(int)

    g = global_importance_df.copy()
    score_col = "Mean_ROC_AUC_Drop" if "Mean_ROC_AUC_Drop" in g.columns else g.columns[1]
    g = g[["Feature", score_col]].rename(columns={score_col: "Global_Importance"})
    g["Global_Rank"] = g["Global_Importance"].rank(ascending=False, method="min").astype(int)

    merged = local.merge(g, on="Feature", how="left")
    merged["Rank_Shift"] = merged["Global_Rank"] - merged["Local_Rank"]
    return merged.sort_values("Local_Rank").head(n).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Recommendations driven by local attribution
# ---------------------------------------------------------------------------

RECOMMENDATION_RULES = {
    "Fatigue or chronic tiredness":
        "Track the fatigue pattern over a week -- MG fatigue characteristically worsens "
        "with sustained exertion and improves after rest, unlike constant fatigue.",
    "Muscle weakness":
        "Document which muscle groups weaken and whether weakness worsens through the day "
        "-- diurnal variation is a key MG feature.",
    "Double vision":
        "Ocular symptoms are the most common MG presentation. Note whether double vision "
        "worsens with sustained upward gaze; prioritise neurology/ophthalmology review.",
    "Drooping eyelid":
        "Ptosis that worsens with sustained upgaze is a classic MG sign -- flag explicitly "
        "in the neurology referral.",
    "Difficulty swallowing":
        "Bulbar symptoms warrant prompt review; assess swallowing safety and aspiration risk.",
    "Shortness of breath":
        "Respiratory involvement raises urgency. Assess respiratory function; escalate "
        "if breathlessness is progressive or occurs at rest.",
    "Slurred speech":
        "Note whether speech deteriorates with prolonged talking -- fatigable dysarthria "
        "supports a bulbar MG picture.",
    "Joint pain":
        "Characterise the joint pattern; consider rheumatology input to rule out an "
        "overlapping autoimmune condition.",
    "Stiffness in the joints":
        "Persistent stiffness -- especially morning-predominant -- warrants rheumatology review.",
    "Dry eyes and/or mouth":
        "Sicca symptoms raise the possibility of Sjogren's overlap; consider screening alongside MG workup.",
    "Weight loss":
        "Arrange nutritional review and thyroid function testing; thyroid disease commonly co-occurs with MG.",
    "Low-grade fever":
        "Monitor for infection or a systemic inflammatory cause before attributing to MG.",
    "Skin rash":
        "Document rash distribution; helps discriminate between overlapping autoimmune diagnoses.",
    "Hair loss":
        "Consider thyroid and autoimmune screening panels.",
    "ESR":
        "Elevated ESR points toward an inflammatory process -- pursue inflammatory workup, "
        "as MG alone does not typically raise ESR.",
    "CRP":
        "Elevated CRP suggests active inflammation or infection; investigate before attributing "
        "symptoms to MG alone.",
    "ANA":
        "A positive ANA supports an autoimmune process and raises the value of a full autoantibody panel.",
    "Age":
        "MG has a bimodal age distribution (younger women, older men) -- factor age into pre-test probability.",
    "Gender":
        "Sex influences MG pre-test probability and typical age of onset; interpret alongside age.",
}

GENERIC_FOLLOW_UP = (
    "Neurology referral with AChR and MuSK antibody testing is the appropriate confirmatory "
    "pathway if this case is flagged Moderate or High."
)


def generate_local_recommendations(
    contrib_df,
    patient_raw_values,
    probability,
    threshold,
    max_recs=6,
    min_contribution=1e-4,
):
    """
    Recommendations ordered by each feature's contribution to THIS patient's score.

    The old version walked a fixed global top-10 list and emitted a rule if the
    patient happened to be positive on that feature. Two problems: the ordering
    was identical for every patient, and a feature could trigger a
    recommendation while contributing nothing to that patient's actual risk.

    Here the ordering IS the local attribution, so the first recommendation
    addresses the input that most raised this specific person's probability.
    Each item carries the contribution that justified it, which is what makes
    the explanation auditable rather than decorative.
    """
    drivers = contrib_df[
        (contrib_df["Contribution"] > min_contribution)
    ].sort_values("Contribution", ascending=False)

    recs = []
    for _, row in drivers.iterrows():
        feat = row["Feature"]
        if feat not in RECOMMENDATION_RULES:
            continue

        raw_val = patient_raw_values.get(feat, None)
        # Skip features the patient is explicitly negative/absent on -- a rule
        # about fatigue is noise for someone who reported no fatigue.
        if raw_val is not None and isinstance(raw_val, (int, float, np.integer, np.floating)):
            if float(raw_val) == 0.0 and row["Patient_Value_Scaled"] <= 0:
                continue

        recs.append({
            "rank": len(recs) + 1,
            "feature": feat,
            "contribution": float(row["Contribution"]),
            "recommendation": RECOMMENDATION_RULES[feat],
            "basis": (
                f"Raised this patient's estimated probability by "
                f"{row['Contribution']:.3f} ({row['Contribution'] * 100:.1f} percentage points)."
            ),
        })
        if len(recs) >= max_recs:
            break

    if not recs:
        recs.append({
            "rank": 1,
            "feature": None,
            "contribution": 0.0,
            "recommendation": (
                "No individual input materially raised this patient's risk estimate. "
                "Maintain routine follow-up and re-screen if new symptoms develop."
            ),
            "basis": "No feature exceeded the minimum local contribution threshold.",
        })

    if probability >= threshold:
        recs.append({
            "rank": len(recs) + 1,
            "feature": None,
            "contribution": None,
            "recommendation": GENERIC_FOLLOW_UP,
            "basis": f"Case flagged at or above the screening threshold ({threshold:.2f}).",
        })

    return recs


def recommendations_to_dataframe(recs):
    """Tabular form for the UI and the PDF report."""
    return pd.DataFrame([
        {
            "#": r["rank"],
            "Driver": r["feature"] or "-",
            "Contribution": (f"{r['contribution']:+.3f}"
                             if r["contribution"] not in (None, 0.0) else "-"),
            "Recommendation": r["recommendation"],
        }
        for r in recs
    ])
