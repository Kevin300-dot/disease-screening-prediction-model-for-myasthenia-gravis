"""Generate the MG screening dataset (simulated patients).

Why this exists: the original dataset only recorded ten generic symptoms (fever, fatigue,
joint pain ...) that look the same in MG and in every other disease, so no model could
separate them (ROC-AUC ~0.55). This script builds a larger, MG-specific dataset: every
simulated patient has the full symptom checklist used by the app (ocular, bulbar, limb,
fatigability pattern, respiratory, general symptoms and medical history).

Symptom frequencies follow the textbook clinical picture of MG (ptosis and diplopia in most
patients, fatigable weakness that is worse in the evening and better with rest, bulbar and
limb involvement in generalized disease, bimodal age/sex onset) and of the conditions that
are commonly confused with it (Lambert-Eaton, thyroid eye disease, Guillain-Barre, myositis,
multiple sclerosis, lupus, rheumatoid arthritis, Sjogren, hypothyroidism, fibromyalgia ...).
Real patients are messier than a checklist, so the simulation deliberately overlaps the classes:
mild / atypical / seronegative MG, "worried well" people, non-MG patients who also have drooping
eyelids or double vision, per-patient over- and under-reporting of symptoms, and recording errors.
All probabilities below are MODELLING ASSUMPTIONS written from the clinical literature, not
measurements - the metrics you get from this data describe the simulation, never real patients.

Run:  python generate_dataset.py      (deterministic, seed 42)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mg import config as C

SEED = 42
N_MG, N_HEALTHY, N_OTHER = 3200, 2400, 2400
FLIP_NOISE = 0.03           # chance that any single yes/no answer is recorded the opposite way

MG_KEYS = list(C.MG_SIGNS)
GEN_KEYS = list(C.GENERAL_SYMPTOMS)
HIS_KEYS = list(C.HISTORY_ITEMS)
BINARY_KEYS = MG_KEYS + GEN_KEYS + HIS_KEYS
BLOOD_KEYS = list(C.BLOOD_FLAGS)       # flags derived from blood details (antibodies, CBC, ESR, CRP, TSH, CK)

# Probability that each blood flag is "abnormal / positive" (0 when the test is not done).
# Order follows C.BLOOD_FLAGS.  Antibody values include the chance that the test was not done.
_BLOOD_BASE_HEALTHY = dict(achr_antibody_positive=.002, musk_antibody_positive=.0005, low_hemoglobin=.08,
                           abnormal_wbc=.05, abnormal_platelets=.03, high_esr=.06, high_crp=.05,
                           abnormal_tsh=.05, high_ck=.04)
_BLOOD_BASE_OTHER = dict(achr_antibody_positive=.004, musk_antibody_positive=.001, low_hemoglobin=.22,
                         abnormal_wbc=.15, abnormal_platelets=.10, high_esr=.35, high_crp=.30,
                         abnormal_tsh=.15, high_ck=.08)
# MG: (ocular, generalized, bulbar-predominant) - AChR+ in ~50% / 85% / 70% of tested patients, ~65% tested
BLOOD_MG = dict(achr_antibody_positive=(.25, .50, .40), musk_antibody_positive=(.01, .03, .10),
                low_hemoglobin=(.10,) * 3, abnormal_wbc=(.07,) * 3, abnormal_platelets=(.04,) * 3,
                high_esr=(.08,) * 3, high_crp=(.07,) * 3, abnormal_tsh=(.17,) * 3, high_ck=(.05,) * 3)
BLOOD_OTHER_OVERRIDES = {
    "Lambert-Eaton myasthenic syndrome": dict(achr_antibody_positive=.01, high_esr=.10, high_crp=.08),
    "Thyroid eye disease (Graves)": dict(abnormal_tsh=.95, high_esr=.15, high_crp=.10, low_hemoglobin=.10),
    "Guillain-Barre syndrome": dict(high_esr=.15, high_crp=.20, high_ck=.10, abnormal_wbc=.10),
    "Polymyositis / dermatomyositis": dict(high_ck=.90, high_esr=.55, high_crp=.40, achr_antibody_positive=0),
    "Multiple sclerosis": dict(high_esr=.10, high_crp=.08, low_hemoglobin=.08, abnormal_tsh=.08),
    "Systemic lupus erythematosus": dict(low_hemoglobin=.55, abnormal_wbc=.50, abnormal_platelets=.40,
                                         high_esr=.85, high_crp=.50),
    "Rheumatoid arthritis": dict(high_esr=.85, high_crp=.80, low_hemoglobin=.40),
    "Sjogren syndrome": dict(high_esr=.60, high_crp=.25, abnormal_wbc=.20),
    "Hashimoto thyroiditis": dict(abnormal_tsh=.98, low_hemoglobin=.25),
    "Fibromyalgia / chronic fatigue syndrome": dict(high_esr=.08, high_crp=.08, abnormal_tsh=.10,
                                                    low_hemoglobin=.10, abnormal_wbc=.05, abnormal_platelets=.04),
    "Celiac disease / IBD / vasculitis": dict(low_hemoglobin=.60, high_esr=.60, high_crp=.60,
                                              abnormal_platelets=.30, abnormal_wbc=.30),
}


def _blood_flags(rng, df: pd.DataFrame) -> pd.DataFrame:
    """Draw the blood flags for every patient from the cohort / diagnosis they belong to."""
    n = len(df)
    P = np.zeros((n, len(BLOOD_KEYS)))
    diag = df["Diagnosis"].to_numpy()
    for j, k in enumerate(BLOOD_KEYS):
        col = np.full(n, _BLOOD_BASE_OTHER[k], dtype=float)
        col[df["Cohort"].to_numpy() == C.HEALTHY_COHORT] = _BLOOD_BASE_HEALTHY[k]
        for i, d in enumerate(("Myasthenia gravis (ocular)", "Myasthenia gravis (generalized)",
                               "Myasthenia gravis (bulbar-predominant)")):
            col[diag == d] = BLOOD_MG[k][i]
        for name, over in BLOOD_OTHER_OVERRIDES.items():
            if k in over:
                col[diag == name] = over[k]
        P[:, j] = col
    return pd.DataFrame((rng.random(P.shape) < P).astype(int), columns=BLOOD_KEYS, index=df.index)

# ------------------------------------------------------------------ healthy
HEALTHY = dict(
    droopy_eyelids=.02, double_vision=.01, difficulty_chewing=.01, slurred_nasal_speech=.01,
    voice_fatigue=.02, facial_weakness=.01, difficulty_swallowing=.01, weak_neck_head_drop=.01,
    limb_weakness=.02, difficulty_raising_arms=.02, difficulty_climbing_stairs=.02,
    hand_grip_weakness=.01, fatigable_weakness=.05, worse_end_of_day=.20, improves_with_rest=.15,
    shortness_of_breath=.02, weak_cough=.01,
    fatigue=.12, low_grade_fever=.03, dizziness=.06, weight_loss=.02, rashes_skin_lesions=.04,
    joint_pain=.06, joint_stiffness=.05, muscle_pain=.05, hair_loss=.05, dry_eyes_mouth=.05,
    headache=.10, numbness_tingling=.04, general_unwell=.06,
    family_history_autoimmune=.08, other_autoimmune_condition=.03, thyroid_disease=.05,
    known_thymus_abnormality=.002, recent_infection_or_stress=.10,
)

# ------------------------------------------------------------------ MG phenotypes
# (ocular, generalized, bulbar-predominant)
MG_PHENOTYPE_P = [0.20, 0.62, 0.18]
MG_SIGN_P = {
    "droopy_eyelids":      (.96, .82, .55), "double_vision":      (.88, .68, .35),
    "difficulty_chewing":  (.03, .45, .85), "slurred_nasal_speech": (.03, .40, .80),
    "voice_fatigue":       (.04, .40, .70), "facial_weakness":    (.08, .45, .50),
    "difficulty_swallowing": (.02, .30, .70), "weak_neck_head_drop": (.02, .40, .30),
    "limb_weakness":       (.03, .75, .35), "difficulty_raising_arms": (.02, .62, .30),
    "difficulty_climbing_stairs": (.02, .55, .28), "hand_grip_weakness": (.02, .35, .18),
    "fatigable_weakness":  (.80, .88, .82), "worse_end_of_day":   (.70, .75, .70),
    "improves_with_rest":  (.65, .72, .65), "shortness_of_breath": (.00, .18, .22),
    "weak_cough":          (.00, .12, .20),
}
MG_GENERAL = dict(
    fatigue=.78, low_grade_fever=.04, dizziness=.10, weight_loss=.08, rashes_skin_lesions=.04,
    joint_pain=.07, joint_stiffness=.05, muscle_pain=.08, hair_loss=.06, dry_eyes_mouth=.10,
    headache=.10, numbness_tingling=.03, general_unwell=.22,
    family_history_autoimmune=.20, other_autoimmune_condition=.14, thyroid_disease=.15,
    known_thymus_abnormality=.12, recent_infection_or_stress=.30,
)
MILD_MG_SHARE = 0.20        # mild / atypical presentations (fewer classic signs)
MILD_MG_FACTOR = 0.45
REPORTER_SD = 0.9           # per-patient tendency to over/under-report symptoms (logit scale)

# ------------------------------------------------------------------ look-alike conditions
# (name, share, overrides on top of a mildly elevated "other autoimmune" baseline)
OTHER_BASE_BOOST = 1.4
OTHER_PROFILES = [
    ("Lambert-Eaton myasthenic syndrome", .06, dict(
        limb_weakness=.85, difficulty_climbing_stairs=.85, difficulty_raising_arms=.60,
        fatigable_weakness=.35, worse_end_of_day=.20, improves_with_rest=.15, droopy_eyelids=.28,
        double_vision=.08, dry_eyes_mouth=.60, fatigue=.75, weight_loss=.35, muscle_pain=.30,
        dizziness=.30, numbness_tingling=.20, other_autoimmune_condition=.20)),
    ("Thyroid eye disease (Graves)", .08, dict(
        double_vision=.55, droopy_eyelids=.06, dry_eyes_mouth=.50, weight_loss=.45, fatigue=.60,
        headache=.30, thyroid_disease=.95, other_autoimmune_condition=.50, hair_loss=.30, dizziness=.20)),
    ("Guillain-Barre syndrome", .05, dict(
        limb_weakness=.92, numbness_tingling=.85, facial_weakness=.50, difficulty_climbing_stairs=.85,
        difficulty_raising_arms=.60, hand_grip_weakness=.60, difficulty_swallowing=.22,
        shortness_of_breath=.18, weak_neck_head_drop=.20, recent_infection_or_stress=.70,
        fatigable_weakness=.25, improves_with_rest=.05, worse_end_of_day=.10)),
    ("Polymyositis / dermatomyositis", .10, dict(
        limb_weakness=.85, difficulty_raising_arms=.80, difficulty_climbing_stairs=.80,
        weak_neck_head_drop=.45, difficulty_swallowing=.28, muscle_pain=.60, rashes_skin_lesions=.60,
        joint_pain=.50, fatigue=.75, fatigable_weakness=.22, other_autoimmune_condition=.30)),
    ("Multiple sclerosis", .10, dict(
        double_vision=.35, numbness_tingling=.80, fatigue=.80, limb_weakness=.50, dizziness=.45,
        slurred_nasal_speech=.22, worse_end_of_day=.35, improves_with_rest=.30, fatigable_weakness=.28,
        headache=.30)),
    ("Systemic lupus erythematosus", .12, dict(
        joint_pain=.80, rashes_skin_lesions=.75, fatigue=.80, low_grade_fever=.40, hair_loss=.50,
        dry_eyes_mouth=.30, joint_stiffness=.50, weight_loss=.20, headache=.30, general_unwell=.50,
        other_autoimmune_condition=.50, family_history_autoimmune=.30)),
    ("Rheumatoid arthritis", .12, dict(
        joint_pain=.95, joint_stiffness=.90, fatigue=.65, hand_grip_weakness=.35, dry_eyes_mouth=.30,
        general_unwell=.35, other_autoimmune_condition=.30)),
    ("Sjogren syndrome", .08, dict(
        dry_eyes_mouth=.95, fatigue=.75, joint_pain=.50, rashes_skin_lesions=.20,
        numbness_tingling=.20, other_autoimmune_condition=.40)),
    ("Hashimoto thyroiditis", .08, dict(
        fatigue=.85, hair_loss=.60, limb_weakness=.25, muscle_pain=.30, thyroid_disease=1.0,
        dizziness=.15, other_autoimmune_condition=.35)),
    ("Fibromyalgia / chronic fatigue syndrome", .09, dict(
        fatigue=.95, muscle_pain=.85, headache=.60, general_unwell=.70, dizziness=.40,
        worse_end_of_day=.50, improves_with_rest=.40, fatigable_weakness=.45, joint_pain=.50,
        limb_weakness=.25, numbness_tingling=.30)),
    ("Cranial nerve palsy / other eye-muscle disorder (non-MG)", .08, dict(
        droopy_eyelids=.50, double_vision=.75, headache=.35, dizziness=.20, fatigue=.30,
        worse_end_of_day=.25, improves_with_rest=.20, fatigable_weakness=.15,
        recent_infection_or_stress=.20)),
    ("Celiac disease / IBD / vasculitis", .12, dict(
        fatigue=.60, weight_loss=.35, joint_pain=.30, rashes_skin_lesions=.30, general_unwell=.40,
        low_grade_fever=.25, other_autoimmune_condition=.35)),
]


# ------------------------------------------------------------------ helpers
def _shift(p, z):
    """Move probability p on the logit scale by z (per-patient reporting tendency)."""
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return 1.0 / (1.0 + np.exp(-(np.log(p / (1 - p)) + z)))


def _draw(rng, probs: dict, n: int) -> pd.DataFrame:
    z = rng.normal(0, REPORTER_SD, n)      # same patient over/under-reports across all questions
    return pd.DataFrame({k: (rng.random(n) < _shift(probs[k], z)).astype(int) for k in BINARY_KEYS})


def _age_sex(rng, n, kind: str):
    if kind == "mg":
        female = rng.random(n) < 0.60
        early = rng.random(n) < np.where(female, 0.70, 0.30)
        age = np.where(early, rng.normal(32, 9, n), rng.normal(65, 9, n))
    elif kind == "healthy":
        female = rng.random(n) < 0.50
        age = rng.normal(42, 16, n)
    else:
        female = rng.random(n) < 0.65
        age = rng.normal(46, 15, n)
    return np.clip(np.round(age), 10, 92).astype(int), np.where(female, "Female", "Male")


def _duration(rng, n, median, sigma, zero_share=0.0):
    d = np.exp(rng.normal(np.log(median), sigma, n))
    d = np.clip(d, 0.25, 120)
    return np.where(rng.random(n) < zero_share, 0.0, np.round(d, 1))


def make_mg(rng, n):
    pheno = rng.choice(3, size=n, p=MG_PHENOTYPE_P)
    cols = {}
    mild = rng.random(n) < MILD_MG_SHARE
    z = rng.normal(0, REPORTER_SD, n)
    for k in MG_KEYS:
        p = np.array(MG_SIGN_P[k])[pheno]
        p = np.where(mild, p * MILD_MG_FACTOR, p)
        cols[k] = (rng.random(n) < _shift(p, z)).astype(int)
    for k in GEN_KEYS + HIS_KEYS:
        p = MG_GENERAL[k]
        if k == "weight_loss":
            p = np.where(pheno == 2, 0.25, p)
        cols[k] = (rng.random(n) < _shift(p, z)).astype(int)
    df = pd.DataFrame(cols)
    df["Diagnosis"] = np.array(["Myasthenia gravis (ocular)", "Myasthenia gravis (generalized)",
                                "Myasthenia gravis (bulbar-predominant)"])[pheno]
    df["Cohort"] = C.MG_COHORT
    return df


def make_healthy(rng, n):
    df = _draw(rng, HEALTHY, n)
    worried = rng.random(n) < 0.08                      # "worried well": more symptoms, no disease
    boosted = _draw(rng, {k: min(.45, v * 4) for k, v in HEALTHY.items()}, n)
    for k in BINARY_KEYS:
        df[k] = np.where(worried, boosted[k], df[k])
    df["Diagnosis"] = "Normal"
    df["Cohort"] = C.HEALTHY_COHORT
    return df


def make_other(rng, n):
    names = [p[0] for p in OTHER_PROFILES]
    share = np.array([p[1] for p in OTHER_PROFILES]); share = share / share.sum()
    which = rng.choice(len(OTHER_PROFILES), size=n, p=share)
    base = {k: min(.6, v * OTHER_BASE_BOOST) for k, v in HEALTHY.items()}
    parts = []
    for i, (_, _, over) in enumerate(OTHER_PROFILES):
        m = int((which == i).sum())
        probs = {**base, **over}
        sub = _draw(rng, probs, m)
        sub["Diagnosis"] = names[i]
        parts.append(sub)
    df = pd.concat(parts, ignore_index=True)
    df["Cohort"] = C.OTHER_COHORT
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


def build() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    frames = []
    for maker, n, kind, dur in [
        (make_mg, N_MG, "mg", dict(median=6, sigma=0.8)),
        (make_healthy, N_HEALTHY, "healthy", dict(median=3, sigma=0.8, zero_share=0.75)),
        (make_other, N_OTHER, "other", dict(median=18, sigma=1.0)),
    ]:
        df = maker(rng, n)
        df["age"], df["Gender"] = _age_sex(rng, n, kind)
        df["duration_months"] = _duration(rng, n, **dur)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df[BLOOD_KEYS] = _blood_flags(rng, df)

    # recording noise: a few answers are recorded the opposite way round
    flips = rng.random((len(df), len(BINARY_KEYS))) < FLIP_NOISE
    df[BINARY_KEYS] = np.where(flips, 1 - df[BINARY_KEYS].to_numpy(), df[BINARY_KEYS].to_numpy())
    lab_keys = [k for k in BLOOD_KEYS if not k.endswith("antibody_positive")]   # antibody results are not mis-recorded
    lab_flips = rng.random((len(df), len(lab_keys))) < FLIP_NOISE
    df[lab_keys] = np.where(lab_flips, 1 - df[lab_keys].to_numpy(), df[lab_keys].to_numpy())

    df[C.TARGET] = (df["Cohort"] == C.MG_COHORT).astype(int)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    df.insert(0, "Patient_ID", np.arange(1, len(df) + 1))
    first = ["Patient_ID", "age", "Gender", "Cohort", "Diagnosis", "duration_months"]
    return df[first + BINARY_KEYS + BLOOD_KEYS + [C.TARGET]]


if __name__ == "__main__":
    out = build()
    out.to_csv(C.DATA_PATH, index=False)
    print(f"Wrote {len(out)} simulated patients -> {C.DATA_PATH}")
    print(out["Cohort"].value_counts().to_string())
