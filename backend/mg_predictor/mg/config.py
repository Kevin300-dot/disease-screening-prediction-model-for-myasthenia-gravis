"""Central configuration: paths, feature definitions, training settings.

Single source of truth for every symptom the app asks about. The front-end list in
`src/lib/mg-options.ts` mirrors these keys (tests/test_pipeline.py checks they match).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "mg_screening_dataset.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
BUNDLE_PATH = MODEL_DIR / "mg_bundle.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# --- Target -----------------------------------------------------------------
TARGET = "MG_Label"                 # 1 = Myasthenia gravis, 0 = anything else
DIAGNOSIS_COL = "Diagnosis"
COHORT_COL = "Cohort"
MG_COHORT = "Myasthenia gravis"
HEALTHY_COHORT = "Healthy (Normal)"
OTHER_COHORT = "Other autoimmune diseases"

# --- MG-specific signs ("red_flags" in the API) -----------------------------
# key -> (label, group, urgent)
MG_SIGNS = {
    # Ocular
    "droopy_eyelids":          ("Drooping eyelid(s)", "Ocular", False),
    "double_vision":           ("Double vision", "Ocular", False),
    # Bulbar (face / mouth / throat)
    "difficulty_chewing":      ("Jaw tires or weakens while chewing", "Bulbar", False),
    "slurred_nasal_speech":    ("Slurred or nasal speech", "Bulbar", False),
    "voice_fatigue":           ("Voice becomes weak or hoarse during long conversations", "Bulbar", False),
    "facial_weakness":         ("Facial weakness (flat expression, weak smile)", "Bulbar", False),
    "difficulty_swallowing":   ("Difficulty swallowing or choking on food or liquid", "Bulbar", True),
    # Limb and neck
    "weak_neck_head_drop":     ("Weak neck or difficulty holding the head up", "Limb and neck", False),
    "limb_weakness":           ("Weak arms or legs", "Limb and neck", False),
    "difficulty_raising_arms": ("Trouble lifting arms overhead (e.g. brushing hair)", "Limb and neck", False),
    "difficulty_climbing_stairs": ("Trouble climbing stairs or rising from a chair", "Limb and neck", False),
    "hand_grip_weakness":      ("Weak hand grip or dropping objects", "Limb and neck", False),
    # Fatigability pattern
    "fatigable_weakness":      ("Weakness that worsens with repeated activity", "Fatigability pattern", False),
    "worse_end_of_day":        ("Symptoms worse in the evening than in the morning", "Fatigability pattern", False),
    "improves_with_rest":      ("Symptoms improve after rest or sleep", "Fatigability pattern", False),
    # Respiratory
    "shortness_of_breath":     ("Shortness of breath or trouble breathing", "Respiratory", True),
    "weak_cough":              ("Weak cough or difficulty clearing the throat", "Respiratory", False),
}
MG_SIGN_GROUPS = ["Ocular", "Bulbar", "Limb and neck", "Fatigability pattern", "Respiratory"]

# --- General symptoms ("symptoms" in the API) --------------------------------
GENERAL_SYMPTOMS = {
    "fatigue":             "Fatigue / chronic tiredness",
    "low_grade_fever":     "Low-grade fever",
    "dizziness":           "Dizziness",
    "weight_loss":         "Unintentional weight loss",
    "rashes_skin_lesions": "Rashes or skin lesions",
    "joint_pain":          "Joint pain",
    "joint_stiffness":     "Stiffness in the joints",
    "muscle_pain":         "Muscle pain or cramps",
    "hair_loss":           "Brittle hair or hair loss",
    "dry_eyes_mouth":      "Dry eyes and/or dry mouth",
    "headache":            "Frequent headaches",
    "numbness_tingling":   "Numbness or tingling",
    "general_unwell":      "General 'unwell' feeling",
}

# --- Medical history ("symptoms" in the API, shown as its own section) --------
HISTORY_ITEMS = {
    "family_history_autoimmune":  "Family history of autoimmune disease",
    "other_autoimmune_condition": "Existing autoimmune condition (e.g. lupus, rheumatoid arthritis)",
    "thyroid_disease":            "Thyroid disease",
    "known_thymus_abnormality":   "Known thymus gland abnormality",
    "recent_infection_or_stress": "Recent infection, surgery or major stress",
}

# --- Blood details ------------------------------------------------------------
# Blood group is recorded on the report (it matters for transfusion / IVIG / plasma exchange)
# but is NOT a model input: there is no established link between ABO/Rh group and MG.
BLOOD_GROUPS = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]

# Optional lab values the patient can enter.  key -> (label, unit, (lo_male, hi_male),
# (lo_female, hi_female), (min_valid, max_valid))
BLOOD_TESTS = {
    "hemoglobin": ("Hemoglobin", "g/dL", (13.5, 17.5), (12.0, 15.5), (3, 25)),
    "wbc":        ("White blood cell count", "x10^3/uL", (4.0, 11.0), (4.0, 11.0), (0.5, 100)),
    "platelets":  ("Platelet count", "x10^3/uL", (150, 450), (150, 450), (5, 1500)),
    "esr":        ("ESR", "mm/hr", (0, 15), (0, 20), (0, 150)),
    "crp":        ("C-reactive protein (CRP)", "mg/L", (0, 5), (0, 5), (0, 500)),
    "tsh":        ("TSH (thyroid)", "mIU/L", (0.4, 4.0), (0.4, 4.0), (0.01, 150)),
    "ck":         ("Creatine kinase (CK)", "U/L", (39, 308), (26, 192), (1, 50000)),
}
# Antibody results: "positive" | "negative" | "not_tested"
ANTIBODIES = {
    "achr": "Anti-AChR antibody",
    "musk": "Anti-MuSK antibody",
}

# Binary model inputs derived from the blood details (0 when a value is blank / not tested).
# key -> label shown in the report
BLOOD_FLAGS = {
    "achr_antibody_positive": "Anti-AChR antibody positive",
    "musk_antibody_positive": "Anti-MuSK antibody positive",
    "low_hemoglobin":         "Low hemoglobin (anaemia)",
    "abnormal_wbc":           "Abnormal white cell count",
    "abnormal_platelets":     "Abnormal platelet count",
    "high_esr":               "Raised ESR",
    "high_crp":               "Raised CRP",
    "abnormal_tsh":           "Abnormal TSH (thyroid)",
    "high_ck":                "Raised creatine kinase",
}

# Keys accepted in the API's "symptoms" list (general symptoms + history)
SYMPTOMS = {**GENERAL_SYMPTOMS, **HISTORY_ITEMS}
SYMPTOM_LABELS = dict(SYMPTOMS)

# Duration of symptoms (months) offered in the UI -> numeric value for the model
DURATION_OPTIONS = {
    "none": 0.0, "lt1": 0.5, "1to3": 2.0, "3to6": 4.5, "6to12": 9.0, "1to2y": 18.0, "gt2y": 36.0,
}
DEFAULT_DURATION_MONTHS = 2.0

# Model input columns (order matters and is stored with the trained bundle)
FEATURES = list(MG_SIGNS) + list(SYMPTOMS) + list(BLOOD_FLAGS) + ["age", "gender_male", "duration_months"]
