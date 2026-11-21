"""Inference engine: builds the full screening report for one patient."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Iterable, Mapping

import joblib
import pandas as pd

from . import config as C
from . import knowledge as K
from .models import positive_score

POSITIVE, NEGATIVE = "POSITIVE", "NEGATIVE"


def _as_keys(value, valid: Iterable[str], kind: str) -> set:
    """Accept a list/set of keys or a {key: bool} mapping; reject unknown keys."""
    valid = set(valid)
    if value is None:
        return set()
    if isinstance(value, Mapping):
        keys = {k for k, v in value.items() if v}
        unknown = set(value) - valid
    else:
        keys = set(value)
        unknown = keys - valid
    if unknown:
        raise ValueError(f"Unknown {kind}: {sorted(unknown)}. Valid options: {sorted(valid)}")
    return keys


def _number(value, field, lo, hi, optional=True):
    if value is None or value == "":
        if optional:
            return None
        raise ValueError(f"{field} is required")
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be a number") from None
    if not lo <= x <= hi:
        raise ValueError(f"{field} must be between {lo:g} and {hi:g}")
    return x


def bmi_category(bmi: float) -> str:
    return ("Underweight" if bmi < 18.5 else "Normal" if bmi < 25
            else "Overweight" if bmi < 30 else "Obese")


def duration_label(months: float) -> str:
    for limit, text in [(0, "No current symptoms"), (0.5, "Less than 1 month"), (2, "1-3 months"),
                        (4.5, "3-6 months"), (9, "6-12 months"), (18, "1-2 years")]:
        if months <= limit:
            return text
    return "More than 2 years"


def _blood_group(value):
    if value in (None, "", "unknown", "Unknown", "not_known"):
        return None
    g = str(value).strip().upper().replace("POSITIVE", "+").replace("NEGATIVE", "-").replace(" ", "")
    if g not in C.BLOOD_GROUPS:
        raise ValueError(f"blood_group must be one of {', '.join(C.BLOOD_GROUPS)}")
    return g


def _blood_panel(blood_tests, antibodies, male: int) -> dict:
    """Validate the optional blood details and turn them into report rows + model flags."""
    blood_tests = blood_tests or {}
    antibodies = antibodies or {}
    if not isinstance(blood_tests, Mapping) or not isinstance(antibodies, Mapping):
        raise ValueError("blood_tests and antibodies must be objects")
    unknown = (set(blood_tests) - set(C.BLOOD_TESTS)) | (set(antibodies) - set(C.ANTIBODIES))
    if unknown:
        raise ValueError(f"Unknown blood detail(s): {sorted(unknown)}")

    panel, flags = [], set()
    for key, (label, unit, rng_m, rng_f, (vmin, vmax)) in C.BLOOD_TESTS.items():
        value = _number(blood_tests.get(key), label, vmin, vmax)
        if value is None:
            continue
        lo, hi = rng_m if male else rng_f
        status = "Low" if value < lo else "High" if value > hi else "Normal"
        panel.append({"key": key, "label": label, "value": value, "unit": unit,
                      "reference": f"{lo:g} - {hi:g}", "status": status})
        if key == "hemoglobin" and status == "Low":
            flags.add("low_hemoglobin")
        elif key == "wbc" and status != "Normal":
            flags.add("abnormal_wbc")
        elif key == "platelets" and status != "Normal":
            flags.add("abnormal_platelets")
        elif key == "esr" and status == "High":
            flags.add("high_esr")
        elif key == "crp" and status == "High":
            flags.add("high_crp")
        elif key == "tsh" and status != "Normal":
            flags.add("abnormal_tsh")
        elif key == "ck" and status == "High":
            flags.add("high_ck")

    ab_rows = []
    for key, label in C.ANTIBODIES.items():
        status = str(antibodies.get(key) or "not_tested").strip().lower()
        if status not in {"positive", "negative", "not_tested"}:
            raise ValueError(f"{label} must be positive, negative or not_tested")
        ab_rows.append({"key": key, "label": label,
                        "status": {"positive": "Positive", "negative": "Negative", "not_tested": "Not tested"}[status]})
        if status == "positive":
            flags.add(f"{key}_antibody_positive")
    return {"panel": panel, "antibodies": ab_rows, "flags": flags}


# The models are trained on a simulated cohort, so the app never claims certainty: every displayed
# probability is limited to 3-97 %. (Raw model output can be ~0 or ~1 on textbook-perfect input.)
PCT_FLOOR, PCT_CEIL = 3.0, 97.0


def _clip_pct(p: float) -> float:
    return round(min(PCT_CEIL, max(PCT_FLOOR, p * 100)), 1)


def _confidence_band(confidence: float, n_agree: int, n_models: int, nothing_entered: bool) -> str:
    """High / Moderate / Borderline. Lowered when the five models disagree or almost nothing was entered."""
    band = 0 if confidence >= 90 else 1 if confidence >= 75 else 2
    if n_agree < n_models - 1:        # 3-2 split or worse
        band = max(band, 2)
    elif n_agree < n_models:          # 4-1 split
        band = max(band, 1)
    if nothing_entered:
        band = 2
    return ("High", "Moderate", "Borderline")[band]


class MGPredictor:
    def __init__(self, bundle_path=C.BUNDLE_PATH):
        try:
            self.bundle = joblib.load(bundle_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"No trained model at {bundle_path}. Run `python train.py` first."
            ) from exc
        self.features = self.bundle["features"]
        self.best = self.bundle["best_model"]

    # ------------------------------------------------------------ validation
    @staticmethod
    def _validate_demographics(age, gender):
        age = _number(age, "age", 1, 120, optional=False)
        g = str(gender).strip().lower()
        if g not in {"male", "female", "m", "f"}:
            raise ValueError("gender must be 'male' or 'female'")
        return age, int(g.startswith("m"))

    @staticmethod
    def _patient_block(name, age, male, weight_kg, height_cm, duration_months, blood_group=None) -> dict:
        name = (str(name).strip() if name else "")[:80] or "Not provided"
        weight = _number(weight_kg, "weight_kg", 2, 400)
        height = _number(height_cm, "height_cm", 40, 250)
        bmi = round(weight / (height / 100) ** 2, 1) if weight and height else None
        return {
            "name": name,
            "age": int(round(age)),
            "gender": "Male" if male else "Female",
            "weight_kg": weight,
            "height_cm": height,
            "bmi": bmi,
            "bmi_category": bmi_category(bmi) if bmi else None,
            "symptom_duration": duration_label(duration_months),
            "blood_group": blood_group,
        }

    def _row(self, age, male, duration, symptoms: set, flags: set, blood_flags: set = frozenset()) -> pd.DataFrame:
        row = {k: int(k in flags) for k in C.MG_SIGNS}
        row.update({k: int(k in symptoms) for k in C.SYMPTOMS})
        row.update({k: int(k in blood_flags) for k in C.BLOOD_FLAGS})
        row.update({"age": age, "gender_male": male, "duration_months": duration})
        return pd.DataFrame([row])[self.features]

    # ---------------------------------------------------------------- scoring
    def _score_all_models(self, X: pd.DataFrame) -> list:
        out = []
        for name, model in self.bundle["models"].items():
            score = float(positive_score(model, X)[0])
            p = float(self.bundle["calibrators"][name].predict([score])[0])
            positive = p >= 0.5
            out.append({
                "model": name,
                "result": POSITIVE if positive else NEGATIVE,
                "flags_mg": positive,
                "mg_probability_percent": _clip_pct(p),
                "confidence_percent": _clip_pct(p if positive else 1 - p),
                "is_primary": name == self.best,
            })
        return out

    # ---------------------------------------------------------------- findings
    @staticmethod
    def _findings(symptoms: set, flags: set, blood: dict | None = None) -> list:
        sections = []
        for group in C.MG_SIGN_GROUPS:
            items = [v[0] for k, v in C.MG_SIGNS.items() if v[1] == group and k in flags]
            sections.append({"section": group, "kind": "mg", "items": items})
        sections.append({"section": "General symptoms", "kind": "general",
                         "items": [C.GENERAL_SYMPTOMS[k] for k in C.GENERAL_SYMPTOMS if k in symptoms]})
        sections.append({"section": "Medical history", "kind": "history",
                         "items": [C.HISTORY_ITEMS[k] for k in C.HISTORY_ITEMS if k in symptoms]})
        if blood is not None and (blood["panel"] or any(a["status"] != "Not tested" for a in blood["antibodies"])):
            items = [f'{a["label"]}: {a["status"]}' for a in blood["antibodies"] if a["status"] != "Not tested"]
            items += [f'{r["label"]} {r["status"].lower()} ({r["value"]:g} {r["unit"]})'
                      for r in blood["panel"] if r["status"] != "Normal"]
            if not items:
                items = ["All entered blood values within the reference range"]
            sections.append({"section": "Blood investigations", "kind": "blood", "items": items})
        return sections

    @staticmethod
    def _groups(flags: set) -> set:
        return {C.MG_SIGNS[f][1] for f in flags}

    @staticmethod
    def _clinical_pattern(flags: set) -> str:
        groups = {C.MG_SIGNS[f][1] for f in flags}
        involved = [g for g in ("Ocular", "Bulbar", "Limb and neck", "Respiratory") if g in groups]
        if not involved:
            return "Fatigable weakness pattern without localizing signs"
        if involved == ["Ocular"]:
            return "Ocular pattern - eyelid and eye-movement involvement without bulbar or limb weakness"
        text = "Generalized pattern - " + ", ".join(g.lower() for g in involved) + " involvement"
        if "Respiratory" in involved:
            text += " (respiratory involvement: urgent evaluation)"
        return text

    @staticmethod
    def _blood_sentence(blood: dict) -> str:
        bits = [f'{a["label"]} positive' for a in blood["antibodies"] if a["status"] == "Positive"]
        abn = [r for r in blood["panel"] if r["status"] != "Normal"]
        if abn:
            bits.append(", ".join(f'{r["status"].lower()} {r["label"].split(" (")[0].replace("White blood cell count", "WBC").replace("Platelet count", "platelets").replace("Creatine kinase", "CK")}'
                                  for r in abn[:4]))
        if bits:
            return "Blood investigations: " + "; ".join(bits) + ". "
        if blood["panel"]:
            return "The blood values entered are within their reference ranges. "
        return ""

    def _interpretation(self, positive, flags, symptoms, n_pos, n_models, blood=None) -> str:
        n_flags = len(flags)
        labels = [C.MG_SIGNS[k][0] for k in C.MG_SIGNS if k in flags]
        listed = ""
        if labels:
            shown = ", ".join(l.lower() for l in labels[:4])
            listed = f", including {shown}" + (f" and {len(labels) - 4} more" if len(labels) > 4 else "")
        fatigable = {"fatigable_weakness", "worse_end_of_day", "improves_with_rest"} & flags
        if positive:
            text = (f"The reported clinical profile is consistent with myasthenia gravis. "
                    f"{n_flags} of {len(C.MG_SIGNS)} MG-specific signs were reported{listed}. ")
            if len(fatigable) >= 2:
                text += ("Weakness that worsens with activity and improves with rest is the "
                         "characteristic fatigable pattern of the disease. ")
            if blood:
                text += self._blood_sentence(blood)
            text += (f"{n_pos} of {n_models} screening algorithms returned a positive result. "
                     "Confirmatory serological and electrophysiological testing is recommended.")
            return text
        text = (f"The reported clinical profile is not consistent with myasthenia gravis. "
                f"{n_flags} of {len(C.MG_SIGNS)} MG-specific signs were reported{listed}. "
                "No characteristic combination of ocular, bulbar or fatigable limb weakness was identified. ")
        gen = [C.GENERAL_SYMPTOMS[k].lower() for k in C.GENERAL_SYMPTOMS if k in symptoms]
        if gen:
            text += f"The general symptoms reported ({', '.join(gen[:4])}) are non-specific. "
        if blood:
            text += self._blood_sentence(blood)
        text += f"{n_pos} of {n_models} screening algorithms returned a positive result."
        return text

    # ----------------------------------------------------------------- advice
    @staticmethod
    def _specialists(positive: bool, symptoms: set, flags: set) -> list:
        groups = {C.MG_SIGNS[f][1] for f in flags}
        picks = [("pcp", "Start here" if positive else "Routine follow-up")]
        if positive:
            picks.append(("neuro", "Priority referral"))
            if "Ocular" in groups:
                picks.append(("neuro_ophth", "Eye symptoms reported"))
            if "Bulbar" in groups:
                picks.append(("slp", "Chewing / speech / swallowing symptoms reported"))
            if "Respiratory" in groups:
                picks.append(("pulm", "Breathing symptoms reported - urgent"))
            picks.append(("endo", "Thyroid screening is routine when MG is suspected"))
            picks.append(("thoracic", "Only after MG is confirmed (thymus assessment)"))
            if groups & {"Limb and neck", "Fatigability pattern"}:
                picks.append(("physio", "Helpful once diagnosed"))
        if symptoms & {"joint_pain", "joint_stiffness", "rashes_skin_lesions", "dry_eyes_mouth",
                       "other_autoimmune_condition"}:
            picks.append(("rheum", "Joint / skin / dryness / autoimmune history reported"))
        return [{"specialist": K.SPECIALISTS[k][0], "why": K.SPECIALISTS[k][1], "when": when}
                for k, when in picks]

    def _advice(self, positive, confidence, symptoms, flags, age, male) -> dict:
        urgent = any(C.MG_SIGNS[f][2] for f in flags)
        groups = self._groups(flags)
        advice = {"emergency": K.EMERGENCY_TEXT if urgent else None}
        if positive:
            tests = list(K.TESTS_TO_ASK_ABOUT)
            tests.sort(key=lambda t: 0 if (t["groups"] and t["groups"] & groups) else 1)
            tests = [t for t in tests if not (t["groups"] == {"Respiratory"} and "Respiratory" not in groups)]
            urgency = K.NEXT_STEPS_URGENCY["high" if confidence >= 90 else "moderate"]
            advice.update({
                "tone": "positive",
                "next_steps": [urgency, *K.PREPARE_FOR_VISIT],
                "tests_to_ask_about": [t["text"] for t in tests],
                "daily_management": [d["text"] for d in K.DAILY_MANAGEMENT
                                     if d["groups"] is None or d["groups"] & groups],
                "medication_safety": K.MEDICATION_SAFETY,
                "about_treatment": K.TREATMENT_NOTE,
                "age_sex_context": (K.AGE_SEX_NOTE_EARLY if age < 40 and not male
                                    else K.AGE_SEX_NOTE_LATE if age > 60 and male else None),
            })
        else:
            advice.update({
                "tone": "negative",
                "next_steps": list(K.NEGATIVE_NEXT_STEPS),
                "watch_for": [C.MG_SIGNS[f][0] for f in
                              ("droopy_eyelids", "double_vision", "fatigable_weakness",
                               "difficulty_swallowing", "shortness_of_breath")],
            })
        if confidence < 75:
            advice["next_steps"].insert(
                0, "Model confidence for this result is lower than usual - clinical correlation is advised.")
        return advice

    # --------------------------------------------------------------- performance
    def _performance(self) -> dict:
        tm = self.bundle["test_metrics"]
        keys = ("accuracy", "precision", "recall", "specificity", "f1", "roc_auc")
        pct = lambda m: {k: round(float(m[k]) * 100, 1) for k in keys}
        n_test = int(sum(tm[self.best][k] for k in ("tn", "fp", "fn", "tp")))
        return {"primary_model": self.best, "n_test": n_test, "n_train": int(self.bundle["n_train"]),
                "primary": pct(tm[self.best]), "per_model": {name: pct(m) for name, m in tm.items()}}

    # ---------------------------------------------------------------- predict
    def predict(self, age, gender, symptoms=None, red_flags=None, *, name=None, weight_kg=None,
                height_cm=None, duration_months=None, blood_group=None, blood_tests=None,
                antibodies=None, report_id=None, generated_at=None) -> dict:
        age, male = self._validate_demographics(age, gender)
        symptoms = _as_keys(symptoms, C.SYMPTOMS, "symptom")
        flags = _as_keys(red_flags, C.MG_SIGNS, "MG sign")
        duration = _number(duration_months, "duration_months", 0, 600)
        if duration is None:
            duration = C.DEFAULT_DURATION_MONTHS if (symptoms or flags) else 0.0

        blood = _blood_panel(blood_tests, antibodies, male)
        group = _blood_group(blood_group)
        X = self._row(age, male, duration, symptoms, flags, blood["flags"])
        per_model = self._score_all_models(X)
        primary = next(m for m in per_model if m["is_primary"])
        positive = primary["flags_mg"]
        n_agree = sum(m["flags_mg"] == positive for m in per_model)
        n_pos = sum(m["flags_mg"] for m in per_model)
        confidence = primary["confidence_percent"]

        status = POSITIVE if positive else NEGATIVE
        nothing_entered = not (flags or symptoms or blood["panel"]
                               or any(a["status"] != "Not tested" for a in blood["antibodies"]))
        advice = self._advice(positive, confidence, symptoms, flags, age, male)
        if nothing_entered:
            advice["next_steps"].insert(0, "No symptoms or results were entered, so this result carries very little "
                                           "information. Complete the checklist for a meaningful screening.")
        return {
            "report": {
                "report_id": report_id or f"MYA-{datetime.now():%Y%m%d}-{uuid.uuid4().hex[:6].upper()}",
                "generated_at": generated_at or f"{datetime.now():%d %b %Y, %H:%M}",
                "patient": self._patient_block(name, age, male, weight_kg, height_cm, duration, group),
            },
            "result": {
                "test": "Myasthenia gravis screening",
                "status": status,
                "headline": f"Myasthenia gravis: {status.title()}",
                "confidence_percent": confidence,
                "confidence_band": _confidence_band(confidence, n_agree, len(per_model), nothing_entered),
                "mg_probability_percent": primary["mg_probability_percent"],
                "primary_model": self.best,
                "models_positive": f"{n_pos} of {len(per_model)}",
                "models_in_agreement": f"{n_agree} of {len(per_model)}",
                "interpretation": self._interpretation(positive, flags, symptoms, n_pos, len(per_model), blood),
                "clinical_pattern": self._clinical_pattern(flags) if positive else None,
            },
            "findings": self._findings(symptoms, flags, blood),
            "blood": {"blood_group": group, "panel": blood["panel"], "antibodies": blood["antibodies"],
                      "abnormal_count": sum(r["status"] != "Normal" for r in blood["panel"]),
                      "group_note": K.BLOOD_GROUP_NOTE if group else None},
            "counts": {"mg_signs": len(flags), "general_symptoms": len(symptoms & set(C.GENERAL_SYMPTOMS)),
                       "history": len(symptoms & set(C.HISTORY_ITEMS))},
            "per_model": per_model,
            "performance": self._performance(),
            "recommended_specialists": self._specialists(positive, symptoms, flags),
            "advice": advice,
            "footer": K.REPORT_FOOTER,
        }
