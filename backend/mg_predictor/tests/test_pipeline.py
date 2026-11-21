"""Run after training:  python -m unittest discover -s tests -v"""
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api import create_app                      # noqa: E402
from mg import config as C                      # noqa: E402
from mg.data import build_xy, load_dataset      # noqa: E402
from mg.predictor import MGPredictor            # noqa: E402
from mg import hospitals as H                   # noqa: E402
from mg.report_pdf import build_report_pdf      # noqa: E402

CLASSIC_MG = ["droopy_eyelids", "double_vision", "fatigable_weakness", "worse_end_of_day",
              "improves_with_rest", "difficulty_chewing", "limb_weakness"]


class DataTests(unittest.TestCase):
    def test_features_and_labels(self):
        X, y = build_xy(load_dataset())
        self.assertEqual(list(X.columns), C.FEATURES)
        banned = {"Patient_ID", "Diagnosis", "Cohort", "MG_Label"}
        self.assertFalse(banned & set(X.columns))
        self.assertGreater(int(y.sum()), 1000)


class PerformanceTests(unittest.TestCase):
    """Sanity checks only. The data is simulated, so we do NOT assert flattering metric targets."""

    def test_models_clearly_beat_chance_on_the_test_set(self):
        perf = MGPredictor()._performance()
        for name, m in perf["per_model"].items():
            self.assertGreater(m["roc_auc"], 90.0, name)
            self.assertGreater(m["recall"], 80.0, name)

    def test_models_are_not_suspiciously_perfect(self):
        perf = MGPredictor()._performance()
        for name, m in perf["per_model"].items():
            self.assertLess(m["accuracy"], 99.0, f"{name} looks too good for noisy data")


class ConfidenceHonestyTests(unittest.TestCase):
    """Probabilities must stay graded: ambiguous input must not look certain."""

    @classmethod
    def setUpClass(cls):
        cls.p = MGPredictor()

    def test_displayed_probabilities_never_show_certainty(self):
        r = self.p.predict(30, "female", ["fatigue"],
                           ["droopy_eyelids", "double_vision", "difficulty_chewing", "limb_weakness",
                            "fatigable_weakness", "worse_end_of_day", "improves_with_rest"])
        for m in r["per_model"]:
            self.assertGreaterEqual(m["mg_probability_percent"], 3.0)
            self.assertLessEqual(m["mg_probability_percent"], 97.0)

    def test_partial_presentation_is_not_high_confidence(self):
        r = self.p.predict(40, "male", [], ["droopy_eyelids"])
        self.assertNotEqual(r["result"]["confidence_band"], "High")

    def test_empty_form_is_borderline_and_says_so(self):
        r = self.p.predict(30, "female")
        self.assertEqual(r["result"]["confidence_band"], "Borderline")
        self.assertIn("very little information", r["advice"]["next_steps"][0])

    def test_probabilities_are_spread_across_cases(self):
        cases = [([], []), (["fatigue"], []), ([], ["droopy_eyelids"]), ([], ["droopy_eyelids", "double_vision"]),
                 ([], ["fatigable_weakness", "worse_end_of_day", "improves_with_rest"])]
        probs = {self.p.predict(40, "female", s, f)["result"]["mg_probability_percent"] for s, f in cases}
        self.assertGreaterEqual(len(probs), 4)


class PredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = MGPredictor()

    def test_all_five_models_present(self):
        names = {m["model"] for m in self.p.predict(40, "female", ["fatigue"])["per_model"]}
        self.assertEqual(names, {"Naive Bayes", "Decision Tree", "Linear Regression",
                                 "Logistic Regression", "Random Forest"})

    def test_classic_mg_is_positive_with_high_confidence(self):
        r = self.p.predict(34, "female", ["fatigue"], CLASSIC_MG, name="Test Patient",
                           weight_kg=60, height_cm=165)["result"]
        self.assertEqual(r["status"], "POSITIVE")
        self.assertGreaterEqual(r["confidence_percent"], 90)
        self.assertEqual(r["models_positive"], "5 of 5")

    def test_no_symptoms_is_negative_with_high_confidence(self):
        r = self.p.predict(30, "male", [], [])["result"]
        self.assertEqual(r["status"], "NEGATIVE")
        self.assertGreaterEqual(r["confidence_percent"], 90)

    def test_interpretation_count_matches_models_positive(self):
        for flags in (CLASSIC_MG, []):
            r = self.p.predict(40, "male", [], flags)["result"]
            self.assertIn(f"{r['models_positive']} screening algorithms returned a positive result",
                          r["interpretation"])

    def test_non_mg_autoimmune_picture_is_negative(self):
        r = self.p.predict(45, "female", ["joint_pain", "rashes_skin_lesions", "fatigue", "hair_loss"], [])
        self.assertEqual(r["result"]["status"], "NEGATIVE")

    def test_patient_block_and_bmi(self):
        rep = self.p.predict(34, "female", [], [], name="  Priya Raman ", weight_kg=58, height_cm=162)["report"]
        self.assertEqual(rep["patient"]["name"], "Priya Raman")
        self.assertEqual(rep["patient"]["bmi"], 22.1)
        self.assertEqual(rep["patient"]["bmi_category"], "Normal")
        self.assertTrue(rep["report_id"].startswith("MYA-"))

    def test_breathing_or_swallowing_triggers_emergency_text(self):
        res = self.p.predict(50, "male", [], ["shortness_of_breath"])
        self.assertIsNotNone(res["advice"]["emergency"])
        self.assertIsNone(self.p.predict(25, "female", [], [])["advice"]["emergency"])

    def test_positive_gets_workup_negative_gets_watch_list(self):
        pos = self.p.predict(34, "female", [], CLASSIC_MG)["advice"]
        neg = self.p.predict(25, "female", [], [])["advice"]
        self.assertIn("tests_to_ask_about", pos)
        self.assertIn("watch_for", neg)
        self.assertNotIn("tests_to_ask_about", neg)

    def test_bad_input_rejected(self):
        for kwargs in (dict(age=-5, gender="male"), dict(age=30, gender="robot"),
                       dict(age=30, gender="male", symptoms=["not_a_symptom"]),
                       dict(age=30, gender="male", weight_kg=5000),
                       dict(age=30, gender="male", height_cm="tall")):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                self.p.predict(**kwargs)


class BloodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = MGPredictor()

    def test_blood_group_and_panel_in_report(self):
        r = self.p.predict(34, "female", [], [], blood_group="o+",
                           blood_tests={"hemoglobin": 9.8, "esr": 40, "tsh": 2.0},
                           antibodies={"achr": "positive"})
        self.assertEqual(r["report"]["patient"]["blood_group"], "O+")
        status = {row["key"]: row["status"] for row in r["blood"]["panel"]}
        self.assertEqual(status, {"hemoglobin": "Low", "esr": "High", "tsh": "Normal"})
        self.assertEqual(r["blood"]["abnormal_count"], 2)
        self.assertIsNotNone(r["blood"]["group_note"])
        self.assertTrue(any(s["section"] == "Blood investigations" for s in r["findings"]))

    def test_reference_ranges_depend_on_sex(self):
        low_f = self.p.predict(30, "female", [], [], blood_tests={"hemoglobin": 13.0})["blood"]["panel"][0]
        low_m = self.p.predict(30, "male", [], [], blood_tests={"hemoglobin": 13.0})["blood"]["panel"][0]
        self.assertEqual((low_f["status"], low_m["status"]), ("Normal", "Low"))

    def test_antibody_positive_raises_mg_probability(self):
        base = self.p.predict(34, "female", [], ["droopy_eyelids", "fatigable_weakness"])["result"]
        withab = self.p.predict(34, "female", [], ["droopy_eyelids", "fatigable_weakness"],
                                antibodies={"achr": "positive"})["result"]
        self.assertGreater(withab["mg_probability_percent"], base["mg_probability_percent"])

    def test_blood_is_optional_and_blank_is_ignored(self):
        r = self.p.predict(34, "female", [], [], blood_tests={"hemoglobin": ""}, blood_group="")
        self.assertEqual(r["blood"]["panel"], [])
        self.assertIsNone(r["report"]["patient"]["blood_group"])

    def test_bad_blood_input_rejected(self):
        for kw in (dict(blood_group="Z+"), dict(blood_tests={"hemoglobin": 900}),
                   dict(blood_tests={"sodium": 140}), dict(antibodies={"achr": "maybe"})):
            with self.assertRaises(ValueError, msg=str(kw)):
                self.p.predict(30, "male", **kw)

    def test_pdf_with_blood_panel_and_api(self):
        body = {"name": "Priya", "age": 34, "gender": "female", "blood_group": "B-",
                "blood_tests": {"hemoglobin": 11, "wbc": 7, "platelets": 250},
                "antibodies": {"achr": "positive", "musk": "negative"}, "red_flags": CLASSIC_MG}
        client = create_app().test_client()
        self.assertEqual(client.post("/predict", json=body).get_json()["blood"]["blood_group"], "B-")
        pdf = client.post("/report-pdf", json=body)
        self.assertTrue(pdf.data.startswith(b"%PDF"))
        self.assertEqual(client.post("/predict", json={**body, "blood_group": "Q"}).status_code, 400)


class HospitalLocatorTests(unittest.TestCase):
    BODY = {"age": 34, "gender": "female", "symptoms": ["fatigue"], "red_flags": CLASSIC_MG,
            "name": "Loc Test", "report_id": "MYA-TEST", "generated_at": "16 Oct 2026, 12:00"}

    @classmethod
    def setUpClass(cls):
        cls.client = create_app().test_client()

    def test_every_area_resolves_and_returns_real_hospitals(self):
        names = {h["name"] for h in H.HOSPITALS}
        for key in H.AREAS:
            r = H.locate({"area": key})
            self.assertIn(r["nearest"]["name"], names)
            self.assertIn(r["best"]["name"], names)
            self.assertEqual(len(r["all_hospitals"]), len(H.HOSPITALS))
            d = [h["distance_km"] for h in r["all_hospitals"]]
            self.assertLessEqual(r["nearest"]["distance_km"], min(d) + 1e-9)

    def test_nearest_is_really_the_closest(self):
        self.assertEqual(H.locate({"area": "velachery"})["nearest"]["id"], "prashanth")
        self.assertEqual(H.locate({"area": "chromepet"})["nearest"]["id"], "rela")
        self.assertEqual(H.locate({"area": "adyar"})["nearest"]["id"], "mgm_malar")

    def test_best_has_highest_evidence_score(self):
        r = H.locate({"area": "tambaram"})
        top = max(H.evidence_score(h) for h in H.HOSPITALS)
        self.assertEqual(r["best"]["evidence_score"], top)

    def test_outside_chennai_rejected(self):
        for loc in ({"lat": 19.07, "lng": 72.87}, {"lat": 28.6, "lng": 77.2}, {"area": "mumbai"}, {}):
            with self.assertRaises(ValueError):
                H.locate(loc)

    def test_prediction_is_identical_with_or_without_location(self):
        a = self.client.post("/predict", json=self.BODY).get_json()
        b = self.client.post("/predict", json={**self.BODY, "location": {"area": "velachery"}}).get_json()
        self.assertIsNone(a["care_locator"])
        self.assertEqual(b["care_locator"]["origin"]["label"], "Velachery")
        for key in ("result", "per_model", "findings", "blood", "counts", "performance", "advice"):
            self.assertEqual(a[key], b[key], key)

    def test_api_rejects_bad_location_and_pdf_includes_hospitals(self):
        bad = self.client.post("/predict", json={**self.BODY, "location": {"area": "mumbai"}})
        self.assertEqual(bad.status_code, 400)
        r = self.client.post("/report-pdf", json={**self.BODY, "location": {"area": "adyar"}})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.data.startswith(b"%PDF"))
        self.assertEqual(self.client.get("/hospital-areas").status_code, 200)
        self.assertEqual(len(self.client.get("/hospitals").get_json()["hospitals"]), len(H.HOSPITALS))


class FrontendParityTests(unittest.TestCase):
    """src/lib/mg-options.ts must list exactly the keys the backend accepts."""

    def test_option_keys_match(self):
        ts = (Path(__file__).resolve().parents[3] / "src" / "lib" / "mg-options.ts").read_text()
        keys = set(re.findall(r'key:\s*"([a-z_0-9]+)"', ts))
        self.assertEqual(keys, set(C.MG_SIGNS) | set(C.SYMPTOMS))

    def test_blood_options_match(self):
        ts = (Path(__file__).resolve().parents[3] / "src" / "lib" / "mg-blood.ts").read_text()
        self.assertEqual(set(re.findall(r'test:\s*"([a-z_0-9]+)"', ts)), set(C.BLOOD_TESTS))
        self.assertEqual(set(re.findall(r'group:\s*"([A-Z+-]+)"', ts)), set(C.BLOOD_GROUPS))

    def test_location_areas_and_bounds_match(self):
        ts = (Path(__file__).resolve().parents[3] / "src" / "lib" / "mg-locations.ts").read_text()
        self.assertEqual(set(re.findall(r'key:\s*"([a-z_]+)"', ts)), set(H.AREAS))
        nums = {k: float(v) for k, v in re.findall(r'(latMin|latMax|lngMin|lngMax):\s*([0-9.]+)', ts)}
        self.assertEqual(nums, {"latMin": H.BOUNDS["lat_min"], "latMax": H.BOUNDS["lat_max"],
                                "lngMin": H.BOUNDS["lng_min"], "lngMax": H.BOUNDS["lng_max"]})


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = create_app().test_client()

    def test_predict_ok(self):
        r = self.client.post("/predict", json={"name": "A B", "age": 45, "gender": "female",
                                               "symptoms": ["fatigue", "joint_pain"],
                                               "red_flags": ["fatigable_weakness"]})
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        for key in ("report", "result", "findings", "per_model", "performance", "advice"):
            self.assertIn(key, body)

    def test_predict_validation_errors(self):
        self.assertEqual(self.client.post("/predict", json={"gender": "male"}).status_code, 400)
        self.assertEqual(self.client.post("/predict", json={"age": "x", "gender": "male"}).status_code, 400)
        self.assertEqual(self.client.post("/predict", data="nope").status_code, 400)

    def test_pdf_download(self):
        r = self.client.post("/report-pdf", json={"name": "Priya Raman", "age": 34, "gender": "female",
                                                  "weight_kg": 58, "height_cm": 162,
                                                  "symptoms": ["fatigue"], "red_flags": CLASSIC_MG})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.mimetype, "application/pdf")
        self.assertTrue(r.data.startswith(b"%PDF"))
        self.assertIn("Priya_Raman", r.headers["Content-Disposition"])

    def test_pdf_keeps_report_id_from_screen(self):
        r = self.client.post("/report-pdf", json={"age": 30, "gender": "male", "report_id": "MYA-TEST-1"})
        self.assertIn("MYA-TEST-1", r.headers["Content-Disposition"])

    def test_pdf_builds_for_both_outcomes(self):
        p = MGPredictor()
        for flags in (CLASSIC_MG, []):
            self.assertTrue(build_report_pdf(p.predict(40, "male", [], flags, name="X")).startswith(b"%PDF"))

    def test_support_endpoints(self):
        for path in ("/health", "/options", "/metrics", "/dataset-breakdown", "/learning-curve"):
            self.assertEqual(self.client.get(path).status_code, 200, path)


if __name__ == "__main__":
    unittest.main()
