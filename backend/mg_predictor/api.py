"""REST backend.

Run:   python api.py            (serves on http://127.0.0.1:5000)

POST /predict       full screening report as JSON
POST /report-pdf    the same report rendered as a downloadable PDF
  {
    "name": "Priya Raman",                 # optional
    "age": 34,
    "gender": "female",
    "weight_kg": 58, "height_cm": 162,     # optional
    "duration_months": 4.5,                # optional
    "blood_group": "O+",                   # optional: A+ A- B+ B- AB+ AB- O+ O-
    "blood_tests": {"hemoglobin": 12.8, "wbc": 7.1, "platelets": 250, "esr": 12, "crp": 3,
                    "tsh": 2.1, "ck": 90},   # optional, any subset
    "antibodies": {"achr": "positive", "musk": "not_tested"},   # optional
    "symptoms": ["fatigue", "thyroid_disease"],        # general symptoms + medical history
    "red_flags": ["droopy_eyelids", "double_vision"],  # MG-specific signs
    "report_id": "...", "generated_at": "..."          # optional (/report-pdf: keep PDF == screen)
  }

    + optional  "location": {"area": "velachery"}   or   {"lat": 12.98, "lng": 80.22}
      (Chennai only) -> response gains "care_locator": nearest + best hospital for MG

GET /hospital-areas      Chennai localities for the location picker
GET /hospitals           the verified hospital list used by the locator
GET /options             valid symptom / MG-sign keys, grouped
GET /metrics             held-out test metrics for all five algorithms
GET /dataset-breakdown   MG vs. Healthy vs. Other-autoimmune counts (pie chart)
GET /learning-curve      ROC-AUC vs. training-set size for all five algorithms
GET /health
"""
from __future__ import annotations

import io
import json
import re

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from flask import Flask, jsonify, request, send_file

try:  # flask-cors is optional: without it we add the CORS headers ourselves
    from flask_cors import CORS
except ImportError:  # pragma: no cover
    CORS = None

from mg import config as C
from mg import hospitals as H
from mg.data import dataset_breakdown, load_dataset
from mg.predictor import MGPredictor
from mg.report_pdf import build_report_pdf


def _predict_from_body(pred: MGPredictor, body):
    if not isinstance(body, dict):
        raise ValueError("Request body must be a JSON object")
    for field in ("age", "gender"):
        if field not in body:
            raise ValueError(f"Missing required field: {field}")
    # Validate the (optional) location first so a bad location fails fast with a clear message.
    care = H.locate(body["location"]) if body.get("location") else None
    result = pred.predict(
        body["age"], body["gender"], body.get("symptoms"), body.get("red_flags"),
        name=body.get("name"), weight_kg=body.get("weight_kg"), height_cm=body.get("height_cm"),
        duration_months=body.get("duration_months"), blood_group=body.get("blood_group"),
        blood_tests=body.get("blood_tests"), antibodies=body.get("antibodies"),
        report_id=body.get("report_id") or None, generated_at=body.get("generated_at") or None,
    )
    # The care locator is added AFTER prediction and never influences it.
    result["care_locator"] = care
    return result


def _load_predictor() -> MGPredictor:
    """Load the saved models; if they are missing or were saved by a different scikit-learn
    version (a common cause of a backend that will not start), retrain automatically."""
    try:
        return MGPredictor()
    except Exception as exc:  # noqa: BLE001
        print(f"[MyaScreen] Could not load the saved model ({exc!r}). Retraining now (about a minute)...",
              flush=True)
        if not C.DATA_PATH.exists():
            subprocess.check_call([sys.executable, str(C.ROOT / "generate_dataset.py")], cwd=C.ROOT)
        subprocess.check_call([sys.executable, str(C.ROOT / "train.py")], cwd=C.ROOT)
        return MGPredictor()


def create_app(predictor: MGPredictor | None = None) -> Flask:
    app = Flask(__name__)
    # The frontend runs on a different origin (Vite dev server / hosted preview), so CORS must be on.
    if CORS is not None:
        CORS(app, expose_headers=["Content-Disposition"])
    else:
        @app.after_request
        def _cors(resp):
            resp.headers["Access-Control-Allow-Origin"] = "*"
            resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
            resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            resp.headers["Access-Control-Expose-Headers"] = "Content-Disposition"
            return resp

    @app.after_request
    def _private_network(resp):
        resp.headers["Access-Control-Allow-Private-Network"] = "true"
        return resp

    pred = predictor or _load_predictor()
    breakdown_cache = dataset_breakdown(load_dataset())

    @app.get("/health")
    def health():
        return jsonify(status="ok", primary_model=pred.best)

    @app.get("/options")
    def options():
        return jsonify(
            symptoms=C.GENERAL_SYMPTOMS,
            history=C.HISTORY_ITEMS,
            red_flags={k: v[0] for k, v in C.MG_SIGNS.items()},
            red_flag_groups={g: [k for k, v in C.MG_SIGNS.items() if v[1] == g] for g in C.MG_SIGN_GROUPS},
            duration_options=C.DURATION_OPTIONS,
            blood_groups=C.BLOOD_GROUPS,
            blood_tests={k: {"label": v[0], "unit": v[1], "male": v[2], "female": v[3]}
                         for k, v in C.BLOOD_TESTS.items()},
            antibodies=C.ANTIBODIES,
            gender=["male", "female"],
            chennai_areas=H.area_options(),
        )

    @app.get("/hospital-areas")
    def hospital_areas():
        return jsonify(region=H.REGION, areas=H.area_options(), bounds=H.BOUNDS)

    @app.get("/hospitals")
    def hospitals():
        return jsonify(region=H.REGION, disclaimer=H.DISCLAIMER, hospitals=[
            {k: h[k] for k in ("id", "name", "area", "address", "phone", "department", "evidence", "source")}
            for h in H.HOSPITALS])

    @app.get("/metrics")
    def metrics():
        b = pred.bundle
        return jsonify(primary_model=b["best_model"], cv_roc_auc=b["cv_roc_auc"],
                       test_metrics=b["test_metrics"], performance=pred._performance())

    @app.get("/dataset-breakdown")
    def dataset_breakdown_route():
        return jsonify(breakdown_cache)

    @app.get("/learning-curve")
    def learning_curve_route():
        path = C.OUTPUT_DIR / "learning_curve.json"
        if not path.exists():
            return jsonify(error="Learning curve not found. Run `python train.py` to generate it."), 404
        with open(path) as f:
            return jsonify(json.load(f))

    @app.post("/predict")
    def predict():
        try:
            result = _predict_from_body(pred, request.get_json(silent=True))
        except (ValueError, TypeError) as e:
            return jsonify(error=str(e)), 400
        return jsonify(result)

    @app.post("/report-pdf")
    def report_pdf():
        try:
            result = _predict_from_body(pred, request.get_json(silent=True))
        except (ValueError, TypeError) as e:
            return jsonify(error=str(e)), 400
        pdf = build_report_pdf(result)
        who = re.sub(r"[^A-Za-z0-9]+", "_", result["report"]["patient"]["name"]).strip("_") or "Patient"
        filename = f"MyaScreen_Report_{who}_{result['report']['report_id']}.pdf"
        return send_file(io.BytesIO(pdf), mimetype="application/pdf", as_attachment=True,
                         download_name=filename)

    return app


if __name__ == "__main__":
    # 0.0.0.0 so the server also accepts connections through a tunnel (e.g. ngrok) when the
    # front-end is hosted in the cloud (like Lovable's preview).
    create_app().run(host="0.0.0.0", port=5000, debug=False)
