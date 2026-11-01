# MyaScreen backend - Myasthenia Gravis screening report

A Python backend that takes a patient's details and symptom checklist and returns a full
screening report: **Myasthenia gravis POSITIVE / NEGATIVE**, the model confidence, the findings,
recommended work-up and specialists - and can render the same report as a **PDF**.

## Quick start

```bash
pip install -r requirements.txt
python generate_dataset.py       # (re)creates data/mg_screening_dataset.csv  (already included)
python train.py                  # trains 5 models, writes charts + metrics to outputs/ (already included)
python api.py                    # REST API on http://127.0.0.1:5000
python -m unittest discover -s tests -v
```

Command line (prints the report and can save the PDF):

```bash
python predict_cli.py                     # interactive
python predict_cli.py --name "Priya Raman" --age 34 --gender female --weight 58 --height 162 \
    --symptoms fatigue --red-flags droopy_eyelids,double_vision,fatigable_weakness \
    --blood-group O+ --blood hemoglobin=12.8,esr=12 --achr positive --pdf report.pdf
python predict_cli.py --list              # every valid symptom key
```

## API

| Endpoint | Purpose |
|---|---|
| `POST /predict` | full screening report as JSON |
| `POST /report-pdf` | the same report as a downloadable PDF |
| `GET /options` `/metrics` `/dataset-breakdown` `/learning-curve` `/health` | supporting data |

```bash
curl -X POST http://127.0.0.1:5000/report-pdf -H "Content-Type: application/json" -o report.pdf \
  -d '{"name":"Priya Raman","age":34,"gender":"female","weight_kg":58,"height_cm":162,
       "duration_months":4.5,"blood_group":"O+","blood_tests":{"hemoglobin":12.8,"esr":12},"antibodies":{"achr":"positive"},
       "symptoms":["fatigue"],"red_flags":["droopy_eyelids","double_vision"]}'
```

* `blood_group` (optional) = `A+ A- B+ B- AB+ AB- O+ O-`. Recorded on the report only - there is no known link between
  ABO/Rh group and MG, so it is deliberately **not** a model input.
* `blood_tests` (optional, any subset) = `hemoglobin, wbc, platelets, esr, crp, tsh, ck`. Each value is compared with a
  sex-specific reference range and shown as Low / Normal / High in the report; abnormal values feed the model.
* `antibodies` (optional) = `{"achr": "positive|negative|not_tested", "musk": ...}`; a positive result feeds the model.
* `red_flags` = MG-specific signs (17 keys: eyes, face/throat, neck/limbs, fatigability pattern, breathing)
* `symptoms` = general symptoms (13) and medical history (5)
* `name`, `weight_kg`, `height_cm`, `duration_months` are optional; `age` and `gender` are required.

## The dataset

`data/mg_screening_dataset.csv` (built by `generate_dataset.py`, seed 42) contains **8,000 simulated patients**:

| Cohort | Patients |
|---|---|
| Myasthenia gravis (ocular / generalized / bulbar-predominant) | 3,200 |
| Healthy | 2,400 |
| Look-alike conditions (Lambert-Eaton, thyroid eye disease, Guillain-Barre, myositis, MS, lupus, RA, Sjogren, Hashimoto, fibromyalgia, ...) | 2,400 |

Every patient has the full checklist the app asks (47 model inputs). Symptom frequencies follow the textbook
picture of MG (ptosis, diplopia, fatigable weakness that is worse in the evening and better with rest, bulbar and
limb involvement, bimodal age/sex onset). Atypical/mild MG cases, "worried well" patients and 1.2% answer noise
are mixed in so the classes overlap slightly - results are high but not a perfect 100%.

**Because the patients are simulated, the accuracy figures describe the simulation, not real-world performance.**

## Algorithms and results (simulated data)

Naive Bayes, Decision Tree, Linear Regression (as a linear probability model), Logistic Regression, Random Forest.
80% development / 20% untouched test split (stratified); calibrated with Platt scaling on 5-fold out-of-fold
predictions; decision rule: calibrated P(MG) >= 50%. The primary model is picked by cross-validated ROC-AUC.

Hold-out test set (1,600 patients), see `outputs/metrics.csv`:

| Algorithm | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Naive Bayes | 95.9% | 93.0% | 97.2% | 95.0% | 99.3% |
| Decision Tree | 95.2% | 94.5% | 93.6% | 94.0% | 98.4% |
| Linear Regression | 97.9% | 98.4% | 96.4% | 97.4% | 99.4% |
| **Logistic Regression (primary)** | **99.0%** | **98.9%** | **98.6%** | **98.7%** | **99.7%** |
| Random Forest | 98.1% | 97.7% | 97.5% | 97.6% | 99.6% |

These figures describe the simulated cohort, not real patients. For confidence intervals, calibration, predictive values at
realistic prevalence and robustness to noisy answers, run `python evaluate_honest.py` and read `../../docs/EVALUATION.md`.

## The report

`result` in the response holds: `status` (POSITIVE/NEGATIVE), `confidence_percent`, `confidence_band`,
`mg_probability_percent`, the interpretation text and (for positive results) the clinical pattern.
Around it: `report` (ID, date, patient name/age/sex/weight/height/BMI/duration), `findings` (what the patient
reported, grouped), `per_model`, `performance`, `recommended_specialists` and `advice`
(next steps, confirmatory tests, day-to-day management, medication safety; or a watch-list for negative results).
Breathing or swallowing difficulty adds a critical-finding box.

## Outputs (`outputs/`)

`confusion_matrices.png`, `model_comparison_bar.png`, `accuracy_precision_bar.png`, `dataset_pie.png`,
`outcome_pie.png`, `roc_curves.png`, `feature_importance_bar.png`, `learning_curve.json`, `metrics.csv/json`.

## Project layout

```
generate_dataset.py   builds the simulated dataset
train.py              training + evaluation + charts
predict_cli.py        interactive / CLI report (+ --pdf)
api.py                Flask REST backend
mg/config.py          paths + every symptom definition (single source of truth)
mg/data.py            loading + feature building
mg/models.py          the 5 algorithms + calibration
mg/evaluation.py      metrics and charts
mg/predictor.py       builds the report (verdict, confidence, findings, advice)
mg/knowledge.py       clinical text: tests, specialists, advice
mg/report_pdf.py      PDF layout (ReportLab)
tests/                unit, API, PDF and front-end parity tests
```


## Blood details

The simulated dataset includes nine blood-derived flags (anti-AChR / anti-MuSK antibodies, low hemoglobin,
abnormal WBC / platelets / TSH, raised ESR / CRP / CK) with disease-specific frequencies - e.g. AChR antibodies in
~45-55% of simulated MG patients, raised CK in myositis, raised ESR/CRP in lupus and rheumatoid arthritis, abnormal
TSH in thyroid disease. As with every other column these are simulated, so they describe the simulation, not real labs.

## If the frontend says "Could not reach the screening backend"

The browser cannot connect to `http://127.0.0.1:5000`. Start the backend first (`start-backend.bat` on Windows,
`./start-backend.sh` elsewhere), keep that window open, then reload the page. `api.py` works even if
`flask-cors` is missing, and retrains the model automatically if the saved one cannot be loaded.
