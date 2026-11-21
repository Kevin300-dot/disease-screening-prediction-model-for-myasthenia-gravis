# MyaScreen

A web-based **myasthenia gravis (MG) screening aid**. A patient (or front-line clinician) enters symptoms,
MG-specific signs, medical history and optional blood results. Five machine-learning models score the
case, and the app returns a medical-style report - **POSITIVE or NEGATIVE with a calibrated confidence**,
findings, suggested investigations and specialists, and a downloadable PDF. With an optional location it also
names the **nearest hospital and the best-evidenced hospital for MG care in Chennai**.

> **Decision-support prototype, not a diagnostic device.** The models are trained on a *simulated* cohort
> (no real patient data), so the metrics below describe how well the models separate that simulation, not
> clinical accuracy. It needs validation on real, de-identified records before any clinical use.

## Features
- 5-step guided intake: patient details, 17 MG-specific signs, 13 general symptoms, history, optional blood tests
  (including anti-AChR / anti-MuSK antibodies) -> a 47-value feature vector
- Five algorithms compared side by side: Naive Bayes, Decision Tree, Linear Regression, Logistic Regression,
  Random Forest; Platt-scaled probabilities; the model with the best cross-validated ROC-AUC (currently Random Forest) decides the verdict (P(MG) >= 50%)
- Urgent banner when swallowing or breathing difficulty is ticked
- On-screen report plus PDF export (ReportLab) with model performance charts
- **Chennai hospital finder**: 50 localities across the city, 10 hospitals, transparent evidence-based ranking
  (see [HOSPITALS.md](HOSPITALS.md))

## Results (simulated hold-out set, 1,600 patients)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Naive Bayes | 0.908 | 0.897 | 0.870 | 0.883 | 0.966 |
| Decision Tree | 0.919 | 0.892 | 0.906 | 0.899 | 0.962 |
| Linear Regression | 0.938 | 0.929 | 0.914 | 0.921 | 0.978 |
| Logistic Regression | 0.938 | 0.921 | 0.925 | 0.923 | 0.979 |
| **Random Forest** (primary) | **0.941** | **0.926** | **0.925** | **0.926** | **0.983** |

The cohort is deliberately noisy (mild and seronegative MG, non-MG patients with eye signs, symptom over- and
under-reporting, recording errors), so scores are in the low 90s rather than near 100%. Displayed probabilities
are limited to 3-97% and the confidence band drops to *Moderate* or *Borderline* when the five models disagree or
the form is almost empty. Metrics are point estimates from one split, and were not tuned to hit a target.

## Architecture
React + TanStack Start front end -> `POST /predict` -> Flask API (`backend/mg_predictor/api.py`) -> `MGPredictor`
-> report JSON; `POST /report-pdf` renders the PDF server-side. Details: [AGENTS.md](AGENTS.md) and
[backend/mg_predictor/README.md](backend/mg_predictor/README.md).

## Run it
```sh
# backend (Python 3.10+)
cd backend/mg_predictor
pip install -r requirements.txt
python train.py          # optional: retrains and rewrites models/ and outputs/
python api.py            # or ./start-backend.sh from the repo root

# front end (Node 22+ / bun)
bun install && bun run dev
```
Tests: `python -m unittest tests.test_pipeline` (backend) and `bun run test` (front end).

## Limitations
- Simulated training data; point estimates from one split and seed, no confidence intervals
- Hospital data comes from each hospital's own website (see `source` in `mg/hospitals.py`) and can change;
  distances are straight-line estimates; always phone the hospital first
- A NEGATIVE result does not exclude MG
