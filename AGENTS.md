# Architecture notes
- The app is a medical-style screening report: 4-step form (patient, MG symptoms, history, report) -> `POST /predict` -> on-screen report (`src/components/report/ReportView.tsx`) -> `POST /report-pdf` for the PDF download.
- Symptom keys live in `backend/mg_predictor/mg/config.py` and are mirrored in `src/lib/mg-options.ts`; `backend/mg_predictor/tests/test_pipeline.py` fails if the two lists drift apart.
- `src/lib/mg-api.ts` mirrors the backend response shape (`mg/predictor.py`). Change both together. (Earlier versions required this file to stay byte-identical; that restriction ended when the report redesign changed the API.)
- The PDF is rendered by the backend (`mg/report_pdf.py`, ReportLab), not in the browser, so the front-end needs no PDF dependency.
- Express the visual direction through global semantic tokens in `src/styles.css` and shared Button variants. The report itself uses the paper-style `.report-*` classes.
