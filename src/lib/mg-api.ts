// Client for the Flask backend in backend/mg_predictor/api.py.
// Types below mirror MGPredictor.predict()'s return shape
// (backend/mg_predictor/mg/predictor.py) -- keep in sync if that changes.

const API_URL = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:5000";

export interface PredictRequest {
  age: number;
  gender: "male" | "female";
  symptoms: string[];
  red_flags: string[];
  name?: string | undefined;
  weight_kg?: number | undefined;
  height_cm?: number | undefined;
  duration_months?: number | undefined;
  blood_group?: string | undefined;
  blood_tests?: Record<string, number> | undefined;
  antibodies?: Record<string, string> | undefined;
  // Optional. South Chennai only: either a named area or GPS coordinates.
  location?: { area: string } | { lat: number; lng: number } | undefined;
  // Only sent when downloading the PDF, so the PDF carries the same ID/time as the screen.
  report_id?: string | undefined;
  generated_at?: string | undefined;
}

export interface PatientBlock {
  name: string;
  age: number;
  gender: "Male" | "Female";
  weight_kg: number | null;
  height_cm: number | null;
  bmi: number | null;
  bmi_category: string | null;
  symptom_duration: string;
  blood_group: string | null;
}

export interface BloodPanelRow {
  key: string;
  label: string;
  value: number;
  unit: string;
  reference: string;
  status: "Low" | "Normal" | "High";
}

export interface BloodBlock {
  blood_group: string | null;
  panel: BloodPanelRow[];
  antibodies: { key: string; label: string; status: "Positive" | "Negative" | "Not tested" }[];
  abnormal_count: number;
  group_note: string | null;
}

export interface ReportMeta {
  report_id: string;
  generated_at: string;
  patient: PatientBlock;
}

export interface ResultBlock {
  test: string;
  status: "POSITIVE" | "NEGATIVE";
  headline: string;
  confidence_percent: number;
  confidence_band: "High" | "Moderate" | "Borderline";
  mg_probability_percent: number;
  primary_model: string;
  models_positive: string;
  models_in_agreement: string;
  interpretation: string;
  clinical_pattern: string | null;
}

export interface FindingSection {
  section: string;
  kind: "mg" | "general" | "history" | "blood";
  items: string[];
}

export interface PerModelResult {
  model: string;
  result: "POSITIVE" | "NEGATIVE";
  flags_mg: boolean;
  mg_probability_percent: number;
  confidence_percent: number;
  is_primary: boolean;
}

export interface PerformanceMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  specificity: number;
  f1: number;
  roc_auc: number;
}

export interface Performance {
  primary_model: string;
  n_test: number;
  n_train: number;
  primary: PerformanceMetrics;
  per_model: Record<string, PerformanceMetrics>;
}

export interface Specialist {
  specialist: string;
  why: string;
  when: string;
}

export interface Advice {
  emergency: string | null;
  tone: "positive" | "negative";
  next_steps: string[];
  watch_for?: string[];
  tests_to_ask_about?: string[];
  daily_management?: string[];
  medication_safety?: string;
  about_treatment?: string;
  age_sex_context?: string | null;
}

export interface HospitalResult {
  id: string;
  name: string;
  area: string;
  address: string;
  phone: string | null;
  department: string;
  evidence: string[];
  evidence_score: number;
  max_score: number;
  mg_named: boolean;
  distance_km: number;
  maps_url: string;
  source: string;
}

export interface CareLocator {
  region: string;
  origin: { label: string; lat: number; lng: number; source: "area" | "gps" };
  nearest: HospitalResult;
  best: HospitalResult;
  same_hospital: boolean;
  summary: string;
  all_hospitals: HospitalResult[];
  method: string;
  disclaimer: string;
}

export interface PredictResponse {
  report: ReportMeta;
  result: ResultBlock;
  findings: FindingSection[];
  blood: BloodBlock;
  // null unless the user gave a South Chennai location
  care_locator: CareLocator | null;
  counts: { mg_signs: number; general_symptoms: number; history: number };
  per_model: PerModelResult[];
  performance: Performance;
  recommended_specialists: Specialist[];
  advice: Advice;
  footer: string;
}

export interface MetricsResponse {
  primary_model: string;
  cv_roc_auc: Record<string, number>;
  performance: Performance;
}

export interface DatasetBreakdown {
  "Myasthenia gravis": number;
  "Healthy (Normal)": number;
  "Other autoimmune diseases": number;
}

export interface LearningCurveModel {
  train_sizes: number[];
  train_roc_auc_mean: number[];
  val_roc_auc_mean: number[];
  val_roc_auc_std: number[];
}

export interface LearningCurveResponse {
  metric: string;
  n_dev: number;
  curves: Record<string, LearningCurveModel>;
}

export class MGApiError extends Error {
  constructor(message: string, public status?: number) {
    super(message);
    this.name = "MGApiError";
  }
}

const unreachable = () =>
  new MGApiError(
    `Could not reach the screening backend at ${API_URL}. Is it running? ` +
      "Start it with start-backend.bat (Windows) or ./start-backend.sh, or run `python api.py` inside backend/mg_predictor."
  );

async function getJSON<T>(path: string): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`);
  } catch {
    throw unreachable();
  }
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    throw new MGApiError(data?.error ?? `Backend returned an error (${response.status}).`, response.status);
  }
  return data as T;
}

function postJSON(path: string, body: PredictRequest): Promise<Response> {
  return fetch(`${API_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function predictMG(body: PredictRequest): Promise<PredictResponse> {
  let response: Response;
  try {
    response = await postJSON("/predict", body);
  } catch {
    throw unreachable();
  }

  const data = await response.json().catch(() => null);
  if (!response.ok) {
    throw new MGApiError(data?.error ?? `Backend returned an error (${response.status}).`, response.status);
  }
  return data as PredictResponse;
}

// Asks the backend to render the report as a PDF and returns the file for the browser to save.
export async function downloadReportPdf(body: PredictRequest): Promise<{ blob: Blob; filename: string }> {
  let response: Response;
  try {
    response = await postJSON("/report-pdf", body);
  } catch {
    throw unreachable();
  }
  if (!response.ok) {
    const data = await response.json().catch(() => null);
    throw new MGApiError(data?.error ?? `Could not create the PDF (${response.status}).`, response.status);
  }
  const disposition = response.headers.get("Content-Disposition") ?? "";
  const match = /filename\*?=(?:UTF-8'')?"?([^";]+)"?/i.exec(disposition);
  return { blob: await response.blob(), filename: match?.[1] ?? "MyaScreen_Report.pdf" };
}

export const getDatasetBreakdown = () => getJSON<DatasetBreakdown>("/dataset-breakdown");
export const getLearningCurve = () => getJSON<LearningCurveResponse>("/learning-curve");
export const getMetrics = () => getJSON<MetricsResponse>("/metrics");
