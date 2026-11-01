// Blood details shown in the form. Keys must match BLOOD_TESTS / BLOOD_GROUPS / ANTIBODIES in
// backend/mg_predictor/mg/config.py (tests/test_pipeline.py checks that they agree).

export const bloodGroups = [
  { group: "A+" }, { group: "A-" }, { group: "B+" }, { group: "B-" },
  { group: "AB+" }, { group: "AB-" }, { group: "O+" }, { group: "O-" },
];

export interface BloodTestField {
  test: string;
  label: string;
  unit: string;
  male: string;
  female: string;
  placeholder: string;
  step: string;
}

// Reference ranges are the ones the backend uses to flag a value as Low / High.
export const bloodTests: BloodTestField[] = [
  { test: "hemoglobin", label: "Hemoglobin", unit: "g/dL", male: "13.5 - 17.5", female: "12.0 - 15.5", placeholder: "e.g. 13.2", step: "0.1" },
  { test: "wbc", label: "White blood cell count", unit: "x10³/µL", male: "4.0 - 11.0", female: "4.0 - 11.0", placeholder: "e.g. 7.2", step: "0.1" },
  { test: "platelets", label: "Platelet count", unit: "x10³/µL", male: "150 - 450", female: "150 - 450", placeholder: "e.g. 260", step: "1" },
  { test: "esr", label: "ESR", unit: "mm/hr", male: "0 - 15", female: "0 - 20", placeholder: "e.g. 10", step: "1" },
  { test: "crp", label: "C-reactive protein (CRP)", unit: "mg/L", male: "0 - 5", female: "0 - 5", placeholder: "e.g. 2", step: "0.1" },
  { test: "tsh", label: "TSH (thyroid)", unit: "mIU/L", male: "0.4 - 4.0", female: "0.4 - 4.0", placeholder: "e.g. 2.1", step: "0.01" },
  { test: "ck", label: "Creatine kinase (CK)", unit: "U/L", male: "39 - 308", female: "26 - 192", placeholder: "e.g. 120", step: "1" },
];

export const antibodyFields = [
  { key: "achr", label: "Anti-AChR antibody" },
  { key: "musk", label: "Anti-MuSK antibody" },
] as const;

export type AntibodyStatus = "not_tested" | "positive" | "negative";
export const antibodyStatusOptions: { value: AntibodyStatus; label: string }[] = [
  { value: "not_tested", label: "Not tested" },
  { value: "positive", label: "Positive" },
  { value: "negative", label: "Negative" },
];
