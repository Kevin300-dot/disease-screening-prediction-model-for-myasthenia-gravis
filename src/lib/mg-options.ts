// Symptom checklist shown in the form. Keys must match backend/mg_predictor/mg/config.py exactly
// (the backend rejects unknown keys; tests/test_pipeline.py checks that the two lists agree).

export interface Option {
  key: string;
  label: string;
  urgent?: boolean;
}

export interface OptionGroup {
  title: string;
  hint: string;
  options: Option[];
}

// MG-specific signs -> sent as `red_flags`
export const mgSignGroups: OptionGroup[] = [
  {
    title: "Eyes",
    hint: "Ocular involvement is often the first sign of myasthenia gravis.",
    options: [
      { key: "droopy_eyelids", label: "Drooping eyelid(s)" },
      { key: "double_vision", label: "Double vision" },
    ],
  },
  {
    title: "Face, mouth and throat",
    hint: "Bulbar muscles control chewing, speech and swallowing.",
    options: [
      { key: "difficulty_chewing", label: "Jaw tires or weakens while chewing" },
      { key: "slurred_nasal_speech", label: "Slurred or nasal speech" },
      { key: "voice_fatigue", label: "Voice becomes weak or hoarse during long conversations" },
      { key: "facial_weakness", label: "Facial weakness (flat expression, weak smile)" },
      { key: "difficulty_swallowing", label: "Difficulty swallowing or choking on food or liquid", urgent: true },
    ],
  },
  {
    title: "Neck, arms and legs",
    hint: "Limb involvement usually affects the muscles closest to the trunk.",
    options: [
      { key: "weak_neck_head_drop", label: "Weak neck or difficulty holding the head up" },
      { key: "limb_weakness", label: "Weak arms or legs" },
      { key: "difficulty_raising_arms", label: "Trouble lifting arms overhead (e.g. brushing hair)" },
      { key: "difficulty_climbing_stairs", label: "Trouble climbing stairs or rising from a chair" },
      { key: "hand_grip_weakness", label: "Weak hand grip or dropping objects" },
    ],
  },
  {
    title: "Fatigability pattern",
    hint: "Weakness that builds with use and eases with rest is the hallmark of MG.",
    options: [
      { key: "fatigable_weakness", label: "Weakness that worsens with repeated activity" },
      { key: "worse_end_of_day", label: "Symptoms worse in the evening than in the morning" },
      { key: "improves_with_rest", label: "Symptoms improve after rest or sleep" },
    ],
  },
  {
    title: "Breathing",
    hint: "Breathing difficulty needs urgent attention.",
    options: [
      { key: "shortness_of_breath", label: "Shortness of breath or trouble breathing", urgent: true },
      { key: "weak_cough", label: "Weak cough or difficulty clearing the throat" },
    ],
  },
];

// General symptoms -> sent as `symptoms`
export const generalSymptoms: Option[] = [
  { key: "fatigue", label: "Fatigue / chronic tiredness" },
  { key: "low_grade_fever", label: "Low-grade fever" },
  { key: "dizziness", label: "Dizziness" },
  { key: "weight_loss", label: "Unintentional weight loss" },
  { key: "rashes_skin_lesions", label: "Rashes or skin lesions" },
  { key: "joint_pain", label: "Joint pain" },
  { key: "joint_stiffness", label: "Stiffness in the joints" },
  { key: "muscle_pain", label: "Muscle pain or cramps" },
  { key: "hair_loss", label: "Brittle hair or hair loss" },
  { key: "dry_eyes_mouth", label: "Dry eyes and/or dry mouth" },
  { key: "headache", label: "Frequent headaches" },
  { key: "numbness_tingling", label: "Numbness or tingling" },
  { key: "general_unwell", label: "General 'unwell' feeling" },
];

// Medical history -> sent as `symptoms`
export const historyItems: Option[] = [
  { key: "family_history_autoimmune", label: "Family history of autoimmune disease" },
  { key: "other_autoimmune_condition", label: "Existing autoimmune condition (e.g. lupus, rheumatoid arthritis)" },
  { key: "thyroid_disease", label: "Thyroid disease" },
  { key: "known_thymus_abnormality", label: "Known thymus gland abnormality" },
  { key: "recent_infection_or_stress", label: "Recent infection, surgery or major stress" },
];

// Values are months, matching DURATION_OPTIONS in the backend config.
export const durationOptions = [
  { value: "0", label: "No current symptoms" },
  { value: "0.5", label: "Less than 1 month" },
  { value: "2", label: "1-3 months" },
  { value: "4.5", label: "3-6 months" },
  { value: "9", label: "6-12 months" },
  { value: "18", label: "1-2 years" },
  { value: "36", label: "More than 2 years" },
];
