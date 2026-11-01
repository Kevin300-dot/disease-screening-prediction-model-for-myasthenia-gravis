"""Clinical content used in the report: advice, investigations and specialists.

Rule-based reference text (not learned from the dataset).
"""
from __future__ import annotations

from .config import MG_SIGNS as RED_FLAGS   # key -> (label, group, urgent)  (re-exported)

EMERGENCY_TEXT = (
    "Difficulty breathing or swallowing, a weak cough, or sudden worsening weakness can signal "
    "a myasthenic crisis, which is a medical emergency. Call your local emergency number "
    "(112 in India) or go to the nearest emergency department now."
)

SPECIALISTS = {
    "pcp": ("General physician / primary care doctor",
            "First point of contact: examination, basic blood work, and referral."),
    "neuro": ("Neurologist (ideally neuromuscular specialist)",
              "Leads MG diagnosis (antibody tests, nerve conduction / single-fibre EMG) and treatment."),
    "neuro_ophth": ("Neuro-ophthalmologist / ophthalmologist",
                    "Evaluates drooping eyelids and double vision; ice-pack test; eye-focused management."),
    "slp": ("Speech-language pathologist (with ENT if needed)",
            "Assesses swallowing, chewing and speech difficulties and teaches safe-swallow strategies."),
    "pulm": ("Pulmonologist",
             "Checks breathing muscle strength (spirometry) when breathlessness or weak cough is present."),
    "thoracic": ("Thoracic surgeon",
                 "After confirmation: assesses the thymus (chest CT/MRI) for thymoma or thymectomy."),
    "endo": ("Endocrinologist",
             "Thyroid disease commonly co-exists with MG and can worsen symptoms."),
    "rheum": ("Rheumatologist / clinical immunologist",
              "Joint, skin or dry-eye/mouth symptoms can point to other autoimmune conditions "
              "that overlap with MG."),
    "physio": ("Physiotherapist / occupational therapist",
               "Energy-pacing and safe strengthening once the diagnosis is established."),
}

# ------------------------------------------------------------ positive result
NEXT_STEPS_URGENCY = {
    "high": "Arrange a neurology evaluation within the next 1-2 weeks (sooner if weakness is progressing).",
    "moderate": "Arrange a neurology evaluation within the next 2-4 weeks and repeat the screening if "
                "symptoms change.",
}

PREPARE_FOR_VISIT = [
    "Describe exactly when symptoms appear, what makes them worse, and what makes them better.",
    "Keep a short symptom diary (time of day, activity, which muscles) and take photos or videos of "
    "eyelid droop or speech changes - they help the specialist.",
    "Take a full list of current medicines and supplements to every appointment.",
]

# Every test stays listed; the ones most relevant to the reported findings come first.
TESTS_TO_ASK_ABOUT = [
    {"text": "Serum acetylcholine-receptor (AChR) antibodies; if negative, MuSK and LRP4 antibodies",
     "groups": None},
    {"text": "Repetitive nerve stimulation and/or single-fibre EMG (confirms the neuromuscular-junction "
             "problem)", "groups": None},
    {"text": "Bedside tests such as the ice-pack test (for eyelid droop) and a clinical fatigability exam",
     "groups": {"Ocular", "Fatigability pattern"}},
    {"text": "Chest CT or MRI to assess the thymus gland", "groups": None},
    {"text": "Thyroid function tests (thyroid disease is common alongside MG)", "groups": None},
    {"text": "Spirometry (forced vital capacity) to check breathing-muscle strength",
     "groups": {"Respiratory"}},
]

DAILY_MANAGEMENT = [
    {"text": "Pace your energy: schedule demanding tasks for when you feel strongest (often mornings) "
             "and rest in between.", "groups": None},
    {"text": "Sleep well and treat infections early - illness, stress and exhaustion commonly trigger "
             "flare-ups.", "groups": None},
    {"text": "Avoid overheating (hot baths, intense sun) if it worsens your weakness.", "groups": None},
    {"text": "Eat softer foods, take smaller bites and sit upright when eating if chewing or swallowing "
             "is tiring.", "groups": {"Bulbar"}},
    {"text": "For double vision, discuss patching or prism glasses with an eye specialist and avoid "
             "driving while affected.", "groups": {"Ocular"}},
    {"text": "Ask about a gentle, supervised exercise plan - avoid pushing to exhaustion.", "groups": None},
]

MEDICATION_SAFETY = (
    "Some medicines can worsen myasthenia gravis in some people (for example fluoroquinolone, "
    "aminoglycoside and macrolide antibiotics, beta-blockers, and IV magnesium). Never stop a medicine "
    "yourself, but make sure every doctor, dentist and pharmacist knows about this result."
)

TREATMENT_NOTE = (
    "If MG is confirmed, effective treatments exist (such as pyridostigmine, immunosuppressive therapy, "
    "newer targeted biologics, IV immunoglobulin or plasma exchange for flares, and thymectomy in "
    "selected patients). These are chosen and monitored by the treating neurologist."
)

AGE_SEX_NOTE_EARLY = (
    "Early-onset myasthenia gravis (before about 40 years) is diagnosed more often in women, which fits "
    "the age and sex entered."
)
AGE_SEX_NOTE_LATE = (
    "Late-onset myasthenia gravis (after about 60 years) is diagnosed more often in men, which fits "
    "the age and sex entered."
)

# ------------------------------------------------------------ negative result
NEGATIVE_NEXT_STEPS = [
    "No MG-specific work-up is indicated on the information provided.",
    "If symptoms persist for more than 2-4 weeks, or any of the warning signs below appear, repeat the "
    "screening and consult a physician.",
    "Maintain regular sleep, balanced nutrition and gentle activity; a routine check-up with basic blood "
    "tests is reasonable if general symptoms continue.",
]

REPORT_FOOTER = (
    "Automated screening report generated by MyaScreen. Results come from machine-learning models "
    "trained on a simulated clinical dataset and have not been validated on real patients. The tool is "
    "meant for people who already have suspicious symptoms: because myasthenia gravis is rare, a POSITIVE "
    "result in a general population would usually be a false alarm. Results must be correlated with clinical "
    "examination and laboratory confirmation."
)


BLOOD_GROUP_NOTE = (
    "Recorded for the patient file. It does not influence the screening result, but it is needed if "
    "IVIG or plasma exchange (used in myasthenic flares) is ever required."
)
