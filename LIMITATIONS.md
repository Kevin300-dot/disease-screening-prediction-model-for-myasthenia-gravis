# Limitations and Threats to Validity

**Hybrid Quantum Machine Learning for Myasthenia Gravis Screening**

This section is deliberately explicit. Every limitation below is one we identified
ourselves during development. Stating them plainly is part of the contribution:
a screening prototype whose failure modes are documented is more useful than one
whose weaknesses have to be discovered by whoever deploys it.

---

## 1. Data provenance

**The dataset is not real prospectively-collected clinical data.** It is a
structured dataset of patient records across ~116 diagnosis categories. It has not
been validated against a clinical registry, and the mechanism by which symptoms and
lab values were associated with diagnoses is not something we can independently
verify.

Consequences:

- Feature–outcome relationships may be cleaner and more separable than in real
  clinical data, where presentations are ambiguous, comorbid, and incomplete.
- Real-world missingness is *informative* (a test is ordered because a clinician
  already suspects something). Median imputation assumes missingness is benign,
  which would not hold in a real EHR.
- Absolute performance numbers should be read as **upper bounds** under favourable
  conditions, not as expected deployment performance.

## 2. Test set size

The held-out test set contains roughly 30 patients, of whom only a small number are
MG cases. This is the single largest statistical limitation.

- Every point estimate is fragile: one patient reclassified moves sensitivity by
  several percentage points.
- This is why all headline metrics are now reported with **bootstrap 95% confidence
  intervals** rather than as bare numbers. The intervals are wide, and that width
  is the honest signal.
- Sensitivity and specificity intervals in particular may span a large range. A
  point estimate quoted without its interval would overstate what the evaluation
  actually establishes.

## 3. Low specificity is a deliberate design choice, not an accident

The screening threshold is selected to hold sensitivity at or above 0.80, then
maximise specificity subject to that floor.

- This is correct for a **screening / triage** tool: missing an MG case (false
  negative) is clinically worse than an unnecessary neurology referral.
- But it means the tool generates false positives by construction, and its
  positive predictive value in a low-prevalence population would be poor.
- It is **not a diagnostic test** and must not be evaluated as one. Its intended
  role is prioritising who receives confirmatory AChR/MuSK antibody testing.

## 4. The quantum component is not demonstrated to help

The hybrid model uses a 2-qubit variational circuit with 6 trainable quantum
parameters, run on a noiseless simulator (`default.qubit`).

- We ran an explicit **classical twin ablation**: an identical architecture with
  the quantum layer replaced by `Dense(2, relu)`, trained on the same split with
  the same recipe and the same threshold-selection procedure.
- The comparison is also reported with a **paired bootstrap confidence interval**
  on the ROC-AUC difference. If that interval includes zero, the two models are
  statistically indistinguishable on this test set and we say so.
- At this scale, no quantum advantage should be expected on theoretical grounds
  either. A 2-qubit circuit is classically simulable in microseconds. The quantum
  layer is a *feasibility and integration* demonstration, not a performance claim.
- Results come from a **noiseless simulator**. Real quantum hardware introduces
  decoherence and gate error that would degrade these results, not improve them.

## 5. Explainability is an approximation

Two explainability layers are reported, and each has a boundary:

- **Global**: permutation importance on the deployed hybrid model. Permutation
  breaks feature correlations, so when features are correlated it can attribute
  importance in ways that do not reflect a causal mechanism.
- **Local (per-patient)**: counterfactual occlusion — neutralise one input,
  re-score, attribute the change. This is a **one-feature-at-a-time**
  approximation. It does not decompose interaction effects the way exact Shapley
  values would, and contributions will not sum exactly to the prediction.
- Neither method establishes causation. "This feature raised the model's estimate"
  is a statement about the model, not about the patient's biology.

## 6. Recommendations are not clinically validated

The recommendation layer is a rule-based mapping from locally-important features to
suggested follow-up actions.

- The rules were written from general clinical reasoning, **not derived from
  guidelines and not reviewed by a neurologist**.
- They are illustrative of how explainability can drive action. They are not
  evidence-based clinical recommendations and should not be presented as such.
- Ordering is now personalised (driven by each patient's own local attributions),
  but the underlying text of each rule remains generic.

## 7. Generalisation beyond the observed differential

A standard random split shows the model every disease category during training. A
model can therefore score well by learning the boundary of *this specific* set of
~115 competing diagnoses rather than learning MG-like patterns.

- We address this with **Leave-Diseases-Out (LDO) validation**: whole disease
  categories are withheld from training entirely, and the model is tested on MG
  versus conditions it has never seen.
- If LDO performance drops materially relative to the random split, that drop
  quantifies how much of the internal score depended on having seen the full
  differential. We report the gap either way.
- LDO is run using the **classical twin**, not the quantum hybrid, because each
  repeat retrains from scratch and the quantum layer's per-sample execution is too
  slow for repeated training. This is a stated approximation.
- LDO still uses a single dataset. It simulates the *spirit* of external validation.
  It is **not a substitute** for testing on a second institution's data.

## 8. Calibration

Predicted probabilities are assessed with a reliability curve and Expected
Calibration Error alongside Brier score.

- Calibration is measured on the same small test set, so the calibration curve is
  itself uncertain — quantile binning with few bins is used to avoid nonsense from
  near-empty bins.
- If ECE is high, the outputs should be treated as a **ranking score**, not as
  literal probabilities. A displayed "62%" would then not mean 62 of 100 such
  patients have MG.
- No post-hoc recalibration (Platt scaling, isotonic regression) has been applied;
  there is not enough held-out data to fit one without overfitting it.

## 9. Scope of the screening question

The model distinguishes MG from other diagnoses **within a population already under
clinical workup**. It has not been evaluated as a general-population screening tool,
where MG prevalence is on the order of 1 in 5,000 and the false-positive burden
would be dramatically higher.

## 10. Reproducibility caveats

- A single random seed (42) is used throughout. Results under other seeds have not
  been characterised; with a test set this small, seed sensitivity is likely to be
  non-trivial.
- Cross-validation uses 3 folds, chosen for runtime with the quantum layer. More
  folds would give a more stable estimate.
- `Sickness_Duration_Months` is excluded from features by default, following the
  dataset's own availability flag and consistent with an early-screening framing.
  Including it would likely raise measured performance while weakening the claim
  that the tool works at *first presentation*.

---

## What would be needed to move this beyond a prototype

1. Prospective, real clinical data from at least two institutions.
2. A test set large enough for confidence intervals narrow enough to support a
   clinical claim (hundreds of MG cases, not tens).
3. Neurologist review of both the feature set and the recommendation rules.
4. Genuine external validation on a second site's patient population.
5. Prospective evaluation of whether the tool changes referral decisions or
   time-to-diagnosis — the outcomes that would actually matter.
6. Execution on real quantum hardware if any quantum claim is to be made at all.
