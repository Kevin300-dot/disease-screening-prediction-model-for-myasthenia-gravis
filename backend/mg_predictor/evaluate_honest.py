"""Honest evaluation of the shipped model bundle.

The headline test-set numbers from train.py only say how well the models separate the *simulated*
cohort. This script adds the checks a reader needs to interpret them:

  1. bootstrap 95% confidence intervals,
  2. calibration (Brier score, expected calibration error, reliability curve),
  3. PPV / NPV at realistic MG prevalence (the simulation has 40% MG; real screening populations do not),
  4. sensitivity by MG form and specificity by look-alike condition,
  5. robustness when the patient's answers are noisy.

Run:  python evaluate_honest.py      (writes outputs/honest_evaluation.json, .png and docs/EVALUATION.md)
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from mg import config as C
from mg.data import build_xy, load_dataset
from mg.models import positive_score

RNG = np.random.default_rng(C.RANDOM_STATE)
N_BOOT = 2000
PREVALENCES = [0.0002, 0.01, 0.05, 0.20, 0.40]
PREV_LABELS = {0.0002: "general population (~0.02%)", 0.01: "1%", 0.05: "5%", 0.20: "20%",
               0.40: "40% (the simulated cohort)"}
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20]
DOCS = C.ROOT.parent.parent / "docs"


def _ci(values):
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def _confusion(y, pred):
    tp = int(((y == 1) & (pred == 1)).sum()); tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum()); fn = int(((y == 1) & (pred == 0)).sum())
    return tp, tn, fp, fn


def _rates(y, pred):
    tp, tn, fp, fn = _confusion(y, pred)
    return {"accuracy": (tp + tn) / len(y), "sensitivity": tp / max(tp + fn, 1),
            "specificity": tn / max(tn + fp, 1)}


def bootstrap(y, score, pred):
    n = len(y)
    out = {k: [] for k in ("accuracy", "sensitivity", "specificity", "roc_auc")}
    for _ in range(N_BOOT):
        idx = RNG.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        r = _rates(yb, pred[idx])
        for k in ("accuracy", "sensitivity", "specificity"):
            out[k].append(r[k])
        out["roc_auc"].append(roc_auc_score(yb, score[idx]))
    point = _rates(y, pred); point["roc_auc"] = float(roc_auc_score(y, score))
    return {k: {"value": point[k], "ci95": _ci(v)} for k, v in out.items()}


def calibration(y, prob, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    ids = np.clip(np.digitize(prob, edges) - 1, 0, bins - 1)
    rows, ece = [], 0.0
    for b in range(bins):
        m = ids == b
        if m.sum() == 0:
            continue
        conf, obs = float(prob[m].mean()), float(y[m].mean())
        ece += m.mean() * abs(conf - obs)
        rows.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(m.sum()),
                     "mean_predicted": conf, "observed_rate": obs})
    return {"brier": float(np.mean((prob - y) ** 2)), "ece": float(ece), "bins": rows}


def predictive_values(sens, spec):
    rows = []
    for p in PREVALENCES:
        ppv = sens * p / (sens * p + (1 - spec) * (1 - p))
        npv = spec * (1 - p) / (spec * (1 - p) + (1 - sens) * p)
        rows.append({"prevalence": p, "ppv": float(ppv), "npv": float(npv),
                     "false_positives_per_true_positive": float((1 - ppv) / ppv) if ppv else None})
    return rows


def noisy(X, flip, binary_cols):
    Xn = X.copy()
    if flip > 0:
        for c in binary_cols:
            m = RNG.random(len(Xn)) < flip
            Xn.loc[m, c] = 1 - Xn.loc[m, c]
    return Xn


def main():
    bundle = joblib.load(C.BUNDLE_PATH)
    primary = bundle["best_model"]
    model, cal, thr = bundle["models"][primary], bundle["calibrators"][primary], bundle["thresholds"][primary]

    df = load_dataset(); X, y = build_xy(df)
    idx = np.arange(len(df))
    _, te, _, _ = train_test_split(idx, y, test_size=C.TEST_SIZE, stratify=y, random_state=C.RANDOM_STATE)
    Xt, yt = X.iloc[te], y.iloc[te].to_numpy(); meta = df.iloc[te]

    result = {"primary_model": primary, "n_test": int(len(yt)), "n_mg": int(yt.sum()),
              "bootstrap_resamples": N_BOOT}

    # 1) confidence intervals for every model
    result["intervals"] = {}
    for name, m in bundle["models"].items():
        s = positive_score(m, Xt); pred = (s >= bundle["thresholds"][name]).astype(int)
        result["intervals"][name] = bootstrap(yt, s, pred)

    # 2) calibration of the primary model
    score = positive_score(model, Xt); prob = cal.predict(score); pred = (score >= thr).astype(int)
    result["calibration"] = calibration(yt, prob)

    # 3) predictive values at realistic prevalence
    sens, spec = result["intervals"][primary]["sensitivity"]["value"], result["intervals"][primary]["specificity"]["value"]
    result["predictive_values"] = predictive_values(sens, spec)

    # 4) subgroup behaviour
    mg_rows = meta[C.TARGET] == 1
    result["sensitivity_by_mg_form"] = {
        str(form): {"n": int((meta[C.DIAGNOSIS_COL] == form).sum()),
                    "sensitivity": float(pred[(meta[C.DIAGNOSIS_COL] == form).to_numpy()].mean())}
        for form in sorted(meta.loc[mg_rows, C.DIAGNOSIS_COL].unique())}
    result["specificity_by_non_mg_group"] = {
        str(g): {"n": int((meta[C.DIAGNOSIS_COL] == g).sum()),
                 "specificity": float(1 - pred[(meta[C.DIAGNOSIS_COL] == g).to_numpy()].mean())}
        for g in sorted(meta.loc[~mg_rows, C.DIAGNOSIS_COL].unique())}

    # 5) robustness to noisy answers (random flips of every yes/no answer)
    binary = [c for c in C.FEATURES if c not in ("age", "duration_months")]
    result["noise_robustness"] = []
    for f in NOISE_LEVELS:
        Xn = noisy(Xt, f, binary); s = positive_score(model, Xn); p = (s >= thr).astype(int)
        r = _rates(yt, p); r["roc_auc"] = float(roc_auc_score(yt, s)); r["flip_probability"] = f
        result["noise_robustness"].append(r)

    Path(C.OUTPUT_DIR / "honest_evaluation.json").write_text(json.dumps(result, indent=2))
    _plots(result, yt, prob)
    DOCS.mkdir(exist_ok=True)
    (DOCS / "EVALUATION.md").write_text(_markdown(result))
    print("wrote outputs/honest_evaluation.json, outputs/honest_evaluation.png and docs/EVALUATION.md")


def _plots(res, y, prob):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    b = res["calibration"]["bins"]
    ax[0].plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    ax[0].plot([r["mean_predicted"] for r in b], [r["observed_rate"] for r in b], "-", color="#C0262D", lw=1, alpha=.6)
    ax[0].scatter([r["mean_predicted"] for r in b], [r["observed_rate"] for r in b],
                  s=[20 + 0.5 * r["n"] for r in b], color="#C0262D", label=f'{res["primary_model"]} (marker size = patients in bin)')
    ax[0].set(xlabel="Predicted probability of MG", ylabel="Observed MG rate",
              title="Reliability curve (simulated test set; sparse middle bins are noisy)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    pv = res["predictive_values"]
    ax[1].semilogx([r["prevalence"] for r in pv], [r["ppv"] for r in pv], "o-", color="#2B6CB0")
    ax[1].set(xlabel="MG prevalence in the screened group (log scale)", ylabel="Positive predictive value",
              title="How often a POSITIVE is right depends on prevalence", ylim=(0, 1.02))
    ax[1].grid(alpha=.3, which="both")
    fig.tight_layout(); fig.savefig(C.OUTPUT_DIR / "honest_evaluation.png", dpi=150); plt.close(fig)


def _pct(x):
    return f"{100 * x:.1f}%"


def _markdown(r):
    p = r["primary_model"]; iv = r["intervals"]
    L = [f"# Evaluation notes\n",
         "> Everything below is measured on a **simulated** cohort (8,000 patients, 40% with MG) built from published clinical "
         "frequencies. It shows how consistent the models are with that simulation. It does not show how they would perform "
         "on real patients. Generated by `backend/mg_predictor/evaluate_honest.py`.\n",
         f"Primary model: **{p}**. Test set: {r['n_test']:,} patients ({r['n_mg']} with MG). Intervals are 95% bootstrap "
         f"intervals ({r['bootstrap_resamples']:,} resamples).\n",
         "## 1. Test-set performance with uncertainty\n",
         "| Algorithm | Accuracy | Sensitivity (recall) | Specificity | ROC-AUC |", "|---|---|---|---|---|"]
    for n, m in iv.items():
        cell = lambda k: f"{_pct(m[k]['value'])} ({_pct(m[k]['ci95'][0])} to {_pct(m[k]['ci95'][1])})" if k != "roc_auc" else \
            f"{m[k]['value']:.3f} ({m[k]['ci95'][0]:.3f} to {m[k]['ci95'][1]:.3f})"
        L.append(f"| {n}{' (primary)' if n == p else ''} | {cell('accuracy')} | {cell('sensitivity')} | {cell('specificity')} | {cell('roc_auc')} |")
    L += ["\nThe intervals only capture sampling error on one simulated test set. They do not include the much larger "
          "gap between simulated and real patients.\n", "## 2. Calibration\n"]
    c = r["calibration"]
    L += [f"Brier score {c['brier']:.4f}; expected calibration error {c['ece']:.4f}. The reliability curve is in "
          "`backend/mg_predictor/outputs/honest_evaluation.png`.\n",
          "| Predicted probability | Patients | Mean predicted | Observed MG rate |", "|---|---|---|---|"]
    L += [f"| {b['bin']} | {b['n']} | {_pct(b['mean_predicted'])} | {_pct(b['observed_rate'])} |" for b in c["bins"]]
    L += ["\nProbabilities are calibrated on simulated data, so they are only meaningful for patients who resemble it.\n",
          "## 3. What a POSITIVE means at realistic prevalence\n",
          f"Sensitivity {_pct(iv[p]['sensitivity']['value'])} and specificity {_pct(iv[p]['specificity']['value'])} are taken "
          "from the test set above. Predictive values depend on how common MG is in the group being screened. Myasthenia "
          "gravis is rare (published prevalence estimates are roughly 10 to 20 per 100,000 people, and vary by country), "
          "so the 40% used in the simulation is far from a real screening population.\n",
          "| MG prevalence in the screened group | PPV (a POSITIVE is truly MG) | NPV | False alarms per true positive |", "|---|---|---|---|"]
    for v in r["predictive_values"]:
        fp = f"{v['false_positives_per_true_positive']:.1f}" if v["false_positives_per_true_positive"] is not None else "n/a"
        L.append(f"| {PREV_LABELS[v['prevalence']]} | {_pct(v['ppv'])} | {_pct(v['npv'])} | {fp} |")
    L += ["\nIn practice this means the tool is only sensible for people who already have suspicious symptoms. Applied to "
          "the general population, almost every POSITIVE would be a false alarm.\n", "## 4. Where it does better and worse\n",
          "| MG form in the test set | Patients | Sensitivity |", "|---|---|---|"]
    L += [f"| {k} | {v['n']} | {_pct(v['sensitivity'])} |" for k, v in r["sensitivity_by_mg_form"].items()]
    L += ["", "| Non-MG group in the test set | Patients | Specificity |", "|---|---|---|"]
    L += [f"| {k} | {v['n']} | {_pct(v['specificity'])} |" for k, v in r["specificity_by_non_mg_group"].items()]
    L += ["\n## 5. Robustness to noisy answers\n",
          "Every yes/no answer in the test set was flipped at random with the probability shown (training data contains only "
          "1.2% answer noise).\n", "| Answers flipped | Accuracy | Sensitivity | Specificity | ROC-AUC |", "|---|---|---|---|---|"]
    L += [f"| {_pct(n['flip_probability'])} | {_pct(n['accuracy'])} | {_pct(n['sensitivity'])} | {_pct(n['specificity'])} | {n['roc_auc']:.3f} |"
          for n in r["noise_robustness"]]
    L += ["\n## Takeaway\n",
          "On the simulated cohort the models look strong, but that mostly reflects how cleanly the simulation separates the "
          "classes. The numbers that matter for any real use are the ones this simulation cannot provide: performance on real, "
          "de-identified patients from the population the tool is meant for.\n"]
    return "\n".join(L)


if __name__ == "__main__":
    main()
