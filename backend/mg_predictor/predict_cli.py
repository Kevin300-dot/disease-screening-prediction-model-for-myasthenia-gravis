"""Command-line myasthenia gravis screening.

Interactive:      python predict_cli.py
Non-interactive:  python predict_cli.py --name "Priya Raman" --age 34 --gender female \\
                      --weight 58 --height 162 --symptoms fatigue \\
                      --red-flags droopy_eyelids,double_vision,fatigable_weakness --pdf report.pdf
List options:     python predict_cli.py --list
Machine output:   add --json
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap

from mg import config as C
from mg.predictor import MGPredictor


def ask_yes_no(prompt: str) -> bool:
    while True:
        ans = input(f"  {prompt} [y/N]: ").strip().lower()
        if ans in {"", "n", "no"}:
            return False
        if ans in {"y", "yes"}:
            return True
        print("    Please answer y or n.")


def interactive_inputs():
    print("\n=== Myasthenia gravis screening ===\n")
    name = input("Patient name: ").strip()

    def number(prompt, lo, hi, optional=False):
        while True:
            raw = input(prompt).strip()
            if optional and not raw:
                return None
            try:
                value = float(raw)
                if lo <= value <= hi:
                    return value
            except ValueError:
                pass
            print(f"  Enter a number between {lo:g} and {hi:g}.")

    age = number("Age: ", 1, 120)
    while True:
        gender = input("Gender (male/female): ").strip().lower()
        if gender in {"male", "female", "m", "f"}:
            break
        print("  Enter 'male' or 'female'.")
    weight = number("Weight in kg (Enter to skip): ", 2, 400, optional=True)
    height = number("Height in cm (Enter to skip): ", 40, 250, optional=True)
    print("\nMG-specific signs - do you currently have:")
    flags = [k for k, (label, _, _) in C.MG_SIGNS.items() if ask_yes_no(label)]
    print("\nGeneral symptoms and history:")
    symptoms = [k for k in C.SYMPTOMS if ask_yes_no(C.SYMPTOM_LABELS[k])]
    return dict(age=age, gender=gender, symptoms=symptoms, red_flags=flags,
                name=name or None, weight_kg=weight, height_cm=height)


def wrap(text, indent="  "):
    return textwrap.fill(text, width=92, initial_indent=indent, subsequent_indent=indent + "  ")


def show(res: dict):
    rep, r = res["report"], res["result"]
    p = rep["patient"]
    bar = "=" * 70
    print(f"\n{bar}\nMYASCREEN REPORT  {rep['report_id']}  ({rep['generated_at']})\n{bar}")
    print(f"Patient : {p['name']}  |  {p['age']} y  |  {p['gender']}")
    if p["bmi"]:
        print(f"Body    : {p['weight_kg']:g} kg, {p['height_cm']:g} cm, BMI {p['bmi']} ({p['bmi_category']})")
    print(f"Symptom duration: {p['symptom_duration']}  |  Blood group: {p.get('blood_group') or '-'}")
    for row in res["blood"]["panel"]:
        print(f"  {row['label']:<28} {row['value']:g} {row['unit']:<10} ref {row['reference']:<10} {row['status']}")
    for a in res["blood"]["antibodies"]:
        if a["status"] != "Not tested":
            print(f"  {a['label']:<28} {a['status']}")
    if res["advice"]["emergency"]:
        print("\n!! CRITICAL FINDING !!")
        print(wrap(res["advice"]["emergency"]))
    print(f"\n  MYASTHENIA GRAVIS: {r['status']}")
    print(f"  Confidence {r['confidence_percent']}% ({r['confidence_band']})  |  "
          f"MG probability {r['mg_probability_percent']}%  |  positive in {r['models_positive']} algorithms")
    print(wrap(r["interpretation"], ""))
    if r["clinical_pattern"]:
        print(f"\nClinical pattern: {r['clinical_pattern']}")

    print("\nPer-algorithm results:")
    for m in res["per_model"]:
        star = "  <- primary" if m["is_primary"] else ""
        print(f"  {m['model']:<20} {m['result']:<9} P(MG) {m['mg_probability_percent']:>5}%{star}")

    print("\nSpecialists to contact:")
    for s in res["recommended_specialists"]:
        print(f"  * {s['specialist']}  [{s['when']}]")
        print(wrap(s["why"], "      "))

    a = res["advice"]
    for title, key in [("What to do next", "next_steps"), ("Watch for", "watch_for"),
                       ("Confirmatory tests", "tests_to_ask_about"), ("Daily management", "daily_management")]:
        if a.get(key):
            print(f"\n{title}:")
            for item in a[key]:
                print(wrap("- " + item))
    for title, key in [("Medication safety", "medication_safety"), ("About treatment", "about_treatment")]:
        if a.get(key):
            print(f"\n{title}:\n{wrap(a[key])}")
    print(f"\n{bar}\n{res['footer']}\n{bar}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name")
    p.add_argument("--age", type=float)
    p.add_argument("--gender")
    p.add_argument("--weight", type=float, help="kg")
    p.add_argument("--height", type=float, help="cm")
    p.add_argument("--symptoms", default="", help="comma-separated general symptom / history keys")
    p.add_argument("--red-flags", default="", help="comma-separated MG-specific sign keys")
    p.add_argument("--blood-group", help="A+ A- B+ B- AB+ AB- O+ O-")
    p.add_argument("--blood", default="", help="comma list e.g. hemoglobin=12.5,wbc=7,platelets=250,esr=10,crp=2,tsh=2,ck=90")
    p.add_argument("--achr", choices=["positive", "negative", "not_tested"], default="not_tested")
    p.add_argument("--musk", choices=["positive", "negative", "not_tested"], default="not_tested")
    p.add_argument("--pdf", metavar="FILE", help="also save the report as a PDF")
    p.add_argument("--json", action="store_true", help="print raw JSON")
    p.add_argument("--list", action="store_true", help="list valid symptom / MG-sign keys")
    args = p.parse_args(argv)

    if args.list:
        print("MG-specific signs (--red-flags):")
        for k, (label, group, _) in C.MG_SIGNS.items():
            print(f"  {k:<28} [{group}] {label}")
        print("\nGeneral symptoms and history (--symptoms):")
        for k in C.SYMPTOMS:
            print(f"  {k:<28} {C.SYMPTOM_LABELS[k]}")
        return 0

    try:
        predictor = MGPredictor()
        if args.age is None or args.gender is None:
            inputs = interactive_inputs()
        else:
            split = lambda s: [x.strip() for x in s.split(",") if x.strip()]
            inputs = dict(age=args.age, gender=args.gender, symptoms=split(args.symptoms),
                          red_flags=split(args.red_flags), name=args.name,
                          weight_kg=args.weight, height_cm=args.height)
        inputs.update(blood_group=args.blood_group, antibodies={"achr": args.achr, "musk": args.musk},
                      blood_tests={k.strip(): float(v) for k, v in
                                   (kv.split("=") for kv in split(args.blood))} or None)
        res = predictor.predict(**inputs)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    print(json.dumps(res, indent=2)) if args.json else show(res)
    if args.pdf:
        from mg.report_pdf import build_report_pdf
        with open(args.pdf, "wb") as f:
            f.write(build_report_pdf(res))
        print(f"\nPDF saved -> {args.pdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
