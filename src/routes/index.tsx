import { createFileRoute } from "@tanstack/react-router";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  ArrowLeft, ArrowRight, Check, ChevronDown, Download, Loader2, LocateFixed, MapPin, RotateCcw, ShieldAlert, Stethoscope,
} from "lucide-react";
import { useState } from "react";

import junctionImage from "@/assets/neuromuscular-junction.jpg";
import { DatasetCompositionCard, LearningCurveChart, LoadingLine, PerformanceBars, PerformanceCard } from "@/components/report/ModelCharts";
import { ReportView } from "@/components/report/ReportView";
import { Button } from "@/components/ui/button";
import {
  downloadReportPdf, getDatasetBreakdown, getLearningCurve, getMetrics, MGApiError, predictMG,
  type PredictRequest, type PredictResponse,
} from "@/lib/mg-api";
import { antibodyFields, antibodyStatusOptions, bloodGroups, bloodTests, type AntibodyStatus } from "@/lib/mg-blood";
import { insideChennai, REGION, chennaiAreas } from "@/lib/mg-locations";
import { durationOptions, generalSymptoms, historyItems, mgSignGroups, type Option } from "@/lib/mg-options";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "MyaScreen | Myasthenia Gravis Screening Report" },
      { name: "description", content: "Enter patient details and symptoms to generate a myasthenia gravis screening report." },
      { property: "og:title", content: "MyaScreen | Myasthenia Gravis Screening Report" },
      { property: "og:description", content: "Enter patient details and symptoms to generate a myasthenia gravis screening report." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Index,
});

type Step = 1 | 2 | 3 | 4 | 5;
const STEP_LABELS = ["Patient", "MG symptoms", "History", "Blood", "Report"];

const urgentKeys = new Set(mgSignGroups.flatMap((g) => g.options.filter((o) => o.urgent).map((o) => o.key)));

const inputClass =
  "h-12 w-full rounded-md border border-input bg-background px-4 font-display text-base outline-none focus:ring-2 focus:ring-ring";

function Index() {
  const [step, setStep] = useState<Step>(1);
  const [name, setName] = useState("");
  const [age, setAge] = useState("");
  const [gender, setGender] = useState<"male" | "female">("female");
  const [weight, setWeight] = useState("");
  const [height, setHeight] = useState("");
  const [duration, setDuration] = useState("2");
  const [bloodGroup, setBloodGroup] = useState("");
  const [bloodValues, setBloodValues] = useState<Record<string, string>>({});
  const [antibodies, setAntibodies] = useState<Record<string, AntibodyStatus>>({ achr: "not_tested", musk: "not_tested" });
  // Hospital finder (optional): a Chennai area, or the browser's GPS position if it is inside Chennai.
  const [area, setArea] = useState("");
  const [gps, setGps] = useState<{ lat: number; lng: number } | null>(null);
  const [gpsMessage, setGpsMessage] = useState("");
  const [selectedFlags, setSelectedFlags] = useState<string[]>([]);
  const [selectedSymptoms, setSelectedSymptoms] = useState<string[]>([]);

  const ageN = Number(age);
  const weightN = Number(weight);
  const heightN = Number(height);
  const bmi = weightN > 0 && heightN > 0 ? weightN / (heightN / 100) ** 2 : null;
  const profileValid =
    name.trim().length > 0 && ageN >= 1 && ageN <= 120 && weightN >= 2 && weightN <= 400 && heightN >= 40 && heightN <= 250;

  // Instant, client-side-only alert for breathing / swallowing difficulty -- shown before the API call.
  const urgent = selectedFlags.some((flag) => urgentKeys.has(flag));

  const enteredBlood = Object.fromEntries(
    Object.entries(bloodValues).filter(([, v]) => v.trim() !== "" && Number.isFinite(Number(v))).map(([k, v]) => [k, Number(v)]),
  );

  const request: PredictRequest = {
    name: name.trim(),
    age: ageN,
    gender,
    weight_kg: weightN,
    height_cm: heightN,
    duration_months: Number(duration),
    symptoms: selectedSymptoms,
    red_flags: selectedFlags,
    ...(bloodGroup ? { blood_group: bloodGroup } : {}),
    ...(Object.keys(enteredBlood).length ? { blood_tests: enteredBlood } : {}),
    antibodies,
    ...(gps ? { location: { lat: gps.lat, lng: gps.lng } } : area ? { location: { area } } : {}),
  };

  const useMyLocation = () => {
    setGpsMessage("");
    if (!navigator.geolocation) {
      setGpsMessage("This browser cannot share its location. Please pick your area from the list.");
      return;
    }
    setGpsMessage("Finding your location...");
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const { latitude, longitude } = pos.coords;
        if (insideChennai(latitude, longitude)) {
          setGps({ lat: latitude, lng: longitude });
          setArea("");
          setGpsMessage("Using your current location.");
        } else {
          setGps(null);
          setGpsMessage(`Your location is outside ${REGION}. The hospital finder covers ${REGION} only - please pick your area from the list.`);
        }
      },
      () => setGpsMessage("Could not read your location. Please pick your area from the list."),
      { enableHighAccuracy: false, timeout: 10000 },
    );
  };

  const mutation = useMutation({ mutationFn: () => predictMG(request) });

  const metricsQuery = useQuery({ queryKey: ["metrics"], queryFn: getMetrics, staleTime: Infinity });
  const datasetQuery = useQuery({ queryKey: ["dataset-breakdown"], queryFn: getDatasetBreakdown, staleTime: Infinity });

  const toggle = (value: string, selected: string[], setter: (values: string[]) => void) => {
    setter(selected.includes(value) ? selected.filter((item) => item !== value) : [...selected, value]);
  };

  const reset = () => {
    setStep(1); setName(""); setAge(""); setGender("female"); setWeight(""); setHeight(""); setDuration("2");
    setBloodGroup(""); setBloodValues({}); setAntibodies({ achr: "not_tested", musk: "not_tested" });
    setSelectedFlags([]); setSelectedSymptoms([]);
    setArea(""); setGps(null); setGpsMessage("");
    mutation.reset();
  };

  const generateReport = () => {
    setStep(5);
    mutation.mutate();
  };

  const showSidebar = step !== 5;

  return (
    <div className="screen-grid text-sm text-foreground">
      <div className="screen-content">
        <header className="screen-header">
          <div className="flex items-center gap-3">
            <div className="neon-glow grid size-11 place-items-center rounded-md border border-primary/40 bg-primary/10 font-display text-lg font-bold text-primary">Mx</div>
            <div>
              <div className="font-display text-xl font-bold">MyaScreen<span className="text-primary">.</span></div>
              <div className="section-label mt-1">Myasthenia gravis screening report</div>
            </div>
          </div>
          <div className="hidden items-center gap-3 sm:flex">
            <span className="section-label flex items-center gap-2 rounded-md border border-border bg-card px-3 py-2">
              <Stethoscope className="size-3.5 text-primary" /> Neuromuscular screening
            </span>
          </div>
        </header>

        <nav aria-label="Assessment progress" className="mt-6 flex items-center gap-3 border-b border-border pb-4 sm:gap-6">
          {STEP_LABELS.map((label, index) => {
            const number = (index + 1) as Step;
            return (
              <div className="contents" key={label}>
                <Button variant="step" onClick={() => number < step && setStep(number)} aria-current={step === number ? "step" : undefined}>
                  <span className={`grid size-7 place-items-center rounded-md border font-mono text-xs ${step >= number ? "border-primary/40 bg-primary/10 text-primary" : "border-border bg-card text-muted-foreground"}`}>
                    {step > number ? <Check className="size-3.5" /> : `0${number}`}
                  </span>
                  <span className="hidden text-xs font-medium sm:inline">{label}</span>
                </Button>
                {index < STEP_LABELS.length - 1 && <span className="h-px flex-1 bg-border" />}
              </div>
            );
          })}
        </nav>

        <main className={showSidebar ? "screen-layout" : "screen-layout screen-layout-single"}>
          <div className="min-w-0">
            {step === 1 && (
              <section className="glass-panel rise-in rounded-lg p-5 sm:p-7">
                <p className="font-mono text-[11px] uppercase tracking-[0.12em] text-primary">Step 1 of 5</p>
                <h1 className="mt-2 screen-title">Patient details</h1>
                <p className="mt-2 max-w-xl text-sm leading-6 text-muted-foreground">
                  These details appear on the screening report. Nothing is stored after you close the page.
                </p>
                <div className="mt-7 grid gap-5 sm:grid-cols-2">
                  <label className="space-y-2 sm:col-span-2">
                    <span className="section-label">Patient name</span>
                    <input value={name} onChange={(e) => setName(e.target.value)} aria-label="Patient name" type="text" maxLength={80} autoComplete="off" placeholder="Full name" className={inputClass} />
                  </label>
                  <label className="space-y-2">
                    <span className="section-label">Age · 1–120 years</span>
                    <input value={age} onChange={(e) => setAge(e.target.value)} aria-label="Age" min="1" max="120" type="number" inputMode="numeric" placeholder="e.g. 34" className={inputClass} />
                  </label>
                  <label className="space-y-2">
                    <span className="section-label">Sex</span>
                    <select aria-label="Gender" value={gender} onChange={(e) => setGender(e.target.value as "male" | "female")} className={inputClass}>
                      <option value="female">Female</option>
                      <option value="male">Male</option>
                    </select>
                  </label>
                  <label className="space-y-2 sm:col-span-2">
                    <span className="section-label">Blood group</span>
                    <select aria-label="Blood group" value={bloodGroup} onChange={(e) => setBloodGroup(e.target.value)} className={inputClass}>
                      <option value="">Not known</option>
                      {bloodGroups.map((b) => <option key={b.group} value={b.group}>{b.group}</option>)}
                    </select>
                  </label>
                  <label className="space-y-2">
                    <span className="section-label">Weight · kg</span>
                    <input value={weight} onChange={(e) => setWeight(e.target.value)} aria-label="Weight in kilograms" min="2" max="400" type="number" inputMode="decimal" placeholder="e.g. 62" className={inputClass} />
                  </label>
                  <label className="space-y-2">
                    <span className="section-label">Height · cm</span>
                    <input value={height} onChange={(e) => setHeight(e.target.value)} aria-label="Height in centimetres" min="40" max="250" type="number" inputMode="decimal" placeholder="e.g. 165" className={inputClass} />
                  </label>
                  <label className="space-y-2 sm:col-span-2">
                    <span className="section-label">How long have the symptoms been present?</span>
                    <select aria-label="Symptom duration" value={duration} onChange={(e) => setDuration(e.target.value)} className={inputClass}>
                      {durationOptions.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
                    </select>
                  </label>
                  <div className="space-y-2 sm:col-span-2">
                    <span className="section-label">Your location · {REGION} only (for the nearest and best hospital)</span>
                    <div className="flex flex-col gap-3 sm:flex-row">
                      <select
                        aria-label="Your area in Chennai"
                        value={gps ? "__gps" : area}
                        onChange={(e) => { setArea(e.target.value === "__gps" ? "" : e.target.value); setGps(null); setGpsMessage(""); }}
                        className={inputClass}
                      >
                        <option value="">Skip (no hospital suggestion)</option>
                        {gps && <option value="__gps">Current location</option>}
                        {chennaiAreas.map((a) => <option key={a.key} value={a.key}>{a.label}</option>)}
                      </select>
                      <Button type="button" variant="ghost" onClick={useMyLocation} className="h-12 shrink-0">
                        <LocateFixed className="size-4" /> Use my location
                      </Button>
                    </div>
                    {gpsMessage && <p className="text-xs text-muted-foreground">{gpsMessage}</p>}
                  </div>
                </div>
                {bmi !== null && (
                  <p className="mt-4 text-xs text-muted-foreground">Body mass index: <span className="font-semibold text-foreground">{bmi.toFixed(1)}</span></p>
                )}
                <div className="mt-8 flex justify-end border-t border-border pt-5">
                  <Button onClick={() => setStep(2)} disabled={!profileValid}>Continue to symptoms <ArrowRight className="size-4" /></Button>
                </div>
              </section>
            )}

            {step === 2 && (
              <div className="space-y-5">
                <section className="glass-panel rise-in rounded-lg p-5 sm:p-6">
                  <p className="font-mono text-[11px] uppercase tracking-[0.12em] text-primary">Step 2 of 5</p>
                  <h1 className="mt-2 text-xl font-semibold">Myasthenia gravis symptoms</h1>
                  <p className="mt-1 text-xs leading-5 text-muted-foreground">Select every symptom the patient currently has.</p>
                </section>
                {mgSignGroups.map((group) => (
                  <section key={group.title} className="glass-panel rise-in rounded-lg p-5 sm:p-6">
                    <h2 className="text-[13px] font-semibold uppercase tracking-[0.1em] text-primary">{group.title}</h2>
                    <p className="mt-1 text-xs leading-5 text-muted-foreground">{group.hint}</p>
                    <div className="mt-4 grid gap-2 sm:grid-cols-2">
                      {group.options.map((o) => <Choice key={o.key} option={o} selected={selectedFlags.includes(o.key)} onClick={() => toggle(o.key, selectedFlags, setSelectedFlags)} />)}
                    </div>
                  </section>
                ))}
                {urgent && <UrgentBanner />}
                <div className="flex justify-between">
                  <Button variant="ghost" onClick={() => setStep(1)}><ArrowLeft className="size-4" /> Back</Button>
                  <Button onClick={() => setStep(3)}>Continue <ArrowRight className="size-4" /></Button>
                </div>
              </div>
            )}

            {step === 3 && (
              <div className="space-y-5">
                <section className="glass-panel rise-in rounded-lg p-5 sm:p-6">
                  <p className="font-mono text-[11px] uppercase tracking-[0.12em] text-primary">Step 3 of 5</p>
                  <h1 className="mt-2 text-xl font-semibold">General symptoms</h1>
                  <p className="mt-1 text-xs leading-5 text-muted-foreground">Other symptoms the patient has noticed.</p>
                  <div className="mt-4 grid gap-2 sm:grid-cols-2">
                    {generalSymptoms.map((o) => <Choice key={o.key} option={o} selected={selectedSymptoms.includes(o.key)} onClick={() => toggle(o.key, selectedSymptoms, setSelectedSymptoms)} />)}
                  </div>
                </section>
                <section className="glass-panel rise-in rounded-lg p-5 sm:p-6">
                  <h2 className="text-lg font-semibold">Medical history</h2>
                  <p className="mt-1 text-xs leading-5 text-muted-foreground">Select anything that applies.</p>
                  <div className="mt-4 grid gap-2 sm:grid-cols-2">
                    {historyItems.map((o) => <Choice key={o.key} option={o} selected={selectedSymptoms.includes(o.key)} onClick={() => toggle(o.key, selectedSymptoms, setSelectedSymptoms)} />)}
                  </div>
                </section>
                {urgent && <UrgentBanner />}
                <div className="flex justify-between">
                  <Button variant="ghost" onClick={() => setStep(2)}><ArrowLeft className="size-4" /> Back</Button>
                  <Button onClick={() => setStep(4)}>Continue to blood details <ArrowRight className="size-4" /></Button>
                </div>
              </div>
            )}

            {step === 4 && (
              <div className="space-y-5">
                <section className="glass-panel rise-in rounded-lg p-5 sm:p-6">
                  <p className="font-mono text-[11px] uppercase tracking-[0.12em] text-primary">Step 4 of 5</p>
                  <h1 className="mt-2 text-xl font-semibold">Blood details</h1>
                  <p className="mt-1 text-xs leading-5 text-muted-foreground">
                    Optional. Fill in only the results you have from a recent lab report; leave the rest blank. Values outside the reference range are flagged on the report.
                  </p>
                </section>
                <section className="glass-panel rise-in rounded-lg p-5 sm:p-6">
                  <h2 className="text-[13px] font-semibold uppercase tracking-[0.1em] text-primary">Antibody tests</h2>
                  <p className="mt-1 text-xs leading-5 text-muted-foreground">Anti-AChR antibodies are the key blood test for myasthenia gravis.</p>
                  <div className="mt-4 grid gap-4 sm:grid-cols-2">
                    {antibodyFields.map((f) => (
                      <label key={f.key} className="space-y-2">
                        <span className="section-label">{f.label}</span>
                        <select aria-label={f.label} value={antibodies[f.key]} onChange={(e) => setAntibodies({ ...antibodies, [f.key]: e.target.value as AntibodyStatus })} className={inputClass}>
                          {antibodyStatusOptions.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
                        </select>
                      </label>
                    ))}
                  </div>
                </section>
                <section className="glass-panel rise-in rounded-lg p-5 sm:p-6">
                  <h2 className="text-[13px] font-semibold uppercase tracking-[0.1em] text-primary">Blood count and markers</h2>
                  <div className="mt-4 grid gap-4 sm:grid-cols-2">
                    {bloodTests.map((t) => (
                      <label key={t.test} className="space-y-2">
                        <span className="section-label">{t.label} · {t.unit}</span>
                        <input
                          value={bloodValues[t.test] ?? ""}
                          onChange={(e) => setBloodValues({ ...bloodValues, [t.test]: e.target.value })}
                          aria-label={t.label}
                          type="number" inputMode="decimal" step={t.step} min="0" placeholder={t.placeholder}
                          className={inputClass}
                        />
                        <span className="block text-[11px] text-muted-foreground">Reference: {gender === "male" ? t.male : t.female} {t.unit}</span>
                      </label>
                    ))}
                  </div>
                </section>
                {urgent && <UrgentBanner />}
                <div className="flex justify-between">
                  <Button variant="ghost" onClick={() => setStep(3)}><ArrowLeft className="size-4" /> Back</Button>
                  <Button onClick={generateReport}>Generate report <ArrowRight className="size-4" /></Button>
                </div>
              </div>
            )}

            {step === 5 && (
              <Results
                request={request}
                mutation={mutation}
                onRetry={() => mutation.mutate()}
                onBack={() => setStep(4)}
                onReset={reset}
              />
            )}
          </div>

          {showSidebar && (
            <aside className="min-w-0 space-y-6">
              <section className="border-b border-border pb-6">
                <div className="flex items-center justify-between">
                  <h2 className="font-display text-sm font-semibold uppercase">Patient summary</h2>
                  <span className="section-label text-primary">In progress</span>
                </div>
                <img src={junctionImage} alt="Scientific illustration of a nerve communicating with a muscle at a neuromuscular junction" width={1024} height={768} className="science-image" />
                <div className="text-[15px] font-semibold">{name.trim() || "Patient name"}</div>
                <div className="mt-1 text-xs text-muted-foreground">
                  {age ? `${age} years` : "Age —"} · {gender === "female" ? "Female" : "Male"}
                  {weightN > 0 ? ` · ${weightN} kg` : ""}{heightN > 0 ? ` · ${heightN} cm` : ""}{bmi !== null ? ` · BMI ${bmi.toFixed(1)}` : ""}{bloodGroup ? ` · Blood group ${bloodGroup}` : ""}
                </div>
                {(gps || area) && (
                  <div className="mt-2 flex items-center gap-1.5 text-xs text-muted-foreground">
                    <MapPin className="size-3.5" /> {gps ? "Current location" : chennaiAreas.find((a) => a.key === area)?.label}, {REGION}
                  </div>
                )}
                <div className="mt-3 text-xs text-muted-foreground">
                  {selectedFlags.length} MG symptoms · {selectedSymptoms.length} general symptoms / history items · {Object.keys(enteredBlood).length + Object.values(antibodies).filter((a) => a !== "not_tested").length} blood results
                </div>
              </section>
              {urgent && <UrgentBanner />}
              {metricsQuery.data && <PerformanceCard performance={metricsQuery.data.performance} />}
              {datasetQuery.data && <DatasetCompositionCard breakdown={datasetQuery.data} />}
              {(metricsQuery.isLoading || datasetQuery.isLoading) && <LoadingLine>Loading model information…</LoadingLine>}
              {(metricsQuery.isError || datasetQuery.isError) && (
                <div className="border-t border-border pt-4">
                  <div className="section-label text-caution">Model information unavailable</div>
                  <p className="mt-2 text-xs leading-5 text-muted-foreground">The screening service could not be reached. Start the backend before generating a report.</p>
                </div>
              )}
            </aside>
          )}
        </main>
      </div>
    </div>
  );
}

function UrgentBanner() {
  return (
    <div role="alert" className="flex gap-3 rounded-md bg-emergency p-4 text-emergency-foreground">
      <ShieldAlert className="mt-0.5 size-5 shrink-0" />
      <p className="text-sm leading-5">
        <strong>Seek emergency help now.</strong> Breathing or swallowing difficulty can signal a myasthenic crisis. Call your local emergency number or go to the nearest emergency department.
      </p>
    </div>
  );
}

function Choice({ option, selected, onClick }: { option: Option; selected: boolean; onClick: () => void }) {
  const critical = option.urgent === true;
  return (
    <Button variant="choice" type="button" aria-pressed={selected} data-critical={critical} onClick={onClick}>
      <span className="check-box">{selected && <Check className="size-3" />}</span>
      <span className="text-xs leading-5">{option.label}</span>
      {critical && <span className="ml-auto shrink-0 font-mono text-[9px] uppercase text-emergency">urgent</span>}
    </Button>
  );
}

function Results({
  request, mutation, onRetry, onBack, onReset,
}: {
  request: PredictRequest;
  mutation: { isPending: boolean; isError: boolean; error: unknown; data?: PredictResponse | undefined };
  onRetry: () => void;
  onBack: () => void;
  onReset: () => void;
}) {
  // Hooks must run unconditionally, before the early returns below.
  const [showAnalytics, setShowAnalytics] = useState(false);
  const learningCurveQuery = useQuery({
    queryKey: ["learning-curve"],
    queryFn: getLearningCurve,
    enabled: showAnalytics,
    staleTime: Infinity,
  });

  const pdfMutation = useMutation({
    mutationFn: async (data: PredictResponse) => {
      // Same inputs plus the report ID / time shown on screen, so the PDF is the same report.
      const { blob, filename } = await downloadReportPdf({
        ...request,
        report_id: data.report.report_id,
        generated_at: data.report.generated_at,
      });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    },
  });

  if (mutation.isPending) {
    return (
      <section className="glass-panel rise-in flex flex-col items-center justify-center gap-3 rounded-lg p-10 text-center">
        <Loader2 className="size-6 animate-spin text-primary" />
        <p className="text-sm text-muted-foreground">Analysing symptoms and generating the report...</p>
      </section>
    );
  }

  if (mutation.isError || !mutation.data) {
    const message = mutation.error instanceof MGApiError ? mutation.error.message : "Something went wrong while contacting the screening backend.";
    return (
      <section className="glass-panel rise-in rounded-lg p-6">
        <div className="flex items-start gap-3 rounded-md bg-emergency/10 p-4 ring-1 ring-emergency/25">
          <ShieldAlert className="mt-0.5 size-5 shrink-0 text-emergency" />
          <div>
            <div className="font-semibold text-emergency">Couldn't generate the report</div>
            <p className="mt-1 text-xs leading-5 text-foreground/80">{message}</p>
          </div>
        </div>
        <div className="mt-6 flex flex-wrap justify-between gap-3">
          <Button variant="ghost" onClick={onBack}><ArrowLeft className="size-4" /> Edit symptoms</Button>
          <Button onClick={onRetry}>Try again</Button>
        </div>
      </section>
    );
  }

  const data = mutation.data;
  const pdfError = pdfMutation.error instanceof MGApiError ? pdfMutation.error.message : pdfMutation.isError ? "The PDF could not be created." : null;

  return (
    <div className="rise-in mx-auto w-full max-w-[920px] space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <Button variant="ghost" onClick={onBack}><ArrowLeft className="size-4" /> Edit details</Button>
        <div className="flex flex-wrap gap-3">
          <Button variant="secondary" onClick={onReset}><RotateCcw className="size-4" /> New patient</Button>
          <Button onClick={() => pdfMutation.mutate(data)} disabled={pdfMutation.isPending}>
            {pdfMutation.isPending ? <Loader2 className="size-4 animate-spin" /> : <Download className="size-4" />}
            Download PDF report
          </Button>
        </div>
      </div>
      {pdfError && <p role="alert" className="rounded-md bg-emergency/10 p-3 text-xs text-emergency ring-1 ring-emergency/25">{pdfError}</p>}

      <ReportView data={data} />

      <Button
        variant="ghost"
        type="button"
        onClick={() => setShowAnalytics((v) => !v)}
        className="flex w-full items-center justify-between rounded-md bg-surface/45 p-4 text-left ring-1 ring-border transition-colors hover:bg-surface/70"
      >
        <span className="font-mono text-[10px] uppercase tracking-[0.12em] text-muted-foreground">Model analytics: accuracy and learning curve</span>
        <ChevronDown className={`size-4 text-muted-foreground transition-transform ${showAnalytics ? "rotate-180" : ""}`} />
      </Button>

      {showAnalytics && (
        <div className="space-y-4">
          <div className="rounded-md bg-surface/60 p-4 ring-1 ring-border">
            <div className="font-mono text-[10px] uppercase tracking-[0.12em] text-muted-foreground">Test-set performance by algorithm</div>
            <PerformanceBars performance={data.performance} />
          </div>
          <div className="rounded-md bg-surface/60 p-4 ring-1 ring-border">
            <div className="font-mono text-[10px] uppercase tracking-[0.12em] text-muted-foreground">Learning curve: generalization check</div>
            <p className="mt-1 text-xs leading-5 text-muted-foreground">
              Validation ROC-AUC vs. training-set size, from cross-validation on the development set. A line that rises and then flattens means the model has learned the pattern and more data would help only a little.
            </p>
            {learningCurveQuery.isLoading && <LoadingLine>Loading...</LoadingLine>}
            {learningCurveQuery.data && <LearningCurveChart data={learningCurveQuery.data} />}
          </div>
        </div>
      )}
    </div>
  );
}
