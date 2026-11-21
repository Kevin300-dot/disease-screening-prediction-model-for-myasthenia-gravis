import { ExternalLink, MapPin, ShieldAlert } from "lucide-react";
import type { ReactNode } from "react";

import type { HospitalResult, PredictResponse } from "@/lib/mg-api";

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section>
      <h2 className="report-section-title">{title}</h2>
      {children}
    </section>
  );
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="report-fact-label">{label}</div>
      <div className="report-fact-value">{value}</div>
    </div>
  );
}

function BulletList({ items }: { items: string[] }) {
  return (
    <ul className="report-list">
      {items.map((item, i) => (
        <li key={i}>{item}</li>
      ))}
    </ul>
  );
}

function HospitalCard({ tag, h }: { tag: string; h: HospitalResult }) {
  return (
    <div className="report-hospital">
      <div className="report-hospital-tag">{tag}</div>
      <div className="report-hospital-name">{h.name}</div>
      <div className="report-muted">{h.department}</div>
      <div className="report-hospital-line">
        <MapPin className="size-3.5 shrink-0" /> {h.address}
      </div>
      {h.phone && <div className="report-hospital-line">Phone: {h.phone}</div>}
      <div className="report-hospital-dist">{h.distance_km} km <span className="report-muted">away (approx., straight line)</span></div>
      <p className="mt-1">{h.evidence[0]}</p>
      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[12px]">
        <a href={h.maps_url} target="_blank" rel="noreferrer" className="report-link">
          Directions on Google Maps <ExternalLink className="inline size-3" />
        </a>
        <a href={h.source} target="_blank" rel="noreferrer" className="report-link">
          Hospital's own page <ExternalLink className="inline size-3" />
        </a>
      </div>
    </div>
  );
}

// On-screen version of the screening report. The downloadable PDF (backend/mg_predictor/mg/report_pdf.py)
// is laid out the same way.
export function ReportView({ data }: { data: PredictResponse }) {
  const { report, result, findings, blood, per_model, performance, recommended_specialists, advice, footer } = data;
  const care = data.care_locator;
  const positive = result.status === "POSITIVE";
  const p = report.patient;
  const metricKeys = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc"] as const;
  const metricHeads = ["Accuracy", "Precision", "Recall", "Specificity", "F1-score", "ROC-AUC"];

  return (
    <article className="report-paper" aria-label="Myasthenia gravis screening report">
      <header className="report-head">
        <div>
          <div className="report-brand">MyaScreen</div>
          <div className="report-brand-sub">Neuromuscular screening laboratory</div>
        </div>
        <div className="report-head-right">
          <div className="report-head-title">Screening Report</div>
          <div>Report ID {report.report_id}</div>
          <div>{report.generated_at}</div>
        </div>
      </header>

      <div className="report-body">
        <div className="report-grid" aria-label="Patient information">
          <Fact label="Patient name" value={p.name} />
          <Fact label="Age" value={`${p.age} years`} />
          <Fact label="Sex" value={p.gender} />
          <Fact label="Weight" value={p.weight_kg ? `${p.weight_kg} kg` : "-"} />
          <Fact label="Height" value={p.height_cm ? `${p.height_cm} cm` : "-"} />
          <Fact label="Body mass index" value={p.bmi ? `${p.bmi} (${p.bmi_category})` : "-"} />
          <Fact label="Symptom duration" value={p.symptom_duration} />
          <Fact label="Blood group" value={p.blood_group ?? "-"} />
          <Fact label="Test requested" value="Myasthenia gravis screening" />
          <Fact label="Date of report" value={report.generated_at} />
        </div>

        <div className={`report-result ${positive ? "positive" : "negative"}`} role="status">
          <div className="report-result-main">
            <div className="report-result-test">Myasthenia gravis</div>
            <div className="report-verdict">{result.status}</div>
            <div className="report-result-meta">
              Primary model: {result.primary_model} &nbsp;|&nbsp; Algorithms returning positive: {result.models_positive}
            </div>
          </div>
          <div className="report-result-confidence">
            <div className="report-fact-label">Model confidence</div>
            <div className="report-confidence-value">{result.confidence_percent}%</div>
            <div className="report-result-meta">
              {result.confidence_band} confidence
              <br />
              MG probability {result.mg_probability_percent}%
            </div>
          </div>
        </div>

        {advice.emergency && (
          <div className="report-critical" role="alert">
            <ShieldAlert className="size-5 shrink-0" />
            <p>
              <strong>Critical finding.</strong> {advice.emergency}
            </p>
          </div>
        )}

        <Section title="Clinical interpretation">
          <p>{result.interpretation}</p>
          {result.clinical_pattern && (
            <p className="mt-2">
              <strong>Clinical pattern:</strong> {result.clinical_pattern}
            </p>
          )}
          {advice.age_sex_context && <p className="report-muted mt-2">{advice.age_sex_context}</p>}
        </Section>

        <Section title="Reported findings">
          <div className="report-scroll">
            <table className="report-table">
              <thead>
                <tr>
                  <th>Category</th>
                  <th>Reported by patient</th>
                </tr>
              </thead>
              <tbody>
                {findings.map((f) => (
                  <tr key={f.section}>
                    <td className="font-semibold">{f.section}</td>
                    <td className={f.items.length ? "" : "report-muted"}>
                      {f.items.length ? f.items.join("; ") : "None reported"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>

        {(blood.panel.length > 0 || blood.antibodies.some((a) => a.status !== "Not tested")) && (
          <Section title="Blood investigations">
            <div className="report-scroll">
              <table className="report-table">
                <thead>
                  <tr>
                    <th>Test</th>
                    <th className="text-center">Result</th>
                    <th className="text-center">Unit</th>
                    <th className="text-center">Reference range</th>
                    <th className="text-center">Flag</th>
                  </tr>
                </thead>
                <tbody>
                  {blood.panel.map((r) => (
                    <tr key={r.key}>
                      <td>{r.label}</td>
                      <td className="text-center font-semibold">{r.value}</td>
                      <td className="text-center">{r.unit}</td>
                      <td className="text-center">{r.reference}</td>
                      <td className={`text-center font-bold ${r.status === "Normal" ? "report-neg" : "report-pos"}`}>{r.status.toUpperCase()}</td>
                    </tr>
                  ))}
                  {blood.antibodies.filter((a) => a.status !== "Not tested").map((a) => (
                    <tr key={a.key}>
                      <td>{a.label}</td>
                      <td className={`text-center font-bold ${a.status === "Positive" ? "report-pos" : "report-neg"}`}>{a.status.toUpperCase()}</td>
                      <td className="text-center">-</td>
                      <td className="text-center">Negative</td>
                      <td />
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {blood.group_note && <p className="report-muted mt-2">Blood group {blood.blood_group}: {blood.group_note}</p>}
          </Section>
        )}

        <Section title="Screening algorithm results">
          <div className="report-scroll">
            <table className="report-table">
              <thead>
                <tr>
                  <th>Algorithm</th>
                  <th className="text-center">Result</th>
                  <th className="text-center">MG probability</th>
                  <th className="text-center">Confidence</th>
                </tr>
              </thead>
              <tbody>
                {per_model.map((m) => (
                  <tr key={m.model}>
                    <td className={m.is_primary ? "font-semibold" : ""}>
                      {m.model}
                      {m.is_primary ? " (primary)" : ""}
                    </td>
                    <td className={`text-center font-bold ${m.flags_mg ? "report-pos" : "report-neg"}`}>{m.result}</td>
                    <td className="text-center">{m.mg_probability_percent}%</td>
                    <td className="text-center">{m.confidence_percent}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>

        <Section title="Model performance">
          <p className="report-muted mb-2">
            Evaluated on {performance.n_test.toLocaleString()} held-out patients not used for training (* = primary
            model used for this report).
          </p>
          <div className="report-scroll">
            <table className="report-table">
              <thead>
                <tr>
                  <th>Algorithm</th>
                  {metricHeads.map((h) => (
                    <th key={h} className="text-center">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(performance.per_model).map(([name, m]) => (
                  <tr key={name}>
                    <td className={name === performance.primary_model ? "font-semibold" : ""}>
                      {name}
                      {name === performance.primary_model ? " *" : ""}
                    </td>
                    {metricKeys.map((k) => (
                      <td key={k} className="text-center">
                        {m[k]}%
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>

        <Section title="Recommendations">
          <h3 className="report-sub">Next steps</h3>
          <BulletList items={advice.next_steps} />
          {advice.tests_to_ask_about && advice.tests_to_ask_about.length > 0 && (
            <>
              <h3 className="report-sub">Suggested confirmatory investigations</h3>
              <BulletList items={advice.tests_to_ask_about} />
            </>
          )}
          {advice.watch_for && advice.watch_for.length > 0 && (
            <>
              <h3 className="report-sub">Warning signs to monitor</h3>
              <BulletList items={advice.watch_for} />
            </>
          )}
          {advice.daily_management && advice.daily_management.length > 0 && (
            <>
              <h3 className="report-sub">Day-to-day management</h3>
              <BulletList items={advice.daily_management} />
            </>
          )}
          {advice.medication_safety && (
            <>
              <h3 className="report-sub">Medication safety</h3>
              <p>{advice.medication_safety}</p>
            </>
          )}
          {advice.about_treatment && (
            <>
              <h3 className="report-sub">About treatment</h3>
              <p>{advice.about_treatment}</p>
            </>
          )}
        </Section>

        <Section title="Recommended specialists">
          <div className="report-scroll">
            <table className="report-table">
              <thead>
                <tr>
                  <th>Specialist</th>
                  <th>When</th>
                  <th>Why</th>
                </tr>
              </thead>
              <tbody>
                {recommended_specialists.map((s) => (
                  <tr key={s.specialist}>
                    <td className="font-semibold">{s.specialist}</td>
                    <td>{s.when}</td>
                    <td>{s.why}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>


        <Section title={`Nearest and best hospital for myasthenia gravis${care ? ` (${care.region})` : ""}`}>
          {care ? (
            <>
              <p className="report-muted mb-2">From {care.origin.label}. {care.summary}</p>
              <div className="report-hospital-grid">
                <HospitalCard tag={care.same_hospital ? "Nearest and best for MG" : "Nearest"} h={care.nearest} />
                {!care.same_hospital && <HospitalCard tag="Best for MG" h={care.best} />}
              </div>
              <h3 className="report-sub">All hospitals compared</h3>
              <div className="report-scroll">
                <table className="report-table">
                  <thead>
                    <tr>
                      <th>Hospital</th>
                      <th>Location</th>
                      <th className="text-center">Distance</th>
                      <th className="text-center">MG evidence score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {care.all_hospitals.map((h) => (
                      <tr key={h.id}>
                        <td className="font-semibold">{h.name}</td>
                        <td>{h.address}</td>
                        <td className="text-center">{h.distance_km} km</td>
                        <td className="text-center">{h.evidence_score} / {h.max_score}{h.mg_named ? " (MG named)" : ""}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="report-muted mt-2">{care.method}</p>
              <p className="report-muted mt-1">{care.disclaimer}</p>
            </>
          ) : (
            <p className="report-muted">
              No location was entered. Choose your area in Chennai on the first screen to see the nearest and
              best hospital for myasthenia gravis care.
            </p>
          )}
        </Section>

        <p className="report-footer">{footer}</p>
      </div>
    </article>
  );
}
