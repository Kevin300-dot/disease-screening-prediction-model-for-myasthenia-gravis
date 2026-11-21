import { afterEach, describe, expect, it, vi } from "vitest";
import { downloadReportPdf, getDatasetBreakdown, getLearningCurve, getMetrics, predictMG } from "@/lib/mg-api";

afterEach(() => vi.unstubAllGlobals());

describe("MG API client", () => {
  it("sends the existing patient and warning-sign keys to /predict", async () => {
    const request = { age: 34, gender: "female" as const, symptoms: ["fatigue"], red_flags: ["droopy_eyelids"] };
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) });
    vi.stubGlobal("fetch", fetchMock);
    await predictMG(request);
    const [url, options] = fetchMock.mock.calls[0] ?? [];
    expect(url).toMatch(/\/predict$/);
    expect(options.method).toBe("POST");
    expect(JSON.parse(options.body)).toEqual(request);
    expect(options.headers).toEqual({ "Content-Type": "application/json" });
  });

  it("keeps the existing dataset-breakdown endpoint", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) });
    vi.stubGlobal("fetch", fetchMock);
    await getDatasetBreakdown();
    expect(fetchMock.mock.calls[0]?.[0]).toMatch(/\/dataset-breakdown$/);
  });

  it("keeps the existing learning-curve endpoint", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) });
    vi.stubGlobal("fetch", fetchMock);
    await getLearningCurve();
    expect(fetchMock.mock.calls[0]?.[0]).toMatch(/\/learning-curve$/);
  });

  it("sends the new patient fields alongside the symptom keys", async () => {
    const request = {
      name: "Priya Raman", age: 34, gender: "female" as const, weight_kg: 58, height_cm: 162,
      duration_months: 4.5, symptoms: ["fatigue"], red_flags: ["droopy_eyelids"],
    };
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) });
    vi.stubGlobal("fetch", fetchMock);
    await predictMG(request);
    expect(JSON.parse(fetchMock.mock.calls[0]?.[1].body)).toEqual(request);
  });

  it("sends blood group, blood tests and antibody results", async () => {
    const request = {
      age: 34, gender: "female" as const, symptoms: [], red_flags: [], blood_group: "O+",
      blood_tests: { hemoglobin: 12.4, esr: 18 }, antibodies: { achr: "positive", musk: "not_tested" },
    };
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) });
    vi.stubGlobal("fetch", fetchMock);
    await predictMG(request);
    expect(JSON.parse(fetchMock.mock.calls[0]?.[1].body)).toEqual(request);
  });

  it("requests the PDF report and reads the filename from the response", async () => {
    const request = { age: 34, gender: "female" as const, symptoms: [], red_flags: [], report_id: "MYA-1" };
    const blob = new Blob(["%PDF-1.4"], { type: "application/pdf" });
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      blob: async () => blob,
      headers: new Headers({ "Content-Disposition": "attachment; filename=MyaScreen_Report_Test.pdf" }),
    });
    vi.stubGlobal("fetch", fetchMock);
    const file = await downloadReportPdf(request);
    const [url, options] = fetchMock.mock.calls[0] ?? [];
    expect(url).toMatch(/\/report-pdf$/);
    expect(options.method).toBe("POST");
    expect(JSON.parse(options.body)).toEqual(request);
    expect(file.filename).toBe("MyaScreen_Report_Test.pdf");
    expect(file.blob).toBe(blob);
  });

  it("keeps the metrics endpoint", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) });
    vi.stubGlobal("fetch", fetchMock);
    await getMetrics();
    expect(fetchMock.mock.calls[0]?.[0]).toMatch(/\/metrics$/);
  });

  it("sends an optional Chennai location alongside the patient data", async () => {
    const request = {
      age: 34, gender: "female" as const, symptoms: ["fatigue"], red_flags: ["droopy_eyelids"],
      location: { area: "velachery" },
    };
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) });
    vi.stubGlobal("fetch", fetchMock);
    await predictMG(request);
    expect(JSON.parse(fetchMock.mock.calls[0]?.[1].body)).toEqual(request);
  });
});
