import { Loader2 } from "lucide-react";
import {
  Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, Pie, PieChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";

import type { DatasetBreakdown, LearningCurveResponse, Performance } from "@/lib/mg-api";

// Fixed categorical palette for the 5-algorithm comparison charts (data-viz
// colors, independent of the semantic UI accents like --emergency/--success).
export const MODEL_COLORS: Record<string, string> = {
  "Naive Bayes": "var(--chart-1)",
  "Decision Tree": "var(--chart-2)",
  "Linear Regression": "var(--chart-3)",
  "Logistic Regression": "var(--chart-4)",
  "Random Forest": "var(--chart-5)",
};

const PIE_COLORS = {
  "Myasthenia gravis": "var(--emergency)",
  "Healthy (Normal)": "var(--success)",
  "Other autoimmune diseases": "var(--primary)",
};

const shortName = (name: string) => name.replace(" Regression", " Reg.").replace(" Forest", " For.");

// Headline accuracy numbers for the primary model, shown in the sidebar from the first screen on.
export function PerformanceCard({ performance }: { performance: Performance }) {
  const m = performance.primary;
  const items: [string, number][] = [
    ["Accuracy", m.accuracy],
    ["Precision", m.precision],
    ["Recall", m.recall],
    ["F1-score", m.f1],
  ];
  return (
    <div className="border-t border-border pt-5">
      <div className="font-mono text-[11px] uppercase tracking-[0.12em] text-muted-foreground">Model performance</div>
      <div className="mt-1 text-xs text-muted-foreground">
        {performance.primary_model} · tested on {performance.n_test.toLocaleString()} unseen patients
      </div>
      <div className="mt-3 grid grid-cols-2 gap-3">
        {items.map(([label, value]) => (
          <div key={label} className="rounded-md border border-border bg-background px-3 py-2.5">
            <div className="text-[11px] text-muted-foreground">{label}</div>
            <div className="mt-0.5 font-display text-xl font-semibold text-success">{value}%</div>
          </div>
        ))}
      </div>
    </div>
  );
}

export function DatasetCompositionCard({ breakdown }: { breakdown: DatasetBreakdown }) {
  const total = Object.values(breakdown).reduce((a, b) => a + b, 0);
  const data = Object.entries(breakdown).map(([name, value]) => ({ name, value }));
  return (
    <div className="border-t border-border pt-5">
      <div className="font-mono text-[11px] uppercase tracking-[0.12em] text-muted-foreground">
        Training dataset composition
      </div>
      <div className="mt-2 h-40">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie data={data} dataKey="value" nameKey="name" innerRadius={36} outerRadius={60} paddingAngle={2}>
              {data.map((entry) => (
                <Cell key={entry.name} fill={PIE_COLORS[entry.name as keyof typeof PIE_COLORS] ?? "var(--border)"} />
              ))}
            </Pie>
            <Tooltip
              formatter={(value: number, name: string) => [
                `${value.toLocaleString()} (${((value / total) * 100).toFixed(1)}%)`,
                name,
              ]}
            />
          </PieChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-2 space-y-1">
        {data.map((d) => (
          <div key={d.name} className="flex items-center justify-between text-[11px]">
            <span className="flex items-center gap-1.5 text-muted-foreground">
              <span
                className="size-2 rounded-full"
                style={{ background: PIE_COLORS[d.name as keyof typeof PIE_COLORS] }}
              />
              {d.name}
            </span>
            <span className="font-medium text-foreground">{d.value.toLocaleString()}</span>
          </div>
        ))}
      </div>
      <p className="mt-2 text-[10px] leading-4 text-muted-foreground">
        {total.toLocaleString()} simulated patient records with the full MG symptom checklist.
      </p>
    </div>
  );
}

// Grouped bars: accuracy / precision / recall / F1 for each of the five algorithms.
export function PerformanceBars({ performance }: { performance: Performance }) {
  const data = Object.entries(performance.per_model).map(([name, m]) => ({
    name: shortName(name),
    Accuracy: m.accuracy,
    Precision: m.precision,
    Recall: m.recall,
    "F1-score": m.f1,
  }));
  const series: [string, string][] = [
    ["Accuracy", "var(--chart-1)"],
    ["Precision", "var(--chart-2)"],
    ["Recall", "var(--chart-3)"],
    ["F1-score", "var(--chart-5)"],
  ];
  return (
    <div className="mt-3 h-64">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 4, right: 8, left: -12, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
          <XAxis dataKey="name" tick={{ fontSize: 10, fill: "var(--muted-foreground)" }} interval={0} />
          <YAxis domain={[80, 100]} tick={{ fontSize: 10, fill: "var(--muted-foreground)" }} unit="%" />
          <Tooltip formatter={(value: number) => `${value}%`} />
          <Legend wrapperStyle={{ fontSize: 10 }} />
          {series.map(([key, color]) => (
            <Bar key={key} dataKey={key} fill={color} radius={[3, 3, 0, 0]} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function LearningCurveChart({ data }: { data: LearningCurveResponse }) {
  const modelNames = Object.keys(data.curves);
  // Reshape {model: {train_sizes, val_roc_auc_mean}} into one row per train
  // size with every model's score as a column, which is what recharts wants.
  const firstModel = modelNames[0];
  const sizes = firstModel ? data.curves[firstModel]?.train_sizes ?? [] : [];
  const chartData = sizes.map((size, i) => {
    const row: Record<string, number> = { size };
    for (const name of modelNames) {
      const score = data.curves[name]?.val_roc_auc_mean[i];
      if (score !== undefined) row[name] = score;
    }
    return row;
  });

  return (
    <div className="mt-3 h-56">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 4, right: 8, left: -16, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
          <XAxis
            dataKey="size"
            tick={{ fontSize: 10, fill: "var(--muted-foreground)" }}
            label={{ value: "Training examples", position: "insideBottom", offset: -2, fontSize: 10, fill: "var(--muted-foreground)" }}
          />
          <YAxis domain={[0.8, 1]} tick={{ fontSize: 10, fill: "var(--muted-foreground)" }} />
          <Tooltip formatter={(value: number) => value.toFixed(3)} />
          <Legend wrapperStyle={{ fontSize: 10 }} />
          {modelNames.map((name) => (
            <Line
              key={name}
              type="monotone"
              dataKey={name}
              stroke={MODEL_COLORS[name] ?? "var(--primary)"}
              strokeWidth={2}
              dot={{ r: 2 }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function LoadingLine({ children }: { children: string }) {
  return (
    <div className="mt-3 flex items-center gap-2 text-xs text-muted-foreground">
      <Loader2 className="size-3.5 animate-spin" /> {children}
    </div>
  );
}
