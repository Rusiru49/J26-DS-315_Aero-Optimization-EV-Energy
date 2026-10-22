import { useMemo, useState } from "react";
import { api } from "../api";
import type { Anova, Corr, Desc, ExplainItem, FactorsOK, LiveCard, Reg, SampleStats, TTest } from "../api";
import { useApi } from "../lib/useApi";
import { num, pval, shortName, signed } from "../lib/format";
import { Card, Chip, ErrorBox, Loading, TD, TH } from "../components/ui";
import { Icon } from "../components/icons";
import type { IconName } from "../components/icons";
import PlotlyChart from "../components/PlotlyChart";

type Tab = "factors" | "live" | "demo";

const TABS: { key: Tab; icon: IconName; title: string; text: string }[] = [
  { key: "factors", icon: "layers", title: "What Affects The Saving?", text: "The factors behind the result, explained in plain words." },
  { key: "live", icon: "flask", title: "Statistical Tests", text: "The four tests on the results from C3. Each one shows its n." },
  { key: "demo", icon: "play", title: "Method Demo", text: "The same four tests on a larger example: 6 driving conditions × 8 designs = 48 results." },
];

export default function StatisticsPage() {
  const [tab, setTab] = useState<Tab>("factors");
  return (
    <div className="flex h-full flex-col gap-3">
      <div className="grid flex-none grid-cols-3 gap-3">
        {TABS.map((t) => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className={`flex items-center gap-3 rounded-2xl border p-3 text-left transition ${
              tab === t.key
                ? "border-brand bg-brand/10 shadow-[0_1px_6px_rgba(29,127,166,0.18)]"
                : "border-line bg-white hover:border-brand/40"
            }`}
          >
            <span
              className={`flex h-10 w-10 flex-none items-center justify-center rounded-xl ${
                tab === t.key ? "bg-brand text-white" : "bg-bg text-brand"
              }`}
            >
              <Icon name={t.icon} size={21} />
            </span>
            <span className="min-w-0">
              <span className="block text-sm font-semibold text-ink">{t.title}</span>
              <span className="block text-xs leading-snug text-muted">{t.text}</span>
            </span>
          </button>
        ))}
      </div>
      <div className="min-h-0 flex-1">{tab === "factors" ? <Factors /> : tab === "live" ? <Live /> : <Demo />}</div>
    </div>
  );
}

/* ============================================================== factors */

const BAR_COLORS = { condition: "#1d7fa6", design: "#19a58e", rest: "#c9d5de" };

function Factors() {
  const res = useApi(() => api.factors(), []);
  if (res.error) return <ErrorBox message={res.error} />;
  if (!res.data) return <Loading />;
  if (!res.data.available) return <ErrorBox message={res.data.message} />;
  return <FactorsBody d={res.data} />;
}

function FactorsBody({ d }: { d: FactorsOK }) {
  const v = d.variation;
  const rows: [string, string, number, string][] = v
    ? [
        ["Driving condition", "speed, stops and distance differ between conditions", v.condition, BAR_COLORS.condition],
        ["Design", "reduction in drag area (Cd × frontal area)", v.design, BAR_COLORS.design],
        ["Left over", "not explained by these two", v.rest, BAR_COLORS.rest],
      ]
    : [];

  const condData = useMemo(
    () => [
      {
        type: "bar",
        x: d.by_condition.map((c) => `${shortName(c.label)}<br>${c.avg_speed_kmh ?? "?"} km/h`),
        y: d.by_condition.map((c) => c.mean_saving_pct),
        text: d.by_condition.map((c) => `${c.mean_saving_pct.toFixed(1)} %`),
        textposition: "outside",
        marker: { color: BAR_COLORS.condition },
        hovertemplate: "%{x}: %{y:.2f} %<extra></extra>",
      },
    ],
    [d],
  );
  const designData = useMemo(
    () => [
      {
        type: "bar",
        x: d.by_design.map((x) => `${x.config_id}<br>drag area −${x.cda_reduction_pct.toFixed(1)} %`),
        y: d.by_design.map((x) => x.mean_saving_pct),
        text: d.by_design.map((x) => `${x.mean_saving_pct.toFixed(1)} %`),
        textposition: "outside",
        marker: { color: BAR_COLORS.design },
        hovertemplate: "%{x}: %{y:.2f} %<extra></extra>",
      },
    ],
    [d],
  );
  const ymax = Math.max(...d.by_condition.map((c) => c.mean_saving_pct), ...d.by_design.map((x) => x.mean_saving_pct));
  const layout = useMemo(
    () => ({ yaxis: { title: { text: "Energy saving (%)" }, range: [0, ymax * 1.25] }, margin: { t: 20 } }),
    [ymax],
  );

  return (
    <div className="grid h-full grid-cols-12 grid-rows-2 gap-3">
      <Card
        className="col-span-7"
        title="Layer 1: What The Data Shows"
        subtitle={`Results from C3: ${d.n_pairs} baseline-optimized pairs (${d.n_conditions} driving conditions, ${d.n_designs} designs).`}
      >
        {v ? (
          <div className="space-y-2">
            {rows.map(([name, note, pct, color]) => (
              <div key={name}>
                <div className="mb-0.5 flex flex-wrap items-baseline justify-between gap-2 text-[13px]">
                  <span>
                    <b>{name}</b> <span className="text-muted">· {note}</span>
                  </span>
                  <b>{num(pct, 0)} %</b>
                </div>
                <div className="h-2.5 w-full overflow-hidden rounded-full bg-bg">
                  <div className="h-full rounded-full" style={{ width: `${Math.max(pct, 1)}%`, background: color }} />
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="rounded-xl bg-warn/10 p-2.5 text-[13px]">{d.variation_note}</p>
        )}
        <ul className="mt-3 space-y-1 text-[13px] leading-snug">
          {d.insights.map((x) => (
            <li key={x} className="flex gap-2">
              <span className="mt-1.5 h-1.5 w-1.5 flex-none rounded-full bg-brand" />
              <span>{x}</span>
            </li>
          ))}
        </ul>
      </Card>

      <Card
        className="col-span-5 row-span-2"
        title="Layer 2: Apart From Cd, What Else Could Affect It?"
        subtitle="Each line is a number from C3's results or a setting C3 reported. It shows how the results go together, not the cause."
      >
        <ExplainList items={d.explain} />
        <div className="mt-3 rounded-xl border border-dashed border-soon bg-white/60 p-2.5">
          <div className="flex items-center justify-between">
            <b className="text-[13px]">Layer 3: Explainable AI</b>
            <Chip tone="soon">Later</Chip>
          </div>
          <p className="mt-1 text-xs text-muted">{d.explain_later}</p>
        </div>
      </Card>

      <div className="col-span-7 grid min-h-0 grid-cols-2 gap-3">
        <Card title="By Driving Condition" subtitle="Energy saving of the design made for each condition.">
          <PlotlyChart data={condData} layout={layout} />
        </Card>
        <Card title="By Design" subtitle="Energy saving, with the reduction in drag area (Cd × frontal area).">
          <PlotlyChart data={designData} layout={layout} />
        </Card>
      </div>
    </div>
  );
}

const STATUS: Record<ExplainItem["status"], { label: string; tone: "ok" | "warn" | "soon" | "brand" | "bench" }> = {
  in_data: { label: "In the data", tone: "ok" },
  not_separable: { label: "Cannot separate yet", tone: "warn" },
  needs_data: { label: "Needs more data", tone: "soon" },
  fixed: { label: "Same for all results", tone: "brand" },
  cost_only: { label: "Cost only", tone: "bench" },
};

function ExplainList({ items }: { items: ExplainItem[] }) {
  return (
    <div className="space-y-2">
      {items.map((it) => {
        const st = STATUS[it.status];
        return (
          <div key={it.factor} className="rounded-xl border border-line p-2.5">
            <div className="flex items-start justify-between gap-2">
              <div>
                <b className="text-[13px]">{it.factor}</b>
                <span className="ml-2 text-[11px] text-muted">{it.group}</span>
              </div>
              <Chip tone={st.tone}>{st.label}</Chip>
            </div>
            <p className="mt-1 text-xs leading-snug text-muted">{it.text}</p>
          </div>
        );
      })}
    </div>
  );
}

/* ================================================================== live */

function Live() {
  const res = useApi(() => api.statsLive(), []);
  if (res.error) return <ErrorBox message={res.error} />;
  if (!res.data) return <Loading />;
  const d = res.data;
  return (
    <div className="flex h-full flex-col gap-3">
      <Card
        className="flex-none"
        title="Descriptive summary: energy saving (%)"
        subtitle={`Across ${d.n_pairs} baseline–optimized pairs (${d.conditions_with_data.length} driving conditions). Each test runs as soon as its own minimum is met and shows n.`}
      >
        <DescTiles d={d.descriptive.energy_saving_pct} />
      </Card>
      <div className="grid min-h-0 flex-1 grid-cols-2 content-start gap-3 overflow-auto">
        {d.tests.map((c) => (
          <TestCard key={c.id} card={c} />
        ))}
      </div>
    </div>
  );
}

function DescTiles({ d }: { d: Desc }) {
  const items: [string, string][] = [
    ["n", String(d.n)],
    ["Mean", num(d.mean, 2)],
    ["Std. dev.", num(d.std, 2)],
    ["Min", num(d.min, 2)],
    ["Median", num(d.median, 2)],
    ["Max", num(d.max, 2)],
  ];
  return (
    <div className="grid grid-cols-3 gap-3 md:grid-cols-6">
      {items.map(([k, v]) => (
        <div key={k} className="rounded-xl border border-line bg-bg p-3 text-center">
          <div className="text-xs text-muted">{k}</div>
          <div className="text-lg font-bold">{v}</div>
        </div>
      ))}
    </div>
  );
}

function TestCard({ card }: { card: LiveCard }) {
  const ready = card.status === "ready";
  return (
    <Card
      title={card.name}
      right={ready ? <Chip tone="ok">Ready</Chip> : <Chip tone="soon">🔒 Locked</Chip>}
      className={ready ? "" : "border-dashed bg-white/60"}
    >
      {ready ? (
        <TestBody card={card} />
      ) : (
        <div className="space-y-2 text-sm">
          <p className="text-muted">{card.reason}</p>
          <p><span className="text-muted">Needs:</span> {card.requires}</p>
          <p><span className="text-muted">Has now:</span> {card.have.replace(/[{}']/g, "")}</p>
        </div>
      )}
    </Card>
  );
}

function TestBody({ card }: { card: LiveCard }) {
  switch (card.id) {
    case "paired_t":
      return <TTestView r={card.result as TTest} />;
    case "correlation":
      return <CorrView r={card.result as Corr} />;
    case "regression":
      return <RegView r={card.result as Reg} />;
    case "anova":
      return <AnovaView r={card.result as Anova} />;
  }
}

function KV({ rows }: { rows: [string, string][] }) {
  return (
    <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-sm">
      {rows.map(([k, v]) => (
        <div key={k} className="contents">
          <dt className="text-muted">{k}</dt>
          <dd className="text-right font-medium">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

function Verdict({ p, alpha, n }: { p: number | null; alpha: number; n: number }) {
  if (p === null) return null;
  const below = p < alpha;
  return (
    <p className={`mt-3 rounded-xl p-3 text-sm ${below ? "bg-opt/10 text-ink" : "bg-warn/10 text-ink"}`}>
      p {pval(p).startsWith("<") ? pval(p) : `= ${pval(p)}`} is {below ? "below" : "not below"} {alpha}
      {below ? ": the result is statistically significant." : ": not significant at this sample size."} (n = {n})
    </p>
  );
}

/** The test result in plain words. Only says what the p-value shows. It never claims a cause. */
const ALPHA = 0.05; // same significance level the backend uses
const TERM_NAMES: Record<string, string> = {
  cd_reduction_pct: "Cd reduction (drag)",
  avg_speed_kmh: "Average speed",
};

function Meaning({ lines }: { lines: string[] }) {
  return (
    <div className="mt-2 rounded-xl border border-brand/20 bg-brand/5 p-2.5 text-xs leading-snug">
      <b>In plain words:</b>
      <ul className="mt-1 space-y-1">
        {lines.map((t) => (
          <li key={t} className="flex gap-2">
            <span className="mt-1.5 h-1.5 w-1.5 flex-none rounded-full bg-brand" />
            <span>{t}</span>
          </li>
        ))}
        <li className="flex gap-2 text-muted">
          <span className="mt-1.5 h-1.5 w-1.5 flex-none rounded-full bg-soon" />
          <span>"Significant" means unlikely to be just chance. It does not prove the cause.</span>
        </li>
      </ul>
    </div>
  );
}

function TTestView({ r }: { r: TTest }) {
  return (
    <>
      <div className="mb-2"><Chip tone="brand">n = {r.n_pairs} pairs · df = {r.df}</Chip></div>
      <KV
        rows={[
          ["Mean energy, baseline", `${num(r.mean_before, 1)} Wh/km`],
          ["Mean energy, optimized", `${num(r.mean_after, 1)} Wh/km`],
          ["Mean difference", `${num(r.mean_difference, 2)} Wh/km`],
          [`${Math.round(r.ci_level * 100)} % confidence interval`, `${num(r.ci_low, 2)} to ${num(r.ci_high, 2)}`],
          ["t statistic", num(r.t_statistic, 3)],
          ["Effect size (Cohen's dz)", num(r.cohens_dz, 2)],
          ["Wilcoxon p-value", pval(r.wilcoxon?.p_value)],
        ]}
      />
      <Verdict p={r.p_value} alpha={r.alpha} n={r.n_pairs} />
      {r.p_value !== null && r.mean_before !== null && r.mean_after !== null && (
        <Meaning
          lines={[
            r.p_value < r.alpha
              ? `The optimized design uses ${r.mean_after < r.mean_before ? "less" : "more"} energy than the baseline (${num(r.mean_before, 1)} → ${num(r.mean_after, 1)} Wh/km), and this is unlikely to be just chance.`
              : `The data does not clearly show a difference in energy between the baseline and the optimized design. With ${r.n_pairs} pairs it could be chance.`,
          ]}
        />
      )}
      {r.note && <p className="mt-2 text-xs text-muted">{r.note}</p>}
    </>
  );
}

function CorrView({ r }: { r: Corr }) {
  return (
    <>
      <div className="mb-2"><Chip tone="brand">n = {r.n} points</Chip></div>
      <KV
        rows={[
          ["Pearson r", num(r.pearson_r, 3)],
          ["Pearson p-value", pval(r.pearson_p)],
          ["Spearman ρ", num(r.spearman_rho, 3)],
          [
            "Spearman p-value",
            r.spearman_rho !== null && Math.abs(Math.abs(r.spearman_rho) - 1) < 1e-12
              ? "not defined (ρ = ±1)"
              : pval(r.spearman_p),
          ],
        ]}
      />
      <p className="mt-3 text-xs text-muted">{r.note}</p>
      {r.pearson_r !== null && r.pearson_p !== null && (
        <Meaning
          lines={[
            r.pearson_p < ALPHA
              ? `There is a clear relationship between speed and saving: when the average speed is higher, the saving is ${r.pearson_r > 0 ? "larger" : "smaller"} (r = ${num(r.pearson_r, 2)}). It is unlikely to be just chance.`
              : `No clear relationship between speed and saving in this data (r = ${num(r.pearson_r, 2)}). Any link seen could be chance.`,
          ]}
        />
      )}
    </>
  );
}

function RegView({ r }: { r: Reg }) {
  return (
    <>
      <div className="mb-2"><Chip tone="brand">n = {r.n} rows · {r.predictors.length} predictor(s)</Chip></div>
      <KV rows={[["R²", num(r.r_squared, 4)], ["Adjusted R²", num(r.adj_r_squared, 4)], ["Model p-value", pval(r.f_p_value)]]} />
      <table className="mt-3 w-full">
        <thead>
          <tr className="border-b border-line">
            <th className={TH}>Term</th>
            <th className={TH}>Coefficient</th>
            <th className={TH}>Std. error</th>
            <th className={TH}>p</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-line">
          {r.coefficients.map((c) => (
            <tr key={c.name}>
              <td className={TD}>{c.name === "const" ? "Intercept" : c.name}</td>
              <td className={TD}>{num(c.coef, 4)}</td>
              <td className={TD}>{num(c.std_err, 4)}</td>
              <td className={TD}>{pval(c.p_value)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {r.warnings.map((w) => (
        <p key={w} className="mt-2 text-xs text-[#9a6a14]">{w}</p>
      ))}
      <p className="mt-3 text-xs text-muted">{r.note}</p>
      {r.f_p_value !== null && (
        <Meaning
          lines={[
            r.f_p_value < ALPHA
              ? `Together, ${r.predictors.map((x) => TERM_NAMES[x] ?? x).join(" and ")} explain part of the saving (R² = ${num(r.r_squared, 2)}). This is unlikely to be just chance.`
              : "These factors together do not clearly explain the saving in this data.",
            ...r.coefficients
              .filter((c) => c.name !== "const" && c.p_value !== null && c.coef !== null)
              .map((c) =>
                (c.p_value as number) < ALPHA
                  ? `${TERM_NAMES[c.name] ?? c.name} has a clear link with the saving: a higher value goes with a ${(c.coef as number) > 0 ? "larger" : "smaller"} saving.`
                  : `${TERM_NAMES[c.name] ?? c.name}: no clear link with the saving.`,
              ),
          ]}
        />
      )}
    </>
  );
}

function AnovaView({ r }: { r: Anova }) {
  const groups = Object.entries(r.groups);
  const data = useMemo(
    () => [
      {
        type: "bar",
        x: groups.map(([k]) => k),
        y: groups.map(([, g]) => g.mean),
        error_y: { type: "data", array: groups.map(([, g]) => g.std), visible: true, color: "#5d7589" },
        marker: { color: "#19a58e" },
      },
    ],
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [r],
  );
  const layout = useMemo(() => ({ yaxis: { title: { text: "Energy saving (%)" }, rangemode: "tozero" } }), []);
  return (
    <>
      <div className="mb-2">
        <Chip tone="brand">
          {groups.length} conditions · n = {groups.reduce((s, [, g]) => s + g.n, 0)} · df = {r.df_between}, {r.df_within}
        </Chip>
      </div>
      <KV rows={[["F statistic", num(r.f_statistic, 3)], ["Effect size (η²)", num(r.eta_squared, 3)]]} />
      <Verdict p={r.p_value} alpha={r.alpha} n={groups.reduce((s, [, g]) => s + g.n, 0)} />
      {r.p_value !== null && (
        <Meaning
          lines={[
            r.p_value < r.alpha
              ? "The saving is not the same in every driving condition. The table below shows which conditions differ."
              : "No clear difference in saving between the driving conditions in this data.",
          ]}
        />
      )}
      <div className="mt-2">
        <PlotlyChart data={data} layout={layout} height={200} />
        <p className="text-center text-xs text-muted">Mean energy saving per condition (bars) and standard deviation (lines)</p>
      </div>
      <h4 className="mb-1 mt-4 text-sm font-semibold">Tukey HSD: which conditions differ?</h4>
      <div>
        <table className="w-full">
          <thead className="sticky top-0 bg-white">
            <tr className="border-b border-line">
              <th className={TH}>Pair</th>
              <th className={TH}>Mean diff</th>
              <th className={TH}>Adj. p</th>
              <th className={TH}>Different?</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-line">
            {r.tukey_hsd.map((t) => (
              <tr key={`${t.group1}-${t.group2}`}>
                <td className={TD}>{t.group1} vs {t.group2}</td>
                <td className={TD}>{signed(t.mean_diff, 2)}</td>
                <td className={TD}>{pval(t.p_adj)}</td>
                <td className={TD}>{t.reject ? <Chip tone="ok">Yes</Chip> : <Chip tone="soon">No</Chip>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

/* ================================================================== demo */

function Demo() {
  const res = useApi(() => api.statsSample(), []);
  if (res.error) return <ErrorBox message={res.error} />;
  if (!res.data) return <Loading />;
  return <DemoBody d={res.data} />;
}

function DemoBody({ d }: { d: SampleStats }) {
  const scatter = useMemo(
    () => [
      {
        type: "scatter",
        mode: "markers+text",
        x: d.by_condition.map((c) => c.avg_speed_kmh),
        y: d.by_condition.map((c) => c.mean_saving_pct),
        text: d.by_condition.map((c) => c.condition),
        textposition: d.by_condition.map((_, i) => (i % 2 === 0 ? "top center" : "bottom center")),
        marker: { size: 11, color: "#19a58e" },
        hovertemplate: "%{text}: %{y:.2f} %<extra></extra>",
      },
    ],
    [d],
  );
  const layout = useMemo(
    () => ({
      xaxis: { title: { text: "Average speed of the condition (km/h)" } },
      yaxis: { title: { text: "Mean energy saving (%)" }, range: [0, 6.5] },
      margin: { t: 30 },
    }),
    [],
  );
  return (
    <div className="grid h-full grid-cols-2 content-start gap-3 overflow-auto">
      <div className="col-span-2 rounded-2xl border border-brand/25 bg-brand/5 p-3 text-[13px] leading-snug">
        <b>What is 6 × 8?</b> This example has <b>6 driving conditions</b> and <b>8 designs</b>, so <b>48 results</b>.
        The results from C3 have fewer driving conditions so far, and some tests need more. With 48 results, all four
        tests can run, so you can see how each one works and how to read it.
      </div>
      <Card title="Paired t-test (baseline vs optimized energy)" subtitle={d.design} right={<Chip tone="ok">Ready</Chip>}>
        <TTestView r={d.paired_t} />
      </Card>
      <Card title="ANOVA + Tukey HSD (saving % across driving conditions)" right={<Chip tone="ok">Ready</Chip>}>
        <AnovaView r={d.anova} />
      </Card>
      <Card title="Correlation (saving % vs condition average speed)" right={<Chip tone="ok">Ready</Chip>}>
        <CorrView r={d.correlation} />
        <div className="mt-2 h-[200px]">
          <PlotlyChart data={scatter} layout={layout} />
        </div>
      </Card>
      <Card title="Regression (saving % ~ Cd reduction + average speed)" right={<Chip tone="ok">Ready</Chip>}>
        <RegView r={d.regression} />
      </Card>
    </div>
  );
}