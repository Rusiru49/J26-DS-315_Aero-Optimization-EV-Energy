import { useMemo, useState } from "react";
import { api } from "../api";
import type { Anova, Corr, Desc, FactorsOK, LiveCard, Reg, SampleStats, TTest } from "../api";
import { useApi } from "../lib/useApi";
import { condShort, num, pval, shortName, signed } from "../lib/format";
import { Card, Chip, ErrorBox, Loading, Segmented, TD, TH } from "../components/ui";
import PlotlyChart from "../components/PlotlyChart";

type Tab = "factors" | "live" | "demo";

export default function StatisticsPage() {
  const [tab, setTab] = useState<Tab>("factors");
  return (
    <div className="flex h-full flex-col gap-3">
      <div className="flex flex-none flex-wrap items-center justify-between gap-2">
        <h2 className="text-base font-semibold">Statistical analysis</h2>
        <Segmented
          value={tab}
          onChange={setTab}
          options={[
            ["factors", "What affects the saving?"],
            ["live", "Statistical tests (for the report)"],
            ["demo", "Method demonstration (6 conditions × 8 designs)"],
          ]}
        />
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
        title="What goes with the differences in energy saving?"
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
        className="col-span-5"
        title="Other factors"
        subtitle="Factors an engineer may ask about. They are tested as soon as the data allows."
      >
        <div className="space-y-2">
          {d.vehicle_factors.map((f) => (
            <div key={f.factor} className={`rounded-xl border p-2.5 ${f.tested ? "border-line" : "border-dashed border-soon bg-white/60"}`}>
              <div className="flex items-center justify-between">
                <b className="text-[13px]">{f.factor}</b>
                {f.tested ? <Chip tone="ok">Tested</Chip> : <Chip tone="soon">Not tested yet</Chip>}
              </div>
              <p className="mt-1 text-xs text-muted">
                {f.tested ? `Correlation with the saving across ${f.n_evs} EVs: r = ${num(f.r, 2)}.` : f.reason}
              </p>
            </div>
          ))}
          <div className="rounded-xl border border-dashed border-soon bg-white/60 p-2.5">
            <div className="flex items-center justify-between">
              <b className="text-[13px]">Speed, idle share, distance</b>
              {d.conditions_move_together ? <Chip tone="soon">Cannot be separated yet</Chip> : <Chip tone="soon">Needs more conditions</Chip>}
            </div>
            <p className="mt-1 text-xs text-muted">
              {d.conditions_move_together
                ? `They rise together across these ${d.n_conditions} conditions. The other SLDC conditions will make it possible to separate them.`
                : `Only ${d.n_conditions} conditions have results (${d.by_condition.map((c) => condShort(c.condition_id)).join(", ")}). More conditions are needed.`}
            </p>
          </div>
          <p className="text-[11px] text-muted">p-values and test results are in the tab “Statistical tests (for the report)”.</p>
        </div>
      </Card>

      <Card className="col-span-6" title="By driving condition" subtitle="Energy saving of the design made for each condition.">
        <PlotlyChart data={condData} layout={layout} />
      </Card>
      <Card className="col-span-6" title="By design" subtitle="Energy saving, with the reduction in drag area (Cd × frontal area).">
        <PlotlyChart data={designData} layout={layout} />
      </Card>
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