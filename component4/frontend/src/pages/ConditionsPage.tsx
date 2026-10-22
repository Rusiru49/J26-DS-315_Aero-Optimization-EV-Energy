import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import type { Catalog, ConditionsCompare, SensitivityOK } from "../api";
import { useApi } from "../lib/useApi";
import { num, shortName, signed } from "../lib/format";
import { Card, Chip, ErrorBox, Loading, Segmented, TD, TH } from "../components/ui";
import PlotlyChart from "../components/PlotlyChart";

type Metric = "energy" | "range" | "cost";
const METRICS: Record<Metric, { label: string; axis: string; digits: number }> = {
  energy: { label: "Energy saving", axis: "Energy saving (%)", digits: 2 },
  range: { label: "Range gain", axis: "Range gain (km)", digits: 1 },
  cost: { label: "Cost saving", axis: "Cost saving (LKR / 100 km)", digits: 1 },
};

const BASE = "#4a86c5";
const OPT = "#19a58e";
const BENCH = "#7a6fc8";

/** Function 2: put the results of the chosen driving conditions next to each other. */
export default function ConditionsPage({
  evId,
  conditionIds,
}: {
  evId: string;
  conditionIds: string[];
  catalog: Catalog;
}) {
  const key = conditionIds.join(",");
  const cmp = useApi(() => api.conditions(evId, conditionIds), [evId, key]);
  if (cmp.error) return <ErrorBox message={cmp.error} />;
  if (!cmp.data) return <Loading />;
  const d = cmp.data;

  return (
    <div className="grid h-full grid-cols-12 gap-3">
      <div className="col-span-8 flex min-h-0 flex-col gap-3">
        <div className="flex flex-none flex-wrap items-center gap-2">
          <h2 className="text-[16px] font-semibold text-ink">
            {d.conditions.length > 1 ? "Chosen Driving Conditions Side By Side" : "Chosen Driving Condition"}
          </h2>
          {d.conditions.map((c) => (
            <Chip key={c.condition_id} tone="opt">
              {shortName(c.label)}: {c.design_id}
            </Chip>
          ))}
        </div>
        <ConditionCharts d={d} />
        <ConditionTable d={d} />
      </div>

      <div className="col-span-4 flex min-h-0 flex-col gap-3">
        <Insights d={d} />
        <Sensitivity evId={evId} options={d.conditions.map((c) => ({ id: c.condition_id, label: shortName(c.label) }))} />
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------- */

function ConditionCharts({ d }: { d: ConditionsCompare }) {
  const [metric, setMetric] = useState<Metric>("energy");
  const m = METRICS[metric];
  const rows = d.conditions;
  const names = rows.map((r) => shortName(r.label));
  const values = rows.map((r) =>
    metric === "energy" ? r.saving_pct : metric === "range" ? r.range_gain_km : r.cost_saving_lkr_per_100km,
  );

  const savingData = useMemo(
    () => [
      {
        type: "bar",
        x: names,
        y: values,
        marker: { color: OPT },
        text: values.map((v) => num(v, m.digits)),
        textposition: "outside",
        cliponaxis: false,
        hovertemplate: "%{x}: %{y:." + m.digits + "f}<extra></extra>",
      },
    ],
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [d, metric],
  );
  const savingLayout = useMemo(
    () => ({ yaxis: { title: { text: m.axis }, rangemode: "tozero" }, margin: { t: 24 }, showlegend: false }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [metric],
  );

  // three bars for each condition: the baseline car on WLTP, the baseline car on the condition, the optimized car on the condition
  const energyData = useMemo(
    () => [
      {
        type: "bar", name: "Baseline on WLTP", x: names, y: rows.map((r) => r.benchmark_energy_wh_per_km),
        marker: { color: BENCH }, text: rows.map((r) => (r.benchmark_energy_wh_per_km === null ? "" : num(r.benchmark_energy_wh_per_km, 1))),
        textposition: "outside", cliponaxis: false,
      },
      {
        type: "bar", name: "Baseline", x: names, y: rows.map((r) => r.energy_baseline_wh_per_km),
        marker: { color: BASE }, text: rows.map((r) => num(r.energy_baseline_wh_per_km, 1)),
        textposition: "outside", cliponaxis: false,
      },
      {
        type: "bar", name: "Optimized", x: names, y: rows.map((r) => r.energy_optimized_wh_per_km),
        marker: { color: OPT }, text: rows.map((r) => num(r.energy_optimized_wh_per_km, 1)),
        textposition: "outside", cliponaxis: false,
      },
    ],
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [d],
  );
  const energyLayout = useMemo(
    () => ({ barmode: "group", yaxis: { title: { text: "Energy consumed (Wh/km)" }, rangemode: "tozero" }, margin: { t: 34 } }),
    [],
  );

  return (
    <div className="grid min-h-0 flex-1 grid-cols-2 gap-3">
      <Card
        title="Savings By Driving Condition"
        right={<Segmented value={metric} onChange={setMetric} options={(Object.keys(METRICS) as Metric[]).map((k) => [k, METRICS[k].label])} />}
      >
        <PlotlyChart data={savingData} layout={savingLayout} />
      </Card>
      <Card title="Energy Consumed: WLTP, Baseline And Optimized" subtitle="WLTP is the standard cycle. Its result is different for each driving condition.">
        <PlotlyChart data={energyData} layout={energyLayout} />
      </Card>
    </div>
  );
}

function ConditionTable({ d }: { d: ConditionsCompare }) {
  return (
    <Card className="flex-none" title="Results For Each Chosen Condition" subtitle="Each condition uses the optimized design made for it.">
      <table className="w-full">
        <thead>
          <tr className="border-b border-line">
            <th className={TH}>Condition</th>
            <th className={TH}>Design</th>
            <th className={TH}>Avg speed</th>
            <th className={TH}>Baseline (Wh/km)</th>
            <th className={`${TH} text-bench`}>Baseline vs WLTP</th>
            <th className={TH}>Energy saving</th>
            <th className={TH}>Range gain</th>
            <th className={TH}>Cost saving</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-line">
          {d.conditions.map((c) => (
            <tr key={c.condition_id}>
              <td className={`${TD} font-medium`}>{shortName(c.label)}</td>
              <td className={TD}>{c.design_id}</td>
              <td className={TD}>{num(c.avg_speed_kmh, 1)} km/h</td>
              <td className={TD}>{num(c.energy_baseline_wh_per_km, 1)}</td>
              <td className={TD}>{c.baseline_vs_benchmark_pct === null ? "—" : `${signed(c.baseline_vs_benchmark_pct, 1)} %`}</td>
              <td className={TD}>
                <b>{num(c.saving_pct, 2)} %</b> <span className="text-muted">({num(c.energy_saving_wh_per_km, 1)} Wh/km)</span>
              </td>
              <td className={TD}>+{num(c.range_gain_km, 1)} km</td>
              <td className={TD}>{num(c.cost_saving_lkr_per_100km, 1)} LKR/100 km</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  );
}

function Insights({ d }: { d: ConditionsCompare }) {
  const byId = Object.fromEntries(d.conditions.map((c) => [c.condition_id, c]));
  return (
    <Card className="min-h-0 flex-1" title="What The Numbers Show" subtitle="Plain statements of the results. They do not explain causes.">
      <ul className="space-y-1.5 text-[13px] leading-snug">
        {d.insights.map((t, i) => (
          <li key={i} className="flex gap-2">
            <span className="mt-1.5 h-1.5 w-1.5 flex-none rounded-full bg-opt" />
            <span>{t}</span>
          </li>
        ))}
      </ul>
      {d.ranking.length > 1 && (
        <div className="mt-3 border-t border-line pt-2">
          <div className="mb-1 text-xs font-semibold text-muted">Ranking by energy saving</div>
          <ol className="space-y-1">
            {d.ranking.map((id, i) => (
              <li key={id} className="flex items-center gap-2 rounded-lg bg-bg px-2 py-1 text-[13px]">
                <span className="flex h-5 w-5 items-center justify-center rounded-full bg-brand text-[11px] font-bold text-white">{i + 1}</span>
                <span className="font-medium">{shortName(byId[id].label)}</span>
                <span className="ml-auto text-muted">{num(byId[id].saving_pct, 2)} %</span>
              </li>
            ))}
          </ol>
        </div>
      )}
    </Card>
  );
}

/* ---------------------------------------------------------------- */

function Sensitivity({ evId, options }: { evId: string; options: { id: string; label: string }[] }) {
  const [tariff, setTariff] = useState(0);
  const [debounced, setDebounced] = useState(0);
  const [cid, setCid] = useState(options[0]?.id ?? "SLDC_Urban");
  const valid = options.some((o) => o.id === cid) ? cid : options[0]?.id ?? cid;
  useEffect(() => {
    const t = setTimeout(() => setDebounced(tariff), 250);
    return () => clearTimeout(t);
  }, [tariff]);

  const res = useApi(() => api.sensitivity(evId, valid, debounced), [evId, valid, debounced]);

  return (
    <Card
      className="flex-none"
      title="Sensitivity: Electricity Tariff"
      subtitle="How the cost saving changes if the tariff changes."
      right={
        options.length > 1 ? (
          <select className="!py-0.5 text-xs" value={valid} onChange={(e) => setCid(e.target.value)}>
            {options.map((o) => (
              <option key={o.id} value={o.id}>{o.label}</option>
            ))}
          </select>
        ) : undefined
      }
      bodyClass="overflow-hidden"
    >
      <label className="block text-xs">
        <div className="mb-1 flex justify-between">
          <span className="text-muted">Tariff change</span>
          <span className="font-semibold">{signed(tariff, 0)} %</span>
        </div>
        <input type="range" min={-30} max={50} step={5} value={tariff} onChange={(e) => setTariff(Number(e.target.value))} className="w-full" />
      </label>
      {res.error && <ErrorBox message={res.error} />}
      {!res.data && !res.error && <Loading />}
      {res.data && res.data.available && <SensitivityBody s={res.data} />}
    </Card>
  );
}

function SensitivityBody({ s }: { s: SensitivityOK }) {
  const t = s.tariff.selected;
  return (
    <div className="mt-2 space-y-2">
      <div className="grid grid-cols-3 gap-2 text-center">
        <Mini label="Tariff" value={num(t.tariff_lkr_per_kwh, 2)} unit="LKR/kWh" />
        <Mini label="Baseline cost" value={num(t.baseline_cost_lkr_per_100km, 1)} unit="LKR/100 km" tone="base" />
        <Mini label="Optimized cost" value={num(t.optimized_cost_lkr_per_100km, 1)} unit="LKR/100 km" tone="opt" />
      </div>
      <div className="rounded-xl bg-opt/10 p-2 text-center text-xs">
        Cost saving: <b className="text-opt">{num(t.cost_saving_lkr_per_100km, 1)} LKR per 100 km</b> ({s.condition_label})
      </div>
      <p className="text-[11px] leading-snug text-muted">{s.tariff.note}</p>
    </div>
  );
}

function Mini({ label, value, unit, tone }: { label: string; value: string; unit: string; tone?: "base" | "opt" }) {
  const color = tone === "base" ? "text-base" : tone === "opt" ? "text-opt" : "text-ink";
  return (
    <div className="rounded-xl border border-line p-2">
      <div className="text-[11px] text-muted">{label}</div>
      <div className={`text-base font-bold ${color}`}>{value}</div>
      <div className="text-[10px] text-muted">{unit}</div>
    </div>
  );
}