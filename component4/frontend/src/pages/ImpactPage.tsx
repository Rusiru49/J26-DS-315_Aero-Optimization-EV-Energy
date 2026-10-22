import { useMemo } from "react";
import { api } from "../api";
import type { Catalog, ComparisonOK } from "../api";
import { useApi } from "../lib/useApi";
import { condShort, num, prettyParam, shortName, signed, unitLabel } from "../lib/format";
import { Card, Chip, ComingSoon, ErrorBox, Kpi, Loading, TD, TH } from "../components/ui";
import Viewer3D from "../components/Viewer3D";
import PlotlyChart from "../components/PlotlyChart";

/** Function 1: baseline vs the optimized design made for the selected EV and driving condition. */
export default function ImpactPage({
  evId,
  conditionId,
  catalog,
}: {
  evId: string;
  conditionId: string;
  catalog: Catalog;
}) {
  const cmp = useApi(() => api.comparison(evId, conditionId), [evId, conditionId]);
  const ev = catalog.evs.find((e) => e.ev_id === evId);
  const baseCfg = ev?.configurations.find((c) => c.type === "baseline");
  const cond = catalog.conditions.find((c) => c.condition_id === conditionId);

  if (!ev || !baseCfg) return <ErrorBox message="This EV has no baseline design." />;
  if (cmp.error) return <ErrorBox message={cmp.error} />;
  if (!cmp.data) return <Loading />;

  const d = cmp.data;
  const optCfg = ev.configurations.find((c) => c.config_id === d.optimized_config_id);
  if (!d.available || !optCfg) {
    return <ComingSoon title={cond?.label ?? conditionId} message={d.available ? "No optimized design." : d.message} />;
  }

  return (
    <div className="grid h-full grid-cols-12 gap-3">
      <div className="col-span-7 flex min-h-0 flex-col gap-3">
        <div className="flex flex-none flex-wrap items-center gap-2">
          <h2 className="text-base font-semibold">
            {shortName(ev.ev_model)} · {shortName(cond?.label ?? conditionId)}
          </h2>
          <Chip tone="opt">
            Optimized design {optCfg.config_id}
            {optCfg.optimized_for ? ` · made for ${condShort(optCfg.optimized_for)}` : ""}
          </Chip>
        </div>
        <Kpis d={d} />
        <div className="grid min-h-0 flex-1 grid-cols-2 gap-3">
          <SideBySide d={d} />
          <BenchmarkCard d={d} />
        </div>
      </div>

      <div className="col-span-5 flex min-h-0 flex-col gap-3">
        <Card
          className="min-h-0 flex-1"
          bodyClass="overflow-hidden"
          title="3D shape"
          subtitle={`Baseline and ${optCfg.config_id}`}
        >
          <Viewer3D baseline={baseCfg} optimized={optCfg} />
        </Card>
        <ShapeAndAero d={d} />
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------- */

function Kpis({ d }: { d: ComparisonOK }) {
  const s = d.savings;
  const b = d.baseline;
  const o = d.optimized;
  return (
    <div className="grid flex-none grid-cols-3 gap-3">
      <Kpi
        icon="energy"
        title="Energy saving"
        value={num(s.energy_saving_pct, 2)}
        unit="%"
        secondary={`${num(s.energy_saving_wh_per_km, 1)} Wh/km less energy`}
        from={num(b.energy_wh_per_km, 1)}
        to={num(o.energy_wh_per_km, 1)}
        unitFromTo="Wh/km"
      />
      <Kpi
        icon="range"
        title="Range gain"
        value={`+${num(s.range_gain_km, 1)}`}
        unit="km"
        secondary={`${num(s.range_gain_pct, 2)} % more range`}
        from={num(b.range_km, 1)}
        to={num(o.range_km, 1)}
        unitFromTo="km"
      />
      <Kpi
        icon="cost"
        title="Cost saving"
        value={num(s.cost_saving_lkr_per_100km, 1)}
        unit="LKR / 100 km"
        secondary={`${num(s.cost_saving_pct, 2)} % lower running cost`}
        from={num(b.cost_lkr_per_100km, 1)}
        to={num(o.cost_lkr_per_100km, 1)}
        unitFromTo="LKR/100 km"
      />
    </div>
  );
}

function SideBySide({ d }: { d: ComparisonOK }) {
  const s = d.savings;
  const b = d.baseline;
  const o = d.optimized;
  return (
    <Card title="Baseline and optimized side by side" subtitle="Values come from C3. Differences are calculated by C4.">
      <table className="w-full">
        <thead>
          <tr className="border-b border-line">
            <th className={TH}>Measure</th>
            <th className={`${TH} text-base`}>Baseline</th>
            <th className={`${TH} text-opt`}>Optimized</th>
            <th className={TH}>Change</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-line">
          <Row label="Cd" b={num(b.cd, 3)} o={num(o.cd, 3)} change={signed(d.aerodynamics.cd_change, 3)} />
          <Row label="Frontal area (m²)" b={num(b.frontal_area_m2, 2)} o={num(o.frontal_area_m2, 2)} change={signed(d.aerodynamics.frontal_area_change_m2, 2)} />
          <Row label="Energy (Wh/km)" b={num(b.energy_wh_per_km, 1)} o={num(o.energy_wh_per_km, 1)} change={signed(-s.energy_saving_wh_per_km, 1)} />
          <Row label="Range (km)" b={num(b.range_km, 1)} o={num(o.range_km, 1)} change={signed(s.range_gain_km, 1)} />
          <Row label="Cost (LKR/100 km)" b={num(b.cost_lkr_per_100km, 1)} o={num(o.cost_lkr_per_100km, 1)} change={signed(-s.cost_saving_lkr_per_100km, 1)} />
        </tbody>
      </table>
      <p className="mt-3 rounded-xl border border-brand/20 bg-brand/5 p-2.5 text-xs leading-relaxed">
        <b>In short:</b> on {d.condition_label}, the optimized design uses {num(s.energy_saving_wh_per_km, 1)} Wh/km less
        energy ({num(s.energy_saving_pct, 2)} %), drives {num(s.range_gain_km, 1)} km further on a full battery and costs
        LKR {num(s.cost_saving_lkr_per_100km, 1)} less per 100 km.
      </p>
    </Card>
  );
}

/** The baseline car under the WLTP benchmark next to the same baseline car under the selected condition. */
function BenchmarkCard({ d }: { d: ComparisonOK }) {
  const bm = d.benchmark;
  const sel = shortName(d.condition_label);
  const data = useMemo(() => {
    if (!bm) return [];
    return [
      {
        type: "bar",
        x: [shortName(bm.condition_label), sel],
        y: [bm.baseline.energy_wh_per_km, d.baseline.energy_wh_per_km],
        text: [num(bm.baseline.energy_wh_per_km, 1), num(d.baseline.energy_wh_per_km, 1)],
        textposition: "outside",
        cliponaxis: false,
        marker: { color: ["#7a6fc8", "#4a86c5"] },
        hovertemplate: "%{x}: %{y:.1f} Wh/km<extra></extra>",
      },
    ];
  }, [bm, d, sel]);
  const layout = useMemo(
    () => ({
      showlegend: false,
      margin: { t: 22, b: 28 },
      yaxis: { title: { text: "Wh/km" }, rangemode: "tozero" },
    }),
    [],
  );
  if (!bm) {
    return (
      <Card title="Baseline car: WLTP vs this condition">
        <p className="text-sm text-muted">No WLTP result for the baseline car yet.</p>
      </Card>
    );
  }
  const bn = shortName(bm.condition_label);
  const b = d.baseline;
  const diff = bm.difference;
  const rows: [string, string, string, string][] = [
    ["Energy (Wh/km)", num(bm.baseline.energy_wh_per_km, 1), num(b.energy_wh_per_km, 1), signed(diff.baseline_energy_wh_per_km, 1)],
    ["Range (km)", num(bm.baseline.range_km, 1), num(b.range_km, 1), signed(diff.baseline_range_km, 1)],
    ["Cost (LKR/100 km)", num(bm.baseline.cost_lkr_per_100km, 1), num(b.cost_lkr_per_100km, 1), signed(diff.baseline_cost_lkr_per_100km, 1)],
  ];
  return (
    <Card title={`Baseline car: ${bn} vs ${sel}`} subtitle={`${bn} is the standard test cycle.`} bodyClass="flex flex-col">
      <p className="mb-2 flex-none rounded-xl border border-bench/25 bg-bench/5 p-2.5 text-xs leading-relaxed">
        The baseline car uses <b>{num(b.energy_wh_per_km, 1)} Wh/km</b> under {sel} but{" "}
        <b>{num(bm.baseline.energy_wh_per_km, 1)} Wh/km</b> under {bn} ({signed(diff.baseline_energy_pct, 1)} %). So {bn}{" "}
        does not describe this condition.
      </p>
      <table className="w-full flex-none">
        <thead>
          <tr className="border-b border-line">
            <th className={TH}>Baseline</th>
            <th className={`${TH} text-bench`}>{bn}</th>
            <th className={`${TH} text-brand`}>{sel}</th>
            <th className={TH}>Diff.</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-line">
          {rows.map(([label, w, l, df]) => (
            <tr key={label}>
              <td className={TD}>{label}</td>
              <td className={TD}>{w}</td>
              <td className={TD}>{l}</td>
              <td className={TD}>{df}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mt-1 min-h-[110px] flex-1">
        <PlotlyChart data={data} layout={layout} />
      </div>
    </Card>
  );
}

function Row({ label, b, o, change }: { label: string; b: string; o: string; change: string }) {
  return (
    <tr>
      <td className={`${TD} font-medium`}>{label}</td>
      <td className={TD}>{b}</td>
      <td className={TD}>{o}</td>
      <td className={`${TD} text-muted`}>{change}</td>
    </tr>
  );
}

/* ---------------------------------------------------------------- */

function ShapeAndAero({ d }: { d: ComparisonOK }) {
  const a = d.aerodynamics;
  const unc = a.cd_uncertainty_optimized;
  const times = unc ? Math.abs(a.cd_change) / unc : null;
  return (
    <Card
      className="flex-none"
      title="What changed in the shape"
      subtitle={
        unc
          ? `Cd changed by ${signed(a.cd_change, 3)} (${signed(a.cd_change_pct, 1)} %), ${num(times, 1)} × C3's error (±${num(unc, 3)}).`
          : `Cd changed by ${signed(a.cd_change, 3)}.`
      }
      right={
        a.saving_within_uncertainty === null ? undefined : a.saving_within_uncertainty ? (
          <Chip tone="warn">Within C3's error</Chip>
        ) : (
          <Chip tone="ok">Larger than C3's error</Chip>
        )
      }
      bodyClass="overflow-hidden"
    >
      <table className="w-full">
        <thead>
          <tr className="border-b border-line">
            <th className={TH}>Parameter</th>
            <th className={`${TH} text-base`}>Baseline</th>
            <th className={`${TH} text-opt`}>Optimized</th>
            <th className={TH}>Change</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-line">
          {d.shape_changes.map((c) => (
            <tr key={c.name} title={c.description}>
              <td className={TD}>{prettyParam(c.name)}</td>
              <td className={TD}>
                {num(c.baseline, 0)}
                {unitLabel(c.unit)}
              </td>
              <td className={`${TD} font-semibold text-opt`}>
                {num(c.optimized, 0)}
                {unitLabel(c.unit)}
              </td>
              <td className={`${TD} text-muted`}>
                {signed(c.delta, 0)}
                {unitLabel(c.unit)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  );
}