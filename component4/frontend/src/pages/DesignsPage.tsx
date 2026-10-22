import { useMemo, useState } from "react";
import { api } from "../api";
import type { Catalog, DesignRow, RankBy } from "../api";
import { useApi } from "../lib/useApi";
import { condShort, num, prettyParam, shortName, signed, unitLabel } from "../lib/format";
import { Card, Chip, ErrorBox, Loading, Segmented, TD, TH } from "../components/ui";
import PlotlyChart from "../components/PlotlyChart";
import Viewer3D from "../components/Viewer3D";

const RANKS: [RankBy, string][] = [
  ["overall", "Overall"],
  ["energy", "Energy saving"],
  ["range", "Range gain"],
  ["cost", "Cost saving"],
];

/** Best design: the optimized designs of one EV, ranked on energy, range and cost. */
export default function DesignsPage({ evId, catalog }: { evId: string; catalog: Catalog }) {
  const [rankBy, setRankBy] = useState<RankBy>("overall");
  const [picked, setPicked] = useState<string | null>(null);
  const res = useApi(() => api.designs(evId, rankBy), [evId, rankBy]);
  const ev = catalog.evs.find((e) => e.ev_id === evId);
  const baseCfg = ev?.configurations.find((c) => c.type === "baseline");

  if (res.error) return <ErrorBox message={res.error} />;
  if (!res.data || !ev || !baseCfg) return <Loading />;
  const d = res.data;
  if (d.designs.length === 0 || !d.best_config_id) return <ErrorBox message="No optimized design has results yet." />;

  const best = d.designs[0];
  const shown = d.designs.find((x) => x.config_id === (picked ?? best.config_id)) ?? best;
  const shownCfg = ev.configurations.find((c) => c.config_id === shown.config_id);

  return (
    <div className="grid h-full grid-cols-12 gap-3">
      <div className="col-span-7 flex min-h-0 flex-col gap-3">
        <div className="flex flex-none flex-wrap items-center gap-2">
          <h2 className="text-base font-semibold">Best design · {shortName(ev.ev_model)}</h2>
          <span className="ml-2 text-xs font-medium text-muted">Rank designs by</span>
          <Segmented
            value={rankBy}
            options={RANKS}
            onChange={(k) => {
              setRankBy(k);
              setPicked(null);
            }}
          />
        </div>

        <Hero best={best} d={d} />

        <Card className="flex-none" title="All optimized designs, ranked" subtitle={d.rule}>
          <table className="w-full">
            <thead>
              <tr className="border-b border-line">
                <th className={TH}>Rank</th>
                <th className={TH}>Design</th>
                <th className={TH}>Made for</th>
                <th className={TH}>Cd</th>
                <th className={TH}>Energy saving</th>
                <th className={TH}>Range gain</th>
                <th className={TH}>Cost saving</th>
                <th className={TH}>Overall</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-line">
              {d.designs.map((x) => (
                <tr key={x.config_id} className={x.rank === 1 ? "bg-opt/10" : ""}>
                  <td className={`${TD} font-bold`}>{x.rank === 1 ? "★ 1" : x.rank}</td>
                  <td className={`${TD} font-semibold`}>{x.config_id}</td>
                  <td className={TD}>{x.optimized_for ? condShort(x.optimized_for) : "—"}</td>
                  <td className={TD}>{num(x.cd, 3)}</td>
                  <td className={TD}>{num(x.mean_energy_saving_pct, 2)} %</td>
                  <td className={TD}>{num(x.mean_range_gain_pct, 2)} %</td>
                  <td className={TD}>{num(x.mean_cost_saving_pct, 2)} %</td>
                  <td className={`${TD} font-semibold`}>{num(x.overall_score_pct, 2)} %</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>

        <SavingsChart designs={d.designs} />
      </div>

      <div className="col-span-5 flex min-h-0 flex-col gap-3">
        <Card
          className="min-h-0 flex-1"
          bodyClass="overflow-hidden"
          title="3D view: baseline and a chosen design"
          right={
            <div className="flex flex-wrap gap-1">
              {d.designs.map((x) => (
                <button
                  key={x.config_id}
                  onClick={() => setPicked(x.config_id)}
                  className={`rounded-full border px-2.5 py-0.5 text-[11px] font-medium transition ${
                    x.config_id === shown.config_id
                      ? "border-opt bg-opt text-white"
                      : "border-line bg-white text-ink hover:border-opt/50"
                  }`}
                >
                  {x.config_id}
                  {x.rank === 1 ? " ★" : ""}
                </button>
              ))}
            </div>
          }
        >
          {shownCfg && <Viewer3D baseline={baseCfg} optimized={shownCfg} />}
        </Card>
        <ShapeTable base={baseCfg.shape_parameters} design={shown} />
      </div>
    </div>
  );
}

function Hero({ best, d }: { best: DesignRow; d: { tie_for_first: boolean; rank_by: RankBy } }) {
  const tiles: [string, string][] = [
    ["Energy saving", `${num(best.mean_energy_saving_pct, 2)} %`],
    ["Range gain", `+${num(best.mean_range_gain_pct, 2)} %`],
    ["Cost saving", `${num(best.mean_cost_saving_pct, 2)} %`],
    ["Overall score", `${num(best.overall_score_pct, 2)} %`],
  ];
  return (
    <div className="flex-none rounded-2xl border border-opt/30 bg-gradient-to-br from-white to-opt/10 p-3">
      <div className="flex flex-wrap items-center gap-2">
        <Chip tone="ok">★ Most promising design</Chip>
        <Chip tone="brand">ranked by {RANKS.find(([k]) => k === d.rank_by)?.[1]}</Chip>
        {d.tie_for_first && <Chip tone="warn">Two designs share first place</Chip>}
        <span className="text-lg font-bold">{best.config_id}</span>
        {best.optimized_for && <span className="text-xs text-muted">made for {condShort(best.optimized_for)}</span>}
      </div>
      <div className="mt-2 grid grid-cols-4 gap-2">
        {tiles.map(([t, v]) => (
          <div key={t} className="rounded-xl border border-line bg-white px-3 py-1.5">
            <div className="text-[11px] text-muted">{t}</div>
            <div className="text-xl font-bold text-ink">{v}</div>
          </div>
        ))}
      </div>
      <p className="mt-2 text-xs text-muted">
        Cd {num(best.cd, 3)} ({signed(best.cd_change, 3)} from the baseline) · frontal area {num(best.frontal_area_m2, 2)} m².
        The design an engineer would pick first, based on C3's results.
      </p>
    </div>
  );
}

function SavingsChart({ designs }: { designs: DesignRow[] }) {
  const data = useMemo(
    () => [
      { name: "Energy saving (%)", y: designs.map((x) => x.mean_energy_saving_pct), color: "#19a58e" },
      { name: "Range gain (%)", y: designs.map((x) => x.mean_range_gain_pct), color: "#4a86c5" },
      { name: "Cost saving (%)", y: designs.map((x) => x.mean_cost_saving_pct), color: "#d9962b" },
    ].map((s) => ({
      type: "bar",
      name: s.name,
      x: designs.map((x) => `${x.config_id}${x.optimized_for ? ` (${condShort(x.optimized_for)})` : ""}`),
      y: s.y,
      marker: { color: s.color },
      text: s.y.map((v) => v.toFixed(1)),
      textposition: "outside",
      cliponaxis: false,
    })),
    [designs],
  );
  const layout = useMemo(() => ({ barmode: "group", yaxis: { title: { text: "%" }, rangemode: "tozero" } }), []);
  return (
    <Card className="min-h-0 flex-1" title="Energy, range and cost for each design" subtitle="Each design against the baseline, under the condition it was made for.">
      <PlotlyChart data={data} layout={layout} />
    </Card>
  );
}

function ShapeTable({ base, design }: { base: Record<string, number>; design: DesignRow }) {
  const names = Object.keys(base);
  return (
    <Card className="flex-none" title={`Shape parameters: baseline vs ${design.config_id}`} bodyClass="overflow-hidden">
      <table className="w-full">
        <thead>
          <tr className="border-b border-line">
            <th className={TH}>Parameter</th>
            <th className={`${TH} text-base`}>Baseline</th>
            <th className={`${TH} text-opt`}>{design.config_id}</th>
            <th className={TH}>Change</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-line">
          {names.map((n) => {
            const unit = n.endsWith("_mm") ? "mm" : "deg";
            const o = design.shape_parameters[n];
            return (
              <tr key={n}>
                <td className={TD}>{prettyParam(n)}</td>
                <td className={TD}>{num(base[n], 0)}{unitLabel(unit)}</td>
                <td className={TD}>{num(o, 0)}{unitLabel(unit)}</td>
                <td className={TD}>{signed(o - base[n], 0)}{unitLabel(unit)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </Card>
  );
}