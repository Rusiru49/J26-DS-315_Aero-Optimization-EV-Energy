import { api } from "../api";
import type { Catalog } from "../api";
import { useApi } from "../lib/useApi";
import { Card, Chip, ErrorBox, Loading } from "../components/ui";

const ICON = { pass: "✓", warn: "!", fail: "✕" } as const;
const TONE = { pass: "bg-opt/10 text-opt", warn: "bg-warn/15 text-[#9a6a14]", fail: "bg-bad/10 text-bad" } as const;

/** Notes C3 sends with its results. Only the fields below are shown. */
const NOTE_LABELS: [string, string][] = [
  ["energy_definition", "Energy"],
  ["range_definition", "Range"],
  ["includes_regen_braking", "Includes regenerative braking"],
  ["includes_auxiliary_load", "Includes auxiliary load"],
  ["road_gradient", "Road gradient"],
  ["cd_source", "Cd source"],
  ["optimized_cd_method", "Optimized Cd method"],
  ["tariff_lkr_per_kwh", "Tariff (LKR/kWh)"],
  ["charging_efficiency", "Charging efficiency"],
  ["experimentally_validated", "Experimentally validated"],
];

export default function QualityPage({ catalog }: { catalog: Catalog }) {
  const res = useApi(() => api.quality(), []);
  if (res.error) return <ErrorBox message={res.error} />;
  if (!res.data) return <Loading />;
  const q = res.data;

  const show = (v: unknown) =>
    typeof v === "boolean" ? (v ? "Yes" : "No") : String(v ?? "—").replace(/\s*\(dummy\)/gi, "");

  return (
    <div className="flex h-full flex-col gap-3">
      <div className="flex flex-none flex-wrap items-center gap-2">
        <h2 className="text-[16px] font-semibold text-ink">Data Checks Before Any Analysis</h2>
        <Chip tone="ok">✓ {q.counts.pass} passed</Chip>
        {q.counts.warn > 0 && <Chip tone="warn">! {q.counts.warn} warnings</Chip>}
        {q.counts.fail > 0 && <Chip tone="bad">✕ {q.counts.fail} failed</Chip>}
      </div>

      <div className="grid min-h-0 flex-1 grid-cols-12 gap-3">
        <Card className="col-span-8" title="Checks On The Data Received From C3" subtitle="C4 checks every response. Bad data is reported here and never used silently.">
          <ul className="grid grid-cols-2 gap-x-6">
            {q.checks.map((c) => (
              <li key={c.id} className="flex items-start gap-2 border-b border-line py-2">
                <span className={`mt-0.5 flex h-5 w-5 flex-none items-center justify-center rounded-full text-xs font-bold ${TONE[c.status]}`}>
                  {ICON[c.status]}
                </span>
                <div>
                  <div className="text-[13px] font-medium">{c.name}</div>
                  <div className="text-xs leading-snug text-muted">{c.detail}</div>
                </div>
              </li>
            ))}
          </ul>
        </Card>

        <Card className="col-span-4" title="How C3 Defines Its Results" subtitle="Notes sent by C3 together with the numbers.">
          <dl className="text-[13px]">
            {NOTE_LABELS.map(([k, label]) => (
              <div key={k} className="flex justify-between gap-3 border-b border-line py-1.5">
                <dt className="text-muted">{label}</dt>
                <dd className="text-right font-medium">{show(catalog.model_notes[k])}</dd>
              </div>
            ))}
          </dl>
        </Card>
      </div>
    </div>
  );
}