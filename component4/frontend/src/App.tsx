import { useEffect, useState } from "react";
import { api } from "./api";
import { useApi } from "./lib/useApi";
import { shortName } from "./lib/format";
import { Chip, Loading } from "./components/ui";
import { Icon, conditionIcon } from "./components/icons";
import type { IconName } from "./components/icons";
import ImpactPage from "./pages/ImpactPage";
import ConditionsPage from "./pages/ConditionsPage";
import DesignsPage from "./pages/DesignsPage";
import StatisticsPage from "./pages/StatisticsPage";
import QualityPage from "./pages/QualityPage";

type Page = "impact" | "conditions" | "designs" | "stats" | "quality";
const PAGES: [Page, string, IconName, string][] = [
  ["impact", "Impact", "impact", "Baseline against the optimized design, for one driving condition"],
  ["conditions", "Compare Conditions", "compare", "Put the results of several driving conditions next to each other"],
  ["designs", "Best Design", "trophy", "The optimized designs ranked on energy, range and cost"],
  ["stats", "Statistics", "stats", "Which factors go with the saving, and the tests behind it"],
  ["quality", "Data Quality", "shield", "Checks run on everything C3 sent"],
];

function Logo() {
  return (
    <div className="flex items-center gap-2.5">
      <svg width="34" height="34" viewBox="0 0 40 40" aria-hidden>
        <defs>
          <linearGradient id="lg" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="#1d7fa6" />
            <stop offset="1" stopColor="#19a58e" />
          </linearGradient>
        </defs>
        <rect width="40" height="40" rx="11" fill="url(#lg)" />
        <path d="M7 24c3-8 9-11 16-11 4 0 7 1 10 4l-2 2-6 1-3 4H12l-3 3-2-3Z" fill="#fff" opacity=".95" />
        <path d="M6 30c5-2 11-2 16-1M10 34c4-1 9-1 13 0" stroke="#fff" strokeWidth="1.6" strokeLinecap="round" opacity=".7" fill="none" />
      </svg>
      <div className="leading-tight">
        <div className="text-[17px] font-bold tracking-tight text-ink">
          AeroEV<span className="text-brand">Lanka</span>
        </div>
        <div className="text-[10px] text-muted">Aerodynamic Optimization Of EVs · Sri Lanka</div>
      </div>
    </div>
  );
}

/** Placeholder for the signed-in user. The sign-in itself is added later. */
function ProfileButton() {
  return (
    <button
      type="button"
      title="Your profile (coming soon)"
      aria-label="Your profile (coming soon)"
      className="flex h-9 w-9 flex-none items-center justify-center rounded-full border border-white bg-gradient-to-br from-brand to-opt text-white shadow-[0_1px_4px_rgba(22,52,77,0.25)] ring-2 ring-brand/15 transition hover:ring-brand/40"
    >
      <Icon name="user" size={19} />
    </button>
  );
}

export default function App() {
  const catalog = useApi(() => api.catalog(), []);
  const health = useApi(() => api.health(), []);
  const [page, setPage] = useState<Page>("impact");
  const [evId, setEvId] = useState("");
  const [conditionId, setConditionId] = useState("SLDC_Urban"); // Impact page: one condition
  const [compareIds, setCompareIds] = useState<string[]>(["SLDC_Urban", "SLDC_Expressway"]); // Compare page: any number

  useEffect(() => {
    if (catalog.data && !evId) setEvId(catalog.data.evs[0]?.ev_id ?? "");
  }, [catalog.data, evId]);

  const showEv = page === "impact" || page === "conditions" || page === "designs";
  const showConditionBar = page === "impact" || page === "conditions";
  const multi = page === "conditions";
  // WLTP is the standard cycle, not a choice: it is shown next to the results of the chosen condition(s).
  const choices = catalog.data?.conditions.filter((c) => c.group !== "standard") ?? [];
  const pageInfo = PAGES.find((p) => p[0] === page)!;

  function pick(id: string) {
    if (!multi) return setConditionId(id);
    setCompareIds((prev) => (prev.includes(id) ? (prev.length > 1 ? prev.filter((x) => x !== id) : prev) : [...prev, id]));
  }

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <header className="flex-none border-b border-line bg-gradient-to-r from-[#dff0f7] via-white to-[#dff5ee] shadow-[0_1px_6px_rgba(22,52,77,0.06)]">
        <div className="flex items-center justify-between gap-4 px-4 py-2">
          <Logo />
          <nav className="flex gap-1 rounded-2xl border border-line/80 bg-white/70 p-1">
            {PAGES.map(([k, label, icon]) => (
              <button
                key={k}
                onClick={() => setPage(k)}
                className={`flex items-center gap-1.5 rounded-xl px-3 py-1.5 text-[13px] font-semibold transition ${
                  page === k ? "bg-brand text-white shadow-sm" : "text-muted hover:bg-white hover:text-ink"
                }`}
              >
                <Icon name={icon} size={16} />
                {label}
              </button>
            ))}
          </nav>
          <div className="flex items-center gap-3">
            <div className="hidden text-right leading-tight xl:block">
              <div className="text-xs font-semibold text-ink">Your Profile</div>
              <div className="text-[10px] text-muted">Coming soon</div>
            </div>
            <ProfileButton />
          </div>
        </div>

        <div className="flex min-h-[40px] flex-wrap items-center gap-x-5 gap-y-1 border-t border-line/70 bg-white/55 px-4 py-1.5">
          {catalog.data && evId && showEv ? (
            <label className="flex items-center gap-2 text-xs">
              <span className="font-medium text-muted">Electric Vehicle</span>
              <select value={evId} onChange={(e) => setEvId(e.target.value)} className="!py-0.5 text-xs font-medium">
                {catalog.data.evs.map((ev) => (
                  <option key={ev.ev_id} value={ev.ev_id}>
                    {ev.ev_id} · {shortName(ev.ev_model)}
                  </option>
                ))}
              </select>
            </label>
          ) : (
            <span className="text-xs text-muted">{pageInfo[3]}</span>
          )}

          {catalog.data && evId && showConditionBar && (
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="text-xs font-medium text-muted">
                {multi ? "Driving Conditions To Compare" : "Driving Condition"}
              </span>
              {choices.map((c) => {
                const selected = multi ? compareIds.includes(c.condition_id) : c.condition_id === conditionId;
                const disabled = !c.available;
                return (
                  <button
                    key={c.condition_id}
                    disabled={disabled}
                    onClick={() => pick(c.condition_id)}
                    title={disabled ? "Results for this condition are not available yet" : ""}
                    className={`flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-medium transition ${
                      disabled
                        ? "cursor-not-allowed border-dashed border-soon bg-white/40 text-muted/60"
                        : selected
                          ? "border-brand bg-brand text-white shadow-sm"
                          : "border-line bg-white text-ink hover:border-brand/50"
                    }`}
                  >
                    <Icon name={conditionIcon(c.condition_id)} size={15} />
                    {shortName(c.label)}
                    {multi && !disabled && selected && <Icon name="check" size={13} />}
                    {disabled && <span className="ml-0.5 text-[9px] font-semibold">Soon</span>}
                  </button>
                );
              })}
            </div>
          )}

          <div className="ml-auto flex items-center gap-2">
            {health.data && (
              <Chip tone="ok">
                <i className="inline-block h-1.5 w-1.5 rounded-full bg-opt" /> C3 Connected ({health.data.c3_mode})
              </Chip>
            )}
            {catalog.data?.meta.predicted_only && <Chip tone="warn">Predicted results, not experimentally validated</Chip>}
          </div>
        </div>
      </header>

      <main className="min-h-0 flex-1 p-3">
        {catalog.error && (
          <div className="rounded-2xl border border-bad/30 bg-bad/5 p-5">
            <h2 className="text-base font-semibold text-bad">Cannot Reach The Backend</h2>
            <p className="mt-1 text-sm">{catalog.error}</p>
            <ol className="mt-2 list-decimal space-y-1 pl-5 text-sm text-muted">
              <li>Start the mock C3 service (port 8001).</li>
              <li>Start the C4 backend (port 8000).</li>
              <li>Reload this page.</li>
            </ol>
          </div>
        )}
        {!catalog.data && !catalog.error && <Loading label="Connecting to the C4 backend…" />}

        {catalog.data && evId && (
          <>
            {page === "impact" && <ImpactPage evId={evId} conditionId={conditionId} catalog={catalog.data} />}
            {page === "conditions" && <ConditionsPage evId={evId} conditionIds={compareIds} catalog={catalog.data} />}
            {page === "designs" && <DesignsPage evId={evId} catalog={catalog.data} />}
            {page === "stats" && <StatisticsPage />}
            {page === "quality" && <QualityPage catalog={catalog.data} />}
          </>
        )}
      </main>
    </div>
  );
}