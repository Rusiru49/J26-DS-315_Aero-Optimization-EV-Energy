import type { ReactNode } from "react";

/* ---------- small building blocks used by every page (dashboard sizes) ---------- */

/** A dashboard panel. It fills the space the grid gives it; long content scrolls inside it. */
export function Card({
  title,
  subtitle,
  right,
  children,
  className = "",
  bodyClass = "",
}: {
  title?: ReactNode;
  subtitle?: ReactNode;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClass?: string;
}) {
  return (
    <section
      className={`flex min-h-0 flex-col rounded-2xl border border-line bg-surface p-3 shadow-[0_1px_2px_rgba(22,52,77,0.04)] ${className}`}
    >
      {(title || right) && (
        <header className="mb-2 flex flex-none flex-wrap items-start justify-between gap-2">
          <div className="min-w-0">
            {title && <h3 className="text-sm font-semibold text-ink">{title}</h3>}
            {subtitle && <p className="text-xs leading-snug text-muted">{subtitle}</p>}
          </div>
          {right}
        </header>
      )}
      <div className={`min-h-0 flex-1 overflow-auto ${bodyClass}`}>{children}</div>
    </section>
  );
}

type Tone = "opt" | "base" | "warn" | "soon" | "bench" | "brand" | "ok" | "bad";
const TONES: Record<Tone, string> = {
  opt: "bg-opt/10 text-opt border-opt/30",
  base: "bg-base/10 text-base border-base/30",
  warn: "bg-warn/10 text-[#9a6a14] border-warn/40",
  soon: "bg-soon/30 text-muted border-soon",
  bench: "bg-bench/10 text-bench border-bench/30",
  brand: "bg-brand/10 text-brand border-brand/30",
  ok: "bg-opt/10 text-opt border-opt/30",
  bad: "bg-bad/10 text-bad border-bad/30",
};

export function Chip({ tone = "brand", children }: { tone?: Tone; children: ReactNode }) {
  return (
    <span className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-medium ${TONES[tone]}`}>
      {children}
    </span>
  );
}

export function Loading({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="flex items-center gap-3 rounded-2xl border border-line bg-surface p-4 text-sm text-muted">
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-line border-t-brand" />
      {label}
    </div>
  );
}

export function ErrorBox({ message }: { message: string }) {
  return (
    <div className="rounded-2xl border border-bad/30 bg-bad/5 p-4 text-sm text-bad">
      <b>Something went wrong.</b> {message}
    </div>
  );
}

export function ComingSoon({ title, message }: { title: string; message?: string }) {
  return (
    <div className="flex h-full items-center justify-center rounded-2xl border-2 border-dashed border-soon bg-white/60 p-6 text-center">
      <div>
        <div className="mx-auto mb-2 flex h-11 w-11 items-center justify-center rounded-full bg-soon/40 text-xl">⏳</div>
        <h3 className="text-base font-semibold text-ink">{title} — coming soon</h3>
        <p className="mx-auto mt-1 max-w-md text-sm text-muted">
          {message ?? "Results for this driving condition are not available yet."}
        </p>
      </div>
    </div>
  );
}

/* ---------- the three headline numbers ---------- */

type IconName = "energy" | "range" | "cost";

function Icon({ name }: { name: IconName }) {
  const common = { width: 18, height: 18, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 2, strokeLinecap: "round" as const, strokeLinejoin: "round" as const };
  if (name === "energy")
    return (
      <svg {...common}>
        <path d="M13 2 4 14h7l-1 8 9-12h-7l1-8Z" />
      </svg>
    );
  if (name === "range")
    return (
      <svg {...common}>
        <path d="M12 21a9 9 0 1 1 9-9" />
        <path d="m12 12 5-4" />
        <circle cx="12" cy="12" r="1.2" />
      </svg>
    );
  return (
    <svg {...common}>
      <circle cx="12" cy="12" r="9" />
      <path d="M14.5 9.2c-.5-.8-1.4-1.2-2.5-1.2-1.4 0-2.5.8-2.5 2s1 1.7 2.5 2 2.5.8 2.5 2-1.1 2-2.5 2c-1.1 0-2-.4-2.5-1.2M12 6.5V8m0 8v1.5" />
    </svg>
  );
}

export function Kpi({
  icon,
  title,
  value,
  unit,
  secondary,
  from,
  to,
  unitFromTo,
}: {
  icon: IconName;
  title: string;
  value: string;
  unit: string;
  secondary: string;
  from: string;
  to: string;
  unitFromTo: string;
}) {
  return (
    <div className="rounded-2xl border border-opt/25 bg-gradient-to-br from-white to-opt/5 p-3">
      <div className="flex items-center gap-2 text-opt">
        <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-opt/10">
          <Icon name={icon} />
        </span>
        <span className="text-xs font-semibold text-ink">{title}</span>
      </div>
      <div className="mt-1.5 flex items-baseline gap-1.5">
        <span className="text-3xl font-bold leading-none tracking-tight text-ink">{value}</span>
        <span className="text-sm font-medium text-opt">{unit}</span>
      </div>
      <div className="mt-0.5 text-xs font-medium text-opt">{secondary}</div>
      <div className="mt-2 flex items-center gap-1.5 border-t border-line pt-2 text-xs text-muted">
        <span className="rounded-md bg-base/10 px-1.5 py-0.5 font-medium text-base">{from}</span>
        <span>→</span>
        <span className="rounded-md bg-opt/10 px-1.5 py-0.5 font-medium text-opt">{to}</span>
        <span className="ml-1 text-[11px]">{unitFromTo}</span>
      </div>
    </div>
  );
}

/* ---------- table styles ---------- */
export const TH = "px-2 py-1.5 text-left text-[11px] font-semibold uppercase tracking-wide text-muted";
export const TD = "px-2 py-1.5 text-[13px]";

/** A row of small buttons (tabs / toggles). */
export function Segmented<T extends string>({
  value,
  options,
  onChange,
}: {
  value: T;
  options: [T, string][];
  onChange: (v: T) => void;
}) {
  return (
    <div className="inline-flex rounded-xl border border-line bg-white p-0.5 text-xs">
      {options.map(([k, label]) => (
        <button
          key={k}
          onClick={() => onChange(k)}
          className={`rounded-lg px-2.5 py-1 font-medium transition ${
            value === k ? "bg-brand text-white shadow-sm" : "text-muted hover:text-ink"
          }`}
        >
          {label}
        </button>
      ))}
    </div>
  );
}