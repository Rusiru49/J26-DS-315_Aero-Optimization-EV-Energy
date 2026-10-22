/** Small line icons used in the header, tabs and condition buttons. */
export type IconName =
  | "impact" | "compare" | "trophy" | "stats" | "shield" | "user" | "check"
  | "urban" | "suburban" | "intercity" | "expressway" | "rural"
  | "layers" | "flask" | "play";

const PATHS: Record<IconName, string[]> = {
  impact: ["M13 2 4 14h7l-1 8 9-12h-7l1-8Z"],
  compare: ["M3 20h18", "M6 20v-8", "M12 20V5", "M18 20v-5"],
  trophy: ["M8 21h8", "M12 17v4", "M7 4h10v5a5 5 0 0 1-10 0V4Z", "M17 5h3v2a3 3 0 0 1-3 3", "M7 5H4v2a3 3 0 0 0 3 3"],
  stats: ["M3 3v18h18", "m7 15 4-5 3 3 5-7"],
  shield: ["M12 3 4 6v6c0 5 3.5 8 8 9 4.5-1 8-4 8-9V6l-8-3Z", "m9 12 2 2 4-4"],
  user: ["M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8Z", "M4 21c0-4 3.5-6 8-6s8 2 8 6"],
  check: ["m5 12 4.5 4.5L19 7"],
  urban: ["M4 21V9l6-3v15", "M10 21V11h10v10", "M3 21h18", "M13 14h1M16 14h1M13 17h1M16 17h1M7 11h1M7 14h1M7 17h1"],
  suburban: ["M2 13 8 8l6 5v8H2v-8Z", "M14 21v-7l4-3.5 4 3.5v7", "M6 21v-4h4v4"],
  intercity: ["M6 19a2 2 0 1 0 0-4 2 2 0 0 0 0 4Z", "M18 9a2 2 0 1 0 0-4 2 2 0 0 0 0 4Z", "M8 17h6a3 3 0 0 0 0-6h-4a3 3 0 0 1 0-6h6"],
  expressway: ["M9 3 4 21", "M15 3l5 18", "M12 4v3", "M12 10.5v3", "M12 17v3"],
  rural: ["M12 21v-6", "M12 3l5 7h-3l4 6H6l4-6H7l5-7Z"],
  layers: ["m12 3 9 5-9 5-9-5 9-5Z", "m3 13 9 5 9-5", "m3 17.5 9 5 9-5"],
  flask: ["M9 3h6", "M10 3v6L4.5 19a2 2 0 0 0 1.8 3h11.4a2 2 0 0 0 1.8-3L14 9V3", "M7.5 15h9"],
  play: ["M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18Z", "m10 8.5 5.5 3.5-5.5 3.5v-7Z"],
};

export function Icon({ name, size = 18, className = "" }: { name: IconName; size?: number; className?: string }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.9}
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden
    >
      {PATHS[name].map((d) => (
        <path key={d} d={d} />
      ))}
    </svg>
  );
}

/** Which icon belongs to which driving condition. */
export function conditionIcon(id: string): IconName {
  if (id.endsWith("Suburban")) return "suburban";
  if (id.endsWith("Intercity")) return "intercity";
  if (id.endsWith("Expressway")) return "expressway";
  if (id.endsWith("Rural")) return "rural";
  return "urban";
}