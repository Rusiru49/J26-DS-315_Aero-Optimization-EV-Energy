/** Small helpers to show numbers and names the same way everywhere. */

export const num = (x: number | null | undefined, digits = 1): string =>
  x === null || x === undefined || !Number.isFinite(x)
    ? "—"
    : x.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });

export const signed = (x: number | null | undefined, digits = 1): string => {
  if (x === null || x === undefined || !Number.isFinite(x)) return "—";
  const sign = x > 0 ? "+" : x < 0 ? "−" : "";
  return sign + num(Math.abs(x), digits);
};

export const pval = (p: number | null | undefined): string => {
  if (p === null || p === undefined) return "—";
  return p < 0.001 ? "< 0.001" : p.toFixed(3);
};

/** "Expressway (Highway)" -> "Expressway", "Sedan (dummy, ...)" -> "Sedan" */
export const shortName = (s: string): string => s.split(" (")[0];

/** "SLDC_Urban" -> "Urban", "SLDC_Suburban" -> "Sub Urban", "WLTP_Urban" -> "WLTP" */
export const condShort = (id: string): string =>
  id.replace(/^SLDC_/, "").replace("Suburban", "Sub Urban").replace(/^WLTP_.*/, "WLTP");

/** "windshield_angle_deg" -> "Windshield angle" */
export const prettyParam = (name: string): string => {
  const t = name.replace(/_(deg|mm)$/, "").replace(/_/g, " ");
  return t.charAt(0).toUpperCase() + t.slice(1);
};

export const unitLabel = (unit: string): string => (unit === "deg" ? "°" : ` ${unit}`);