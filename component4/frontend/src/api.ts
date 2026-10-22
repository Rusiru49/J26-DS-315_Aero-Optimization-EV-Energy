// Everything the frontend knows about the C4 backend: the types of the answers and one function per route.

export const API_BASE: string =
  (import.meta.env.VITE_API_URL as string | undefined) ?? "http://127.0.0.1:8000";

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

type Params = Record<string, string | number | null | undefined>;

async function get<T>(path: string, params: Params = {}): Promise<T> {
  const url = new URL(API_BASE + path);
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null) url.searchParams.set(k, String(v));
  }
  let res: Response;
  try {
    res = await fetch(url);
  } catch {
    throw new ApiError(
      `The C4 backend is not reachable at ${API_BASE}. Start it with: uvicorn app.main:app --port 8000`,
      0,
    );
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = await res.json();
      detail = j.detail ?? detail;
    } catch {
      /* keep statusText */
    }
    throw new ApiError(detail, res.status);
  }
  return (await res.json()) as T;
}

// ------------------------------------------------------------------ types
export interface Meta {
  predicted_only: boolean;
  source_mode: string;
  loaded_at: string | null;
  load_seconds: number | null;
  package_version: string;
}

export type ShapeParams = Record<string, number>;

export interface Config {
  config_id: string;
  type: "baseline" | "optimized";
  optimized_from: string | null;
  optimized_for: string | null;
  frontal_area_m2: number;
  cd: number;
  cd_uncertainty: number | null;
  shape_parameters: ShapeParams;
  geometry_file: string | null;
}

export interface EV {
  ev_id: string;
  ev_model: string;
  mass_kg: number;
  usable_battery_kwh: number;
  configurations: Config[];
}

export interface Condition {
  condition_id: string;
  group: string;
  label: string;
  avg_speed_kmh: number | null;
  max_speed_kmh: number | null;
  idle_fraction: number | null;
  distance_km: number | null;
  available: boolean;
  status: "available" | "coming_soon";
}

export interface Catalog {
  meta: Meta;
  evs: EV[];
  conditions: Condition[];
  model_notes: Record<string, unknown>;
}

export interface Side {
  cd: number;
  cd_uncertainty: number | null;
  frontal_area_m2: number;
  energy_wh_per_km: number;
  range_km: number;
  cost_lkr_per_100km: number;
  geometry_file: string | null;
  shape_parameters: ShapeParams;
}

export interface Savings {
  energy_saving_wh_per_km: number;
  energy_saving_pct: number;
  range_gain_km: number;
  range_gain_pct: number;
  cost_saving_lkr_per_100km: number;
  cost_saving_pct: number;
}

export interface ShapeChange {
  name: string;
  unit: string;
  baseline: number;
  optimized: number;
  delta: number;
  min: number;
  max: number;
  description: string;
}

export interface BenchmarkBlock {
  condition_id: string;
  condition_label: string;
  baseline: { energy_wh_per_km: number; range_km: number; cost_lkr_per_100km: number };
  difference: {
    baseline_energy_wh_per_km: number;
    baseline_energy_pct: number;
    baseline_range_km: number;
    baseline_cost_lkr_per_100km: number;
  };
}

export interface ComparisonOK {
  available: true;
  ev_id: string;
  ev_model: string;
  condition_id: string;
  condition_label: string;
  baseline_config_id: string;
  optimized_config_id: string;
  optimized_for: string | null;
  baseline: Side;
  optimized: Side;
  savings: Savings;
  benchmark: BenchmarkBlock | null;
  aerodynamics: {
    cd_change: number;
    cd_change_pct: number;
    frontal_area_change_m2: number;
    cd_uncertainty_optimized: number | null;
    saving_within_uncertainty: boolean | null;
    uncertainty_note: string | null;
  };
  shape_changes: ShapeChange[];
  caution: string;
  meta: Meta;
}

export interface ComparisonSoon {
  available: false;
  status: "coming_soon";
  optimized_config_id: string;
  condition_id: string;
  condition_label: string;
  message: string;
}

export type Comparison = ComparisonOK | ComparisonSoon;

export interface ConditionRow {
  condition_id: string;
  label: string;
  avg_speed_kmh: number | null;
  idle_fraction: number | null;
  distance_km: number | null;
  design_id: string;
  energy_baseline_wh_per_km: number;
  energy_optimized_wh_per_km: number;
  benchmark_condition_id: string | null;
  benchmark_energy_wh_per_km: number | null;
  baseline_vs_benchmark_pct: number | null;
  saving_pct: number;
  energy_saving_wh_per_km: number;
  range_gain_km: number;
  range_gain_pct: number;
  cost_saving_lkr_per_100km: number;
  cost_saving_pct: number;
}

export interface ConditionsCompare {
  conditions: ConditionRow[];
  coming_soon: { condition_id: string; label: string }[];
  ranking: string[];
  insights: string[];
  meta: Meta;
}

export interface TariffCase {
  tariff_change_pct: number;
  tariff_lkr_per_kwh: number | null;
  baseline_cost_lkr_per_100km: number;
  optimized_cost_lkr_per_100km: number;
  cost_saving_lkr_per_100km: number;
}

export interface SensitivityOK {
  available: true;
  condition_label: string;
  cd_check: {
    baseline_cd: number;
    optimized_cd: number;
    cd_change: number;
    optimized_cd_uncertainty: number | null;
    change_in_uncertainties: number | null;
    within_uncertainty: boolean | null;
  };
  tariff: { selected: TariffCase; sweep: TariffCase[]; note: string };
}
export interface NotAvailable {
  available: false;
  status: "coming_soon";
  message: string;
}
export type Sensitivity = SensitivityOK | NotAvailable;

export interface QualityCheck {
  id: string;
  name: string;
  status: "pass" | "warn" | "fail";
  detail: string;
}
export interface Quality {
  status: "pass" | "warn" | "fail";
  counts: { pass: number; warn: number; fail: number };
  checks: QualityCheck[];
}

export interface Desc {
  n: number;
  mean?: number | null;
  std?: number | null;
  min?: number | null;
  median?: number | null;
  max?: number | null;
}
export interface TTest {
  n_pairs: number;
  df: number;
  mean_before: number | null;
  mean_after: number | null;
  mean_difference: number | null;
  sd_difference: number | null;
  t_statistic: number | null;
  p_value: number | null;
  ci_level: number;
  ci_low: number | null;
  ci_high: number | null;
  cohens_dz: number | null;
  significant: boolean;
  alpha: number;
  wilcoxon: { statistic: number | null; p_value: number | null } | null;
  note: string | null;
}
export interface Corr {
  n: number;
  pearson_r: number | null;
  pearson_p: number | null;
  spearman_rho: number | null;
  spearman_p: number | null;
  note: string;
}
export interface Reg {
  outcome: string;
  predictors: string[];
  n: number;
  r_squared: number | null;
  adj_r_squared: number | null;
  f_p_value: number | null;
  coefficients: { name: string; coef: number | null; std_err: number | null; p_value: number | null }[];
  warnings: string[];
  note: string;
}
export interface Tukey {
  group1: string;
  group2: string;
  mean_diff: number | null;
  p_adj: number | null;
  lower: number | null;
  upper: number | null;
  reject: boolean;
}
export interface Anova {
  groups: Record<string, Desc>;
  df_between: number;
  df_within: number;
  f_statistic: number | null;
  p_value: number | null;
  eta_squared: number | null;
  significant: boolean;
  alpha: number;
  tukey_hsd: Tukey[];
}

export interface LiveCard {
  id: "paired_t" | "anova" | "correlation" | "regression";
  name: string;
  status: "ready" | "locked";
  requires: string;
  have: string;
  reason: string | null;
  result: unknown;
}
export interface LiveStats {
  n_pairs: number;
  conditions_with_data: string[];
  descriptive: { energy_saving_pct: Desc };
  tests: LiveCard[];
}
export interface SampleStats {
  design: string;
  paired_t: TTest;
  anova: Anova;
  correlation: Corr;
  regression: Reg;
  by_condition: { condition: string; mean_saving_pct: number | null; avg_speed_kmh: number }[];
}

export interface FactorsOK {
  available: true;
  n_pairs: number;
  n_conditions: number;
  n_designs: number;
  variation: { condition: number; design: number; rest: number } | null;
  variation_note: string | null;
  by_condition: {
    condition_id: string;
    label: string;
    avg_speed_kmh: number | null;
    idle_fraction: number | null;
    distance_km: number | null;
    mean_saving_pct: number;
    n: number;
  }[];
  by_design: {
    config_id: string;
    optimized_for: string | null;
    cda_reduction_pct: number;
    cd_reduction_pct: number;
    mean_saving_pct: number;
    n: number;
  }[];
  vehicle_factors: { factor: string; tested: boolean; r: number | null; n_evs: number; reason: string | null }[];
  conditions_move_together: boolean;
  explain: ExplainItem[];
  explain_later: string;
  insights: string[];
}
export interface ExplainItem {
  group: string;
  factor: string;
  status: "in_data" | "not_separable" | "needs_data" | "fixed" | "cost_only";
  text: string;
}
export type Factors = FactorsOK | { available: false; message: string };

export type RankBy = "overall" | "energy" | "range" | "cost";
export interface DesignCondition {
  condition_id: string;
  label: string;
  energy_saving_pct: number;
  range_gain_pct: number;
  cost_saving_pct: number;
  energy_saving_wh_per_km: number;
  range_gain_km: number;
  cost_saving_lkr_per_100km: number;
  overall_pct: number;
}
export interface DesignRow {
  rank: number;
  config_id: string;
  optimized_for: string | null;
  optimized_for_label: string | null;
  cd: number;
  cd_change: number;
  cd_uncertainty: number | null;
  frontal_area_m2: number;
  shape_parameters: ShapeParams;
  geometry_file: string | null;
  n_conditions: number;
  mean_energy_saving_pct: number;
  mean_range_gain_pct: number;
  mean_cost_saving_pct: number;
  overall_score_pct: number;
  per_condition: DesignCondition[];
}
export interface Designs {
  ev_id: string;
  baseline_config_id: string;
  rank_by: RankBy;
  rule: string;
  conditions_used: string[];
  designs: DesignRow[];
  best_config_id: string | null;
  tie_for_first: boolean;
  best_by_condition: { condition_id: string; label: string; best_config_id: string; value_pct: number }[];
}

export interface Health {
  status: string;
  c3_mode: string;
  data_loaded: boolean;
  last_error: string | null;
}

// ----------------------------------------------------------------- routes
export const api = {
  health: () => get<Health>("/api/health"),
  catalog: () => get<Catalog>("/api/catalog"),
  // Each condition uses the optimized design that was made for it (the backend picks it).
  comparison: (evId: string, conditionId: string) =>
    get<Comparison>("/api/comparison", { ev_id: evId, condition_id: conditionId }),
  // conditionIds = the conditions the user ticked (empty list = all that have results)
  conditions: (evId: string, conditionIds: string[]) =>
    get<ConditionsCompare>("/api/conditions/compare", {
      ev_id: evId,
      conditions: conditionIds.length ? conditionIds.join(",") : null,
    }),
  sensitivity: (evId: string, conditionId: string, tariffChangePct: number) =>
    get<Sensitivity>("/api/sensitivity", {
      ev_id: evId,
      condition_id: conditionId,
      tariff_change_pct: tariffChangePct,
    }),
  factors: () => get<Factors>("/api/factors"),
  designs: (evId: string, rankBy: RankBy = "overall") =>
    get<Designs>("/api/designs", { ev_id: evId, rank_by: rankBy }),
  quality: () => get<Quality>("/api/data-quality"),
  statsLive: () => get<LiveStats>("/api/statistics/live"),
  statsSample: () => get<SampleStats>("/api/statistics/sample"),
};