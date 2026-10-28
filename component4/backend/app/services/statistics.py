"""Statistics engine: descriptive, paired t-test, ANOVA + Tukey HSD, correlation, regression.

Design rules
- The functions are pure (numbers in, numbers out) so they can be unit-tested against
  hand calculations.
- The live endpoint runs a test as soon as the method's own mathematical minimum is met
  (for example ANOVA needs at least 2 groups with at least 2 values each). Otherwise the test
  is reported as "locked" with the reason (what is needed, what we have).
- Every result reports n (and df), so the sample size is always visible.
"""
import math
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.api import OLS, add_constant
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from ..config import Settings
from .store import CONDITION_LABELS, DataStore


def _f(x) -> Optional[float]:
    """JSON-safe float."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


# ----------------------------------------------------------------- pure tests
def descriptive(values) -> dict:
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean": _f(a.mean()),
            "std": _f(a.std(ddof=1)) if a.size > 1 else None,
            "min": _f(a.min()), "median": _f(np.median(a)), "max": _f(a.max())}


def paired_ttest(before, after, alpha: float = 0.05) -> dict:
    """Paired t-test on (before - after). Positive mean_diff = 'after' is lower (a saving)."""
    b, o = np.asarray(before, float), np.asarray(after, float)
    if b.shape != o.shape or b.size < 2:
        raise ValueError("Paired t-test needs two equal-length samples with at least 2 pairs.")
    d = b - o
    n = int(d.size)
    mean_d, sd_d = float(d.mean()), float(d.std(ddof=1))
    df = n - 1
    if sd_d == 0:
        t_stat, p_val, ci = None, (0.0 if mean_d != 0 else 1.0), (mean_d, mean_d)
        note = "All differences are identical, so the t statistic is undefined."
    else:
        se = sd_d / math.sqrt(n)
        t_stat = mean_d / se
        p_val = float(2 * stats.t.sf(abs(t_stat), df))
        h = stats.t.ppf(1 - alpha / 2, df) * se
        ci, note = (mean_d - h, mean_d + h), None
    wil = None
    try:
        w = stats.wilcoxon(b, o)
        wil = {"statistic": _f(w.statistic), "p_value": _f(w.pvalue)}
    except ValueError:
        pass
    return {
        "test": "paired_t_test", "n_pairs": n, "df": df,
        "mean_before": _f(b.mean()), "mean_after": _f(o.mean()),
        "mean_difference": _f(mean_d), "sd_difference": _f(sd_d),
        "t_statistic": _f(t_stat), "p_value": _f(p_val),
        "ci_level": 1 - alpha, "ci_low": _f(ci[0]), "ci_high": _f(ci[1]),
        "cohens_dz": _f(mean_d / sd_d) if sd_d > 0 else None,
        "significant": bool(p_val < alpha), "alpha": alpha,
        "wilcoxon": wil, "note": note,
    }


def anova_tukey(groups: dict[str, list[float]], alpha: float = 0.05) -> dict:
    """One-way ANOVA across groups (e.g. driving conditions) + Tukey HSD post-hoc."""
    clean = {k: np.asarray(v, float) for k, v in groups.items() if len(v) > 0}
    if len(clean) < 2 or any(v.size < 2 for v in clean.values()):
        raise ValueError("ANOVA needs at least 2 groups with at least 2 values each.")
    f_stat, p_val = stats.f_oneway(*clean.values())
    n_total = sum(v.size for v in clean.values())
    k = len(clean)
    grand = np.concatenate(list(clean.values())).mean()
    ssb = sum(v.size * (v.mean() - grand) ** 2 for v in clean.values())
    sst = sum(((v - grand) ** 2).sum() for v in clean.values())
    values = np.concatenate(list(clean.values()))
    labels = np.concatenate([[name] * v.size for name, v in clean.items()])
    tk = pairwise_tukeyhsd(values, labels, alpha=alpha)
    rows = tk.summary().data[1:]
    pairs = [{"group1": r[0], "group2": r[1], "mean_diff": _f(r[2]), "p_adj": _f(r[3]),
              "lower": _f(r[4]), "upper": _f(r[5]), "reject": bool(r[6])} for r in rows]
    return {
        "test": "one_way_anova", "groups": {k_: descriptive(v) for k_, v in clean.items()},
        "df_between": k - 1, "df_within": n_total - k,
        "f_statistic": _f(f_stat), "p_value": _f(p_val),
        "eta_squared": _f(ssb / sst) if sst > 0 else None,
        "significant": bool(p_val < alpha), "alpha": alpha,
        "tukey_hsd": pairs,
    }


def correlation(x, y) -> dict:
    a, b = np.asarray(x, float), np.asarray(y, float)
    if a.shape != b.shape or a.size < 3:
        raise ValueError("Correlation needs two equal-length samples with at least 3 points.")
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return {"test": "correlation", "n": int(a.size), "pearson_r": None, "pearson_p": None,
                "spearman_rho": None, "spearman_p": None,
                "note": "One of the variables is constant, so correlation is undefined."}
    pr, pp = stats.pearsonr(a, b)
    sr, sp = stats.spearmanr(a, b)
    return {"test": "correlation", "n": int(a.size),
            "pearson_r": _f(pr), "pearson_p": _f(pp),
            "spearman_rho": _f(sr), "spearman_p": _f(sp),
            "note": "Correlation shows association, not cause."}


def regression(df: pd.DataFrame, y: str, xs: list[str]) -> dict:
    """OLS regression. Explanatory (direction and size of effects), not for prediction."""
    d = df[[y] + xs].dropna()
    n, k = len(d), len(xs)
    if n <= k + 1:
        raise ValueError(f"Regression needs more than {k + 1} rows for {k} predictors (have {n}).")
    X = add_constant(d[xs].astype(float), has_constant="add")
    model = OLS(d[y].astype(float), X).fit()
    warn = []
    if k >= 2:
        corr = d[xs].corr().abs().values[np.triu_indices(k, 1)]
        if corr.size and corr.max() > 0.8:
            warn.append("Predictors are strongly correlated with each other; coefficients may be unstable.")
    resid_p = None
    if n >= 3 and np.ptp(model.resid) > 0:
        resid_p = _f(stats.shapiro(model.resid).pvalue)
    return {
        "test": "ols_regression", "outcome": y, "predictors": xs, "n": int(n),
        "r_squared": _f(model.rsquared), "adj_r_squared": _f(model.rsquared_adj),
        "f_p_value": _f(model.f_pvalue),
        "coefficients": [{"name": name, "coef": _f(model.params[name]),
                          "std_err": _f(model.bse[name]), "p_value": _f(model.pvalues[name])}
                         for name in model.params.index],
        "residual_normality_p": resid_p, "warnings": warn,
        "note": "Explanatory model: shows direction and size of each effect, not a forecast.",
    }


# ------------------------------------------------------------- availability
def _card(tid, name, available, requires, have, result=None, reason=None):
    return {"id": tid, "name": name, "status": "ready" if available else "locked",
            "requires": requires, "have": have, "reason": reason, "result": result}


def live_statistics(store: DataStore, settings: Settings) -> dict:
    """Run on the data received from C3. Locks tests whose method minimum is not met."""
    pairs = store.pairs()
    n = len(pairs)
    cond_speed = {c.condition_id: c.avg_speed_kmh for c in store.available_conditions()}
    rows = []
    for p in pairs:
        eb, eo = p["base"].energy_wh_per_km, p["opt"].energy_wh_per_km
        cfg_b = store.baseline_config(p["ev_id"])
        cfg_o = store.optimized_config(p["ev_id"], p["optimized_config_id"])
        rows.append({
            "condition_id": p["condition_id"], "eb": eb, "eo": eo,
            "saving_pct": (eb - eo) / eb * 100,
            "avg_speed_kmh": cond_speed.get(p["condition_id"]),
            "cd_reduction_pct": (cfg_b.cd - cfg_o.cd) / cfg_b.cd * 100,
        })
    df = pd.DataFrame(rows)
    groups = {}
    for r in rows:
        groups.setdefault(r["condition_id"], []).append(r["saving_pct"])
    sizes = {k: len(v) for k, v in groups.items()}

    cards = []
    # paired t-test (method minimum: 2 pairs)
    ok = n >= settings.min_pairs_ttest
    cards.append(_card("paired_t", "Paired t-test (baseline vs optimized energy)", ok,
                       f"at least {settings.min_pairs_ttest} baseline-optimized pairs", f"{n} pairs",
                       paired_ttest(df.eb, df.eo, settings.alpha) if ok else None,
                       None if ok else "Needs at least 2 baseline-optimized pairs."))
    # ANOVA (method minimum: 2 groups with at least 2 values each)
    ok = (len(groups) >= settings.min_groups_anova
          and all(s >= settings.min_per_group_anova for s in sizes.values()))
    cards.append(_card("anova", "ANOVA + Tukey HSD (saving % across driving conditions)", ok,
                       f"at least {settings.min_groups_anova} conditions with at least "
                       f"{settings.min_per_group_anova} values each",
                       f"{len(groups)} conditions, values per condition: {sizes}",
                       anova_tukey({CONDITION_LABELS.get(k, k): v for k, v in groups.items()},
                                   settings.alpha) if ok else None,
                       None if ok else "Each condition has too few values. Needs several designs "
                                       "(or EVs) under each condition."))
    # correlation (method minimum: 3 points)
    d2 = df.dropna(subset=["avg_speed_kmh"])
    ok = len(d2) >= settings.min_points_correlation
    cards.append(_card("correlation", "Correlation (saving % vs condition average speed)", ok,
                       f"at least {settings.min_points_correlation} data points", f"{len(d2)} points",
                       correlation(d2.avg_speed_kmh, d2.saving_pct) if ok else None,
                       None if ok else "Needs at least 3 data points."))
    # regression (method minimum: more rows than predictors + 1; predictors must vary)
    xs = [c for c in ("cd_reduction_pct", "avg_speed_kmh") if df[c].notna().all() and df[c].nunique() > 1]
    ok = len(xs) >= 1 and len(df) > len(xs) + 1
    cards.append(_card("regression", "Regression (saving % ~ factors that vary in the data)", ok,
                       "more data rows than predictors + 1, and predictors that vary",
                       f"{len(df)} rows, varying predictors: {xs or 'none'}",
                       regression(df, "saving_pct", xs) if ok else None,
                       None if ok else "Needs more rows, and factors that vary across the data."))

    desc = {"energy_saving_pct": descriptive(df.saving_pct) if n else {"n": 0}}
    return {"n_pairs": n, "conditions_with_data": list(groups), "descriptive": desc,
            "tests": cards}


# ------------------------------------------------------------- sample data
def sample_dataset(seed: int = 42) -> pd.DataFrame:
    """Generated sample: 5 driving conditions x 8 designs made for each = 40 baseline-optimized pairs
    (used to show the methods; WLTP is the standard cycle, not a condition with designs)."""
    rng = np.random.default_rng(seed)
    speeds = {"Urban": 21.8, "Sub Urban": 37.0, "Rural": 44.0, "Intercity": 63.0, "Expressway": 94.5}
    cd_red = np.linspace(4, 12, 8)  # % Cd reduction of 8 candidate designs
    rows = []
    for cond, v in speeds.items():
        eb = 105 + 1.05 * v
        share = 0.12 + 0.0049 * v
        for i, red in enumerate(cd_red):
            saving = share * red + rng.normal(0, 0.12)
            rows.append({"condition": cond, "avg_speed_kmh": v, "design": f"D{i + 1}",
                         "cd_reduction_pct": float(red), "energy_base": eb,
                         "energy_opt": eb * (1 - saving / 100), "saving_pct": saving})
    return pd.DataFrame(rows)


def sample_statistics(settings: Settings, seed: int = 42) -> dict:
    df = sample_dataset(seed)
    groups = {c: g.saving_pct.tolist() for c, g in df.groupby("condition", sort=False)}
    return {
        "design": "5 driving conditions x 8 designs for each = 40 baseline-optimized pairs",
        "paired_t": paired_ttest(df.energy_base, df.energy_opt, settings.alpha),
        "anova": anova_tukey(groups, settings.alpha),
        "correlation": correlation(df.avg_speed_kmh, df.saving_pct),
        "regression": regression(df, "saving_pct", ["cd_reduction_pct", "avg_speed_kmh"]),
        "by_condition": [{"condition": c, "mean_saving_pct": _f(np.mean(v)),
                          "avg_speed_kmh": float(df[df.condition == c].avg_speed_kmh.iloc[0])}
                         for c, v in groups.items()],
    } 