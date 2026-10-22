"""What affects the saving? A plain view of which factors go with the energy saving.

Everything here is calculated from C3's results. No model is fitted for forecasting.
- Variation shares: how much of the differences in energy saving (%) goes with the driving
  condition and how much goes with the design (two-way analysis of variance, sums of squares).
- By condition / by design: the average saving for each, with the factors that differ.
- Factors that cannot be tested yet are listed with the reason (for example vehicle mass needs
  at least 3 EVs with different masses).
"""
import math
from typing import Optional

import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from .store import CONDITION_LABELS, DataStore

MIN_EVS_FOR_VEHICLE_FACTORS = 3


def _f(x) -> Optional[float]:
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _short(label: str) -> str:
    return label.split(" (")[0]


def _rows(store: DataStore) -> pd.DataFrame:
    cond_info = {c.condition_id: c for c in store.available_conditions()}
    rows = []
    for p in store.pairs():
        ev = store.get_ev(p["ev_id"])
        base = store.baseline_config(ev.ev_id)
        opt = store.optimized_config(ev.ev_id, p["optimized_config_id"])
        c = cond_info[p["condition_id"]]
        eb, eo = p["base"].energy_wh_per_km, p["opt"].energy_wh_per_km
        cda_b, cda_o = base.cd * base.frontal_area_m2, opt.cd * opt.frontal_area_m2
        rows.append({
            "ev": ev.ev_id, "design": opt.config_id, "optimized_for": opt.optimized_for,
            "condition": p["condition_id"], "saving": (eb - eo) / eb * 100,
            "cda_red": (cda_b - cda_o) / cda_b * 100, "cd_red": (base.cd - opt.cd) / base.cd * 100,
            "speed": c.avg_speed_kmh, "idle": c.idle_fraction, "dist": c.distance_km,
            "mass": ev.mass_kg, "battery": ev.usable_battery_kwh,
        })
    return pd.DataFrame(rows)


def _variation(df: pd.DataFrame) -> Optional[dict]:
    nc, nd = df.condition.nunique(), df.design.nunique()
    if nc < 2 or nd < 2 or len(df) - (nc - 1) - (nd - 1) - 1 < 1:
        return None
    table = anova_lm(ols("saving ~ C(condition) + C(design)", data=df).fit(), typ=2)
    ss = {"condition": float(table.loc["C(condition)", "sum_sq"]),
          "design": float(table.loc["C(design)", "sum_sq"]),
          "rest": float(table.loc["Residual", "sum_sq"])}
    total = sum(ss.values())
    if total <= 0:
        return None
    return {k: v / total * 100 for k, v in ss.items()}


def _collinear(df: pd.DataFrame) -> bool:
    """True when speed, idle share and distance rank the conditions in exactly the same way."""
    per = df.groupby("condition")[["speed", "idle", "dist"]].first().dropna()
    if len(per) < 3:
        return False
    for a, b in (("speed", "idle"), ("speed", "dist")):
        rho = stats.spearmanr(per[a], per[b]).statistic
        if not (abs(rho) > 0.999999):
            return False
    return True


def factor_analysis(store: DataStore) -> dict:
    df = _rows(store)
    if df.empty:
        return {"available": False, "message": "No baseline-optimized results yet."}

    var = _variation(df)
    n_cond, n_design = int(df.condition.nunique()), int(df.design.nunique())

    # ---- by driving condition
    by_condition = []
    for cid, g in df.groupby("condition"):
        by_condition.append({
            "condition_id": cid, "label": CONDITION_LABELS.get(cid, cid),
            "avg_speed_kmh": _f(g.speed.iloc[0]), "idle_fraction": _f(g.idle.iloc[0]),
            "distance_km": _f(g.dist.iloc[0]), "mean_saving_pct": _f(g.saving.mean()), "n": int(len(g)),
        })
    by_condition.sort(key=lambda r: (r["avg_speed_kmh"] is None, r["avg_speed_kmh"] or 0))

    # ---- by design
    by_design = []
    for did, g in df.groupby("design"):
        by_design.append({
            "config_id": did, "optimized_for": g.optimized_for.iloc[0],
            "cda_reduction_pct": _f(g.cda_red.iloc[0]), "cd_reduction_pct": _f(g.cd_red.iloc[0]),
            "mean_saving_pct": _f(g.saving.mean()), "n": int(len(g)),
        })
    by_design.sort(key=lambda r: r["cda_reduction_pct"] or 0)

    # ---- vehicle factors (need several EVs)
    vehicle = []
    ev_means = df.groupby("ev").agg(saving=("saving", "mean"), mass=("mass", "first"),
                                    battery=("battery", "first"))
    for key, label, unit in (("mass", "Vehicle mass", "kg"), ("battery", "Battery size", "kWh")):
        distinct = ev_means[key].nunique()
        if len(ev_means) >= MIN_EVS_FOR_VEHICLE_FACTORS and distinct >= MIN_EVS_FOR_VEHICLE_FACTORS:
            r = stats.pearsonr(ev_means[key], ev_means["saving"]).statistic
            vehicle.append({"factor": label, "tested": True, "r": _f(r), "n_evs": int(len(ev_means)),
                            "reason": None})
        else:
            vehicle.append({
                "factor": label, "tested": False, "r": None, "n_evs": int(len(ev_means)),
                "reason": f"Not tested yet: the data has {len(ev_means)} EV(s). "
                          f"It needs at least {MIN_EVS_FOR_VEHICLE_FACTORS} EVs with different values ({unit})."})

    # ---- plain statements
    insights = []
    if var:
        insights.append(
            f"About {var['condition']:.0f}% of the differences in energy saving go with the driving "
            f"condition, {var['design']:.0f}% go with the design, and {var['rest']:.0f}% are left over "
            f"(n = {len(df)} baseline-optimized pairs).")
    if len(by_condition) >= 2:
        top = max(by_condition, key=lambda r: r["mean_saving_pct"])
        low = min(by_condition, key=lambda r: r["mean_saving_pct"])
        insights.append(
            f"On average the saving is highest under {_short(top['label'])} "
            f"({top['mean_saving_pct']:.1f}%) and lowest under {_short(low['label'])} "
            f"({low['mean_saving_pct']:.1f}%).")
    # each design is tested under one condition only: condition and design cannot be separated
    one_cond_per_design = bool((df.groupby("design").condition.nunique() == 1).all())
    if one_cond_per_design and n_design >= 2:
        insights.append(
            "Each design was tested under one driving condition only, so the effect of the condition and "
            "the effect of the design cannot be separated yet.")
    if n_design >= 2 and any(len(g) >= 2 for _, g in df.groupby("condition")):
        agree = True
        for _, g in df.groupby("condition"):
            if len(g) >= 2:
                rho = stats.spearmanr(g.cda_red, g.saving).statistic
                agree = agree and bool(abs(rho - 1) < 1e-9)
        if agree:
            insights.append("Under every driving condition, the design with the larger reduction in "
                            "drag area (Cd × frontal area) has the larger saving.")
    if _collinear(df):
        insights.append(
            f"Speed, idle share and distance rise together across the {n_cond} conditions here, so their "
            "effects cannot be told apart yet. More driving conditions (the other SLDC cycles) are needed.")
    insights.append("This shows how the results go together. It does not prove the cause.")

    return {
        "available": True, "n_pairs": int(len(df)), "n_conditions": n_cond, "n_designs": n_design,
        "variation": var,
        "variation_note": None if var else (
            "Cannot split the variation yet: each design was tested under one condition only."
            if one_cond_per_design else "Needs at least 2 driving conditions and 2 designs."),
        "by_condition": by_condition, "by_design": by_design,
        "vehicle_factors": vehicle,
        "conditions_move_together": _collinear(df),
        "insights": insights, "meta": store.meta(),
    }