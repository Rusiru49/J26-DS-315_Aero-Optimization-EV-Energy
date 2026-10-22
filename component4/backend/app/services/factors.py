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
ALPHA = 0.05 


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


def _explain(store: DataStore, df: pd.DataFrame, by_condition: list, by_design: list,
             vehicle: list, one_cond_per_design: bool, collinear: bool) -> list[dict]:
    """The second layer: apart from Cd, what else could affect the result, and what the data says
    about it. Every line is either a number from C3's results or a fact from C3's model notes.
    Statuses: in_data, not_separable, needs_data, fixed (same for all results), cost_only."""
    notes = store.catalog.model_notes
    out = []

    # 1. drag area (Cd x frontal area)
    if by_design:
        parts = [f"{d['config_id']}: drag area −{d['cda_reduction_pct']:.1f}%, saving {d['mean_saving_pct']:.1f}%"
                 for d in by_design]
        text = "; ".join(parts) + "."
        status = "in_data"
        if one_cond_per_design and len(by_design) >= 2:
            text += (" Each design was tested under one driving condition only, so its effect "
                     "cannot be separated from the condition yet.")
            status = "not_separable"
        out.append({"group": "Shape", "factor": "Cd and frontal area (drag area)",
                    "status": status, "text": text})

    # 2. driving condition
    if len(by_condition) >= 2:
        parts = []
        for c in by_condition:
            bits = []
            if c["avg_speed_kmh"] is not None:
                bits.append(f"{c['avg_speed_kmh']:g} km/h")
            if c["idle_fraction"] is not None:
                bits.append(f"{c['idle_fraction'] * 100:.0f}% idle")
            if c["distance_km"] is not None:
                bits.append(f"{c['distance_km']:g} km")
            parts.append(f"{_short(c['label'])}: saving {c['mean_saving_pct']:.1f}% ({', '.join(bits)})")
        text = "; ".join(parts) + "."
        status = "in_data"
        if collinear:
            text += " Speed, idle share and distance change together here, so they cannot be told apart yet."
            status = "not_separable"
        out.append({"group": "Driving", "factor": "Driving condition (speed, idle share, distance)",
                    "status": status, "text": text})
    else:
        out.append({"group": "Driving", "factor": "Driving condition (speed, idle share, distance)",
                    "status": "needs_data", "text": "Needs results for at least 2 driving conditions."})

    # 3. vehicle
    for v in vehicle:
        if v["tested"]:
            out.append({"group": "Vehicle", "factor": v["factor"], "status": "in_data",
                        "text": f"Correlation with the saving across {v['n_evs']} EVs: r = {v['r']:.2f}."})
        else:
            out.append({"group": "Vehicle", "factor": v["factor"], "status": "needs_data",
                        "text": v["reason"]})

    # 4. settings that C3 keeps the same for every result
    def yes_no(x):
        return None if x is None else ("included" if x else "not included")
    for label, value in (("Regenerative braking", yes_no(notes.includes_regen_braking)),
                         ("Auxiliary load", yes_no(notes.includes_auxiliary_load))):
        if value is None:
            text = "C3 did not say whether this is in the energy figures."
        else:
            text = f"C3 says it is {value} in the energy figures, the same for every result, so its effect is not measured here."
        out.append({"group": "C3 model", "factor": label, "status": "fixed", "text": text})
    if notes.road_gradient:
        out.append({"group": "C3 model", "factor": "Road gradient", "status": "fixed",
                    "text": f"C3 used a '{notes.road_gradient}' road for every result, so the effect of slope is not measured here."})

    # 5. tariff
    out.append({"group": "Cost", "factor": "Electricity tariff", "status": "cost_only",
                "text": "Changes the running cost only, not the energy. Cost = energy × tariff ÷ charging efficiency "
                        "(checked in Data Quality). See Compare Conditions → Sensitivity."})
    return out


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
        
    # does the saving go up or down with the average speed of the condition? (needs 3+ conditions)
    one_cond_per_design = bool((df.groupby("design").condition.nunique() == 1).all())
    if len(by_condition) >= 3:
        xs = [c["avg_speed_kmh"] for c in by_condition]
        ys = [c["mean_saving_pct"] for c in by_condition]
        if all(x is not None for x in xs) and len(set(xs)) > 1 and len(set(ys)) > 1:
            res = stats.pearsonr(xs, ys)
            r, p = float(res.statistic), float(res.pvalue)
            if p < ALPHA:
                insights.append(
                    f"Across the {len(xs)} driving conditions, a higher average speed goes with a "
                    f"{'larger' if r > 0 else 'smaller'} saving (r = {r:.2f}).")
            else:
                insights.append(
                    f"Across the {len(xs)} driving conditions, there is no clear link between "
                    f"average speed and the saving (r = {r:.2f}).")
    if n_design >= 2 and any(len(g) >= 2 for _, g in df.groupby("condition")):
        agree = True
        for _, g in df.groupby("condition"):
            if len(g) >= 2:
                rho = stats.spearmanr(g.cda_red, g.saving).statistic
                agree = agree and bool(abs(rho - 1) < 1e-9)
        if agree:
            insights.append("Under every driving condition, the design with the larger reduction in "
                            "drag area (Cd × frontal area) has the larger saving.")
    insights.append("This shows how the results go together. It does not prove the cause.")

    return {
        "available": True, "n_pairs": int(len(df)), "n_conditions": n_cond, "n_designs": n_design,
        "variation": var,
        "variation_note": None if var else (
            "Cannot split the variation yet: each design was tested under one condition only."
            if one_cond_per_design else "Needs at least 2 driving conditions and 2 designs."),
        "by_condition": by_condition, "by_design": by_design,
        "vehicle_factors": vehicle,
        "explain": _explain(store, df, by_condition, by_design, vehicle,
                            one_cond_per_design, _collinear(df)),
        "explain_later": "Explainable AI (planned): when C3 sends many more varied runs, a method that "
                         "ranks each factor's share in a result can be added. Not available yet.",
        "conditions_move_together": _collinear(df),
        "insights": insights, "meta": store.meta(),
    }