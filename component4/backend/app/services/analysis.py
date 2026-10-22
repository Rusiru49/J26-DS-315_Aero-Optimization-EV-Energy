"""Function 2: driving-condition comparison (WLTP benchmark vs local) and sensitivity analysis.

Everything here is arithmetic on the numbers C3 sent. No modelling assumptions are added:
- Condition comparison: differences and ratios of C3's own results.
- Cd check: the Cd change compared with C3's own `cd_uncertainty` (C3's definition).
- Tariff sensitivity: C3's cost is energy x tariff / charging efficiency (checked by the
  data quality module), so cost changes in direct proportion to the tariff.
- Driving mix: a weighted average of C3's results using the mix the user chooses.
"""
from typing import Optional

from scipy import stats

from .comparison import compare, savings
from .store import CONDITION_LABELS, DataStore


def _short(label: str) -> str:
    return label.split(" (")[0]


# ------------------------------------------------------------------ insights
def _insights(rows: list[dict], benchmark: str, predicted_only: bool) -> list[str]:
    """Plain statements of what the numbers show. No explanations of cause."""
    out = []
    if not rows:
        return ["No driving conditions with results yet."]
    top = max(rows, key=lambda r: r["saving_pct"])
    low = min(rows, key=lambda r: r["saving_pct"])
    if len(rows) > 1:
        out.append(f"The saving is highest under {_short(top['label'])} ({top['saving_pct']:.1f}%) "
                   f"and lowest under {_short(low['label'])} ({low['saving_pct']:.1f}%).")
    bench = next((r for r in rows if r["condition_id"] == benchmark), None)
    if bench is not None:
        for r in rows:
            if r["condition_id"] == benchmark:
                continue
            diff = r["saving_pct"] - bench["saving_pct"]
            word = "above" if diff >= 0 else "below"
            out.append(f"{_short(r['label'])}: {r['saving_pct']:.1f}% saving, {abs(diff):.1f} "
                       f"percentage points {word} the {_short(bench['label'])} benchmark "
                       f"({bench['saving_pct']:.1f}%).")
    urban = next((r for r in rows if r["condition_id"] == "SLDC_Urban"), None)
    exp_ = next((r for r in rows if r["condition_id"] == "SLDC_Expressway"), None)
    if urban and exp_ and urban["saving_pct"] > 0:
        out.append(f"The Expressway saving is {exp_['saving_pct'] / urban['saving_pct']:.1f} times "
                   f"the Urban saving.")
    sp = sorted((r["avg_speed_kmh"], r["saving_pct"], _short(r["label"]))
                for r in rows if r["avg_speed_kmh"] is not None)
    if len(sp) >= 3:
        rho = stats.spearmanr([a for a, _, _ in sp], [b for _, b, _ in sp]).statistic
        if rho == 1.0:
            order = ", ".join(f"{n} {a:.0f} km/h" for a, _, n in sp)
            out.append(f"In this data the saving rises with the condition's average speed ({order}).")
    elif len(sp) == 2 and sp[0][0] != sp[1][0] and sp[1][1] > sp[0][1]:
        out.append(f"The condition with the higher average speed ({sp[1][2]}, {sp[1][0]:.0f} km/h) "
                   f"has the larger saving.")
    if predicted_only:
        out.append("These are predicted results (not experimentally validated).")
    return out


def condition_comparison(store: DataStore, ev_id: str, benchmark: str,
                         optimized_config_id: Optional[str] = None) -> dict:
    store.get_ev(ev_id)
    opt = store.optimized_config(ev_id, optimized_config_id)
    rows = []
    for cond in store.available_conditions():
        c = compare(store, ev_id, cond.condition_id, opt.config_id)
        if not c["available"]:
            continue
        s = c["savings"]
        rows.append({
            "condition_id": cond.condition_id,
            "label": CONDITION_LABELS.get(cond.condition_id, cond.condition_id),
            "group": cond.group, "avg_speed_kmh": cond.avg_speed_kmh,
            "max_speed_kmh": cond.max_speed_kmh, "idle_fraction": cond.idle_fraction,
            "distance_km": cond.distance_km,
            "energy_baseline_wh_per_km": c["baseline"]["energy_wh_per_km"],
            "energy_optimized_wh_per_km": c["optimized"]["energy_wh_per_km"],
            "saving_pct": s["energy_saving_pct"], "energy_saving_wh_per_km": s["energy_saving_wh_per_km"],
            "range_gain_km": s["range_gain_km"], "range_gain_pct": s["range_gain_pct"],
            "cost_saving_lkr_per_100km": s["cost_saving_lkr_per_100km"],
            "cost_saving_pct": s["cost_saving_pct"],
        })
    bench = next((r for r in rows if r["condition_id"] == benchmark), None)
    for r in rows:
        if bench is None:
            r["vs_benchmark_pct_points"], r["vs_benchmark_ratio"] = None, None
        else:
            r["vs_benchmark_pct_points"] = r["saving_pct"] - bench["saving_pct"]
            r["vs_benchmark_ratio"] = (r["saving_pct"] / bench["saving_pct"]
                                       if bench["saving_pct"] else None)
    ranking = [r["condition_id"] for r in sorted(rows, key=lambda r: -r["saving_pct"])]
    return {
        "ev_id": ev_id, "optimized_config_id": opt.config_id,
        "benchmark_condition": benchmark if bench else None,
        "conditions": rows,
        "coming_soon": [{"condition_id": c.condition_id,
                         "label": CONDITION_LABELS.get(c.condition_id, c.condition_id)}
                        for c in store.coming_soon_conditions()],
        "ranking": ranking,
        "insights": _insights(rows, benchmark, store.meta()["predicted_only"]),
        "meta": store.meta(),
    }


# --------------------------------------------------------------- sensitivity
def sensitivity(store: DataStore, ev_id: str, condition_id: str,
                optimized_config_id: Optional[str] = None,
                tariff_change_pct: float = 0.0) -> dict:
    """Cd change vs C3's uncertainty, and how the cost saving moves with the tariff."""
    base = store.baseline_config(ev_id)
    opt = store.optimized_config(ev_id, optimized_config_id)
    rb = store.row(ev_id, condition_id, base.config_id)
    ro = store.row(ev_id, condition_id, opt.config_id)
    if rb is None or ro is None:
        return {"available": False, "status": "coming_soon",
                "message": "Results for this driving condition are not available yet."}

    d_cd = opt.cd - base.cd
    u = opt.cd_uncertainty
    cd_check = {
        "baseline_cd": base.cd, "optimized_cd": opt.cd, "cd_change": d_cd,
        "optimized_cd_uncertainty": u,
        "change_in_uncertainties": (None if not u else abs(d_cd) / u),
        "within_uncertainty": None if u is None else abs(d_cd) < u,
    }

    def tariff_case(pct: float) -> dict:
        f = 1 + pct / 100
        notes = store.catalog.model_notes
        return {
            "tariff_change_pct": pct,
            "tariff_lkr_per_kwh": (None if notes.tariff_lkr_per_kwh is None
                                   else notes.tariff_lkr_per_kwh * f),
            "baseline_cost_lkr_per_100km": rb.cost_lkr_per_100km * f,
            "optimized_cost_lkr_per_100km": ro.cost_lkr_per_100km * f,
            "cost_saving_lkr_per_100km": (rb.cost_lkr_per_100km - ro.cost_lkr_per_100km) * f,
        }

    return {
        "available": True, "status": "available",
        "ev_id": ev_id, "condition_id": condition_id,
        "condition_label": CONDITION_LABELS.get(condition_id, condition_id),
        "optimized_config_id": opt.config_id,
        "cd_check": cd_check,
        "tariff": {"selected": tariff_case(tariff_change_pct),
                   "sweep": [tariff_case(p) for p in (-20, -10, 0, 10, 20)],
                   "note": "C3's cost = energy x tariff / charging efficiency, so cost changes in "
                           "direct proportion to the tariff. The % saving stays the same."},
        "meta": store.meta(),
    }


# ---------------------------------------------------------------- what-if mix
def mix(store: DataStore, ev_id: str, condition_a: str, condition_b: str,
        share_a_pct: float, distance_km: float,
        optimized_config_id: Optional[str] = None) -> dict:
    """Weighted average of C3's results for a driving mix chosen by the user (by distance share)."""
    if not 0 <= share_a_pct <= 100:
        raise ValueError("share_a_pct must be between 0 and 100.")
    if distance_km < 0:
        raise ValueError("distance_km must not be negative.")
    ev = store.get_ev(ev_id)
    base = store.baseline_config(ev_id)
    opt = store.optimized_config(ev_id, optimized_config_id)
    rows = {}
    for key, cid in (("a", condition_a), ("b", condition_b)):
        rb = store.row(ev_id, cid, base.config_id)
        ro = store.row(ev_id, cid, opt.config_id)
        if rb is None or ro is None:
            return {"available": False, "status": "coming_soon",
                    "message": f"Results for {CONDITION_LABELS.get(cid, cid)} are not available yet."}
        rows[key] = (rb, ro)
    wa, wb = share_a_pct / 100, 1 - share_a_pct / 100

    def blend(field, which):
        return wa * getattr(rows["a"][which], field) + wb * getattr(rows["b"][which], field)

    e_b, e_o = blend("energy_wh_per_km", 0), blend("energy_wh_per_km", 1)
    c_b, c_o = blend("cost_lkr_per_100km", 0), blend("cost_lkr_per_100km", 1)
    # range follows C3's definition: usable battery x 1000 / energy per km
    r_b, r_o = ev.usable_battery_kwh * 1000 / e_b, ev.usable_battery_kwh * 1000 / e_o
    s = savings(e_b, e_o, r_b, r_o, c_b, c_o)
    return {
        "available": True, "ev_id": ev_id, "optimized_config_id": opt.config_id,
        "condition_a": {"id": condition_a, "label": CONDITION_LABELS.get(condition_a, condition_a),
                        "share_pct": share_a_pct},
        "condition_b": {"id": condition_b, "label": CONDITION_LABELS.get(condition_b, condition_b),
                        "share_pct": 100 - share_a_pct},
        "distance_km": distance_km,
        "blended": {"baseline_energy_wh_per_km": e_b, "optimized_energy_wh_per_km": e_o,
                    "baseline_range_km": r_b, "optimized_range_km": r_o, **s},
        "over_distance": {"energy_saved_kwh": s["energy_saving_wh_per_km"] * distance_km / 1000,
                          "cost_saving_lkr": s["cost_saving_lkr_per_100km"] * distance_km / 100},
        "note": "The driving mix (share of distance) and the distance are chosen by the user.",
        "meta": store.meta(),
    }