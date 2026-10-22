"""Function 2: compare driving conditions with each other (each with its own optimized design),
plus sensitivity analysis.

Everything here is arithmetic on the numbers C3 sent. No modelling assumptions are added:
- Condition comparison: differences and ratios of C3's own results.
- Cd check: the Cd change compared with C3's own `cd_uncertainty` (C3's definition).
- Tariff sensitivity: C3's cost is energy x tariff / charging efficiency (checked by the
  data quality module), so cost changes in direct proportion to the tariff.
"""
from typing import Optional

from .comparison import compare, pick_design
from .store import CONDITION_LABELS, DataStore


def _short(label: str) -> str:
    return label.split(" (")[0]


# ------------------------------------------------------------------ insights
def _insights(rows: list[dict], bench_energy: Optional[float], predicted_only: bool) -> list[str]:
    """Plain statements of what the numbers show. No explanations of cause."""
    out = []
    if not rows:
        return ["No driving conditions with results yet."]
    for r in rows:
        if r["baseline_vs_benchmark_pct"] is not None and bench_energy:
            diff = r["baseline_vs_benchmark_pct"]
            word = "above" if diff >= 0 else "below"
            out.append(f"{_short(r['label'])}: the baseline car uses {r['energy_baseline_wh_per_km']:.1f} Wh/km, "
                       f"{abs(diff):.1f}% {word} its WLTP value ({bench_energy:.1f} Wh/km).")
    if len(rows) > 1:
        top = max(rows, key=lambda r: r["saving_pct"])
        low = min(rows, key=lambda r: r["saving_pct"])
        out.append(f"The saving is highest under {_short(top['label'])} ({top['saving_pct']:.1f}%) "
                   f"and lowest under {_short(low['label'])} ({low['saving_pct']:.1f}%).")
        out.append("Each condition uses its own optimized design, so a difference between conditions "
                   "also includes the difference between the designs.")
    if predicted_only:
        out.append("These are predicted results (not experimentally validated).")
    return out


def condition_comparison(store: DataStore, ev_id: str, benchmark: str,
                         condition_ids: Optional[list[str]] = None) -> dict:
    """The user picks any number of conditions; each is shown with its own optimized design
    (baseline vs the design found for that condition) next to the others."""
    store.get_ev(ev_id)
    base = store.baseline_config(ev_id)
    for cid in condition_ids or []:
        store.get_condition(cid)        # unknown id -> NotFound (404)
    bench_row = None
    if any(c.condition_id == benchmark for c in store.available_conditions()):
        bench_row = store.row(ev_id, benchmark, base.config_id)
    bench_energy = None if bench_row is None else bench_row.energy_wh_per_km

    rows = []
    for cond in store.available_conditions():
        if cond.condition_id == benchmark:
            continue
        if condition_ids is not None and cond.condition_id not in condition_ids:
            continue
        c = compare(store, ev_id, cond.condition_id)
        if not c["available"]:
            continue
        s = c["savings"]
        eb = c["baseline"]["energy_wh_per_km"]
        rows.append({
            "condition_id": cond.condition_id,
            "label": CONDITION_LABELS.get(cond.condition_id, cond.condition_id),
            "group": cond.group, "avg_speed_kmh": cond.avg_speed_kmh,
            "max_speed_kmh": cond.max_speed_kmh, "idle_fraction": cond.idle_fraction,
            "distance_km": cond.distance_km,
            "design_id": c["optimized_config_id"],
            "energy_baseline_wh_per_km": eb,
            "energy_optimized_wh_per_km": c["optimized"]["energy_wh_per_km"],
            "baseline_vs_benchmark_pct": (None if not bench_energy
                                          else (eb - bench_energy) / bench_energy * 100),
            "saving_pct": s["energy_saving_pct"], "energy_saving_wh_per_km": s["energy_saving_wh_per_km"],
            "range_gain_km": s["range_gain_km"], "range_gain_pct": s["range_gain_pct"],
            "cost_saving_lkr_per_100km": s["cost_saving_lkr_per_100km"],
            "cost_saving_pct": s["cost_saving_pct"],
        })
    ranking = [r["condition_id"] for r in sorted(rows, key=lambda r: -r["saving_pct"])]
    return {
        "ev_id": ev_id,
        "benchmark_condition": benchmark if bench_energy is not None else None,
        "benchmark_baseline_energy_wh_per_km": bench_energy,
        "conditions": rows,
        "coming_soon": [{"condition_id": c.condition_id,
                         "label": CONDITION_LABELS.get(c.condition_id, c.condition_id)}
                        for c in store.coming_soon_conditions()],
        "ranking": ranking,
        "insights": _insights(rows, bench_energy, store.meta()["predicted_only"]),
        "meta": store.meta(),
    }


# --------------------------------------------------------------- sensitivity
def sensitivity(store: DataStore, ev_id: str, condition_id: str,
                optimized_config_id: Optional[str] = None,
                tariff_change_pct: float = 0.0) -> dict:
    """Cd change vs C3's uncertainty, and how the cost saving moves with the tariff."""
    base = store.baseline_config(ev_id)
    opt = pick_design(store, ev_id, condition_id, optimized_config_id)
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