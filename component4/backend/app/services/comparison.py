"""Function 1: baseline vs optimized impact for one EV under one condition.

The three savings are calculated here by C4 from C3's result rows:
  Energy Saving = E_baseline - E_optimized
  Range Gain    = R_optimized - R_baseline
  Cost Saving   = C_baseline - C_optimized
"""
from typing import Optional

from .store import CONDITION_LABELS, DataStore


def _pct(delta: float, base: float) -> Optional[float]:
    return None if base == 0 else delta / base * 100.0
 

def savings(energy_b: float, energy_o: float, range_b: float, range_o: float,
            cost_b: float, cost_o: float) -> dict:
    """Pure calculation of the three savings (and % changes). No rounding here."""
    e, r, c = energy_b - energy_o, range_o - range_b, cost_b - cost_o
    return {
        "energy_saving_wh_per_km": e, "energy_saving_pct": _pct(e, energy_b),
        "range_gain_km": r, "range_gain_pct": _pct(r, range_b),
        "cost_saving_lkr_per_100km": c, "cost_saving_pct": _pct(c, cost_b),
    }


def _shape_changes(store: DataStore, base, opt) -> list[dict]:
    defs = {d.name: d for d in store.catalog.shape_parameter_definitions}
    out = []
    for name in list(base.shape_parameters) + [n for n in opt.shape_parameters
                                               if n not in base.shape_parameters]:
        b, o = base.shape_parameters.get(name), opt.shape_parameters.get(name)
        d = defs.get(name)
        out.append({
            "name": name, "unit": d.unit if d else "", "baseline": b, "optimized": o,
            "delta": None if b is None or o is None else o - b,
            "min": d.min if d else None, "max": d.max if d else None,
            "description": d.effect_on_geometry if d else "",
        })
    return out


def pick_design(store: DataStore, ev_id: str, condition_id: str,
                optimized_config_id: Optional[str] = None):
    """The optimized design to use: the one asked for; else the design found for this condition;
    else (for example WLTP, which has no design of its own) the best overall design."""
    if optimized_config_id is None:
        own = [c for c in store.optimized_configs(ev_id) if c.optimized_for == condition_id]
        if own:
            return own[0]
        from .designs import best_config_id  # imported here to avoid a circular import
        optimized_config_id = best_config_id(store, ev_id)
    return store.optimized_config(ev_id, optimized_config_id)


def compare(store: DataStore, ev_id: str, condition_id: str,
            optimized_config_id: Optional[str] = None) -> dict:
    ev = store.get_ev(ev_id)
    cond = store.get_condition(condition_id)
    base = store.baseline_config(ev_id)
    opt = pick_design(store, ev_id, condition_id, optimized_config_id)

    head = {
        "ev_id": ev_id, "ev_model": ev.ev_model, "mass_kg": ev.mass_kg,
        "usable_battery_kwh": ev.usable_battery_kwh,
        "condition_id": condition_id,
        "condition_label": CONDITION_LABELS.get(condition_id, condition_id),
        "baseline_config_id": base.config_id, "optimized_config_id": opt.config_id,
        "optimized_for": opt.optimized_for,
        "meta": store.meta(),
    }

    rb = store.row(ev_id, condition_id, base.config_id)
    ro = store.row(ev_id, condition_id, opt.config_id)
    if rb is None or ro is None:
        return {**head, "available": False, "status": "coming_soon",
                "message": "Results for this driving condition are not available yet "
                           "(data collection in progress)."}

    s = savings(rb.energy_wh_per_km, ro.energy_wh_per_km, rb.range_km, ro.range_km,
                rb.cost_lkr_per_100km, ro.cost_lkr_per_100km)

    d_cd = opt.cd - base.cd
    unc = opt.cd_uncertainty
    within = None if unc is None else abs(d_cd) < unc

    # cross-check with C3's own comparison row (if C3 sent one)
    c3 = next((c for c in store.prediction(ev_id, condition_id).comparisons
               if c.optimized_config_id == opt.config_id), None)
    cross = None
    if c3 is not None:
        cross = {
            "energy_saving_wh_per_km": c3.energy_saving_wh_per_km - s["energy_saving_wh_per_km"],
            "range_gain_km": c3.range_gain_km - s["range_gain_km"],
            "cost_saving_lkr_per_100km": c3.cost_saving_lkr_per_100km - s["cost_saving_lkr_per_100km"],
        }

    # the baseline car under the benchmark (WLTP), so the user can compare it with this condition
    benchmark = None
    bench_cond = store.benchmark_for(condition_id)
    if bench_cond is not None:
        bb = store.row(ev_id, bench_cond.condition_id, base.config_id)
        if bb is not None:
            benchmark = {
                "condition_id": bench_cond.condition_id,
                "condition_label": CONDITION_LABELS.get(bench_cond.condition_id, bench_cond.condition_id),
                "baseline": {"energy_wh_per_km": bb.energy_wh_per_km, "range_km": bb.range_km,
                             "cost_lkr_per_100km": bb.cost_lkr_per_100km},
                # selected condition minus benchmark
                "difference": {
                    "baseline_energy_wh_per_km": rb.energy_wh_per_km - bb.energy_wh_per_km,
                    "baseline_energy_pct": (rb.energy_wh_per_km - bb.energy_wh_per_km) / bb.energy_wh_per_km * 100,
                    "baseline_range_km": rb.range_km - bb.range_km,
                    "baseline_cost_lkr_per_100km": rb.cost_lkr_per_100km - bb.cost_lkr_per_100km,
                },
            }

    return {
        **head,
        "available": True, "status": "available",
        "condition": cond.model_dump(),
        "baseline": {"cd": base.cd, "cd_uncertainty": base.cd_uncertainty,
                     "frontal_area_m2": base.frontal_area_m2,
                     "energy_wh_per_km": rb.energy_wh_per_km, "range_km": rb.range_km,
                     "cost_lkr_per_100km": rb.cost_lkr_per_100km,
                     "geometry_file": base.geometry_file,
                     "shape_parameters": base.shape_parameters},
        "optimized": {"cd": opt.cd, "cd_uncertainty": opt.cd_uncertainty,
                      "frontal_area_m2": opt.frontal_area_m2,
                      "energy_wh_per_km": ro.energy_wh_per_km, "range_km": ro.range_km,
                      "cost_lkr_per_100km": ro.cost_lkr_per_100km,
                      "geometry_file": opt.geometry_file,
                      "shape_parameters": opt.shape_parameters},
        "savings": s,
        "aerodynamics": {
            "cd_change": d_cd, "cd_change_pct": _pct(d_cd, base.cd),
            "frontal_area_change_m2": opt.frontal_area_m2 - base.frontal_area_m2,
            "cd_uncertainty_optimized": unc,
            "saving_within_uncertainty": within,
            "uncertainty_note": (
                None if within is None else
                "The Cd difference is smaller than the model's error. Treat the saving as "
                "'no clear difference'." if within else
                "The Cd difference is larger than the model's error."),
        },
        "benchmark": benchmark,
        "shape_changes": _shape_changes(store, base, opt),
        "c3_cross_check": cross,
        "caution": "Predicted results (not experimentally validated)."
                   if store.meta()["predicted_only"] else None,
    }