"""Best design: ranks the optimized designs of one EV on energy, range and cost.

Each optimized design is compared with the baseline under the local (SLDC) condition it was made
for (and under any other local condition where C3 sent results for it). WLTP is the benchmark and
is not part of the ranking. For each design C4 takes the average of
  - energy saving (%)
  - range gain (%)
  - cost saving (%)
The overall score is the plain average of these three (equal weights).
`rank_by` can be "overall", "energy", "range" or "cost".
Nothing is estimated: all numbers come from C3's results.
"""
from typing import Optional

from .comparison import savings
from .store import CONDITION_LABELS, DataStore

METRICS = {
    "overall": "overall_score_pct",
    "energy": "mean_energy_saving_pct",
    "range": "mean_range_gain_pct",
    "cost": "mean_cost_saving_pct",
}
RULE = ("Overall score = average of three numbers: energy saving %, range gain % and cost saving %. "
        "Each design is compared with the baseline under the driving condition it was made for. "
        "The three results have equal weight. Because the designs are for different conditions, "
        "the percentages are measured against different baselines.")


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def ranking(store: DataStore, ev_id: str, rank_by: str = "overall") -> dict:
    if rank_by not in METRICS:
        raise ValueError(f"rank_by must be one of: {', '.join(METRICS)}.")
    ev = store.get_ev(ev_id)
    base = store.baseline_config(ev_id)
    available = store.available_conditions()
    conds = [c for c in available if c.group != "standard"]   # local conditions
    key = METRICS[rank_by]

    designs = []
    for opt in store.optimized_configs(ev_id):
        per = []
        for cond in conds:
            rb = store.row(ev_id, cond.condition_id, base.config_id)
            ro = store.row(ev_id, cond.condition_id, opt.config_id)
            if rb is None or ro is None:
                continue
            s = savings(rb.energy_wh_per_km, ro.energy_wh_per_km, rb.range_km, ro.range_km,
                        rb.cost_lkr_per_100km, ro.cost_lkr_per_100km)
            per.append({
                "condition_id": cond.condition_id,
                "label": CONDITION_LABELS.get(cond.condition_id, cond.condition_id),
                "energy_saving_pct": s["energy_saving_pct"], "range_gain_pct": s["range_gain_pct"],
                "cost_saving_pct": s["cost_saving_pct"],
                "energy_saving_wh_per_km": s["energy_saving_wh_per_km"],
                "range_gain_km": s["range_gain_km"],
                "cost_saving_lkr_per_100km": s["cost_saving_lkr_per_100km"],
                "overall_pct": (s["energy_saving_pct"] + s["range_gain_pct"] + s["cost_saving_pct"]) / 3,
            })
        if not per:
            continue
        e = _mean([p["energy_saving_pct"] for p in per])
        r = _mean([p["range_gain_pct"] for p in per])
        c = _mean([p["cost_saving_pct"] for p in per])
        designs.append({
            "config_id": opt.config_id, "optimized_for": opt.optimized_for,
            "optimized_for_label": (CONDITION_LABELS.get(opt.optimized_for, opt.optimized_for)
                                    if opt.optimized_for else None),
            "cd": opt.cd, "cd_change": opt.cd - base.cd,
            "cd_uncertainty": opt.cd_uncertainty, "frontal_area_m2": opt.frontal_area_m2,
            "shape_parameters": opt.shape_parameters, "geometry_file": opt.geometry_file,
            "n_conditions": len(per),
            "mean_energy_saving_pct": e, "mean_range_gain_pct": r, "mean_cost_saving_pct": c,
            "overall_score_pct": (e + r + c) / 3,
            "per_condition": per,
        })

    designs.sort(key=lambda d: -d[key])
    for i, d in enumerate(designs, start=1):
        d["rank"] = i
    tie = len(designs) > 1 and abs(designs[0][key] - designs[1][key]) < 1e-9

    # the winner under each single condition
    pk = {"overall": "overall_pct", "energy": "energy_saving_pct",
          "range": "range_gain_pct", "cost": "cost_saving_pct"}[rank_by]
    by_condition = []
    for cond in conds:
        cand = [(d["config_id"], p[pk]) for d in designs for p in d["per_condition"]
                if p["condition_id"] == cond.condition_id]
        if cand:
            win = max(cand, key=lambda t: t[1])
            by_condition.append({"condition_id": cond.condition_id,
                                 "label": CONDITION_LABELS.get(cond.condition_id, cond.condition_id),
                                 "best_config_id": win[0], "value_pct": win[1]})

    return {
        "ev_id": ev_id, "ev_model": ev.ev_model, "baseline_config_id": base.config_id,
        "rank_by": rank_by, "rule": RULE,
        "conditions_used": [c.condition_id for c in conds],
        "designs": designs,
        "best_config_id": designs[0]["config_id"] if designs else None,
        "tie_for_first": tie,
        "best_by_condition": by_condition,
        "meta": store.meta(),
    }


def best_config_id(store: DataStore, ev_id: str) -> Optional[str]:
    """The best design by the overall score (used as the default where no design is chosen)."""
    return ranking(store, ev_id, "overall")["best_config_id"]