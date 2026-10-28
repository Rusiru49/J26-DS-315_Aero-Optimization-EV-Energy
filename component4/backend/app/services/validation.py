"""Data validation and quality report (FR4, NFR4, NFR5).

Runs on everything received from C3 and returns a list of checks, each with
status pass / warn / fail. Nothing is silently used when it fails.
"""
import math
from collections import Counter

from ..config import Settings 
from .comparison import savings
from .store import DataStore

EXPECTED_UNITS = {
    "energy_wh_per_km": "Wh/km", "range_km": "km", "cost_lkr_per_100km": "LKR/100km",
    "mass_kg": "kg", "usable_battery_kwh": "kWh",
}
KNOWN_CONDITIONS = {"WLTP", "WLTP_Urban", "WLTP_Expressway", "SLDC_Urban", "SLDC_Suburban", "SLDC_Intercity",
                    "SLDC_Expressway", "SLDC_Rural"}

def _finite(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


class _Report:
    def __init__(self):
        self.checks = []

    def add(self, cid, name, status, detail=""):
        self.checks.append({"id": cid, "name": name, "status": status, "detail": detail})

    def to_dict(self):
        counts = Counter(c["status"] for c in self.checks)
        overall = "fail" if counts["fail"] else ("warn" if counts["warn"] else "pass")
        return {"status": overall, "counts": {k: counts.get(k, 0) for k in ("pass", "warn", "fail")},
                "checks": self.checks}


def run_validation(store: DataStore, settings: Settings) -> dict:
    store.ensure_loaded()
    cat = store.catalog
    rep = _Report()

    # 1. units
    units = cat.units or {}
    if not units:
        rep.add("units", "Units declared", "warn", "C3 did not send a units table; cannot confirm units.")
    else:
        bad = {k: (units.get(k), v) for k, v in EXPECTED_UNITS.items()
               if k in units and units[k].replace(" ", "") != v}
        missing = [k for k in EXPECTED_UNITS if k not in units]
        if bad:
            rep.add("units", "Units declared", "fail", f"Unexpected units (got, expected): {bad}")
        elif missing:
            rep.add("units", "Units declared", "warn", f"Units not declared for: {missing}")
        else:
            rep.add("units", "Units declared", "pass", "All key fields use the agreed units.")

    # 2. structure of configurations
    problems = []
    cfg_ids = [c.config_id for ev in cat.evs for c in ev.configurations]
    dup = [k for k, v in Counter(cfg_ids).items() if v > 1]
    if dup:
        problems.append(f"duplicate config_id: {dup}")
    for ev in cat.evs:
        bases = [c for c in ev.configurations if c.type == "baseline"]
        opts = [c for c in ev.configurations if c.type == "optimized"]
        if len(bases) != 1:
            problems.append(f"{ev.ev_id}: expected 1 baseline, found {len(bases)}")
        if not opts:
            problems.append(f"{ev.ev_id}: no optimized configuration")
        for o in opts:
            if o.optimized_for and o.optimized_for not in {c.condition_id for c in cat.conditions}:
                problems.append(f"{o.config_id}: optimized_for '{o.optimized_for}' is not a known condition")
            if bases and o.optimized_from != bases[0].config_id:
                problems.append(f"{o.config_id}: optimized_from '{o.optimized_from}' "
                                f"is not the baseline of {ev.ev_id}")
    rep.add("structure", "EV configurations linked correctly",
            "fail" if problems else "pass", "; ".join(problems) or "Each EV has one baseline and linked optimized design(s).")

    # 3. conditions
    cids = [c.condition_id for c in cat.conditions]
    dup = [k for k, v in Counter(cids).items() if v > 1]
    unknown = [c for c in cids if c not in KNOWN_CONDITIONS]
    bad_link = [f"{c.condition_id}.benchmark_of='{c.benchmark_of}'" for c in cat.conditions
                if c.benchmark_of and (c.benchmark_of not in cids or c.group != "standard")]
    if dup:
        rep.add("conditions", "Driving conditions", "fail", f"duplicate condition_id: {dup}")
    elif bad_link:
        rep.add("conditions", "Driving conditions", "fail",
                f"benchmark_of must name a known driving condition, on a standard-cycle entry: {bad_link}")
    elif unknown:
        rep.add("conditions", "Driving conditions", "warn", f"Unrecognised condition ids: {unknown}")
    else:
        rep.add("conditions", "Driving conditions", "pass",
                f"{len(cids)} conditions listed: "
                f"{len(store.available_conditions())} with results, "
                f"{len(store.coming_soon_conditions())} coming soon.")

    # 4. completeness of the results grid (+ duplicates)
    #    baseline: one row under every available condition.
    #    optimized design: one row under the condition it was made for (every condition if none is named).
    miss, dups = [], []
    for ev in cat.evs:
        for cond in store.available_conditions():
            rows = store.prediction(ev.ev_id, cond.condition_id).results
            counts = Counter(r.config_id for r in rows)
            for c in ev.configurations:
                expected = (c.type == "baseline" or not c.optimized_for
                            or c.optimized_for == cond.condition_id)
                if expected and counts.get(c.config_id, 0) == 0:
                    miss.append(f"{c.config_id}@{cond.condition_id}")
                elif counts.get(c.config_id, 0) > 1:
                    dups.append(f"{c.config_id}@{cond.condition_id}")
    for ev in cat.evs:
        for c in ev.configurations:
            if c.type == "optimized" and c.optimized_for and store.is_available(c.optimized_for) is False:
                miss.append(f"{c.config_id}@{c.optimized_for} (condition has no results)")
    if miss or dups:
        rep.add("completeness", "Results grid complete", "fail",
                f"missing: {miss}; duplicates: {dups}")
    else:
        rep.add("completeness", "Results grid complete", "pass",
                "The baseline has a result row for each available condition, and each optimized design "
                "has one for the condition it was made for.")

    # 5. valid numbers: finite and above zero (a fraction must be between 0 and 1)
    bad = []
    for ev in cat.evs:
        for key in ("mass_kg", "usable_battery_kwh"):
            v = getattr(ev, key)
            if not _finite(v) or v <= 0:
                bad.append(f"{ev.ev_id}.{key}={v}")
        for c in ev.configurations:
            for key in ("cd", "frontal_area_m2"):
                v = getattr(c, key)
                if not _finite(v) or v <= 0:
                    bad.append(f"{c.config_id}.{key}={v}")
    for (ev_id, cid), p in store.predictions.items():
        for r in p.results:
            for key in ("energy_wh_per_km", "range_km", "cost_lkr_per_100km"):
                v = getattr(r, key)
                if not _finite(v) or v <= 0:
                    bad.append(f"{r.config_id}@{cid}.{key}={v}")
    for c in cat.conditions:
        if c.idle_fraction is not None and not (_finite(c.idle_fraction) and 0 <= c.idle_fraction <= 1):
            bad.append(f"{c.condition_id}.idle_fraction={c.idle_fraction}")
    rep.add("ranges", "Numbers are valid", "fail" if bad else "pass",
            ", ".join(bad) or "All values are finite and above zero; idle fractions are between 0 and 1.")

    # 6. range = battery / energy
    bad = []
    batt = {ev.ev_id: ev.usable_battery_kwh for ev in cat.evs}
    for (ev_id, cid), p in store.predictions.items():
        for r in p.results:
            expected = batt[ev_id] * 1000 / r.energy_wh_per_km
            if abs(r.range_km - expected) / expected > settings.range_tolerance:
                bad.append(f"{r.config_id}@{cid}: range {r.range_km} vs {expected:.1f}")
    rep.add("range_check", "Range matches battery / energy", "fail" if bad else "pass",
            "; ".join(bad) or f"Within {settings.range_tolerance:.0%} for every row.")

    # 7. cost = energy x tariff / efficiency
    n = cat.model_notes
    if n.tariff_lkr_per_kwh is None or n.charging_efficiency is None:
        rep.add("cost_check", "Cost matches energy x tariff", "warn",
                "Tariff or charging efficiency not in model notes; cost could not be verified.")
    else:
        bad = []
        for (ev_id, cid), p in store.predictions.items():
            for r in p.results:
                expected = r.energy_wh_per_km / 1000 * 100 * n.tariff_lkr_per_kwh / n.charging_efficiency
                if abs(r.cost_lkr_per_100km - expected) / expected > settings.cost_tolerance:
                    bad.append(f"{r.config_id}@{cid}: cost {r.cost_lkr_per_100km} vs {expected:.2f}")
        rep.add("cost_check", "Cost matches energy x tariff", "fail" if bad else "pass",
                "; ".join(bad) or f"Within {settings.cost_tolerance:.0%} for every row "
                                  f"(tariff {n.tariff_lkr_per_kwh} LKR/kWh, efficiency {n.charging_efficiency}).")

    # 8. shape parameters inside the declared limits
    defs = {d.name: d for d in cat.shape_parameter_definitions}
    bad = []
    for ev in cat.evs:
        for c in ev.configurations:
            for name, v in c.shape_parameters.items():
                d = defs.get(name)
                if d is None:
                    bad.append(f"{c.config_id}: '{name}' has no definition")
                elif not d.min <= v <= d.max:
                    bad.append(f"{c.config_id}: {name}={v} outside [{d.min}, {d.max}]")
    rep.add("shape_params", "Shape parameters inside limits", "fail" if bad else "pass",
            "; ".join(bad) or "All shape parameter values are defined and within min/max.")

    # 9. optimized should not be worse than baseline (C3 promised savings are never negative)
    worse = [f"{p['optimized_config_id']}@{p['condition_id']}" for p in store.pairs()
             if p["opt"].energy_wh_per_km > p["base"].energy_wh_per_km]
    rep.add("not_worse", "Optimized not worse than baseline", "warn" if worse else "pass",
            f"Optimized uses more energy than baseline for: {worse}" if worse
            else "Optimized energy is lower than baseline in every pair.")

    # 10. cross-check C4's savings against C3's own comparison rows
    bad, checked = [], 0
    for p in store.pairs():
        c3 = next((c for c in store.prediction(p["ev_id"], p["condition_id"]).comparisons
                   if c.optimized_config_id == p["optimized_config_id"]), None)
        if c3 is None:
            continue
        checked += 1
        s = savings(p["base"].energy_wh_per_km, p["opt"].energy_wh_per_km,
                    p["base"].range_km, p["opt"].range_km,
                    p["base"].cost_lkr_per_100km, p["opt"].cost_lkr_per_100km)
        tol = settings.crosscheck_tolerance
        if (abs(c3.energy_saving_wh_per_km - s["energy_saving_wh_per_km"]) > tol
                or abs(c3.range_gain_km - s["range_gain_km"]) > tol
                or abs(c3.cost_saving_lkr_per_100km - s["cost_saving_lkr_per_100km"]) > tol):
            bad.append(f"{p['optimized_config_id']}@{p['condition_id']}")
    if checked == 0:
        rep.add("c3_crosscheck", "Savings agree with C3's own", "warn", "C3 sent no comparison rows to check against.")
    else:
        rep.add("c3_crosscheck", "Savings agree with C3's own", "fail" if bad else "pass",
                f"Differs for: {bad}" if bad else f"C4's calculation matches C3 for all {checked} comparison rows.")

    # 11. model notes present / predicted-only reminder
    missing = [k for k in ("energy_definition", "range_definition", "includes_regen_braking",
                           "road_gradient", "cd_source") if getattr(n, k) in (None, "")]
    rep.add("model_notes", "Model notes provided", "warn" if missing else "pass",
            f"Missing: {missing}" if missing else "Model notes are provided.")
    rep.add("validated", "Predicted vs experimentally validated", "pass",
            "Results are experimentally validated." if n.experimentally_validated else
            "Results are PREDICTED only (not experimentally validated) and are labelled that way.")
    return rep.to_dict()