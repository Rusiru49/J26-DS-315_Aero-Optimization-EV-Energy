"""Generate the DUMMY C3 data package used while real C3 data is not available.

Format = the format agreed with C3 (docs/C4_Request_to_C3_Data_Package.md).
Scope for PP1: one sedan, one baseline + one optimized design for each of Urban and Expressway,
  - SLDC_Urban        baseline + its optimized design
  - SLDC_Expressway   baseline + its optimized design
  - WLTP_Urban        standard-cycle result of the baseline, for Urban      (baseline only)
  - WLTP_Expressway   standard-cycle result of the baseline, for Expressway (baseline only)
The other three SLDC conditions are listed but have no results ("coming soon").

Run:  python scripts/generate_dummy_data.py
"""
import datetime
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "c3_package_dummy.json"

TARIFF_LKR_PER_KWH = 45.0   # placeholder (same as C3's placeholder)
CHARGING_EFF = 0.90

EV = {
    "ev_id": "EV01",
    "ev_model": "Sedan (dummy, BYD Seal-based body)",
    "mass_kg": 2000,
    "usable_battery_kwh": 80.0,
}

BASE = {"cd": 0.230, "cd_uncertainty": 0.002, "frontal_area_m2": 2.30}

BASE_SHAPE = {
    "windshield_angle_deg": 28,
    "hood_angle_deg": 10,
    "spoiler_angle_deg": 10,
    "boat_tail_angle_deg": 15,
    "ground_clearance_mm": 150,
}

# One optimized design for each SLDC driving condition that has its own design (Urban, Expressway).
# A design has results only under the condition it was made for.
# WLTP is the benchmark: it has no design of its own; only the baseline has a WLTP result.
# config_id, optimized_for, cd, cd_uncertainty, frontal_area, shape parameters
DESIGNS = [
    ("EV01_OPT1", "SLDC_Urban", 0.208, 0.004, 2.28,
     {"windshield_angle_deg": 25, "hood_angle_deg": 8, "spoiler_angle_deg": 14,
      "boat_tail_angle_deg": 19, "ground_clearance_mm": 130}),
    ("EV01_OPT2", "SLDC_Expressway", 0.201, 0.005, 2.28,
     {"windshield_angle_deg": 23, "hood_angle_deg": 7, "spoiler_angle_deg": 16,
      "boat_tail_angle_deg": 22, "ground_clearance_mm": 120}),
]

SHAPE_DEFS = [
    ("windshield_angle_deg", "deg", 22, 34, 28,
     "Windscreen glass angle, measured from horizontal; a lower value means a more sloped windscreen"),
    ("hood_angle_deg", "deg", 5, 15, 10,
     "Bonnet (hood) slope, measured from horizontal"),
    ("spoiler_angle_deg", "deg", 0, 20, 10,
     "Rear spoiler plate pitch, measured from horizontal; 0 means flush with the body"),
    ("boat_tail_angle_deg", "deg", 5, 25, 15,
     "Taper of the rear body, measured from horizontal; a higher value means the rear narrows more towards the back"),
    ("ground_clearance_mm", "mm", 110, 190, 150,
     "Vertical gap from the ground to the lowest point of the underbody"),
]

# condition_id, group, avg_speed, max_speed, idle_fraction, distance_km
# The two standard-cycle entries each say which driving condition they are the benchmark of.
CONDITIONS = [
    ("WLTP_Urban", "standard", 46.5, 131.3, 0.13, 23.27),
    ("WLTP_Expressway", "standard", 46.5, 131.3, 0.13, 23.27),
    ("SLDC_Urban", "sldc", 21.8, 58.0, 0.24, 9.6),
    ("SLDC_Suburban", "sldc", None, None, None, None),
    ("SLDC_Intercity", "sldc", None, None, None, None),
    ("SLDC_Expressway", "sldc", 94.5, 118.0, 0.01, 26.4),
    ("SLDC_Rural", "sldc", None, None, None, None),
]

# Dummy baseline energy (Wh/km) and the share of that energy that is aerodynamic drag.
# Drag share grows with speed. Values are invented but realistic in size.
BASE_ENERGY = {"WLTP_Urban": 152.0, "WLTP_Expressway": 171.0,
               "SLDC_Urban": 138.0, "SLDC_Expressway": 205.0}
DRAG_SHARE = {"SLDC_Urban": 0.27, "SLDC_Expressway": 0.58}
BENCHMARK_OF = {"WLTP_Urban": "SLDC_Urban", "WLTP_Expressway": "SLDC_Expressway"}


def build() -> dict:
    def row(cfg_id, cond, energy):
        energy = round(energy, 1)
        rng = round(EV["usable_battery_kwh"] * 1000 / energy, 1)
        cost = round(energy / 1000 * 100 * TARIFF_LKR_PER_KWH / CHARGING_EFF, 2)
        return {
            "ev_id": EV["ev_id"], "config_id": cfg_id, "condition_id": cond,
            "energy_wh_per_km": energy, "range_km": rng, "cost_lkr_per_100km": cost,
        }

    results, comparisons = [], []
    base_cda = BASE["cd"] * BASE["frontal_area_m2"]
    for cond in BASE_ENERGY:
        eb = BASE_ENERGY[cond]
        rb = row("EV01_BASE", cond, eb)
        results.append(rb)
        # each optimized design is evaluated only under the condition it was made for.
        # The standard-cycle entries (WLTP) have the baseline only (for now).
        for (cfg_id, _for, cd, unc, area, _shape) in DESIGNS:
            if _for != cond:
                continue
            cda_ratio = (cd * area) / base_cda
            eo = eb * (1 - DRAG_SHARE[cond] * (1 - cda_ratio))
            ro = row(cfg_id, cond, eo)
            results.append(ro)
            comparisons.append({
                "ev_id": EV["ev_id"], "baseline_config_id": "EV01_BASE",
                "optimized_config_id": cfg_id, "condition_id": cond,
                "energy_saving_wh_per_km": round(rb["energy_wh_per_km"] - ro["energy_wh_per_km"], 2),
                "energy_saving_pct": round((rb["energy_wh_per_km"] - ro["energy_wh_per_km"])
                                           / rb["energy_wh_per_km"] * 100, 2),
                "range_gain_km": round(ro["range_km"] - rb["range_km"], 1),
                "cost_saving_lkr_per_100km": round(rb["cost_lkr_per_100km"] - ro["cost_lkr_per_100km"], 2),
                "saving_within_uncertainty": abs(BASE["cd"] - cd) < unc,
            })

    configurations = [
        {"config_id": "EV01_BASE", "type": "baseline", "optimized_from": None,
         "optimized_for": None,
         "frontal_area_m2": BASE["frontal_area_m2"], "cd": BASE["cd"],
         "cd_uncertainty": BASE["cd_uncertainty"],
         "shape_parameters": BASE_SHAPE, "geometry_file": "EV01_BASE.glb"},
    ] + [
        {"config_id": cfg_id, "type": "optimized", "optimized_from": "EV01_BASE",
         "optimized_for": _for,
         "frontal_area_m2": area, "cd": cd, "cd_uncertainty": unc,
         "shape_parameters": shape, "geometry_file": f"{cfg_id}.glb"}
        for (cfg_id, _for, cd, unc, area, shape) in DESIGNS
    ]

    return {
        "package_version": "0.1-dummy",
        "generated_on": datetime.date.today().isoformat(),
        "status": "draft",
        "evs": [{
            **EV,
            "configurations": configurations,
        }],
        "shape_parameter_definitions": [
            {"name": n, "description": d, "unit": u, "min": lo, "max": hi,
             "baseline": b, "effect_on_geometry": d}
            for (n, u, lo, hi, b, d) in SHAPE_DEFS
        ],
        "conditions": [
            {"condition_id": c, "group": g, "avg_speed_kmh": a, "max_speed_kmh": m,
             "idle_fraction": i, "distance_km": d, "benchmark_of": BENCHMARK_OF.get(c)}
            for (c, g, a, m, i, d) in CONDITIONS
        ],
        "results": results,
        "comparisons": comparisons,
        "model_notes": {
            "is_dummy": True,
            "data_origin": "DUMMY - generated by scripts/generate_dummy_data.py (not real C3 results)",
            "energy_definition": "Energy taken from the battery per km over the full cycle, including drivetrain losses (dummy).",
            "range_definition": "usable_battery_kwh x 1000 / energy_wh_per_km",
            "includes_regen_braking": True,
            "includes_auxiliary_load": False,
            "road_gradient": "flat",
            "cd_source": "CFD, 100 km/h, time-averaged (dummy)",
            "optimized_cd_method": "baseline Cd + predicted change",
            "tariff_lkr_per_kwh": TARIFF_LKR_PER_KWH,
            "tariff_source": "placeholder - real CEB tariff and source to be confirmed",
            "charging_efficiency": CHARGING_EFF,
            "experimentally_validated": False,
        },
        "units": {
            "mass_kg": "kg", "usable_battery_kwh": "kWh", "frontal_area_m2": "m2",
            "cd": "dimensionless", "energy_wh_per_km": "Wh/km", "range_km": "km",
            "cost_lkr_per_100km": "LKR/100km", "idle_fraction": "fraction 0-1",
        },
    }


if __name__ == "__main__":
    pkg = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(pkg, indent=2), encoding="utf-8")
    print(f"written {OUT}")
    for c in pkg["comparisons"]:
        print(c["optimized_config_id"], c["condition_id"], c["energy_saving_pct"], "%", c["range_gain_km"], "km",
              c["cost_saving_lkr_per_100km"], "LKR/100km")