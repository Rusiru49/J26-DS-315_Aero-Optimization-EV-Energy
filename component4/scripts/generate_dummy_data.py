"""Generate the DUMMY C3 data package used while real C3 data is not available.

Format = the format agreed with C3 (docs/C4_Request_to_C3_Data_Package.md).
Scope for PP1: one sedan, baseline + one optimized design,
  - WLTP  (sample benchmark)
  - SLDC_Urban
  - SLDC_Expressway
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
OPT = {"cd": 0.208, "cd_uncertainty": 0.004, "frontal_area_m2": 2.28}

BASE_SHAPE = {
    "windshield_angle_deg": 28,
    "hood_angle_deg": 10,
    "spoiler_angle_deg": 10,
    "boat_tail_angle_deg": 15,
    "ground_clearance_mm": 150,
}
OPT_SHAPE = {
    "windshield_angle_deg": 25,
    "hood_angle_deg": 8,
    "spoiler_angle_deg": 14,
    "boat_tail_angle_deg": 19,
    "ground_clearance_mm": 130,
}

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
CONDITIONS = [
    ("WLTP", "standard", 46.5, 131.3, 0.13, 23.27),
    ("SLDC_Urban", "sldc", 21.8, 40.0, 0.24, 9.6),
    ("SLDC_Suburban", "sldc", None, None, None, None),
    ("SLDC_Intercity", "sldc", None, None, None, None),
    ("SLDC_Expressway", "sldc", 94.5, 100.0, 0.01, 26.4),
    ("SLDC_Rural", "sldc", None, None, None, None),
]

# Dummy baseline energy (Wh/km) and the share of that energy that is aerodynamic drag.
# Drag share grows with speed. Values are invented but realistic in size.
BASE_ENERGY = {"WLTP": 152.0, "SLDC_Urban": 138.0, "SLDC_Expressway": 205.0}
DRAG_SHARE = {"WLTP": 0.38, "SLDC_Urban": 0.27, "SLDC_Expressway": 0.58}


def build() -> dict:
    cda_ratio = (OPT["cd"] * OPT["frontal_area_m2"]) / (BASE["cd"] * BASE["frontal_area_m2"])

    def row(cfg_id, cond, energy):
        energy = round(energy, 1)
        rng = round(EV["usable_battery_kwh"] * 1000 / energy, 1)
        cost = round(energy / 1000 * 100 * TARIFF_LKR_PER_KWH / CHARGING_EFF, 2)
        return {
            "ev_id": EV["ev_id"], "config_id": cfg_id, "condition_id": cond,
            "energy_wh_per_km": energy, "range_km": rng, "cost_lkr_per_100km": cost,
        }

    results, comparisons = [], []
    for cond in BASE_ENERGY:
        eb = BASE_ENERGY[cond]
        eo = eb * (1 - DRAG_SHARE[cond] * (1 - cda_ratio))
        rb, ro = row("EV01_BASE", cond, eb), row("EV01_OPT1", cond, eo)
        results += [rb, ro]
        comparisons.append({
            "ev_id": EV["ev_id"], "baseline_config_id": "EV01_BASE",
            "optimized_config_id": "EV01_OPT1", "condition_id": cond,
            "energy_saving_wh_per_km": round(rb["energy_wh_per_km"] - ro["energy_wh_per_km"], 2),
            "energy_saving_pct": round((rb["energy_wh_per_km"] - ro["energy_wh_per_km"])
                                       / rb["energy_wh_per_km"] * 100, 2),
            "range_gain_km": round(ro["range_km"] - rb["range_km"], 1),
            "cost_saving_lkr_per_100km": round(rb["cost_lkr_per_100km"] - ro["cost_lkr_per_100km"], 2),
            "saving_within_uncertainty": abs(BASE["cd"] - OPT["cd"]) < OPT["cd_uncertainty"],
        })

    return {
        "package_version": "0.1-dummy",
        "generated_on": datetime.date.today().isoformat(),
        "status": "draft",
        "evs": [{
            **EV,
            "configurations": [
                {"config_id": "EV01_BASE", "type": "baseline", "optimized_from": None,
                 "frontal_area_m2": BASE["frontal_area_m2"], "cd": BASE["cd"],
                 "cd_uncertainty": BASE["cd_uncertainty"],
                 "shape_parameters": BASE_SHAPE, "geometry_file": "EV01_BASE.glb"},
                {"config_id": "EV01_OPT1", "type": "optimized", "optimized_from": "EV01_BASE",
                 "frontal_area_m2": OPT["frontal_area_m2"], "cd": OPT["cd"],
                 "cd_uncertainty": OPT["cd_uncertainty"],
                 "shape_parameters": OPT_SHAPE, "geometry_file": "EV01_OPT1.glb"},
            ],
        }],
        "shape_parameter_definitions": [
            {"name": n, "description": d, "unit": u, "min": lo, "max": hi,
             "baseline": b, "effect_on_geometry": d}
            for (n, u, lo, hi, b, d) in SHAPE_DEFS
        ],
        "conditions": [
            {"condition_id": c, "group": g, "avg_speed_kmh": a, "max_speed_kmh": m,
             "idle_fraction": i, "distance_km": d}
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
        print(c["condition_id"], c["energy_saving_pct"], "%", c["range_gain_km"], "km",
              c["cost_saving_lkr_per_100km"], "LKR/100km")