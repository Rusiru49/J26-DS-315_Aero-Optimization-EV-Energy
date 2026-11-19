"""
Phase 1 deliverable: run the vehicle energy model against every real
driving-cycle CSV Member 1's pipeline has produced, using the baseline Cd
and placeholder vehicle constants from config.py.

Usage:
    python scripts/run_group2_demo.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from component3 import config
from component3.workflow2.energy_model import evaluate_cd_on_cycle


def main():
    print(f"Baseline Cd = {config.BASELINE_CD}, mass = {config.VEHICLE_MASS_KG} kg, "
          f"battery = {config.BATTERY_CAPACITY_KWH} kWh\n")

    header = f"{'Cycle':<12}{'Distance (km)':>15}{'Wh/km':>10}{'Range (km)':>12}{'Cost (LKR)':>12}"
    print(header)
    print("-" * len(header))

    results = {}
    for cycle_name in config.DRIVING_CYCLE_FILES:
        result = evaluate_cd_on_cycle(cycle_name, cd=config.BASELINE_CD)
        results[cycle_name] = result
        print(
            f"{cycle_name:<12}{result['distance_km']:>15.2f}"
            f"{result['wh_per_km']:>10.1f}{result['range_km']:>12.1f}"
            f"{result['cost_lkr']:>12.2f}"
        )

    # Total Wh/km also depends on the mass (a heavy car pays for stop-start
    # acceleration), so the aero check compares the cost of a 10% higher Cd.
    drag_cost = {c: evaluate_cd_on_cycle(c, cd=1.1 * config.BASELINE_CD)["wh_per_km"] - results[c]["wh_per_km"]
                 for c in ("urban", "expressway")}
    assert drag_cost["expressway"] > drag_cost["urban"], (
        "drag should cost more energy per km on the expressway than in urban driving"
    )
    print(f"\nOK: +10% Cd costs {drag_cost['expressway']:.2f} Wh/km on the expressway and "
          f"{drag_cost['urban']:.2f} Wh/km in urban driving, as expected.")


if __name__ == "__main__":
    main()
