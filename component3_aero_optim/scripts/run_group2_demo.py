"""
Phase 1 deliverable: run the vehicle energy model against every real
driving-cycle CSV Member 1's pipeline has produced, using the baseline Cd
and placeholder vehicle constants from config.py.

Usage:
    python scripts/run_group2_demo.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from energy_model import evaluate_cd_on_cycle


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

    assert results["expressway"]["wh_per_km"] > results["urban"]["wh_per_km"], (
        "expressway should consume more Wh/km than urban (aero dominance at speed)"
    )
    print("\nOK: expressway consumes more Wh/km than urban, as expected.")


if __name__ == "__main__":
    main()
