"""
The energy model was rewritten from a Python loop to NumPy arrays for speed.
This test keeps the ORIGINAL loop as a reference and checks that the fast
version gives the same numbers on every real driving cycle, for several Cd
values, and on a cycle with a bad (non-increasing) timestamp.
"""

import numpy as np
import pandas as pd
import pytest

import config
from energy_model import _instantaneous_forces, load_driving_cycle, prepare_cycle, simulate_cycle


def reference_simulate_cycle(cycle_df, cd, mass_kg=config.VEHICLE_MASS_KG, frontal_area_m2=config.FRONTAL_AREA_M2,
                             battery_kwh=config.BATTERY_CAPACITY_KWH, tariff=config.ELECTRICITY_TARIFF_LKR_PER_KWH):
    """The original, slow, easy-to-read loop (kept only to prove equivalence)."""
    t = cycle_df["time_s"].to_numpy(dtype=float)
    v_ms = cycle_df["speed_kmh"].to_numpy(dtype=float) / 3.6
    traction = aux = dist = 0.0
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        if dt <= 0:
            continue
        v_now, v_avg, accel = v_ms[i], (v_ms[i - 1] + v_ms[i]) / 2.0, (v_ms[i] - v_ms[i - 1]) / dt
        _, _, _, f_prop = _instantaneous_forces(v_now, accel, cd, frontal_area_m2, mass_kg)
        p_wheel = f_prop * v_now
        p_batt = p_wheel / config.DRIVETRAIN_EFFICIENCY if p_wheel >= 0 else p_wheel * config.REGEN_EFFICIENCY
        traction += p_batt * dt / 3600.0
        aux += config.AUX_LOAD_W * dt / 3600.0
        dist += v_avg * dt
    total = traction + aux
    km = dist / 1000.0
    wh_km = total / km
    return {"total_wh": total, "distance_km": km, "wh_per_km": wh_km,
            "range_km": battery_kwh * 1000.0 / wh_km, "cost_lkr": total / 1000.0 * tariff}


@pytest.mark.parametrize("cycle", sorted(config.DRIVING_CYCLE_FILES))
@pytest.mark.parametrize("cd", [0.18, 0.25, 0.32])
def test_fast_version_matches_the_original_loop(cycle, cd):
    df = load_driving_cycle(cycle)
    fast, ref = simulate_cycle(df, cd), reference_simulate_cycle(df, cd)
    for key in ("total_wh", "distance_km", "wh_per_km", "range_km", "cost_lkr"):
        assert fast[key] == pytest.approx(ref[key], rel=1e-9), key


def test_prepared_cycle_gives_the_same_answer_and_can_skip_the_trace():
    df = load_driving_cycle("urban")
    full = simulate_cycle(df, 0.25)
    light = simulate_cycle(prepare_cycle(df), 0.25, return_trace=False)
    assert "trace" not in light and "trace" in full
    assert light["wh_per_km"] == pytest.approx(full["wh_per_km"], rel=1e-12)


def test_a_non_increasing_timestamp_is_skipped_like_before():
    df = pd.DataFrame({"time_s": [0, 1, 2, 2, 3, 4], "speed_kmh": [0, 20, 40, 41, 45, 50]})
    fast, ref = simulate_cycle(df, 0.25), reference_simulate_cycle(df, 0.25)
    assert fast["total_wh"] == pytest.approx(ref["total_wh"], rel=1e-9)
    assert fast["distance_km"] == pytest.approx(ref["distance_km"], rel=1e-9)
