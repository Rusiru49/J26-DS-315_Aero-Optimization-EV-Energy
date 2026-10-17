"""
Unit tests for workflow2/energy_model.py, targeting the class of bug that runs fine
and still produces a plausible-looking wrong number: sign errors in regen,
a unit mix-up in speed, or auxiliary load being incorrectly offset by regen.
"""

import numpy as np
import pandas as pd
import pytest

from component3 import config
from component3.workflow2.energy_model import simulate_cycle


def _constant_speed_cycle(speed_kmh=60.0, duration_s=60):
    t = np.arange(0, duration_s + 1, 1.0)
    v = np.full_like(t, speed_kmh)
    return pd.DataFrame({"time_s": t, "speed_kmh": v})


def _hard_braking_to_stop_cycle(start_kmh=100.0, brake_duration_s=5):
    """Cruise briefly, then decelerate hard to a full stop."""
    cruise_t = np.arange(0, 10, 1.0)
    cruise_v = np.full_like(cruise_t, start_kmh)

    brake_t = np.arange(10, 10 + brake_duration_s + 1, 1.0)
    brake_v = np.linspace(start_kmh, 0.0, brake_duration_s + 1)

    t = np.concatenate([cruise_t, brake_t])
    v = np.concatenate([cruise_v, brake_v])
    return pd.DataFrame({"time_s": t, "speed_kmh": v})


def test_constant_speed_never_shows_regen():
    """At constant speed, F_accel = 0, so F_prop = F_aero + F_roll > 0
    throughout — there is no braking event, so the model must never report
    a negative (regen) power draw during this segment."""
    cycle = _constant_speed_cycle()
    result = simulate_cycle(cycle, cd=config.BASELINE_CD)
    trace = result["trace"]
    assert (trace["p_battery_w"] >= 0).all()
    assert (trace["p_wheel_w"] >= 0).all()


def test_hard_braking_shows_capped_regen():
    """During the braking segment, some energy should be recovered (net
    negative traction power), but never more than the regen efficiency
    fraction of the kinetic energy actually lost — i.e. regen must be
    capped, not treated as free/perfect energy recovery."""
    cycle = _hard_braking_to_stop_cycle()
    result = simulate_cycle(cycle, cd=config.BASELINE_CD)
    trace = result["trace"]

    braking_rows = trace[trace["time_s"] > 10]
    assert (braking_rows["f_accel_n"] < 0).any(), "test cycle should actually decelerate"

    recovering_rows = braking_rows[braking_rows["p_wheel_w"] < 0]
    assert len(recovering_rows) > 0, "hard braking should trigger at least one regen step"
    # Regen efficiency must actually reduce the recovered magnitude relative
    # to the raw wheel power — i.e. p_battery_w (during regen) is smaller in
    # magnitude than p_wheel_w, never equal or larger.
    assert (recovering_rows["p_battery_w"].abs() < recovering_rows["p_wheel_w"].abs()).all()


def test_auxiliary_load_never_offset_by_regen():
    """The auxiliary load must be added in full every second regardless of
    whether that second was net traction draw or net regen — total_aux_wh
    must equal AUX_LOAD_W integrated over the cycle duration exactly, not a
    smaller number because some seconds were regenerating."""
    cycle = _hard_braking_to_stop_cycle()
    result = simulate_cycle(cycle, cd=config.BASELINE_CD)

    duration_s = cycle["time_s"].iloc[-1] - cycle["time_s"].iloc[0]
    expected_aux_wh = config.AUX_LOAD_W * duration_s / 3600.0
    assert result["total_aux_wh"] == pytest.approx(expected_aux_wh, rel=1e-9)


def test_speed_conversion_km_to_ms():
    """60 km/h must be treated as 60/3.6 m/s, not 60 m/s — sanity-checked
    indirectly via the aero drag force at a known constant speed."""
    cycle = _constant_speed_cycle(speed_kmh=60.0, duration_s=2)
    result = simulate_cycle(cycle, cd=config.BASELINE_CD)
    trace = result["trace"]

    expected_v_ms = 60.0 / 3.6
    expected_f_aero = (
        0.5 * config.AIR_DENSITY_KG_M3 * config.BASELINE_CD * config.FRONTAL_AREA_M2 * expected_v_ms ** 2
    )
    assert trace["f_aero_n"].iloc[0] == pytest.approx(expected_f_aero, rel=1e-6)


def test_wh_per_km_is_plausible_for_a_compact_ev():
    """Sanity check against published compact-EV efficiency figures
    (roughly 120-220 Wh/km is a reasonable real-world band) rather than an
    exact target — this test exists to catch gross unit errors (e.g. an
    order-of-magnitude mistake), not to validate precise accuracy."""
    cycle = _constant_speed_cycle(speed_kmh=80.0, duration_s=120)
    result = simulate_cycle(cycle, cd=config.BASELINE_CD)
    assert 50.0 < result["wh_per_km"] < 400.0


def test_expressway_cycle_uses_more_energy_per_km_than_urban_stop_start():
    """Aero drag dominates at higher, steadier speed; this is the physical
    basis for the whole project's cycle-aware optimisation claim, so it's
    worth a direct regression test rather than assuming the physics is
    self-evidently correct."""
    from component3.workflow2.energy_model import evaluate_cd_on_cycle

    expressway = evaluate_cd_on_cycle("expressway", cd=config.BASELINE_CD)
    urban = evaluate_cd_on_cycle("urban", cd=config.BASELINE_CD)
    assert expressway["wh_per_km"] > urban["wh_per_km"]
