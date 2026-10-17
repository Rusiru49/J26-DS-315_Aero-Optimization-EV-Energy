"""
Vehicle Energy Model — pure physics, no ML.

Converts a drag coefficient and a driving cycle (a speed-vs-time table) into
energy consumption, driving range, and running cost for that cycle. This is
Group 2 / Step 5 of the pipeline and has no dependency on the drag surrogate,
POD, or any trained model — it only needs a Cd value (baseline or optimised)
and a cycle to evaluate against.

Physics, per second of the cycle:
    F_aero  = 0.5 * rho * Cd * A * v^2                 (always resistive)
    F_roll  = C_rr * m * g                             (always resistive, v>0)
    F_accel = m * (dv/dt)                              (signed)
    F_prop  = F_aero + F_roll + F_accel                (net force the wheels
                                                         must supply)

If F_prop >= 0, the motor draws power from the battery, penalised by
drivetrain losses. If F_prop < 0, that magnitude is a net braking force;
regenerative braking recovers a fraction of it (not all of it — friction
braking / regen limits absorb the rest, not modelled further here). This
formulation means resistive losses (aero + rolling) automatically reduce how
much of any given deceleration is recoverable, without a separate branch for
it — it falls out of F_prop's sign and magnitude on its own.

A constant auxiliary load (AC, lights, electronics) is added on top of the
traction energy every second, regardless of whether that second was net
draw or net regen — it is never offset by regen.
"""

import numpy as np
import pandas as pd

from component3 import config
# ------------------------------------------------------------------ #
# 1. LOADING A DRIVING CYCLE
# ------------------------------------------------------------------ #
def load_driving_cycle(cycle_name):
    """
    Load one of Member 1's real driving-cycle CSVs (columns: time_s,
    speed_kmh) by name, e.g. "urban", "expressway". Raises a clear error if
    the name isn't one of the five produced by pipeline.py.
    """
    if cycle_name not in config.DRIVING_CYCLE_FILES:
        raise ValueError(
            f"Unknown driving cycle '{cycle_name}'. Expected one of: "
            f"{sorted(config.DRIVING_CYCLE_FILES)}"
        )
    path = config.DRIVING_CYCLE_FILES[cycle_name]
    df = pd.read_csv(path)
    if not {"time_s", "speed_kmh"}.issubset(df.columns):
        raise ValueError(f"{path} is missing expected columns time_s/speed_kmh")
    return df.sort_values("time_s").reset_index(drop=True)


# ------------------------------------------------------------------ #
# 2. PER-SECOND PHYSICS
# ------------------------------------------------------------------ #
def _instantaneous_forces(v_ms, accel_ms2, cd, frontal_area_m2, mass_kg):
    """
    Return (F_aero, F_roll, F_accel, F_prop) in Newtons for one instant.
    F_aero and F_roll are always >= 0 (resistive); F_accel and F_prop carry
    sign.
    """
    f_aero = 0.5 * config.AIR_DENSITY_KG_M3 * cd * frontal_area_m2 * v_ms ** 2
    f_roll = config.ROLLING_RESISTANCE_COEFF * mass_kg * config.GRAVITY_M_S2 if v_ms > 0 else 0.0
    f_accel = mass_kg * accel_ms2
    f_prop = f_aero + f_roll + f_accel
    return f_aero, f_roll, f_accel, f_prop


def prepare_cycle(cycle_df):
    """
    Everything about a driving cycle that does NOT depend on the shape: time
    steps, speeds in m/s, accelerations, average speeds and distance. The
    optimiser calls simulate_cycle() about 100 times on the same cycle, so it
    prepares the cycle once and passes this dict in place of the table.
    Intervals with a non-positive time step (duplicate or out-of-order
    timestamps in real logs) are dropped, never divided by.
    """
    if len(cycle_df) < 2:
        raise ValueError("simulate_cycle needs at least 2 rows to compute dt/accel")
    t = cycle_df["time_s"].to_numpy(dtype=float)
    v_kmh = cycle_df["speed_kmh"].to_numpy(dtype=float)
    v_ms = v_kmh / 3.6  # km/h -> m/s, once, at ingestion

    dt = np.diff(t)
    keep = dt > 0
    dt = dt[keep]
    dv = np.diff(v_ms)[keep]
    v_now = v_ms[1:][keep]
    return {
        "time_s": t[1:][keep],
        "speed_kmh": v_kmh[1:][keep],
        "dt": dt,
        "v_now": v_now,
        "accel": dv / dt,
        "distance_m": float(np.sum((v_ms[:-1][keep] + v_now) / 2.0 * dt)),
    }


def simulate_cycle(
    cycle_df,
    cd,
    mass_kg=config.VEHICLE_MASS_KG,
    frontal_area_m2=config.FRONTAL_AREA_M2,
    battery_kwh=config.BATTERY_CAPACITY_KWH,
    tariff_lkr_per_kwh=config.ELECTRICITY_TARIFF_LKR_PER_KWH,
    return_trace=True,
):
    """
    Integrate the per-second physics above over a full driving cycle, with
    NumPy arrays (one pass over the whole cycle, no Python loop).

    cycle_df is a table with columns time_s, speed_kmh (from
    load_driving_cycle()) or the dict from prepare_cycle(). Returns a dict
    with total_wh, distance_km, wh_per_km, range_km, cost_lkr and, if
    return_trace, a per-second trace DataFrame (for plotting and debugging).
    The optimiser passes return_trace=False because it only needs the summary.
    """
    c = cycle_df if isinstance(cycle_df, dict) else prepare_cycle(cycle_df)
    dt, v, accel = c["dt"], c["v_now"], c["accel"]

    f_aero = 0.5 * config.AIR_DENSITY_KG_M3 * cd * frontal_area_m2 * v ** 2
    f_roll = np.where(v > 0, config.ROLLING_RESISTANCE_COEFF * mass_kg * config.GRAVITY_M_S2, 0.0)
    f_accel = mass_kg * accel
    f_prop = f_aero + f_roll + f_accel
    p_wheel = f_prop * v  # signed wheel power, Watts

    # Driving: draw more from the battery than the wheels need (drivetrain
    # losses). Braking (negative power): recover only a fraction.
    p_battery = np.where(p_wheel >= 0, p_wheel / config.DRIVETRAIN_EFFICIENCY, p_wheel * config.REGEN_EFFICIENCY)
    traction_wh = p_battery * dt / 3600.0
    aux_wh = config.AUX_LOAD_W * dt / 3600.0  # always added, never offset by regen

    total_traction_wh = float(np.sum(traction_wh))
    total_aux_wh = float(np.sum(aux_wh))
    total_wh = total_traction_wh + total_aux_wh
    distance_km = c["distance_m"] / 1000.0
    wh_per_km = total_wh / distance_km if distance_km > 0 else float("nan")
    range_km = (battery_kwh * 1000.0) / wh_per_km if wh_per_km > 0 else float("nan")
    cost_lkr = (total_wh / 1000.0) * tariff_lkr_per_kwh

    result = {
        "cd": cd,
        "total_wh": total_wh,
        "total_traction_wh": total_traction_wh,
        "total_aux_wh": total_aux_wh,
        "distance_km": distance_km,
        "wh_per_km": wh_per_km,
        "range_km": range_km,
        "cost_lkr": cost_lkr,
    }
    if return_trace:
        result["trace"] = pd.DataFrame({
            "time_s": c["time_s"], "speed_kmh": c["speed_kmh"],
            "f_aero_n": f_aero, "f_roll_n": f_roll, "f_accel_n": f_accel, "f_prop_n": f_prop,
            "p_wheel_w": p_wheel, "p_battery_w": p_battery,
            "traction_wh_step": traction_wh, "aux_wh_step": aux_wh,
        })
    return result


# ------------------------------------------------------------------ #
# 3. CONVENIENCE: EVALUATE BY CYCLE NAME
# ------------------------------------------------------------------ #
def evaluate_cd_on_cycle(cycle_name, cd, **vehicle_kwargs):
    """Load a named cycle and simulate it for a given Cd in one call."""
    cycle_df = load_driving_cycle(cycle_name)
    return simulate_cycle(cycle_df, cd, **vehicle_kwargs)
