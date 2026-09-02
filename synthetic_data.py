"""
Synthetic GPS + OBD-II Data Generator
======================================
Simulates raw, imperfect real-world vehicle data (as if collected from a
phone GPS + an OBD-II dongle) for 5 road environment classes:
    Urban, Suburban, Intercity, Expressway, Rural

This stands in for "REAL-WORLD VEHICLE -> GPS / OBD-II" in the pipeline
diagram. Swap this module out later for real logged data with the same
column schema and everything downstream keeps working.

Design choices (why the data looks the way it does):
- Each road type has different target-speed statistics, stop frequency,
  and acceleration aggressiveness (this is what your K-Means step should
  later be able to tell apart).
- Speed is generated with an Ornstein-Uhlenbeck (mean-reverting random
  walk) process -> realistic smooth acceleration/deceleration, not just
  noise.
- GPS is derived from integrating speed + a slowly-varying heading, then
  corrupted with realistic GPS position noise and occasional dropout jumps.
- OBD-II channels (speed, RPM, throttle, engine load) are sampled at a
  DIFFERENT rate than GPS and are correlated with the "true" kinematics
  but have their own sensor noise -> this is what makes the
  DATA SYNCHRONIZATION step in your diagram necessary and non-trivial.
"""

import numpy as np
import pandas as pd

RNG_SEED = 42

# ---- Road-environment kinematic "personalities" -----------------------
# These parameters are the ground-truth behavioral signature per road
# class. The pipeline is NOT told these numbers -- it has to recover
# similar structure from the noisy signals (that's the point of the
# feature extraction + clustering stages).
ROAD_PROFILES = {
    "urban": dict(
        target_speed_kmh=25, speed_std_kmh=10, ou_theta=0.35,
        stop_prob_per_s=0.010, stop_duration_s=(8, 35),
        max_accel=1.8, max_decel=2.2, gps_noise_m=6, base_lat=6.9271, base_lon=79.8612,
    ),
    "suburban": dict(
        target_speed_kmh=45, speed_std_kmh=12, ou_theta=0.25,
        stop_prob_per_s=0.005, stop_duration_s=(5, 20),
        max_accel=1.6, max_decel=1.8, gps_noise_m=5, base_lat=6.9500, base_lon=79.9000,
    ),
    "intercity": dict(
        target_speed_kmh=70, speed_std_kmh=14, ou_theta=0.15,
        stop_prob_per_s=0.0015, stop_duration_s=(10, 40),
        max_accel=1.3, max_decel=1.5, gps_noise_m=4, base_lat=7.1000, base_lon=80.1000,
    ),
    "expressway": dict(
        target_speed_kmh=95, speed_std_kmh=8, ou_theta=0.10,
        stop_prob_per_s=0.0002, stop_duration_s=(15, 60),
        max_accel=1.1, max_decel=1.2, gps_noise_m=4, base_lat=6.8000, base_lon=80.0000,
    ),
    "rural": dict(
        target_speed_kmh=40, speed_std_kmh=18, ou_theta=0.20,
        stop_prob_per_s=0.003, stop_duration_s=(5, 15),
        max_accel=1.7, max_decel=1.9, gps_noise_m=8, base_lat=6.6000, base_lon=80.2000,
    ),
}

GPS_HZ = 1.0        # 1 sample / second (typical phone GPS)
OBD_HZ = 5.0         # 5 samples / second (typical OBD-II polling)
M_PER_DEG_LAT = 111_320.0


def _ou_speed_process(n_steps, dt, target_kmh, std_kmh, theta, max_accel, max_decel, rng):
    """Mean-reverting random walk for speed (m/s), clipped to realistic accel limits."""
    target_ms = target_kmh / 3.6
    std_ms = std_kmh / 3.6
    speed = np.zeros(n_steps)
    speed[0] = max(0.0, rng.normal(target_ms * 0.5, std_ms * 0.3))
    for t in range(1, n_steps):
        drift = theta * (target_ms - speed[t - 1]) * dt
        shock = rng.normal(0, std_ms * np.sqrt(dt) * 0.6)
        v = speed[t - 1] + drift + shock
        accel = (v - speed[t - 1]) / dt
        accel = np.clip(accel, -max_decel, max_accel)
        v = speed[t - 1] + accel * dt
        speed[t] = max(0.0, v)
    return speed


def _inject_stops(speed, dt, stop_prob_per_s, stop_duration_s, rng):
    """Randomly force the vehicle to decelerate to a stop and idle, then resume."""
    n = len(speed)
    t = 0
    while t < n:
        if rng.random() < stop_prob_per_s * dt and speed[t] > 3:
            decel_steps = int(speed[t] / (1.5 * dt))
            decel_steps = max(1, min(decel_steps, n - t - 1))
            ramp = np.linspace(speed[t], 0, decel_steps)
            speed[t:t + decel_steps] = np.minimum(speed[t:t + decel_steps], ramp)
            idle_len = int(rng.uniform(*stop_duration_s) / dt)
            idle_end = min(n, t + decel_steps + idle_len)
            speed[t + decel_steps:idle_end] = 0.0
            t = idle_end
        else:
            t += 1
    return speed


def _integrate_track(speed_ms, dt, base_lat, base_lon, rng):
    """Turn a speed profile into a lat/lon track via a slowly-wandering heading."""
    n = len(speed_ms)
    heading = np.zeros(n)
    heading[0] = rng.uniform(0, 2 * np.pi)
    for t in range(1, n):
        heading[t] = heading[t - 1] + rng.normal(0, 0.05)
    dx = speed_ms * np.cos(heading) * dt
    dy = speed_ms * np.sin(heading) * dt
    lat = base_lat + np.cumsum(dy) / M_PER_DEG_LAT
    lon = base_lon + np.cumsum(dx) / (M_PER_DEG_LAT * np.cos(np.radians(base_lat)))
    return lat, lon


def generate_trip(road_type, duration_min, trip_id, seed):
    """Generate one synthetic trip: raw GPS log + raw OBD-II log (unsynchronized)."""
    rng = np.random.default_rng(seed)
    prof = ROAD_PROFILES[road_type]
    duration_s = int(duration_min * 60)

    # --- ground-truth kinematics at high resolution (10 Hz) then resampled ---
    dt_true = 0.1
    n_true = int(duration_s / dt_true)
    speed_true = _ou_speed_process(
        n_true, dt_true, prof["target_speed_kmh"], prof["speed_std_kmh"],
        prof["ou_theta"], prof["max_accel"], prof["max_decel"], rng,
    )
    speed_true = _inject_stops(
        speed_true, dt_true, prof["stop_prob_per_s"], prof["stop_duration_s"], rng
    )
    lat_true, lon_true = _integrate_track(speed_true, dt_true, prof["base_lat"], prof["base_lon"], rng)
    t_true = np.arange(n_true) * dt_true

    start_time = pd.Timestamp("2026-06-01 08:00:00") + pd.Timedelta(minutes=trip_id * 37)

    # --- GPS log: 1 Hz, position noise + occasional dropout jump ---
    gps_idx = np.round(np.arange(0, duration_s, 1 / GPS_HZ) / dt_true).astype(int)
    gps_idx = gps_idx[gps_idx < n_true]
    gps_noise_deg = prof["gps_noise_m"] / M_PER_DEG_LAT
    lat_noisy = lat_true[gps_idx] + rng.normal(0, gps_noise_deg, len(gps_idx))
    lon_noisy = lon_true[gps_idx] + rng.normal(0, gps_noise_deg, len(gps_idx))
    # occasional GPS jump / dropout artifact (~0.3% of samples)
    dropout_mask = rng.random(len(gps_idx)) < 0.003
    lat_noisy[dropout_mask] += rng.normal(0, 0.002, dropout_mask.sum())
    lon_noisy[dropout_mask] += rng.normal(0, 0.002, dropout_mask.sum())
    gps_speed_noisy = speed_true[gps_idx] * 3.6 + rng.normal(0, 1.2, len(gps_idx))  # km/h, GPS-derived speed

    gps_df = pd.DataFrame({
        "trip_id": trip_id,
        "road_type_true": road_type,   # kept ONLY for later validation, pipeline won't "peek"
        "timestamp": start_time + pd.to_timedelta(gps_idx * dt_true, unit="s"),
        "lat": lat_noisy,
        "lon": lon_noisy,
        "gps_speed_kmh": np.clip(gps_speed_noisy, 0, None),
    })

    # --- OBD-II log: 5 Hz, own sensor noise, jitter in timestamps ---
    obd_idx = np.round(np.arange(0, duration_s, 1 / OBD_HZ) / dt_true).astype(int)
    obd_idx = obd_idx[obd_idx < n_true]
    obd_speed = speed_true[obd_idx] * 3.6 + rng.normal(0, 0.8, len(obd_idx))
    obd_speed = np.clip(obd_speed, 0, None)
    # crude RPM model: idle RPM + proportional to speed, with gear-like steps
    gear_ratio = 1800 / max(prof["target_speed_kmh"], 1)
    rpm = 800 + obd_speed * gear_ratio * rng.uniform(0.85, 1.15, len(obd_idx))
    rpm = np.where(obd_speed < 1, rng.normal(750, 40, len(obd_idx)), rpm)
    accel_est = np.gradient(speed_true[obd_idx], 1 / OBD_HZ)
    throttle = np.clip(20 + accel_est * 18 + rng.normal(0, 5, len(obd_idx)), 0, 100)
    engine_load = np.clip(throttle * 0.7 + rng.normal(0, 6, len(obd_idx)), 5, 100)
    # timestamp jitter to simulate imperfect OBD polling interval
    jitter = rng.normal(0, 0.03, len(obd_idx))
    obd_time = start_time + pd.to_timedelta(obd_idx * dt_true + jitter, unit="s")

    obd_df = pd.DataFrame({
        "trip_id": trip_id,
        "timestamp": obd_time,
        "obd_speed_kmh": obd_speed,
        "rpm": rpm,
        "throttle_pct": throttle,
        "engine_load_pct": engine_load,
    })

    return gps_df, obd_df


def generate_dataset(trips_per_class=4, duration_min_range=(10, 18), seed=RNG_SEED, out_dir="data"):
    """Generate the full raw synthetic dataset across all 5 road classes."""
    rng = np.random.default_rng(seed)
    all_gps, all_obd = [], []
    trip_id = 0
    for road_type in ROAD_PROFILES:
        for _ in range(trips_per_class):
            duration = rng.uniform(*duration_min_range)
            gps_df, obd_df = generate_trip(road_type, duration, trip_id, seed=seed + trip_id)
            all_gps.append(gps_df)
            all_obd.append(obd_df)
            trip_id += 1

    gps_all = pd.concat(all_gps, ignore_index=True)
    obd_all = pd.concat(all_obd, ignore_index=True)

    gps_all.to_csv(f"{out_dir}/raw_gps.csv", index=False)
    obd_all.to_csv(f"{out_dir}/raw_obd.csv", index=False)
    return gps_all, obd_all


if __name__ == "__main__":
    gps_all, obd_all = generate_dataset()
    print(f"Generated {gps_all['trip_id'].nunique()} trips")
    print(f"GPS rows: {len(gps_all)}, OBD rows: {len(obd_all)}")
    print(gps_all.head())
    print(obd_all.head())
