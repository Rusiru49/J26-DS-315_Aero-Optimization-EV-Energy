"""
Driving Cycle Construction Pipeline
=====================================
Implements every stage of the methodology diagram, operating on the
synthetic raw GPS + OBD-II logs produced by synthetic_data.py.

Stages implemented (matching the diagram):
 1. Data Synchronization      -> sync_gps_obd()
 2. Data Preprocessing        -> preprocess()
 3. Route Identification      -> identify_routes()      [map-matching stand-in]
 4. Road Environment Class    -> classify_road_env()     [rule-based on kinematics]
 5. Adaptive Segmentation     -> adaptive_segmentation()
 6. Feature Extraction        -> extract_features()
 7. K-Means Clustering        -> cluster_segments()
 8. Representative Segments   -> select_representative_segments()
 9. Cycle Construction        -> construct_cycle()
10. Statistical Validation    -> validate_cycle()
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

RANDOM_STATE = 42


# ------------------------------------------------------------------ #
# 1. DATA SYNCHRONIZATION
# ------------------------------------------------------------------ #
def sync_gps_obd(gps_df, obd_df, tolerance_s=0.5):
    """
    Merge GPS (1 Hz) and OBD-II (5 Hz) logs onto a single common 1 Hz
    timeline per trip using nearest-timestamp matching (merge_asof).
    """
    synced = []
    for trip_id in gps_df["trip_id"].unique():
        g = gps_df[gps_df["trip_id"] == trip_id].sort_values("timestamp").copy()
        o = obd_df[obd_df["trip_id"] == trip_id].sort_values("timestamp").copy()
        g["timestamp"] = pd.to_datetime(g["timestamp"]).astype("datetime64[ns]")
        o["timestamp"] = pd.to_datetime(o["timestamp"]).astype("datetime64[ns]")
        merged = pd.merge_asof(
            g, o.drop(columns=["trip_id"]),
            on="timestamp", direction="nearest",
            tolerance=pd.Timedelta(seconds=tolerance_s),
        )
        synced.append(merged)
    out = pd.concat(synced, ignore_index=True)
    # fall back to GPS-derived speed where OBD didn't match within tolerance
    out["speed_kmh"] = out["obd_speed_kmh"].fillna(out["gps_speed_kmh"])
    return out.dropna(subset=["speed_kmh"]).reset_index(drop=True)


# ------------------------------------------------------------------ #
# 2. DATA PREPROCESSING
# ------------------------------------------------------------------ #
def preprocess(df, max_plausible_accel=5.0):
    """
    Clean the synchronized signal per trip:
      - drop physically implausible speed jumps (bad GPS fixes)
      - smooth with a Savitzky-Golay filter
      - recompute acceleration/jerk from the cleaned speed
    """
    cleaned = []
    for trip_id, g in df.groupby("trip_id"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        speed_ms = g["speed_kmh"].to_numpy() / 3.6
        dt = g["timestamp"].diff().dt.total_seconds().fillna(1.0).to_numpy().copy()
        dt[dt <= 0] = 1.0
        accel = np.gradient(speed_ms, dt.cumsum())

        implausible = np.abs(accel) > max_plausible_accel
        speed_ms_clean = speed_ms.copy()
        if implausible.any():
            idx = np.arange(len(speed_ms))
            speed_ms_clean[implausible] = np.interp(
                idx[implausible], idx[~implausible], speed_ms[~implausible]
            )

        window = min(11, len(speed_ms_clean) - (1 - len(speed_ms_clean) % 2))
        if window >= 5:
            if window % 2 == 0:
                window -= 1
            speed_ms_smooth = savgol_filter(speed_ms_clean, window_length=window, polyorder=2)
            speed_ms_smooth = np.clip(speed_ms_smooth, 0, None)
        else:
            speed_ms_smooth = speed_ms_clean

        g["speed_ms"] = speed_ms_smooth
        g["speed_kmh_clean"] = speed_ms_smooth * 3.6
        g["accel_ms2"] = np.gradient(speed_ms_smooth, dt.cumsum())
        g["jerk"] = np.gradient(g["accel_ms2"].to_numpy(), dt.cumsum())
        cleaned.append(g)
    return pd.concat(cleaned, ignore_index=True)


# ------------------------------------------------------------------ #
# 3. ROUTE IDENTIFICATION  (map-matching stand-in)
# ------------------------------------------------------------------ #
def identify_routes(df):
    """
    In a full system this snaps GPS points to an OSM road network
    (e.g. via OSMnx or a HMM map-matcher) to retrieve road IDs and
    tagged attributes (speed limit, road class). Here we simulate the
    OUTPUT of that step: a synthetic road-segment id and free-flow
    speed limit derived from local GPS spatial clustering, so the
    downstream classification stage has something concrete to consume.
    """
    df = df.copy()
    df["route_segment_id"] = (
        df["trip_id"].astype(str) + "_" +
        (df.groupby("trip_id").cumcount() // 60).astype(str)  # new "segment" every ~60s
    )
    return df


# ------------------------------------------------------------------ #
# 4. ROAD ENVIRONMENT CLASSIFICATION  (rule-based on kinematics)
# ------------------------------------------------------------------ #
def classify_road_env(df, window_s=60):
    """
    Classify each rolling window of driving into one of the 5 road
    environments using observed speed statistics + stop density --
    this mimics using OSM road tags plus a kinematic sanity check,
    WITHOUT looking at the ground-truth label used by the generator.
    """
    df = df.copy()
    df["env_pred"] = "unclassified"

    def _classify_group(g):
        v = g["speed_kmh_clean"]
        mean_v = v.mean()
        stop_frac = (v < 2).mean()
        std_v = v.std()

        if mean_v >= 80 and std_v < 20:
            label = "expressway"
        elif mean_v >= 58:
            label = "intercity"
        elif mean_v < 35 and stop_frac > 0.12:
            label = "urban"
        elif mean_v < 55 and stop_frac > 0.04:
            label = "suburban"
        else:
            # catch-all: variable/low-traffic speed with few stops -> rural
            label = "rural"
        return label

    labels = df.groupby("route_segment_id", group_keys=False).apply(
        lambda g: pd.Series(_classify_group(g), index=g.index)
    )
    df["env_pred"] = labels
    return df


# ------------------------------------------------------------------ #
# 5. ADAPTIVE SEGMENTATION
# ------------------------------------------------------------------ #
def adaptive_segmentation(df, min_segment_s=20, idle_speed_kmh=2.0, min_idle_s=3):
    """
    Split each trip into 'micro-trips': continuous stretches of driving
    bounded by idle (stationary) periods, which is the standard
    kinematic-segment definition used in driving-cycle literature.
    Segment length is adaptive: short urban segments (frequent stops)
    are kept as-is; long steady segments (expressway) get capped and
    split so no single cluster feature vector spans wildly different
    behavior.
    """
    segments = []
    seg_counter = 0
    for trip_id, g in df.groupby("trip_id"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        is_idle = g["speed_kmh_clean"] < idle_speed_kmh

        idle_run_id = (is_idle != is_idle.shift()).cumsum()
        idle_lengths = g.groupby(idle_run_id)["speed_kmh_clean"].transform("count")
        real_idle = is_idle & (idle_lengths >= min_idle_s)

        boundaries = list(np.where(real_idle.to_numpy())[0])
        cut_points = [0] + boundaries + [len(g)]
        cut_points = sorted(set(cut_points))

        for i in range(len(cut_points) - 1):
            s, e = cut_points[i], cut_points[i + 1]
            if e - s < min_segment_s:
                continue
            seg = g.iloc[s:e].copy()
            # adaptive cap: long high-speed segments split into ~180s chunks
            max_len = 400 if seg["speed_kmh_clean"].mean() > 70 else 240
            for chunk_start in range(0, len(seg), max_len):
                chunk = seg.iloc[chunk_start:chunk_start + max_len]
                if len(chunk) < min_segment_s:
                    continue
                chunk = chunk.copy()
                chunk["segment_id"] = seg_counter
                segments.append(chunk)
                seg_counter += 1
    return pd.concat(segments, ignore_index=True)


# ------------------------------------------------------------------ #
# 6. FEATURE EXTRACTION
# ------------------------------------------------------------------ #
FEATURE_COLS = [
    "avg_speed", "max_speed", "std_speed",
    "avg_accel", "avg_decel", "max_accel", "max_decel",
    "idle_time_pct", "accel_time_pct", "decel_time_pct", "cruise_time_pct",
    "stops_per_km", "duration_s", "distance_km",
]


def extract_features(seg_df):
    """Compute the standard kinematic feature vector per driving segment."""
    rows = []
    for seg_id, g in seg_df.groupby("segment_id"):
        v = g["speed_kmh_clean"].to_numpy()
        a = g["accel_ms2"].to_numpy()
        dt = g["timestamp"].diff().dt.total_seconds().fillna(1.0).to_numpy()
        duration_s = dt.sum()
        distance_km = np.sum(v / 3.6 * dt) / 1000.0

        accel_mask = a > 0.1
        decel_mask = a < -0.1
        idle_mask = v < 2

        stops = int(((v < 2) & (pd.Series(v).shift(1, fill_value=0) >= 2)).sum())

        rows.append({
            "segment_id": seg_id,
            "trip_id": g["trip_id"].iloc[0],
            "env_pred": g["env_pred"].mode()[0] if "env_pred" in g else "unknown",
            "road_type_true": g["road_type_true"].iloc[0] if "road_type_true" in g else "unknown",
            "avg_speed": v.mean(),
            "max_speed": v.max(),
            "std_speed": v.std(),
            "avg_accel": a[accel_mask].mean() if accel_mask.any() else 0.0,
            "avg_decel": a[decel_mask].mean() if decel_mask.any() else 0.0,
            "max_accel": a.max(),
            "max_decel": a.min(),
            "idle_time_pct": idle_mask.mean() * 100,
            "accel_time_pct": accel_mask.mean() * 100,
            "decel_time_pct": decel_mask.mean() * 100,
            "cruise_time_pct": (~accel_mask & ~decel_mask & ~idle_mask).mean() * 100,
            "stops_per_km": stops / distance_km if distance_km > 0.05 else 0.0,
            "duration_s": duration_s,
            "distance_km": distance_km,
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ #
# 7. K-MEANS CLUSTERING
# ------------------------------------------------------------------ #
def cluster_segments(feat_df, k_range=(2, 6)):
    """
    Run K-Means independently within each predicted road-environment
    class (so clusters represent sub-behaviors WITHIN e.g. 'urban',
    not across road types). Picks k via silhouette score.
    """
    feat_df = feat_df.copy()
    feat_df["cluster"] = -1
    models = {}

    for env, g in feat_df.groupby("env_pred"):
        if len(g) < k_range[0] + 1:
            feat_df.loc[g.index, "cluster"] = 0
            continue

        X = StandardScaler().fit_transform(g[FEATURE_COLS])
        best_k, best_score, best_labels = 2, -1, None
        max_k = min(k_range[1], len(g) - 1)
        for k in range(k_range[0], max_k + 1):
            km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
            labels = km.fit_predict(X)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(X, labels)
            if score > best_score:
                best_k, best_score, best_labels = k, score, labels

        if best_labels is None:
            best_labels = np.zeros(len(g), dtype=int)

        feat_df.loc[g.index, "cluster"] = best_labels
        models[env] = dict(k=best_k, silhouette=best_score)

    return feat_df, models


# ------------------------------------------------------------------ #
# 8. REPRESENTATIVE SEGMENTS
# ------------------------------------------------------------------ #
def select_representative_segments(feat_df):
    """For each (env, cluster), pick the segment closest to the cluster centroid (medoid)."""
    reps = []
    for (env, cluster), g in feat_df.groupby(["env_pred", "cluster"]):
        X = StandardScaler().fit_transform(g[FEATURE_COLS])
        centroid = X.mean(axis=0)
        dists = np.linalg.norm(X - centroid, axis=1)
        medoid_idx = g.index[np.argmin(dists)]
        rep = feat_df.loc[medoid_idx].copy()
        rep["cluster_size"] = len(g)
        rep["cluster_weight"] = len(g)  # used to weight cycle construction
        reps.append(rep)
    return pd.DataFrame(reps).reset_index(drop=True)


# ------------------------------------------------------------------ #
# 9. CYCLE CONSTRUCTION
# ------------------------------------------------------------------ #
def construct_cycle(seg_df, reps_df, env, smooth_join_s=2):
    """
    Concatenate the representative segments for one road environment
    (weighted by cluster size = how common that behavior was) into a
    single continuous synthetic driving cycle, smoothing speed
    discontinuities at splice points so the joined trace stays
    physically plausible.
    """
    env_reps = reps_df[reps_df["env_pred"] == env].sort_values(
        "cluster_weight", ascending=False
    )
    if env_reps.empty:
        return None

    pieces = []
    for _, rep in env_reps.iterrows():
        seg = seg_df[seg_df["segment_id"] == rep["segment_id"]].sort_values("timestamp")
        speed = seg["speed_kmh_clean"].to_numpy()
        # repeat each representative segment proportionally to how common
        # its behavior cluster was (min 1x, capped at 3x to bound cycle length)
        reps_count = int(np.clip(round(rep["cluster_weight"]), 1, 3))
        for _ in range(reps_count):
            pieces.append(speed.copy())

    cycle = pieces[0]
    for nxt in pieces[1:]:
        n = min(smooth_join_s, len(cycle), len(nxt))
        if n > 1:
            ramp = np.linspace(0, 1, n)
            blend_tail = cycle[-n:] * (1 - ramp) + nxt[:n] * ramp
            cycle = np.concatenate([cycle[:-n], blend_tail, nxt[n:]])
        else:
            cycle = np.concatenate([cycle, nxt])

    return cycle  # km/h, 1 Hz


# ------------------------------------------------------------------ #
# 10. STATISTICAL VALIDATION
# ------------------------------------------------------------------ #
def validate_cycle(cycle_speed_kmh, reference_seg_df, env):
    """
    Compare characteristic parameters of the constructed cycle against
    the FULL pool of real segments for that road class (not just the
    representatives) -- the standard acceptance check in driving-cycle
    literature, usually requiring <10-15% relative error.
    """
    ref = reference_seg_df[reference_seg_df["env_pred"] == env]
    if ref.empty or cycle_speed_kmh is None:
        return None

    v = cycle_speed_kmh
    a = np.gradient(v / 3.6, 1.0)  # 1 Hz assumed

    cycle_stats = {
        "avg_speed_kmh": v.mean(),
        "max_speed_kmh": v.max(),
        "idle_time_pct": (v < 2).mean() * 100,
        "avg_accel_ms2": a[a > 0.1].mean() if (a > 0.1).any() else 0.0,
    }
    ref_stats = {
        "avg_speed_kmh": ref["avg_speed"].mean(),
        "max_speed_kmh": ref["max_speed"].mean(),
        "idle_time_pct": ref["idle_time_pct"].mean(),
        "avg_accel_ms2": ref["avg_accel"].mean(),
    }

    rows = []
    for key in cycle_stats:
        c, r = cycle_stats[key], ref_stats[key]
        rel_err = abs(c - r) / r * 100 if r != 0 else np.nan
        rows.append({"parameter": key, "cycle_value": c, "reference_value": r,
                      "relative_error_pct": rel_err, "pass_10pct": rel_err <= 10 if not np.isnan(rel_err) else False})
    return pd.DataFrame(rows)
