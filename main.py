"""
End-to-end run of the driving cycle construction pipeline on synthetic data.
"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from synthetic_data import generate_dataset, ROAD_PROFILES
from pipeline import (
    sync_gps_obd, preprocess, identify_routes, classify_road_env,
    adaptive_segmentation, extract_features, cluster_segments,
    select_representative_segments, construct_cycle, validate_cycle,
    FEATURE_COLS,
)

plt.rcParams.update({"figure.dpi": 110, "font.size": 9})

ROAD_COLORS = {
    "urban": "#ef4444",
    "suburban": "#f97316",
    "intercity": "#eab308",
    "expressway": "#22c55e",
    "rural": "#3b82f6",
}

for folder in ["data", "outputs", "figures"]:
    if not os.path.exists(folder):
        os.makedirs(folder)


def main():
    print("=" * 60)
    print("STAGE 0: Generating synthetic raw GPS + OBD-II dataset")
    print("=" * 60)
    gps_df, obd_df = generate_dataset(trips_per_class=5, duration_min_range=(12, 20))
    print(f"  {gps_df['trip_id'].nunique()} trips | GPS rows: {len(gps_df)} | OBD rows: {len(obd_df)}")

    print("\nSTAGE 1: Data Synchronization")
    synced = sync_gps_obd(gps_df, obd_df)
    print(f"  Synced rows: {len(synced)}  (merged onto GPS 1Hz timeline)")

    print("\nSTAGE 2: Data Preprocessing")
    clean = preprocess(synced)
    print(f"  Cleaned rows: {len(clean)}")

    print("\nSTAGE 3: Route Identification (map-matching stand-in)")
    routed = identify_routes(clean)
    print(f"  Route segments identified: {routed['route_segment_id'].nunique()}")

    print("\nSTAGE 4: Road Environment Classification")
    classified = classify_road_env(routed)
    acc = (classified.groupby("route_segment_id")
           .apply(lambda g: g["env_pred"].iloc[0] == g["road_type_true"].mode()[0]))
    print(f"  Rule-based classifier agreement with ground truth: {acc.mean()*100:.1f}%")
    print(classified.groupby("env_pred")["route_segment_id"].nunique())

    print("\nSTAGE 5: Adaptive Segmentation")
    segments = adaptive_segmentation(classified)
    print(f"  Micro-trip segments produced: {segments['segment_id'].nunique()}")

    print("\nSTAGE 6: Feature Extraction")
    feats = extract_features(segments)
    print(f"  Feature matrix: {feats.shape}")
    feats.to_csv("outputs/segment_features.csv", index=False)

    print("\nSTAGE 7: K-Means Clustering (per road environment)")
    feats_clustered, cluster_models = cluster_segments(feats)
    for env, info in cluster_models.items():
        print(f"  {env:12s} -> k={info['k']}, silhouette={info['silhouette']:.3f}")

    print("\nSTAGE 8: Representative Segment Selection")
    reps = select_representative_segments(feats_clustered)
    print(f"  Representative segments selected: {len(reps)}")
    reps.to_csv("outputs/representative_segments.csv", index=False)

    print("\nSTAGE 9: Cycle Construction")
    cycles = {}
    for env in ROAD_PROFILES:
        cycle = construct_cycle(segments, reps, env)
        cycles[env] = cycle
        if cycle is not None:
            print(f"  {env:12s} -> cycle length {len(cycle)} s ({len(cycle)/60:.1f} min)")
        else:
            print(f"  {env:12s} -> no representative segments found")

    print("\nSTAGE 10: Statistical Validation")
    validation_results = {}
    for env in ROAD_PROFILES:
        result = validate_cycle(cycles[env], feats_clustered, env)
        validation_results[env] = result
        if result is not None:
            print(f"\n  --- {env} ---")
            print(result.to_string(index=False))

    # ---------------- ORIGINAL PLOTS ----------------
    print("\nGenerating validation figures...")
    plot_cycles(cycles)
    plot_feature_space(feats_clustered)
    plot_classification_confusion(classified)

    # ---------------- ADDITIONAL PLOTS ----------------
    print("Generating additional diagnostic figures...")
    plot_raw_trip_trace(gps_df, obd_df, trip_id=0)
    plot_gps_tracks(gps_df)
    plot_sync_zoom(synced, trip_id=0)
    plot_preprocessing_effect(synced, clean, trip_id=0)
    plot_feature_distributions(feats_clustered)
    plot_feature_correlation(feats_clustered)
    plot_silhouette_scores(cluster_models)
    plot_cluster_sizes(feats_clustered)
    plot_validation_errors(validation_results)
    plot_speed_acceleration_joint(feats_clustered)
    plot_micro_trip_durations(segments)
    plot_cycle_summary_table(cycles, validation_results)

    # Save cycles to CSV
    for env, cycle in cycles.items():
        if cycle is not None:
            pd.DataFrame({"time_s": np.arange(len(cycle)), "speed_kmh": cycle}) \
                .to_csv(f"outputs/driving_cycle_{env}.csv", index=False)

    print("\nDone. Outputs in ./outputs, figures in ./figures")
    return classified, segments, feats_clustered, reps, cycles, validation_results


# ==================================================================== #
# ORIGINAL PLOTS
# ==================================================================== #
def plot_cycles(cycles):
    fig, axes = plt.subplots(len(cycles), 1, figsize=(9, 10), sharex=False)
    for ax, (env, cycle) in zip(axes, cycles.items()):
        if cycle is None:
            ax.set_title(f"{env} (no data)")
            continue
        t = np.arange(len(cycle))
        ax.plot(t, cycle, color="#2563eb", linewidth=1)
        ax.fill_between(t, cycle, alpha=0.15, color="#2563eb")
        ax.set_title(f"Constructed Driving Cycle — {env.capitalize()}", fontsize=10)
        ax.set_ylabel("Speed (km/h)")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig("figures/constructed_cycles.png")
    plt.close(fig)


def plot_feature_space(feats_clustered):
    envs = feats_clustered["env_pred"].unique()
    fig, axes = plt.subplots(1, len(envs), figsize=(4 * len(envs), 4), sharey=True)
    if len(envs) == 1:
        axes = [axes]
    for ax, env in zip(axes, envs):
        g = feats_clustered[feats_clustered["env_pred"] == env]
        sc = ax.scatter(g["avg_speed"], g["stops_per_km"], c=g["cluster"], cmap="viridis", s=25)
        ax.set_title(env)
        ax.set_xlabel("avg speed (km/h)")
    axes[0].set_ylabel("stops per km")
    fig.suptitle("K-Means clusters per road environment (feature space slice)")
    fig.tight_layout()
    fig.savefig("figures/cluster_feature_space.png")
    plt.close(fig)


def plot_classification_confusion(classified):
    seg_level = classified.groupby("route_segment_id").agg(
        env_pred=("env_pred", "first"), road_type_true=("road_type_true", lambda x: x.mode()[0])
    )
    conf = pd.crosstab(seg_level["road_type_true"], seg_level["env_pred"])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(conf.values, cmap="Blues")
    ax.set_xticks(range(len(conf.columns)))
    ax.set_xticklabels(conf.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(conf.index)))
    ax.set_yticklabels(conf.index)
    ax.set_xlabel("Predicted road environment")
    ax.set_ylabel("Ground-truth road environment")
    ax.set_title("Road Environment Classification Agreement")
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            ax.text(j, i, conf.values[i, j], ha="center", va="center", fontsize=8)
    fig.colorbar(im, label="segment count")
    fig.tight_layout()
    fig.savefig("figures/classification_confusion.png")
    plt.close(fig)


# ==================================================================== #
# ADDITIONAL PLOTS
# ==================================================================== #
def plot_raw_trip_trace(gps_df, obd_df, trip_id=0, out_path="figures/raw_trip_trace.png"):
    """Shows why synchronization is needed: two differently-sampled,
    differently-noisy speed signals for the same physical trip."""
    g = gps_df[gps_df["trip_id"] == trip_id].sort_values("timestamp")
    o = obd_df[obd_df["trip_id"] == trip_id].sort_values("timestamp")
    t0 = g["timestamp"].iloc[0]
    g_t = (g["timestamp"] - t0).dt.total_seconds()
    o_t = (o["timestamp"] - t0).dt.total_seconds()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(g_t, g["gps_speed_kmh"], label="GPS speed (1 Hz)", color="#2563eb", alpha=0.8, linewidth=1)
    ax.plot(o_t, o["obd_speed_kmh"], label="OBD-II speed (5 Hz)", color="#dc2626", alpha=0.6, linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (km/h)")
    road = g["road_type_true"].iloc[0] if "road_type_true" in g else "?"
    ax.set_title(f"Raw GPS vs OBD-II Speed — Trip {trip_id} ({road})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_gps_tracks(gps_df, out_path="figures/gps_tracks.png"):
    """Spatial view of every synthetic trip, colored by road environment."""
    fig, ax = plt.subplots(figsize=(7, 7))
    for road_type, color in ROAD_COLORS.items():
        sub = gps_df[gps_df["road_type_true"] == road_type]
        for trip_id, g in sub.groupby("trip_id"):
            ax.plot(g["lon"], g["lat"], color=color, alpha=0.6, linewidth=1)
    handles = [Patch(color=c, label=r) for r, c in ROAD_COLORS.items()]
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Synthetic GPS Tracks by Road Environment")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_sync_zoom(synced_df, trip_id=0, window_s=120, out_path="figures/sync_zoom.png"):
    """Zoom into a short window post-sync to visually confirm GPS and
    OBD speed streams now share one timeline and largely agree."""
    g = synced_df[synced_df["trip_id"] == trip_id].sort_values("timestamp").reset_index(drop=True)
    g = g.iloc[:window_s]
    t = (g["timestamp"] - g["timestamp"].iloc[0]).dt.total_seconds()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, g["gps_speed_kmh"], "o-", label="GPS speed", color="#2563eb", markersize=3)
    ax.plot(t, g["obd_speed_kmh"], "s-", label="OBD speed", color="#dc2626", markersize=3, alpha=0.7)
    ax.plot(t, g["speed_kmh"], "--", label="Merged speed_kmh (final)", color="#16a34a", linewidth=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (km/h)")
    ax.set_title(f"Synchronized Signal Close-up — Trip {trip_id} (first {window_s}s)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_preprocessing_effect(synced_df, clean_df, trip_id=0, out_path="figures/preprocessing_effect.png"):
    """Overlay raw synced speed against the cleaned/smoothed speed to
    show what the Savitzky-Golay + outlier-interpolation step did."""
    raw = synced_df[synced_df["trip_id"] == trip_id].sort_values("timestamp")
    cln = clean_df[clean_df["trip_id"] == trip_id].sort_values("timestamp")
    t_raw = (raw["timestamp"] - raw["timestamp"].iloc[0]).dt.total_seconds()
    t_cln = (cln["timestamp"] - cln["timestamp"].iloc[0]).dt.total_seconds()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(t_raw, raw["speed_kmh"], color="#94a3b8", linewidth=0.8, label="Raw merged speed")
    ax1.plot(t_cln, cln["speed_kmh_clean"], color="#2563eb", linewidth=1.2, label="Cleaned + smoothed")
    ax1.set_ylabel("Speed (km/h)")
    ax1.set_title(f"Preprocessing Effect — Trip {trip_id}")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(t_cln, cln["accel_ms2"], color="#dc2626", linewidth=0.8)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_ylabel("Accel (m/s²)")
    ax2.set_xlabel("Time (s)")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_feature_distributions(feats_df, features=None, out_path="figures/feature_distributions.png"):
    """Boxplot of key kinematic features grouped by predicted road
    environment — a quick sanity check that classes are separable."""
    if features is None:
        features = ["avg_speed", "stops_per_km", "idle_time_pct", "std_speed"]
    envs = sorted(feats_df["env_pred"].unique())
    fig, axes = plt.subplots(1, len(features), figsize=(4 * len(features), 4))
    if len(features) == 1:
        axes = [axes]
    for ax, feat in zip(axes, features):
        data = [feats_df.loc[feats_df["env_pred"] == e, feat].dropna() for e in envs]
        bp = ax.boxplot(data, labels=envs, patch_artist=True)
        for patch, env in zip(bp["boxes"], envs):
            patch.set_facecolor(ROAD_COLORS.get(env, "#999999"))
            patch.set_alpha(0.6)
        ax.set_title(feat)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Segment Feature Distributions by Road Environment")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_feature_correlation(feats_df, out_path="figures/feature_correlation.png"):
    """Correlation matrix across all extracted kinematic features —
    useful for spotting redundant features before clustering."""
    corr = feats_df[FEATURE_COLS].corr()

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURE_COLS)))
    ax.set_xticklabels(FEATURE_COLS, rotation=90, fontsize=7)
    ax.set_yticks(range(len(FEATURE_COLS)))
    ax.set_yticklabels(FEATURE_COLS, fontsize=7)
    ax.set_title("Feature Correlation Matrix")
    fig.colorbar(im, label="Pearson r")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_silhouette_scores(cluster_models, out_path="figures/silhouette_scores.png"):
    """Bar chart of chosen k and silhouette score per road environment,
    from the dict returned by cluster_segments()."""
    envs = list(cluster_models.keys())
    scores = [cluster_models[e]["silhouette"] for e in envs]
    ks = [cluster_models[e]["k"] for e in envs]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(envs, scores, color=[ROAD_COLORS.get(e, "#999999") for e in envs])
    for bar, k in zip(bars, ks):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"k={k}", ha="center", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Silhouette score")
    ax.set_title("K-Means Cluster Quality per Road Environment")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_cluster_sizes(feats_df, out_path="figures/cluster_sizes.png"):
    """Stacked bar of segment counts per (env, cluster) — shows how
    much weight each representative segment will get in construct_cycle()."""
    counts = feats_df.groupby(["env_pred", "cluster"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    bottom = np.zeros(len(counts))
    n_clusters = counts.shape[1]
    cmap = plt.get_cmap("viridis", n_clusters)
    for i, col in enumerate(counts.columns):
        ax.bar(counts.index, counts[col], bottom=bottom, color=cmap(i), label=f"cluster {col}")
        bottom += counts[col].to_numpy()
    ax.set_ylabel("Segment count")
    ax.set_title("Micro-trip Segments per Behavior Cluster")
    ax.legend(title="Cluster", fontsize=7, ncol=n_clusters)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_validation_errors(validation_results, out_path="figures/validation_errors.png"):
    """Bar chart of relative error per parameter per road environment,
    from the dict of DataFrames returned by validate_cycle()."""
    envs = [e for e, r in validation_results.items() if r is not None]
    if not envs:
        return
    params = validation_results[envs[0]]["parameter"].tolist()

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8 / len(envs)
    x = np.arange(len(params))
    for i, env in enumerate(envs):
        r = validation_results[env].set_index("parameter").loc[params, "relative_error_pct"]
        ax.bar(x + i * width, r.values, width=width, label=env,
               color=ROAD_COLORS.get(env, "#999999"))
    ax.axhline(10, color="red", linestyle="--", linewidth=1, label="10% acceptance threshold")
    ax.set_xticks(x + width * (len(envs) - 1) / 2)
    ax.set_xticklabels(params, rotation=30, ha="right")
    ax.set_ylabel("Relative error (%)")
    ax.set_title("Constructed Cycle vs Reference Segments — Validation Error")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_speed_acceleration_joint(feats_df, out_path="figures/speed_accel_joint.png"):
    """Scatter of avg_speed vs avg_accel per segment, faceted by road
    environment — a lightweight stand-in for a full speed-acceleration
    frequency distribution (SAFD), a standard driving-cycle QA plot."""
    envs = sorted(feats_df["env_pred"].unique())
    fig, axes = plt.subplots(1, len(envs), figsize=(4 * len(envs), 4), sharex=True, sharey=True)
    if len(envs) == 1:
        axes = [axes]
    for ax, env in zip(axes, envs):
        g = feats_df[feats_df["env_pred"] == env]
        ax.scatter(g["avg_speed"], g["avg_accel"], s=20, alpha=0.6,
                   color=ROAD_COLORS.get(env, "#999999"))
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(env)
        ax.set_xlabel("avg speed (km/h)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("avg accel (m/s²)")
    fig.suptitle("Speed vs Acceleration per Segment (by Road Environment)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_micro_trip_durations(segments_df, out_path="figures/segment_durations.png"):
    """Histogram of micro-trip segment lengths (seconds) per road
    environment, from adaptive_segmentation() output."""
    dur = segments_df.groupby(["segment_id", "env_pred"]).size().reset_index(name="duration_s")
    envs = sorted(dur["env_pred"].unique())

    fig, ax = plt.subplots(figsize=(8, 4))
    for env in envs:
        vals = dur.loc[dur["env_pred"] == env, "duration_s"]
        ax.hist(vals, bins=20, alpha=0.5, label=env, color=ROAD_COLORS.get(env, "#999999"))
    ax.set_xlabel("Segment duration (s)")
    ax.set_ylabel("Count")
    ax.set_title("Micro-trip Segment Duration Distribution")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_cycle_summary_table(cycles, validation_results, out_path="figures/cycle_summary_table.png"):
    """Renders a compact table figure summarizing each constructed
    cycle: duration, avg/max speed, and pass/fail count vs reference."""
    rows = []
    for env, cycle in cycles.items():
        if cycle is None:
            rows.append([env, "-", "-", "-", "-"])
            continue
        v = cycle
        result = validation_results.get(env)
        n_pass = int(result["pass_10pct"].sum()) if result is not None else 0
        n_total = len(result) if result is not None else 0
        rows.append([
            env,
            f"{len(v)/60:.1f} min",
            f"{v.mean():.1f} km/h",
            f"{v.max():.1f} km/h",
            f"{n_pass}/{n_total} params <10% err",
        ])

    fig, ax = plt.subplots(figsize=(9, 0.6 * len(rows) + 1))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Road env", "Duration", "Avg speed", "Max speed", "Validation"],
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    ax.set_title("Constructed Driving Cycle Summary", pad=20)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()