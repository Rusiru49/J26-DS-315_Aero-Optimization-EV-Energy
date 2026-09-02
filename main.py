"""
End-to-end run of the driving cycle construction pipeline on synthetic data.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from synthetic_data import generate_dataset, ROAD_PROFILES
from pipeline import (
    sync_gps_obd, preprocess, identify_routes, classify_road_env,
    adaptive_segmentation, extract_features, cluster_segments,
    select_representative_segments, construct_cycle, validate_cycle,
)

plt.rcParams.update({"figure.dpi": 110, "font.size": 9})


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

    # ---------------- PLOTS ----------------
    print("\nGenerating validation figures...")
    plot_cycles(cycles)
    plot_feature_space(feats_clustered)
    plot_classification_confusion(classified)

    # Save cycles to CSV
    for env, cycle in cycles.items():
        if cycle is not None:
            pd.DataFrame({"time_s": np.arange(len(cycle)), "speed_kmh": cycle}) \
                .to_csv(f"outputs/driving_cycle_{env}.csv", index=False)

    print("\nDone. Outputs in ./outputs, figures in ./figures")
    return classified, segments, feats_clustered, reps, cycles, validation_results


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


if __name__ == "__main__":
    main()
