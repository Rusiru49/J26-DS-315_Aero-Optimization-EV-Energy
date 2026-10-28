"""
Driving Cycle Benchmarking against Standard Reference Cycles
================================================================
Compares your constructed cycles (from construct_cycle() in pipeline.py)
against published characteristic parameters of standard drive cycles
(WLTP Class 3, NEDC, IDC, MIDC), and optionally against full reference
speed-time traces if you supply them as CSV.

--------------------------------------------------------------------
LEVEL 1 — Characteristic parameter comparison (works immediately)
--------------------------------------------------------------------
REFERENCE_CYCLES below holds published macroscopic parameters sourced
from standard references (DieselNet, UNECE WLTP DHC documentation,
and driving-cycle survey papers). No trace data needed.

Usage:
    from cycle_benchmark import (
        compute_cycle_characteristics, compare_characteristics_table,
        plot_characteristics_bar, plot_radar_comparison,
    )

    my_stats = {env: compute_cycle_characteristics(cycle)
                for env, cycle in cycles.items() if cycle is not None}

    table = compare_characteristics_table(my_stats, reference_keys=["NEDC", "WLTP_Class3", "MIDC"])
    print(table)

    plot_characteristics_bar(my_stats, reference_keys=["NEDC", "WLTP_Class3", "MIDC"])
    plot_radar_comparison(my_stats, reference_keys=["NEDC", "WLTP_Class3", "MIDC"])

--------------------------------------------------------------------
LEVEL 2 — Full trace overlay (needs a reference CSV you supply)
--------------------------------------------------------------------
If you obtain the official second-by-second WLTP/NEDC/IDC trace as a
CSV with columns [time_s, speed_kmh], you can overlay it directly
against your constructed cycle:

    plot_trace_overlay(cycles["urban"], "wltp_class3_trace.csv",
                        my_label="My Urban Cycle", ref_label="WLTP Class 3")
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 110, "font.size": 9})

# -------------------------------------------------------------------- #
# Published characteristic parameters of standard drive cycles.
# Sources: DieselNet (ECE15/EUDC/NEDC), UNECE WLTP DHC documentation
# (Class 3 phase table), and published driving-cycle comparison tables
# for IDC / MIDC (Indian Drive Cycle / Modified Indian Drive Cycle).
# All idle_time_pct values are (idle_time_s / total_time_s) * 100.
# -------------------------------------------------------------------- #
REFERENCE_CYCLES = {
    "NEDC": dict(
        duration_s=1180, distance_km=10.93,
        avg_speed_kmh=33.35, max_speed_kmh=120,
        idle_time_pct=267 / 1180 * 100,
        avg_accel_ms2=0.506, max_accel_ms2=1.042,
    ),
    "WLTP_Class3": dict(
        duration_s=1801, distance_km=23.266,
        avg_speed_kmh=46.5, max_speed_kmh=131.3,
        idle_time_pct=227 / 1801 * 100,
        avg_accel_ms2=None, max_accel_ms2=1.58,
    ),
    "IDC": dict(
        duration_s=108, distance_km=0.66,
        avg_speed_kmh=21.73, max_speed_kmh=None,
        idle_time_pct=18 / 108 * 100,
        avg_accel_ms2=None, max_accel_ms2=None,
    ),
    "MIDC": dict(
        duration_s=1184, distance_km=10.59,
        avg_speed_kmh=32.17, max_speed_kmh=None,
        idle_time_pct=298 / 1184 * 100,
        avg_accel_ms2=None, max_accel_ms2=None,
    ),
}


# ------------------------------------------------------------------ #
# Compute the same characteristic parameters for one of YOUR
# constructed cycles, so it's directly comparable to REFERENCE_CYCLES.
# ------------------------------------------------------------------ #
def compute_cycle_characteristics(cycle_speed_kmh, dt_s=1.0):
    """cycle_speed_kmh: 1D array of speed at dt_s intervals (as produced
    by construct_cycle() in pipeline.py, which is 1 Hz km/h)."""
    v = np.asarray(cycle_speed_kmh, dtype=float)
    duration_s = len(v) * dt_s
    distance_km = np.sum(v / 3.6 * dt_s) / 1000.0
    a = np.gradient(v / 3.6, dt_s)

    idle_mask = v < 2
    return dict(
        duration_s=duration_s,
        distance_km=distance_km,
        avg_speed_kmh=v.mean(),
        max_speed_kmh=v.max(),
        idle_time_pct=idle_mask.mean() * 100,
        avg_accel_ms2=a[a > 0.1].mean() if (a > 0.1).any() else 0.0,
        max_accel_ms2=a.max(),
    )


# ------------------------------------------------------------------ #
# Table: your cycles vs selected reference cycles, side by side
# ------------------------------------------------------------------ #
def compare_characteristics_table(my_stats, reference_keys=None):
    """my_stats: dict {your_cycle_name: characteristics_dict}
    (from compute_cycle_characteristics). Returns a tidy DataFrame."""
    if reference_keys is None:
        reference_keys = list(REFERENCE_CYCLES.keys())

    rows = []
    for name, stats in my_stats.items():
        row = {"cycle": name, "source": "constructed"}
        row.update(stats)
        rows.append(row)
    for name in reference_keys:
        row = {"cycle": name, "source": "standard"}
        row.update(REFERENCE_CYCLES[name])
        rows.append(row)

    return pd.DataFrame(rows)[
        ["cycle", "source", "duration_s", "distance_km", "avg_speed_kmh",
         "max_speed_kmh", "idle_time_pct", "avg_accel_ms2", "max_accel_ms2"]
    ]


# ------------------------------------------------------------------ #
# Grouped bar chart: your cycles vs standards, per parameter
# ------------------------------------------------------------------ #
def plot_characteristics_bar(my_stats, reference_keys=None,
                              params=("avg_speed_kmh", "max_speed_kmh", "idle_time_pct"),
                              out_path="figures/benchmark_comparison.png"):
    if reference_keys is None:
        reference_keys = list(REFERENCE_CYCLES.keys())

    all_names = list(my_stats.keys()) + reference_keys
    all_sources = ["constructed"] * len(my_stats) + ["standard"] * len(reference_keys)
    all_stats = list(my_stats.values()) + [REFERENCE_CYCLES[k] for k in reference_keys]

    fig, axes = plt.subplots(1, len(params), figsize=(5 * len(params), 4))
    if len(params) == 1:
        axes = [axes]
    for ax, param in zip(axes, params):
        vals = [s.get(param) if s.get(param) is not None else np.nan for s in all_stats]
        colors = ["#2563eb" if src == "constructed" else "#94a3b8" for src in all_sources]
        ax.bar(all_names, vals, color=colors)
        ax.set_title(param)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Constructed Cycles vs Standard Reference Cycles")
    fig.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color="#2563eb", label="Your cycles"),
        plt.Rectangle((0, 0), 1, 1, color="#94a3b8", label="Standard cycles"),
    ], loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ------------------------------------------------------------------ #
# Radar/spider chart: normalized multi-parameter comparison
# ------------------------------------------------------------------ #
def plot_radar_comparison(my_stats, reference_keys=None,
                           params=("avg_speed_kmh", "max_speed_kmh", "idle_time_pct", "avg_accel_ms2"),
                           out_path="figures/benchmark_radar.png"):
    """Normalizes each parameter to [0,1] across all cycles shown, then
    plots a radar chart so shape differences are visually comparable."""
    if reference_keys is None:
        reference_keys = list(REFERENCE_CYCLES.keys())

    names = list(my_stats.keys()) + reference_keys
    stats_list = list(my_stats.values()) + [REFERENCE_CYCLES[k] for k in reference_keys]

    # normalize each parameter across all cycles (skip if all-NaN)
    raw = {p: [s.get(p) if s.get(p) is not None else np.nan for s in stats_list] for p in params}
    norm = {}
    for p, vals in raw.items():
        arr = np.array(vals, dtype=float)
        finite = arr[~np.isnan(arr)]
        if len(finite) == 0 or finite.max() == finite.min():
            norm[p] = np.zeros_like(arr)
        else:
            norm[p] = (arr - finite.min()) / (finite.max() - finite.min())

    angles = np.linspace(0, 2 * np.pi, len(params), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    cmap = plt.get_cmap("tab10", len(names))
    for i, name in enumerate(names):
        vals = [norm[p][i] if not np.isnan(norm[p][i]) else 0 for p in params]
        vals += vals[:1]
        style = "-" if name in my_stats else "--"
        ax.plot(angles, vals, style, linewidth=1.5, label=name, color=cmap(i))
        ax.fill(angles, vals, alpha=0.05, color=cmap(i))
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(params, fontsize=8)
    ax.set_yticklabels([])
    ax.set_title("Normalized Cycle Characteristics (solid = yours, dashed = standard)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ------------------------------------------------------------------ #
# LEVEL 2 — overlay against a real reference trace, if you have one
# ------------------------------------------------------------------ #
def plot_trace_overlay(my_cycle_speed_kmh, reference_csv_path,
                        my_label="Constructed cycle", ref_label="Reference cycle",
                        dt_s=1.0, out_path="figures/trace_overlay.png"):
    """reference_csv_path: CSV with columns [time_s, speed_kmh] for the
    official standard cycle trace (source this yourself, e.g. from
    UNECE/ARAI/EPA publications or an open driving-cycle dataset repo —
    these are regulatory reference traces, not something to hardcode
    here without a verified source)."""
    ref = pd.read_csv(reference_csv_path)
    t_mine = np.arange(len(my_cycle_speed_kmh)) * dt_s

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t_mine, my_cycle_speed_kmh, color="#2563eb", linewidth=1, label=my_label)
    ax.plot(ref["time_s"], ref["speed_kmh"], color="#dc2626", linewidth=1, alpha=0.7, label=ref_label)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (km/h)")
    ax.set_title(f"{my_label} vs {ref_label} — Speed-Time Trace Overlay")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)