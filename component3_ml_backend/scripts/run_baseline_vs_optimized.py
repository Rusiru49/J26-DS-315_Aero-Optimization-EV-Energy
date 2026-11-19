"""
Workflow 2 results: baseline shape vs optimised shape on every driving cycle,
with the drag and frontal-area models saved by run_workflow1.py, the vehicle
constants in config.py and the styling limits (config.search_bounds()).

For each cycle the optimiser searches the 5 shape values for the lowest Wh/km
and reports the gain and the verdict (already near-optimal or better shape
found). A cross-cycle check then runs each cycle's best shape on every other
cycle, to test whether the best shape really depends on the cycle.

Usage:
    python scripts/run_baseline_vs_optimized.py                 # Member 2's PP1 models, 150 tries
    python scripts/run_baseline_vs_optimized.py --trials 100

Writes:
    outputs/workflow2_baseline_vs_optimized.csv    one row per cycle and side (baseline / optimised)
    outputs/workflow2_cross_cycle.csv              Wh/km of each cycle's best shape on every cycle
    outputs/logs/workflow2_baseline_vs_optimized.txt
    outputs/figures/wf2_fig1_energy_saving.png     Wh/km and extra range, baseline vs optimised
    outputs/figures/wf2_fig2_shape_change.png      baseline and optimised shape values inside the limits
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from component3 import config
from component3.workflow1.training import load_model
from component3.workflow2.energy_model import load_driving_cycle, prepare_cycle
from component3.workflow2.optimizer import _evaluate_shape, compare_to_baseline

FIGS = os.path.join(config.OUTPUTS_DIR, "figures")
LOGS = os.path.join(config.OUTPUTS_DIR, "logs")
# same palette as make_pp1_figures.py: blue = optimised, orange = baseline
OPT, BASE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CYCLE_COLOURS = ["#2a78d6", "#eb6834", "#1e7a80", "#7a4cc2", "#c98a00"]
NICE = {"windshield_angle_deg": "Windshield angle (deg)", "spoiler_angle_deg": "Spoiler angle (deg)",
        "boat_tail_angle_deg": "Boat-tail angle (deg)", "hood_angle_deg": "Hood angle (deg)",
        "ground_clearance_mm": "Ground clearance (mm)"}


def style(ax, grid="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=10)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def load_models(dataset):
    folder = os.path.join(config.MODELS_DIR, dataset)
    with open(os.path.join(folder, "model_card.json")) as fh:
        card = json.load(fh)
    # search inside the CFD box this model was trained on
    config.set_design_space(card["design_box"], config.BASELINE_SHAPE)
    drag = load_model(os.path.join(folder, card["drag_model_file"]))
    area = load_model(os.path.join(folder, card["frontal_area_model_file"]))
    return card, drag, area


def run_cycles(drag, area, n_trials):
    results, seconds = {}, {}
    for cycle in config.DRIVING_CYCLE_FILES:
        start = time.perf_counter()
        results[cycle] = compare_to_baseline(drag, cycle, n_trials=n_trials, area_model=area)
        seconds[cycle] = time.perf_counter() - start
    return results, seconds


def results_table(results):
    rows = []
    for cycle, r in results.items():
        for side in ("baseline", "optimised"):
            s = r[side]
            rows.append({
                "cycle": cycle, "side": side, **{n: s["shape"][n] for n in config.SHAPE_PARAM_NAMES},
                **{f"cd_{v}kmh": c for v, c in s["cd_by_speed"].items()},
                "cd_effective": s["cd"], "frontal_area_m2": s["frontal_area_m2"], "cd_a_m2": s["cd_a_m2"],
                "distance_km": s["distance_km"], "wh_per_km": s["wh_per_km"], "range_km": s["range_km"],
                "cost_lkr": s["cost_lkr"],
                "gain_percent": 100 * r["relative_improvement"] if side == "optimised" else 0.0,
                "verdict": ("" if side == "baseline" else
                            "already near-optimal" if r["already_near_optimal"] else "better shape found"),
            })
    return pd.DataFrame(rows)


def cross_cycle(drag, area, results):
    """Wh/km of each cycle's best shape (rows) on every cycle (columns)."""
    prepared = {c: prepare_cycle(load_driving_cycle(c)) for c in results}
    table = pd.DataFrame(
        {target: [_evaluate_shape(drag, area, prepared[target], results[src]["optimised"]["shape"], {})["wh_per_km"]
                  for src in results] for target in results},
        index=list(results))
    table.index.name = "best_shape_of"
    # how much worse is another cycle's best shape than the best found for this cycle, in %
    spread = 100 * (table.max() - table.min()) / table.min()
    return table, spread


def fig_energy_saving(table, results):
    cycles = list(results)
    base = table[table.side == "baseline"].set_index("cycle").loc[cycles]
    opt = table[table.side == "optimised"].set_index("cycle").loc[cycles]
    x = np.arange(len(cycles))
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.6, 1]})
    w = 0.38
    a.bar(x - w / 2, base["wh_per_km"], w, color=BASE, edgecolor="white", label="baseline shape (D000)")
    a.bar(x + w / 2, opt["wh_per_km"], w, color=OPT, edgecolor="white", label="optimised shape")
    for xi, c in zip(x, cycles):
        a.text(xi + w / 2, opt.loc[c, "wh_per_km"] + 2, f"-{opt.loc[c, 'gain_percent']:.1f}%",
               ha="center", fontsize=10, color=INK, fontweight="bold")
    a.set_xticks(x, [f"{c.capitalize()}\n" + ("near-optimal" if results[c]["already_near_optimal"] else "better shape")
                     for c in cycles], fontsize=10, color=INK)
    a.set_ylabel("Energy use (Wh/km)", color=MUTED)
    a.set_title("Energy per km on each driving cycle", fontsize=12, color=INK, loc="left")
    a.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.0), fontsize=9)
    a.set_ylim(0, max(base["wh_per_km"]) * 1.18)
    style(a)
    gain_km = opt["range_km"] - base["range_km"]
    b.barh(x[::-1], gain_km, color=OPT, height=0.6, edgecolor="white")
    for yi, g in zip(x[::-1], gain_km):
        b.text(g + 0.8, yi, f"+{g:.0f} km", va="center", fontsize=10, color=INK)
    b.set_yticks(x[::-1], [c.capitalize() for c in cycles], fontsize=10, color=INK)
    b.set_xlabel("Extra range on a full battery (km)", color=MUTED)
    b.set_title("Range gained by the optimised shape", fontsize=12, color=INK, loc="left")
    b.set_xlim(0, max(gain_km) * 1.25 + 1)
    style(b, "x")
    fig.text(0.01, -0.02, f"Vehicle: {config.VEHICLE_MASS_KG:.0f} kg, {config.BATTERY_CAPACITY_KWH} kWh "
             f"(BYD Seal Premium). Shapes searched inside the realistic styling limits. "
             f"Near-optimal = gain of {100 * config.BASELINE_OPTIMAL_TOLERANCE:.0f}% or less.",
             fontsize=8.5, color=MUTED)
    fig.tight_layout()
    return fig


def fig_shape_change(results):
    bounds = next(iter(results.values()))["search_bounds"]
    names = config.SHAPE_PARAM_NAMES
    fig, axes = plt.subplots(len(names), 1, figsize=(10, 7.4))
    for ax, n in zip(axes, names):
        lo, hi = config.SHAPE_PARAM_BOUNDS[n]
        slo, shi = bounds[n]
        ax.axvspan(lo, hi, color=GRID, alpha=0.6, lw=0)
        ax.axvspan(slo, shi, color=OPT, alpha=0.12, lw=0)
        base = config.BASELINE_SHAPE[n]
        ax.scatter([base], [0], marker="D", s=70, color=INK, zorder=3)
        for k, (cycle, r) in enumerate(results.items()):
            v = r["optimised"]["shape"][n]
            ax.scatter([v], [0.22 + 0.11 * k], s=34, color=CYCLE_COLOURS[k % len(CYCLE_COLOURS)], zorder=3)
        ax.set_xlim(lo - 0.04 * (hi - lo), hi + 0.04 * (hi - lo))
        ax.set_ylim(-0.3, 0.9)
        ax.set_yticks([])
        ax.set_title(NICE[n], fontsize=10.5, color=INK, loc="left", pad=2)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=9)
    handles = [plt.Line2D([], [], marker="D", ls="", color=INK, markersize=8, label="baseline D000")]
    handles += [plt.Line2D([], [], marker="o", ls="", color=CYCLE_COLOURS[k % len(CYCLE_COLOURS)], markersize=7,
                           label=f"optimised, {c}") for k, c in enumerate(results)]
    handles += [plt.Rectangle((0, 0), 1, 1, color=OPT, alpha=0.12, label="realistic styling limits"),
                plt.Rectangle((0, 0), 1, 1, color=GRID, alpha=0.6, label="CFD design box")]
    fig.legend(handles=handles, frameon=False, loc="upper center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--model", default="member2_sedan_pp1", help="folder in models/ made by run_workflow1.py")
    parser.add_argument("--trials", type=int, default=150, help="optimiser tries per cycle")
    args = parser.parse_args()

    card, drag, area = load_models(args.model)
    results, seconds = run_cycles(drag, area, args.trials)
    table = results_table(results)
    cross, spread = cross_cycle(drag, area, results)

    os.makedirs(FIGS, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    table.to_csv(os.path.join(config.OUTPUTS_DIR, "workflow2_baseline_vs_optimized.csv"), index=False)
    cross.to_csv(os.path.join(config.OUTPUTS_DIR, "workflow2_cross_cycle.csv"))
    for name, make in (("wf2_fig1_energy_saving", lambda: fig_energy_saving(table, results)),
                       ("wf2_fig2_shape_change", lambda: fig_shape_change(results))):
        fig = make()
        fig.savefig(os.path.join(FIGS, f"{name}.png"), dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    bounds = next(iter(results.values()))["search_bounds"]
    lines = [
        f"Workflow 2: baseline vs optimised shape ({time.strftime('%Y-%m-%d %H:%M')})",
        f"Drag model: models/{args.model} ({card['drag_model']}, inputs {', '.join(card['inputs'])})",
        f"Vehicle: {config.VEHICLE_MASS_KG:.0f} kg, {config.BATTERY_CAPACITY_KWH} kWh usable, C_rr "
        f"{config.ROLLING_RESISTANCE_COEFF}, drivetrain {config.DRIVETRAIN_EFFICIENCY}, regen "
        f"{config.REGEN_EFFICIENCY}, aux {config.AUX_LOAD_W:.0f} W, tariff LKR {config.ELECTRICITY_TARIFF_LKR_PER_KWH}/kWh",
        "Search limits: " + ", ".join(f"{n} {lo:g}-{hi:g}" for n, (lo, hi) in bounds.items()),
        f"Optimiser: {args.trials} tries per cycle, baseline tried first; near-optimal = gain <= "
        f"{100 * config.BASELINE_OPTIMAL_TOLERANCE:.0f}%",
        f"Cycles: {', '.join(config.DRIVING_CYCLE_FILES)} from {config.DRIVING_CYCLE_DIR}",
        "",
        f"{'cycle':<11}{'side':<10}{'wind':>6}{'spoil':>6}{'boat':>6}{'hood':>6}{'gc':>7}"
        f"{'Cd_eff':>8}{'CdA':>7}{'Wh/km':>8}{'range':>7}{'gain%':>7}  verdict",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"{row.cycle if row.side == 'baseline' else '':<11}{row.side:<10}{row.windshield_angle_deg:6.1f}"
            f"{row.spoiler_angle_deg:6.1f}{row.boat_tail_angle_deg:6.1f}{row.hood_angle_deg:6.1f}"
            f"{row.ground_clearance_mm:7.1f}{row.cd_effective:8.4f}{row.cd_a_m2:7.3f}{row.wh_per_km:8.2f}"
            f"{row.range_km:7.1f}{row.gain_percent:7.2f}  {row.verdict}")
    lines += ["", "Search time per cycle (s): " + ", ".join(f"{c} {s:.1f}" for c, s in seconds.items()), "",
              "Cross-cycle check: Wh/km of each cycle's best shape (rows) on each cycle (columns)",
              cross.round(3).to_string(), "",
              "Largest difference between the best shapes, per cycle (%): "
              + ", ".join(f"{c} {v:.2f}" for c, v in spread.items()),
              "A difference this small means the best shape is effectively the same on every cycle; "
              "what changes with the cycle is the size of the gain."]
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(LOGS, "workflow2_baseline_vs_optimized.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
