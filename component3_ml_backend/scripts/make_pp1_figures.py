"""
PP1 evidence figures for Workflow 1, made from the saved results of
run_workflow1.py (outputs/workflow1_<dataset>_<speed>kmh.json and
..._predictions.csv). Nothing is retrained here.

Usage:
    python scripts/make_pp1_figures.py            # Member 2's PP1 data (30 shapes)

Writes to outputs/figures/:
    pp1_fig1_model_errors.png        cross-validated error of every candidate, both speeds
    pp1_fig2_predicted_vs_cfd.png    chosen drag model: predicted vs CFD Cd per design
    pp1_fig3_more_data.png           24 vs 30 shapes: labeller and models improve with data
    pp1_fig4_helper_versions.png     five helper versions (experiments/helper_study/compare_helpers.py)
    pp1_fig5_augmentation_size.png   error vs number of synthetic shapes
    pp1_fig6_original_vs_augmented.png  slide figure: % error, original vs augmented dataset, per model
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from component3 import config

OUT = os.path.join(config.OUTPUTS_DIR, "figures")
DATASET, OLD_DATASET = "member2_sedan_pp1", "for_component3_full"
SPEEDS = (40, 100)
SPEED_NAME = {40: "40 km/h (urban)", 100: "100 km/h (highway)"}
# reference palette (dataviz skill), light mode: slot 1 blue, slot 2 orange
AUG, REAL = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LABELS = {
    "xgboost+real": "XGBoost, real only", "xgboost+aug": "XGBoost + augmentation",
    "random_forest+real": "Random Forest, real only", "random_forest+aug": "Random Forest + augmentation",
    "ann_mlp+real": "ANN, real only", "ann_mlp+aug": "ANN + augmentation",
    "ann_mlp+aug_weighted": "ANN + augmentation, real weighted",
    "ann_mlp+pretrain_finetune": "ANN + augmentation, pre-train/fine-tune",
}


def load(dataset, speed, suffix=""):
    with open(os.path.join(config.OUTPUTS_DIR, f"workflow1_{dataset}_{speed}kmh{suffix}.json")) as fh:
        summary = json.load(fh)
    return summary, pd.DataFrame(summary["results_counts"]).set_index("candidate")


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_model_errors():
    order = list(LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, speed in zip(axes, SPEEDS):
        summary, t = load(DATASET, speed)
        rows = [c for c in order if c in t.index]
        y = np.arange(len(rows))[::-1]
        colours = [REAL if c.endswith("+real") else AUG for c in rows]
        ax.barh(y, t.loc[rows, "mae"], xerr=t.loc[rows, "mae_spread"], color=colours, height=0.62,
                edgecolor="white", linewidth=2, error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 3})
        for yi, c in zip(y, rows):
            ax.text(t.loc[c, "mae"] + t.loc[c, "mae_spread"] + 0.3, yi, f"{t.loc[c, 'mae']:.1f}",
                    va="center", fontsize=10, color=INK)
        label = t.loc["label", "mae"]
        ax.axvline(label, color=MUTED, linestyle="--", linewidth=1.2)
        ax.text(label, len(rows) - 0.35, f" labeller error {label:.1f}", color=MUTED, fontsize=9, va="bottom")
        ax.set_yticks(y, [LABELS[c] for c in rows], fontsize=10, color=INK)
        ax.set_title(SPEED_NAME[speed], fontsize=12, color=INK, loc="left")
        ax.set_xlabel("Cross-validated error on real CFD designs (drag counts, 1 = 0.001 Cd)", fontsize=10, color=MUTED)
        style(ax)
        chosen = summary["drag_model"]
        ax.text(0.99, 0.02, f"chosen: {LABELS[chosen]}", transform=ax.transAxes, ha="right", fontsize=9, color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=REAL), plt.Rectangle((0, 0), 1, 1, color=AUG)]
    fig.legend(handles, ["trained on the real CFD designs only", "trained with physics-labelled augmentation"],
               loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=10)
    fig.suptitle(f"Workflow 1 on Member 2's real CFD ({summary['n_designs']} shapes, one model with speed as input): which drag model is most accurate?",
                 fontsize=13, color=INK, y=1.09)
    fig.text(0.01, -0.04, f"Validation: {summary['validation']}. Bars = mean over repeats, "
             "whiskers = spread between repeats. Lower is better.", fontsize=9, color=MUTED)
    return fig


def fig_predicted_vs_cfd():
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, speed in zip(axes, SPEEDS):
        summary, _ = load(DATASET, speed)
        chosen = summary["drag_model"]
        oof = pd.read_csv(os.path.join(config.OUTPUTS_DIR, f"workflow1_{DATASET}_{speed}kmh_predictions.csv"))
        per = oof.groupby("run_id")[["cd", chosen]].mean()
        lo, hi = per.min().min() - 0.01, per.max().max() + 0.01
        line = np.array([lo, hi])
        ax.fill_between(line, line - 0.010, line + 0.010, color=GRID, label="within 10 drag counts")
        ax.plot(line, line, color=MUTED, linewidth=1, label="perfect prediction")
        ax.scatter(per["cd"], per[chosen], s=46, color=REAL if chosen.endswith("+real") else AUG,
                   edgecolor="white", linewidth=1.5, zorder=3, label=f"{len(per)} test designs")
        for rid, r in per.iterrows():
            if abs(r[chosen] - r["cd"]) > 0.025:
                ax.annotate(rid, (r["cd"], r[chosen]), xytext=(-34, 4), textcoords="offset points", fontsize=8, color=MUTED)
        # the same measure as the results table: mean error over all held-out predictions
        mae = (oof[chosen] - oof["cd"]).abs().mean() * 1000
        ax.set_title(f"{SPEED_NAME[speed]}\n{LABELS[chosen]}, mean error {mae:.1f} counts", fontsize=11, color=INK, loc="left")
        ax.set_xlim(lo, hi), ax.set_ylim(lo, hi), ax.set_aspect("equal")
        ax.set_xlabel("Cd from CFD (Member 2)", fontsize=10, color=MUTED)
        ax.set_ylabel("Cd predicted (design never seen in training)", fontsize=10, color=MUTED)
        style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("Chosen drag model vs real CFD, each design predicted while it was held out", fontsize=13, color=INK, y=1.03)
    fig.text(0.01, -0.02, "Each dot is one design, its prediction averaged over the 3 repeats. Labelled: designs more than 25 counts off.",
             fontsize=9, color=MUTED)
    return fig


def fig_more_data():
    rows = [("label", "Labeller\n(synthetic labels)"), ("xgboost+aug", "XGBoost\n+ augmentation"),
            ("xgboost+real", "XGBoost,\nreal only"), ("ann_mlp+real", "ANN,\nreal only")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    for ax, speed in zip(axes, SPEEDS):
        # same helper (15 patterns) on both: the H3 improvement is shown separately in figure 4
        old_s, old = load(OLD_DATASET, speed)
        new_s, new = load(DATASET, speed, suffix="_before_H3")
        x = np.arange(len(rows))
        ax.bar(x - 0.19, [old.loc[c, "mae"] for c, _ in rows], 0.36, color="#c3c2b7", edgecolor="white", linewidth=2,
               label=f"{old_s['n_designs']} shapes")
        ax.bar(x + 0.19, [new.loc[c, "mae"] for c, _ in rows], 0.36, color=AUG, edgecolor="white", linewidth=2,
               label=f"{new_s['n_designs']} shapes (PP1 data)")
        for xi, (c, _) in zip(x, rows):
            ax.text(xi - 0.19, old.loc[c, "mae"] + 0.3, f"{old.loc[c, 'mae']:.1f}", ha="center", fontsize=9, color=MUTED)
            ax.text(xi + 0.19, new.loc[c, "mae"] + 0.3, f"{new.loc[c, 'mae']:.1f}", ha="center", fontsize=9, color=INK)
        ax.set_xticks(x, [n for _, n in rows], fontsize=9, color=INK)
        ax.set_title(SPEED_NAME[speed], fontsize=12, color=INK, loc="left")
        ax.set_ylabel("Cross-validated error (drag counts)", fontsize=10, color=MUTED)
        style(ax)
        ax.grid(axis="x", visible=False), ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_ylim(0, 21)
    fig.legend([plt.Rectangle((0, 0), 1, 1, color="#c3c2b7"), plt.Rectangle((0, 0), 1, 1, color=AUG)],
               [f"{old_s['n_designs']} shapes (earlier delivery)", f"{new_s['n_designs']} shapes (PP1 data)"],
               loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=10)
    fig.suptitle("More real CFD shapes make the labeller and the augmented models more accurate", fontsize=13, color=INK, y=1.09)
    return fig


def fig_helper_versions():
    path = os.path.join(config.COMPONENT_ROOT, "experiments", "helper_study", "results", "compare_helpers_{}kmh.json")
    names = {"H0": "H0\ntoday", "H1": "H1\nlinear trend", "H3": "H3 patterns\nchosen by CV",
             "H1+H3": "H1 + H3", "H2": "H2 = H1+H3\n+ correction"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, speed in zip(axes, SPEEDS):
        with open(path.format(speed)) as fh:
            s = json.load(fh)
        t = {r["helper"]: r for r in s["table"]}
        x = np.arange(len(t))
        for off, key, colour, label in ((-0.2, "label", "#c3c2b7", "helper's own labels"),
                                        (0.2, "xgboost", AUG, "XGBoost trained with these labels")):
            vals = [t[h][key][0] for h in t]
            ax.bar(x + off, vals, 0.38, color=colour, edgecolor="white", linewidth=2, label=label)
            for xi, v in zip(x, vals):
                ax.text(xi + off, v + 0.3, f"{v:.1f}", ha="center", fontsize=9, color=INK)
        ann = s["real_only"]["ann_mlp"][0]
        ax.axhline(ann, color=REAL, linestyle="--", linewidth=1.4, label="best model without augmentation (ANN, real only)")
        ax.text(-0.45, ann - 0.9, f"{ann:.1f}", fontsize=9, color=REAL)
        ax.set_xticks(x, [names[h] for h in t], fontsize=9, color=INK)
        ax.set_title(SPEED_NAME[speed], fontsize=12, color=INK, loc="left")
        ax.set_ylabel("Cross-validated error on real CFD (drag counts)", fontsize=10, color=MUTED)
        style(ax)
        ax.grid(axis="x", visible=False), ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_ylim(0, 17)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=10)
    fig.suptitle("Improving the helper: choosing the number of pressure patterns by cross-validation (H3) wins at both speeds",
                 fontsize=13, color=INK, y=1.09)
    return fig


def fig_augmentation_size():
    path = os.path.join(config.COMPONENT_ROOT, "experiments", "augmentation_size", "results", "size_study.json")
    with open(path) as fh:
        s = json.load(fh)
    sizes = [int(n) for n in s["mae_counts"]]
    cols = s["columns"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    styles = {"xgboost_40kmh": (AUG, "-", "o"), "xgboost_100kmh": (AUG, "--", "s"),
              "ann_mlp_40kmh": (MUTED, "-", "o"), "ann_mlp_100kmh": (MUTED, "--", "s")}
    names = {"xgboost_40kmh": "XGBoost, 40 km/h", "xgboost_100kmh": "XGBoost, 100 km/h",
             "ann_mlp_40kmh": "ANN, 40 km/h", "ann_mlp_100kmh": "ANN, 100 km/h"}
    x = np.arange(len(sizes))
    for j, c in enumerate(cols):
        colour, ls, mk = styles[c]
        ax.plot(x, [s["mae_counts"][str(n)][j] for n in sizes], color=colour, linestyle=ls, marker=mk,
                linewidth=2, markersize=8, markeredgecolor="white", label=names[c])
    ax.axvline(list(sizes).index(1200), color=REAL, linestyle=":", linewidth=1.4)
    ax.text(list(sizes).index(1200) + 0.08, 13.5, "used in Workflow 1", color=REAL, fontsize=9)
    ax.set_xticks(x, ["0\n(real only)" if n == 0 else f"{n:,}" for n in sizes], fontsize=10, color=INK)
    ax.set_xlabel("Synthetic shapes added per speed", fontsize=10, color=MUTED)
    ax.set_ylabel("Cross-validated error on real CFD (drag counts)", fontsize=10, color=MUTED)
    ax.set_ylim(0, 20)
    style(ax)
    ax.grid(axis="x", visible=False), ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=9, loc="upper right", ncol=2)
    ax.set_title("Most of the gain comes from the first few hundred synthetic shapes; more than 1,200 adds nothing",
                 fontsize=12, color=INK, loc="left")
    return fig


def fig_original_vs_augmented():
    """Slide figure: mean % error on held-out real CFD, original (real-only) vs augmented
    training, per model, over both speeds."""
    oof = pd.concat([pd.read_csv(os.path.join(config.OUTPUTS_DIR, f"workflow1_{DATASET}_{s}kmh_predictions.csv"))
                     for s in SPEEDS])
    models = (("xgboost", "XGBoost"), ("random_forest", "Random Forest"), ("ann_mlp", "ANN"))
    pct = {c: float(((oof[c] - oof["cd"]).abs() / oof["cd"]).mean() * 100)
           for m, _ in models for c in (f"{m}+real", f"{m}+aug")}
    fig, ax = plt.subplots(figsize=(9, 5))
    x, w = np.arange(len(models)), 0.36
    real = [pct[f"{m}+real"] for m, _ in models]
    aug = [pct[f"{m}+aug"] for m, _ in models]
    ax.bar(x - w / 2, real, w, color=REAL, label="original dataset (real CFD only)", edgecolor="white", linewidth=2)
    ax.bar(x + w / 2, aug, w, color=AUG, label="augmented dataset (real + physics-labelled)", edgecolor="white",
           linewidth=2)
    for xi, r, a in zip(x, real, aug):
        ax.text(xi - w / 2, r + 0.08, f"{r:.1f}%", ha="center", va="bottom", fontsize=11, color=INK)
        ax.text(xi + w / 2, a + 0.08, f"{a:.1f}%", ha="center", va="bottom", fontsize=11, color=INK, weight="bold")
        ax.text(xi, max(r, a) + 0.55, f"error {(r - a) / r:.0%} lower", ha="center", va="bottom", fontsize=10.5,
                color=AUG, weight="bold")
    ax.text(x[0] + w / 2, aug[0] / 2, "final\nmodel", ha="center", va="center", fontsize=10.5, color="white",
            weight="bold")
    ax.set_xticks(x, [n for _, n in models], fontsize=12, color=INK)
    ax.set_ylabel("Mean error on held-out real CFD shapes (%)", fontsize=10.5, color=MUTED)
    ax.set_ylim(0, max(real) + 1.3)
    style(ax)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    ax.set_title("Drag-prediction error: original vs augmented training data", fontsize=13, color=INK, loc="left")
    fig.text(0.01, -0.03, "Cross-validation by design, both speeds; every score is on real CFD shapes the model never "
             "saw. Lower is better.", fontsize=9, color=MUTED)
    return fig


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, make in (("pp1_fig1_model_errors", fig_model_errors), ("pp1_fig2_predicted_vs_cfd", fig_predicted_vs_cfd),
                       ("pp1_fig3_more_data", fig_more_data), ("pp1_fig4_helper_versions", fig_helper_versions),
                       ("pp1_fig5_augmentation_size", fig_augmentation_size),
                       ("pp1_fig6_original_vs_augmented", fig_original_vs_augmented)):
        fig = make()
        path = os.path.join(OUT, f"{name}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("wrote", path)


if __name__ == "__main__":
    main()
