"""
PP1 diagrams for Component 3 (drawn, not computed from results).

Usage:  python scripts/make_pp1_diagrams.py
Writes to outputs/figures/:
    pp1_component3_architecture.png   Workflow 1 (done) and Workflow 2 (next), for the design slide
    pp1_workflow1_methodology.png     how the drag model is built and validated, for the methodology slide
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from component3 import config

OUT = os.path.join(config.OUTPUTS_DIR, "figures")
BG, TEAL, INK, GREY, LINE = "#FCF8F3", "#1E7A80", "#1b1b1b", "#5f5f5a", "#3a3a3a"


def architecture():
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")
    w, h, y1, y2 = 3.45, 2.2, 5.5, 1.3
    xs = [0.3, 4.2, 8.1, 12.0]

    def box(i, y, text, fill, done=True, sub=None):
        x = xs[i]
        face = (TEAL if done else "#d9e6e6") if fill else ("white" if done else BG)
        edge = TEAL if fill else ("#777" if done else "#9a9a94")
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.25", linewidth=1.6,
                                    facecolor=face, edgecolor=edge, linestyle="-" if done else "--"))
        col = "white" if (fill and done) else (INK if done else GREY)
        ax.text(x + w / 2, y + h / 2 + (0.25 if sub else 0), text, ha="center", va="center", fontsize=13, color=col,
                linespacing=1.3)
        if sub:
            ax.text(x + w / 2, y + 0.38, sub, ha="center", va="center", fontsize=10.5, color=col, style="italic")

    def arrow(i, j, y, done=True):
        ax.add_patch(FancyArrowPatch((xs[i] + w, y), (xs[j], y), arrowstyle="-|>", mutation_scale=18, linewidth=1.6,
                                     color=INK if done else "#9a9a94", linestyle="-" if done else "--"))

    ax.text(0.3, 8.35, "Workflow 1 - drag model:  DONE, validated on Member 2's real CFD", fontsize=15, color=TEAL,
            weight="bold")
    box(0, y1, "CFD dataset (Member 2)\nbody shapes at driving\nspeeds + surface pressure", False, sub="automatic data checks")
    box(1, y1, "POD pressure patterns\n+ physics-informed\ndrag labels", True, sub="each label physics-checked")
    box(2, y1, "Train & select by\ncross-validation\nXGBoost / RF / ANN", False, sub="scored on real CFD only")
    box(3, y1, "Drag model\nshape + speed → Cd, area", True, sub="XGBoost + augmentation")
    for i in range(3):
        arrow(i, i + 1, y1 + h / 2)
    ax.text(0.3, 4.2, "Workflow 2 - next, after PP1", fontsize=15, color=GREY, weight="bold")
    box(0, y2, "Member 1\ndriving cycles\n(WLTP + local cycles)", False, done=False)
    box(1, y2, "Energy model\nbattery use, range & cost\nper driving cycle", False, done=False, sub="built; needs final cycles")
    box(2, y2, "Cycle-aware optimiser\nBayesian search within\nstyling limits", True, done=False)
    box(3, y2, "FastAPI / JSON package\n→ Member 4 dashboard", False, done=False, sub="format agreed")
    for i in range(3):
        arrow(i, i + 1, y2 + h / 2, done=False)
    # drag model feeds the energy model: down, left above the Workflow 2 row, down
    mid, grey = 4.65, "#9a9a94"
    ax.plot([xs[3] + w / 2, xs[3] + w / 2, xs[1] + w / 2, xs[1] + w / 2], [y1 - 0.05, mid, mid, y2 + h + 0.4],
            color=grey, linewidth=1.6, linestyle="--")
    ax.add_patch(FancyArrowPatch((xs[1] + w / 2, y2 + h + 0.45), (xs[1] + w / 2, y2 + h + 0.04), arrowstyle="-|>",
                                 mutation_scale=18, linewidth=1.6, color=grey))
    ax.text((xs[1] + xs[3] + w) / 2, mid + 0.18, "drag model feeds the energy model", ha="center", va="bottom",
            fontsize=11, color=GREY, style="italic")
    return fig


def methodology():
    """Two lanes: BUILD (numbered steps, left to right, training shapes only) and
    VALIDATE (held-out real CFD, never used for training), with only the few
    arrows that connect them."""
    fig, ax = plt.subplots(figsize=(17, 9))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.set_xlim(0, 17.1); ax.set_ylim(0, 9); ax.axis("off")
    LIGHT_TEAL, LIGHT_GREY = "#E3EFEF", "#ECEAE4"

    ax.text(8.55, 8.65, "Workflow 1: how the drag model is built and validated", fontsize=17, color=TEAL,
            weight="bold", ha="center", va="center")

    # lanes
    ax.add_patch(FancyBboxPatch((0.1, 4.3), 16.9, 3.95, boxstyle="round,pad=0,rounding_size=0.25",
                                facecolor=LIGHT_TEAL, edgecolor="none"))
    ax.add_patch(FancyBboxPatch((0.1, 0.15), 16.9, 3.75, boxstyle="round,pad=0,rounding_size=0.25",
                                facecolor=LIGHT_GREY, edgecolor="none"))
    ax.text(0.35, 7.95, "BUILD", fontsize=15, color=TEAL, weight="bold", va="center")
    ax.text(1.45, 7.95, "the drag model; held-out shapes stay hidden until the winner is chosen", fontsize=12, color=TEAL, va="center")
    ax.text(0.4, 2.75, "VALIDATE", fontsize=15, color=GREY, weight="bold", va="center")
    ax.text(0.4, 2.05, "against held-out\nreal CFD shapes,\nnever used for\ntraining", fontsize=11.5, color=GREY,
            va="center", linespacing=1.35)

    w, h, y = 2.5, 2.75, 4.65
    xs = [0.3 + i * 2.8 for i in range(6)]
    steps = [
        ("Real CFD data", "Member 2's runs\nbody shapes at\ndriving speeds\n+ surface pressure", "automatic data checks", "white"),
        ("Split by design", "training shapes /\nheld-out shapes\nrepeated\ncross-validation", "hidden at every speed", "white"),
        ("Physics labeller", "POD pressure patterns\n+ Gaussian Process\nlabels new\nsampled shapes", None, "teal"),
        ("Per-label checks", "stagnation limit\npositive pressure drag\nCd in CFD range\nin validated region", "failing labels removed", "check"),
        ("Train & select", "XGBoost / RF / ANN\ntrained with and\nwithout the\nchecked labels", "augmentation only if it helps", "white"),
        ("Final drag model", "winner retrained\non all real shapes\nshape + speed → Cd\n+ model card", None, "teal"),
    ]
    for i, (x, (title, body, note, style)) in enumerate(zip(xs, steps), start=1):
        teal = style == "teal"
        ink = "white" if teal else INK
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.2",
                                    linewidth=2.4 if style == "check" else 1.4, facecolor=TEAL if teal else "white",
                                    edgecolor="#888" if style == "white" else TEAL))
        cx = x + w / 2
        ax.add_patch(plt.Circle((cx, y + h), 0.27, facecolor=TEAL, edgecolor="white", linewidth=2.5, zorder=3))
        ax.text(cx, y + h, str(i), ha="center", va="center", fontsize=13, weight="bold", color="white", zorder=4)
        ax.text(cx, y + h - 0.55, title, ha="center", va="center", fontsize=13, weight="bold", color=ink)
        ax.text(cx, y + h / 2 - (0.05 if note else 0.2), body, ha="center", va="center", fontsize=10.5,
                linespacing=1.35, color=ink)
        if note:
            ax.text(cx, y + 0.3, note, ha="center", va="center", fontsize=9, style="italic",
                    color="white" if teal else TEAL)
        if i < 6:
            ax.add_patch(FancyArrowPatch((x + w + 0.03, y + h / 2), (x + 2.8 - 0.03, y + h / 2), arrowstyle="-|>",
                                         mutation_scale=20, linewidth=2.2, color=INK))

    yv, hv = 0.75, 2.1

    def vbox(x, wv, title, body):
        ax.add_patch(FancyBboxPatch((x, yv), wv, hv, boxstyle="round,pad=0.02,rounding_size=0.2", linewidth=1.4,
                                    facecolor="white", edgecolor="#888", linestyle="--"))
        ax.text(x + wv / 2, yv + hv - 0.4, title, ha="center", va="center", fontsize=12.5, weight="bold", color=INK)
        ax.text(x + wv / 2, yv + hv / 2 - 0.25, body, ha="center", va="center", fontsize=10.5, linespacing=1.35,
                color=INK)

    vbox(xs[1], w, "Held-out shapes", "real CFD Cd\n(the answer key)")
    vbox(xs[2], xs[3] + w - xs[2], "Label validation",
         "labeller's Cd vs real Cd\non shapes it never saw\n→ calibrates each label's uncertainty")
    vbox(xs[4], w, "Model scoring", "every candidate\nvs real Cd\nlowest error wins")

    def link(x, up, label, dashed=True):
        y0, y1 = (yv + hv + 0.03, y - 0.03) if up else (y - 0.03, yv + hv + 0.03)
        ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>", mutation_scale=17, linewidth=1.7,
                                     color=LINE, linestyle="--" if dashed else "-"))
        ax.text(x + 0.1, (y0 + y1) / 2, label, ha="left", va="center", fontsize=10.5, color=GREY,
                bbox=dict(boxstyle="round,pad=0.15", facecolor=BG, edgecolor="none"))

    link(xs[1] + w / 2, False, "held-out", dashed=False)
    link(xs[2] + 0.55, False, "predicted Cd")
    link(xs[3] + w - 0.85, True, "uncertainty")
    link(xs[4] + 0.45, False, "predicted Cd")
    link(xs[4] + w - 0.4, True, "error")
    ax.add_patch(FancyArrowPatch((xs[1] + w + 0.03, yv + hv / 2), (xs[2] - 0.03, yv + hv / 2), arrowstyle="-|>",
                                 mutation_scale=17, linewidth=1.7, color=LINE))
    ax.plot([xs[1] + w / 2, xs[1] + w / 2, xs[4] + w / 2], [yv, 0.42, 0.42], color=LINE, linewidth=1.7)
    ax.add_patch(FancyArrowPatch((xs[4] + w / 2, 0.42), (xs[4] + w / 2, yv - 0.03), arrowstyle="-|>",
                                 mutation_scale=17, linewidth=1.7, color=LINE))
    ax.text((xs[1] + xs[4] + w) / 2, 0.42, "real Cd", ha="center", va="center", fontsize=10.5, color=GREY,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#ECEAE4", edgecolor="none"))

    # legend, in the empty corner under step 6
    lx, ly = xs[5] + 0.1, 2.5
    ax.text(lx, ly + 0.25, "How to read", fontsize=11.5, weight="bold", color=INK, va="center")
    ax.add_patch(FancyArrowPatch((lx, ly - 0.25), (lx + 0.7, ly - 0.25), arrowstyle="-|>", mutation_scale=15,
                                 linewidth=2.0, color=INK))
    ax.text(lx + 0.85, ly - 0.25, "main build flow", fontsize=10, color=INK, va="center")
    ax.add_patch(FancyArrowPatch((lx, ly - 0.75), (lx + 0.7, ly - 0.75), arrowstyle="-|>", mutation_scale=15,
                                 linewidth=1.7, color=LINE, linestyle="--"))
    ax.text(lx + 0.85, ly - 0.75, "checked against\nreal CFD", fontsize=10, color=INK, va="center")
    ax.add_patch(FancyBboxPatch((lx, ly - 1.45), 0.7, 0.3, boxstyle="round,pad=0.01,rounding_size=0.06",
                                facecolor=TEAL, edgecolor=TEAL))
    ax.text(lx + 0.85, ly - 1.3, "core method", fontsize=10, color=INK, va="center")
    return fig


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, make in (("pp1_component3_architecture", architecture), ("pp1_workflow1_methodology", methodology)):
        fig = make()
        path = os.path.join(OUT, f"{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        print("wrote", path)


if __name__ == "__main__":
    main()
