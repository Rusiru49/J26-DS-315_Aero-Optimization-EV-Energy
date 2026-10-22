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
    box(0, y1, "CFD dataset (Member 2)\n30 shapes x 2 speeds\n+ surface pressure", False, sub="25/25 data checks")
    box(1, y1, "POD pressure patterns\n+ physics-informed\ndrag labels", True, sub="1,200 new shapes (LHS)")
    box(2, y1, "Train & select by\ncross-validation\nXGBoost / RF / ANN", False, sub="scored on real CFD only")
    box(3, y1, "Drag model\nshape + speed -> Cd, area", True, sub="XGBoost + augmentation, ~2.5%")
    for i in range(3):
        arrow(i, i + 1, y1 + h / 2)
    ax.text(0.3, 4.2, "Workflow 2 - next, after PP1", fontsize=15, color=GREY, weight="bold")
    box(0, y2, "Member 1\ndriving cycles\n(WLTP + 5 SLDC)", False, done=False)
    box(1, y2, "Energy model\nbattery use, range & cost\nper driving cycle", False, done=False, sub="built; needs final cycles")
    box(2, y2, "Cycle-aware optimiser\nBayesian search within\nstyling limits", True, done=False)
    box(3, y2, "FastAPI / JSON package\n-> Member 4 dashboard", False, done=False, sub="format agreed")
    for i in range(3):
        arrow(i, i + 1, y2 + h / 2, done=False)
    ax.annotate("", xy=(xs[1] + w / 2, y2 + h + 0.05), xytext=(xs[3] + w / 2, y1 - 0.05),
                arrowprops=dict(arrowstyle="-|>", color="#9a9a94", linewidth=1.6, linestyle="--", mutation_scale=18,
                                connectionstyle="angle,angleA=-90,angleB=180,rad=12"))
    return fig


def methodology():
    fig, ax = plt.subplots(figsize=(17, 9.2))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.set_xlim(0, 16.2); ax.set_ylim(0, 9.2); ax.axis("off")
    boxes = {}

    def box(key, x, y, w, h, text, teal=False, sub=None):
        boxes[key] = (x, y, w, h)
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.18", linewidth=1.5,
                                    facecolor=TEAL if teal else "white", edgecolor=TEAL if teal else "#777"))
        col = "white" if teal else INK
        ax.text(x + w / 2, y + h / 2 + (0.2 if sub else 0), text, ha="center", va="center", fontsize=12.5, color=col,
                weight="bold" if teal else "normal", linespacing=1.25)
        if sub:
            ax.text(x + w / 2, y + 0.27, sub, ha="center", va="center", fontsize=10, color=col, style="italic")

    def path(pts, label=None, at=None, dashed=False):
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=LINE, linewidth=1.4, linestyle="--" if dashed else "-", zorder=1)
        ax.annotate("", xy=pts[-1], xytext=pts[-2],
                    arrowprops=dict(arrowstyle="-|>", color=LINE, lw=1.4, mutation_scale=16))
        if label:
            ax.text(at[0], at[1], label, ha="center", va="center", fontsize=10.5, color=GREY,
                    bbox=dict(facecolor=BG, edgecolor="none", pad=1.5))

    y1, y2, y3, h1, h2, h3 = 7.2, 4.4, 1.3, 1.25, 1.35, 1.3
    box("A", 0.2, y1, 2.7, h1, "Real CFD runs\n(Member 2)", sub="30 shapes x 2 speeds")
    box("B", 3.3, y1, 2.7, h1, "Data verification", sub="25 automatic checks")
    box("C", 6.4, y1, 3.0, h1, "Cross-validation split\n(by design, repeated)", sub="hidden at both speeds")
    box("D", 10.4, y1, 2.6, h1, "POD model\n(pressure patterns)", teal=True)
    box("E", 13.4, y1, 2.6, h1, "Helper model\n(Gaussian Process)", teal=True, sub="settings chosen by CV")
    box("K", 0.2, y2, 2.7, h2, "Held-out runs\n(real CFD)", sub="used to score the models")
    box("I", 3.6, y2, 3.3, h2, "Train & select\nXGBoost / RF / ANN", sub="augmentation kept only if it helps")
    box("H", 7.8, y2, 2.9, h2, "Augmented dataset\n(real + synthetic)", sub="1,200 shapes per speed")
    box("F", 12.4, y2, 3.0, h2, "Physics labeller\n(patterns -> Cd)", teal=True)
    box("J", 3.6, y3, 3.3, h3 + 0.2, "Final drag model\n(shape + speed)", teal=True, sub="+ frontal-area model + model card")
    box("L", 7.8, y3, 2.9, h3, "Label validation\n(vs held-out real CFD)", sub="+ physics-trend check")
    box("G", 13.1, y3, 2.6, h3, "Shape generator\n(Latin hypercube)")

    def mid(k):
        return boxes[k][1] + boxes[k][3] / 2

    path([(2.9, mid("A")), (3.3, mid("B"))])
    path([(6.0, mid("B")), (6.4, mid("C"))])
    path([(9.4, mid("C")), (10.4, mid("D"))], "training\nruns", (9.9, mid("C") + 0.38))
    path([(13.0, mid("D")), (13.4, mid("E"))])
    path([(11.7, y1), (11.7, 6.4), (13.3, 6.4), (13.3, y2 + h2)], "pattern amounts", (12.5, 6.62))
    path([(14.7, y1), (14.7, y2 + h2)])
    path([(14.4, y3 + h3), (14.4, y2)], "new shapes", (14.4, 3.5))
    path([(12.4, mid("F")), (10.7, mid("H"))], "physics-based Cd", (11.55, mid("F") + 0.3))
    path([(9.0, y1), (9.0, y2 + h2)], "real training runs", (9.0, 6.3))
    path([(6.8, y1), (6.8, 6.45), (1.55, 6.45), (1.55, y2 + h2)], "held-out runs", (4.2, 6.45))
    path([(7.8, mid("H")), (6.9, mid("I"))])
    path([(2.9, mid("K")), (3.6, mid("I"))])
    path([(5.25, y2), (5.25, y3 + h3 + 0.2)], "with vs without augmentation:\nlowest error wins,\nretrain on all designs", (6.75, 3.5))
    path([(1.55, y2), (1.55, 0.7), (9.25, 0.7), (9.25, y3)], "real Cd", (5.4, 0.7), dashed=True)
    path([(12.65, y2), (12.65, 1.95), (10.7, 1.95)], "predicted Cd", (11.6, 2.2), dashed=True)
    ax.text(0.2, 8.85, "Workflow 1: how the drag model is built and validated", fontsize=16, color=TEAL, weight="bold")
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
