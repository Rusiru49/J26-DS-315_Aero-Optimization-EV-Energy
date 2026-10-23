"""
PP1 diagrams for Component 3 (drawn, not computed from results).

Usage:  python scripts/make_pp1_diagrams.py
Writes to outputs/figures/:
    pp1_component3_architecture.png   Workflow 1 (done) and Workflow 2 (next), for the design slide
    pp1_workflow1_methodology.png     how the drag model is built and validated, for the methodology slide
    pp1_design_excellence.png         key design decisions + my deliverables, for the design excellence slide
    pp1_user_feedback.png             feedback -> action -> status, for the user feedback slide
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


def design_excellence():
    """Design excellence slide, in the team's style: the key design decisions, each with its
    reason, and the deliverables of the individual contribution. (The process itself is on the
    methodology slide.)"""
    fig, ax = plt.subplots(figsize=(18, 7.55))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.set_xlim(0, 18); ax.set_ylim(0, 7.5); ax.axis("off")
    NAVY, WHY_BG, WHY_INK, BORDER = "#14243F", "#E3F3EC", "#1E6B4E", "#D9DEE6"

    def card(x, y, w, h, face="white", edge=BORDER, lw=1.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.15", facecolor=face,
                                    edgecolor=edge, linewidth=lw))

    cw, cg = (17.8 - 2 * 0.25) / 3, 0.25
    ax.text(0.1, 7.27, "KEY DESIGN DECISIONS", fontsize=11, weight="bold", color=NAVY, va="center")
    decisions = [
        ("Labels from pressure physics", "Cd is integrated from the rebuilt surface pressure.",
         "Synthetic labels follow the same physics as CFD."),
        ("Every label is checked", "Physical limits, CFD range and validated region.",
         "No unphysical value ever enters training."),
        ("Hide whole designs, at every speed", "Synthetic data is rebuilt in every fold.",
         "Nothing leaks from test shapes into training."),
        ("Judge only on real CFD", "Synthetic shapes train the model, never test it.",
         "The score measures truth, not the labeller."),
        ("Augmentation only if it helps", "Real-only and augmented models compete.",
         "It cannot be fooled by its own data."),
        ("One model, speed as an input", "One model gives Cd for every speed in the CFD data.",
         "Scales to new speeds and driving cycles."),
    ]
    ch = 1.7
    for i, (title, what, why) in enumerate(decisions):
        cx = 0.1 + (i % 3) * (cw + cg)
        cy = 5.25 - (i // 3) * (ch + 0.2)
        card(cx, cy, cw, ch)
        ax.add_patch(plt.Circle((cx + 0.32, cy + ch - 0.33), 0.19, facecolor="#EEF1F5", edgecolor="none"))
        ax.text(cx + 0.32, cy + ch - 0.33, str(i + 1), fontsize=10, weight="bold", color=NAVY, ha="center",
                va="center")
        ax.text(cx + 0.65, cy + ch - 0.33, title, fontsize=12, weight="bold", color=NAVY, va="center")
        ax.text(cx + 0.2, cy + ch - 0.78, what, fontsize=9.5, color=GREY, va="center")
        ax.add_patch(FancyBboxPatch((cx + 0.15, cy + 0.14), cw - 0.3, 0.5, boxstyle="round,pad=0,rounding_size=0.08",
                                    facecolor=WHY_BG, edgecolor="none"))
        ax.text(cx + 0.28, cy + 0.39, "Why:", fontsize=9.5, weight="bold", color=WHY_INK, va="center")
        ax.text(cx + 0.85, cy + 0.39, why, fontsize=9.5, color=WHY_INK, va="center")

    ax.text(0.1, 3.0, "MY CONTRIBUTION: WHAT I DELIVERED", fontsize=11, weight="bold", color=NAVY, va="center")
    delivered = [
        ("Physics labelling method", "POD pressure modes + Gaussian Process\nlabel unseen shapes (Novelty 1)"),
        ("Label validation layer", "per-label physics checks and uncertainty\ncalibrated on held-out real CFD"),
        ("Validated drag model", "Cd and frontal area for any shape and\nspeed, selected by cross-validation"),
        ("Integration interfaces", "data contract with Member 2 and the\nagreed output format for Member 4"),
        ("Energy model (Workflow 2 start)", "battery use, range and cost per driving\ncycle; awaits the final cycles"),
        ("Reproducible, tested codebase", "Python package, automated tests, model\ncard and a frozen blind-test model"),
    ]
    dh = 1.2
    for i, (title, what) in enumerate(delivered):
        dx = 0.1 + (i % 3) * (cw + cg)
        dy = 1.5 - (i // 3) * (dh + 0.2)
        card(dx, dy, cw, dh, face=NAVY, edge=NAVY)
        ax.text(dx + 0.25, dy + dh - 0.3, title, fontsize=12, weight="bold", color="white", va="center")
        ax.text(dx + 0.25, dy + dh - 0.55, what, fontsize=9.5, color="#B5C3D6", va="top", linespacing=1.4)
    return fig


def feedback():
    """User feedback slide: feedback from the people who consume or supply this component
    (feedback -> action -> status) and the planned task-based user evaluation."""
    import textwrap
    fig, ax = plt.subplots(figsize=(18, 6.7))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.set_xlim(0, 18); ax.set_ylim(0, 6.65); ax.axis("off")
    NAVY, BORDER = "#14243F", "#D9DEE6"
    DONE, WAIT = ("#E3F3EC", "#1E6B4E"), ("#FDF1E4", "#B8650F")
    items = [
        ("M4", "Member 4", "Dashboard developer (consumer of model outputs)", "#A8325E",
         "The dashboard requires quantified energy, range and cost savings relative to the baseline design.",
         "Extended the output schema with baseline-versus-optimised comparisons of energy, range and cost.",
         "Agreed and specified", DONE),
        ("M2", "Member 2", "CFD data provider (supplier of training data)", "#1F5FA8",
         "The delivered CFD dataset uses a revised design space and flags several morphs as low-confidence.",
         "The pipeline now derives the design space from the dataset and restricts low-confidence designs "
         "to training only.",
         "Implemented", DONE),
        ("M1", "Member 1", "Driving-cycle provider (input to the energy model)", "#2E7D4F",
         "Two Sri Lankan driving cycles have been delivered; the remaining cycles are under development.",
         "The energy model accepts cycles in the agreed 1 Hz format, so the delivered cycles are used "
         "directly and the rest are added on release.",
         "2 cycles delivered", WAIT),
    ]
    cw, cg, top, bottom = (17.8 - 2 * 0.25) / 3, 0.25, 6.55, 1.75
    for i, (tag, name, role, accent, said, done, status, (chip_bg, chip_ink)) in enumerate(items):
        x = 0.1 + i * (cw + cg)
        ax.add_patch(FancyBboxPatch((x, bottom), cw, top - bottom, boxstyle="round,pad=0,rounding_size=0.15",
                                    facecolor="white", edgecolor=BORDER, linewidth=1.2))
        ax.add_patch(FancyBboxPatch((x, top - 0.09), cw, 0.09, boxstyle="square,pad=0", facecolor=accent,
                                    edgecolor="none"))
        ax.add_patch(plt.Circle((x + 0.55, top - 0.62), 0.32, facecolor=accent, edgecolor="none"))
        ax.text(x + 0.55, top - 0.62, tag, fontsize=11, weight="bold", color="white", ha="center", va="center")
        ax.text(x + 1.05, top - 0.48, name, fontsize=13, weight="bold", color=NAVY, va="center")
        ax.text(x + 1.05, top - 0.8, role, fontsize=9.5, color=GREY, va="center")
        ax.plot([x + 0.25, x + cw - 0.25], [top - 1.15, top - 1.15], color=BORDER, linewidth=1)
        ax.text(x + 0.3, top - 1.45, "FEEDBACK", fontsize=9, weight="bold", color=accent, va="center")
        ax.text(x + 0.3, top - 1.7, "“" + "\n".join(textwrap.wrap(said, 52)) + "”", fontsize=10.5, style="italic",
                color=NAVY, va="top", linespacing=1.4)
        ax.text(x + cw / 2, top - 2.62, "↓", fontsize=15, color=accent, ha="center", va="center")
        ax.text(x + 0.3, top - 2.95, "ACTION TAKEN", fontsize=9, weight="bold", color=accent, va="center")
        ax.text(x + 0.3, top - 3.2, "\n".join(textwrap.wrap(done, 54)), fontsize=10.5, color=NAVY, va="top",
                linespacing=1.4)
        ax.add_patch(FancyBboxPatch((x + 0.3, bottom + 0.2), 0.55 + 0.11 * len(status), 0.38,
                                    boxstyle="round,pad=0,rounding_size=0.18", facecolor=chip_bg, edgecolor="none"))
        ax.text(x + 0.46, bottom + 0.39, "●  " + status, fontsize=9.5, weight="bold", color=chip_ink, va="center")

    ax.add_patch(FancyBboxPatch((0.1, 0.1), 17.8, 1.4, boxstyle="round,pad=0,rounding_size=0.15",
                                facecolor="#F1EEFA", edgecolor="none"))
    ax.text(0.4, 1.12, "PLANNED USER EVALUATION", fontsize=10, weight="bold", color="#5B45A8", va="center")
    ax.text(0.4, 0.62, "A task-based session with EV design practitioners on the integrated dashboard: evaluate a "
            "candidate shape, interpret the predicted drag and its uncertainty,\nrecord observed difficulties, and "
            "convert the findings into Planner tasks.", fontsize=10.5, color=NAVY, va="center", linespacing=1.4)
    return fig


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, make in (("pp1_component3_architecture", architecture), ("pp1_workflow1_methodology", methodology),
                       ("pp1_design_excellence", design_excellence), ("pp1_user_feedback", feedback)):
        fig = make()
        path = os.path.join(OUT, f"{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print("wrote", path)


if __name__ == "__main__":
    main()
