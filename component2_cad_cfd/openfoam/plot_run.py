#!/usr/bin/env python3
"""Plot a simpleFoam run: residuals + Cd/Cl history (for the panel, and to watch a run live).

  python3 plot_run.py CASE_DIR                 # writes CASE_DIR/run_history.png (use after / during a run)
  python3 plot_run.py CASE_DIR --live          # opens a window that refreshes every 5 s while the solver runs
Needs: numpy matplotlib (no OpenFOAM needed).
"""
import argparse, glob, os, re, sys
import numpy as np

def read_residuals(log):
    it = None; data = {}; its = []
    for ln in open(log, errors="ignore"):
        m = re.match(r"^Time = (\d+)\s*$", ln)
        if m: it = int(m.group(1)); its.append(it); continue
        m = re.match(r"^\s*(?:smoothSolver|GAMG|PCG|PBiCGStab|DILUPBiCG)?:?\s*Solving for (\w+), Initial residual = ([\d.eE+-]+)", ln)
        if m and it is not None:
            data.setdefault(m.group(1), {})[it] = float(m.group(2))
    return data

def read_coeffs(case):
    fc = sorted(glob.glob(os.path.join(case, "postProcessing", "forceCoeffs1", "*", "*.dat")))
    if not fc: return None
    hdr = None; rows = []
    for ln in open(fc[-1], errors="ignore"):
        if ln.startswith("#"):
            if "Time" in ln: hdr = ln.lstrip("# ").split()
        elif ln.strip():
            try: rows.append([float(x) for x in ln.split()])
            except ValueError: pass
    if not rows or not hdr: return None
    rows = np.array([r for r in rows if len(r) == len(hdr)]); col = {h: j for j, h in enumerate(hdr)}
    return rows[:, 0], rows[:, col["Cd"]], rows[:, col["Cl"]]

def draw(case, axes, avg=100):
    ax1, ax2 = axes
    for a in axes: a.clear()
    res = read_residuals(os.path.join(case, "log.simpleFoam"))
    for name, d in sorted(res.items()):
        it = sorted(d); ax1.semilogy(it, [d[i] for i in it], label=name, lw=1.1)
    ax1.axhline(1e-4, color="k", ls="--", lw=0.8); ax1.text(0.99, 1e-4, " target 1e-4", ha="right", va="bottom", transform=ax1.get_yaxis_transform(), fontsize=8)
    ax1.set_xlabel("iteration"); ax1.set_ylabel("initial residual"); ax1.set_title("Residuals (lower = more converged)"); ax1.legend(ncol=3, fontsize=8); ax1.grid(alpha=0.3)
    cf = read_coeffs(case)
    if cf is not None:
        t, cd, cl = cf
        ax2.plot(t, cd, label="Cd", color="#c0392b", lw=1.2); ax2.plot(t, cl, label="Cl", color="#2471a3", lw=1.2)
        if len(cd) > avg:
            m = cd[-avg:].mean(); s = cd[-avg:].std()
            ax2.axvspan(t[-avg], t[-1], color="#c0392b", alpha=0.08)
            ax2.set_title("Cd, Cl history  (Cd last %d it: %.4f ± %.4f = %.1f%%)" % (avg, m, s, 100 * s / abs(m)))
        else: ax2.set_title("Cd, Cl history")
        ax2.legend(fontsize=8)
    else: ax2.set_title("Cd, Cl history (no forceCoeffs data yet)")
    ax2.set_xlabel("iteration"); ax2.grid(alpha=0.3)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("case"); ap.add_argument("--live", action="store_true"); ap.add_argument("--avg", type=int, default=100)
    a = ap.parse_args()
    import matplotlib
    if not a.live: matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if a.live:
        plt.ion(); plt.show(block=False)
        while plt.fignum_exists(fig.number):
            try: draw(a.case, axes, a.avg); fig.tight_layout()
            except Exception as e: print("waiting for data...", e)
            plt.pause(5)
    else:
        draw(a.case, axes, a.avg); fig.suptitle(os.path.basename(os.path.abspath(a.case)), fontsize=10); fig.tight_layout()
        out = os.path.join(a.case, "run_history.png"); fig.savefig(out, dpi=160); print("wrote", out)

if __name__ == "__main__":
    main()
