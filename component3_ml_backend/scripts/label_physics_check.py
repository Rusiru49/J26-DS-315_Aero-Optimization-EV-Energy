"""
Are the generated (synthetic) Cd labels physically consistent with the real CFD?
For each speed: build the Workflow 1 labeller on all real designs, label 1,200
LHS shapes, and compare (1) the Cd range and mean and (2) the effect of each
shape parameter (linear-fit slope x the parameter's range, in drag counts)
between the generated labels and the real CFD.

Usage:  python scripts/label_physics_check.py [data_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np
from sklearn.linear_model import LinearRegression

from component3 import config
from component3.data.loader import CFDDataset
from component3.workflow1.evaluation import augmented_table, build_labeller
from component3.workflow1.pod import dataset_arrays


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else config.REAL_DATA_DIR
    names = config.SHAPE_PARAM_NAMES
    short = [n.replace("_angle_deg", "").replace("_mm", "") for n in names]
    print("=== Are the generated (synthetic) Cd labels physically consistent with the real CFD? ===")
    for s in (40, 100):
        ds = CFDDataset(data_dir, s)
        a = dataset_arrays(ds, ds.runs.reset_index(drop=True))
        pod, helper, (kind, method, k) = build_labeller(a)
        x, y = augmented_table(a, pod, helper, method, config.RANDOM_STATE)
        xs, ys = x[len(a["cd"]):], y[len(a["cd"]):]
        span = np.array([config.SHAPE_PARAM_BOUNDS[n][1] - config.SHAPE_PARAM_BOUNDS[n][0] for n in names])
        real = LinearRegression().fit(a["x_raw"], a["cd"]).coef_ * span * 1000
        syn = LinearRegression().fit(xs, ys).coef_ * span * 1000
        print(f"\n--- {s} km/h (labeller: {kind}+{method}, {k} patterns; {len(ys):,} generated shapes)")
        print(f"   range: generated {ys.min():.4f}-{ys.max():.4f}, real CFD {a['cd'].min():.4f}-{a['cd'].max():.4f}; "
              f"generated mean {ys.mean():.4f}, real mean {a['cd'].mean():.4f}")
        print("   effect of each parameter, min -> max of its range (drag counts):")
        print(f"   {'parameter':18s} {'real CFD (30)':>14s} {'generated':>12s}  same direction?")
        for n, r, g in zip(short, real, syn):
            print(f"   {n:18s} {r:+14.1f} {g:+12.1f}  {'yes' if np.sign(r) == np.sign(g) else 'NO'}")


if __name__ == "__main__":
    main()
