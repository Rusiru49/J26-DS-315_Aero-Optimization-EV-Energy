"""
PP1 demo, key function 1: physics-informed labelling of a NEW car shape, with the
per-label checks.

Builds the Workflow 1 labeller (POD pressure patterns + Gaussian Process) on Member 2's
real CFD at one speed, gives the shape a Cd label with its calibrated uncertainty, and
runs the four checks every synthetic label must pass before it may be used for training.

Usage:
    python scripts/label_shape.py                           # 26 5 12 12 140 at 100 km/h
    python scripts/label_shape.py 24 10 6 13 120            # windshield spoiler boat-tail hood clearance
    python scripts/label_shape.py 24 10 6 13 120 --speed 40
"""

import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")  # GP fitting warnings would bury the demo output

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np

from component3 import config
from component3.data.loader import CFDDataset
from component3.workflow1.evaluation import _unit, build_labeller
from component3.workflow1.label_quality import (CD_RANGE_MARGIN, CP_STAGNATION_LIMIT, calibrated_sigma, check_labels,
                                                predict_labels, region_limit)
from component3.workflow1.pod import dataset_arrays

COUNTS = 1000  # 1 drag count = 0.001 Cd


def main():
    args = sys.argv[1:]
    speed = 100
    if "--speed" in args:
        i = args.index("--speed")
        speed = int(args[i + 1])
        args = args[:i] + args[i + 2:]
    values = list(map(float, args)) if args else [26, 5, 12, 12, 140]
    names = config.SHAPE_PARAM_NAMES
    with open(os.path.join(config.MODELS_DIR, "member2_sedan_pp1", "model_card.json")) as fh:
        cal = json.load(fh)["synthetic_label_uncertainty_calibration"][str(speed)]

    print(f"=== Physics-informed label for a new shape at {speed} km/h ===")
    for n, v in zip(names, values):
        print(f"   {n:22s} {v:8.2f}")

    t0 = time.perf_counter()
    ds = CFDDataset(config.REAL_DATA_DIR, speed)
    a = dataset_arrays(ds, ds.runs.reset_index(drop=True))
    pod, helper, (kind, method, k) = build_labeller(a)
    print(f"\n1. Labeller built on {len(a['cd'])} real CFD shapes: {k} POD pressure patterns + Gaussian Process "
          f"({kind}), chosen by cross-validation  [{time.perf_counter() - t0:.1f} s]")

    u = _unit(np.array([values]))
    amps, cd, sigma = predict_labels(helper, pod, u, method)
    s = float(calibrated_sigma(sigma, (cal["a"], cal["b"]))[0])
    print(f"2. Label: rebuilt the surface pressure from the patterns and integrated it over the car")
    print(f"   Cd = {cd[0]:.4f} +/- {s * COUNTS:.0f} drag counts (uncertainty calibrated on held-out real CFD)")

    c = check_labels(pod, amps, u, cd, a["x_unit"], a["cd"]).iloc[0]
    lo, hi = a["cd"].min() - CD_RANGE_MARGIN, a["cd"].max() + CD_RANGE_MARGIN
    mark = lambda ok: "PASS" if ok else "FAIL"
    print("3. Per-label physics checks:")
    print(f"   [{mark(c.cp_below_stagnation)}] pressure never above the stagnation limit (max Cp {c.max_cp:.2f} "
          f"<= {CP_STAGNATION_LIMIT:.1f})")
    print(f"   [{mark(c.pressure_drag_positive)}] pressure drag is positive (air pushes the car backwards)")
    print(f"   [{mark(c.cd_in_range)}] Cd inside the real CFD range ({lo:.3f} - {hi:.3f})")
    print(f"   [{mark(c.in_validated_region)}] shape inside the validated region (distance to nearest real shape "
          f"{c.distance_to_real:.2f} <= {region_limit(a['x_unit']):.2f})")
    print(f"\n=> label {'KEPT: it may be used for training' if c.keep else 'REMOVED: it is never used for training'}")


if __name__ == "__main__":
    main()
