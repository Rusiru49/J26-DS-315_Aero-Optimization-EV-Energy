"""
PP1 demo, key function 2: instant drag prediction for a NEW car shape.

Give the 5 shape values (and a speed); the trained Workflow 1 model returns Cd,
frontal area and drag area (Cd x A) in milliseconds, instead of hours of CFD.
Warns when a value is outside the CFD design box (the model is not validated
there).

Usage:
    python scripts/predict_shape.py                         # the baseline sedan D000, both speeds
    python scripts/predict_shape.py 24 10 6 13 120          # windshield spoiler boat-tail hood clearance
    python scripts/predict_shape.py 24 10 6 13 120 --speed 100
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import joblib
import numpy as np

from component3 import config

MODEL_DIR = os.path.join(config.MODELS_DIR, "member2_sedan_pp1")


def main():
    args = sys.argv[1:]
    speeds = [40, 100]
    if "--speed" in args:
        i = args.index("--speed")
        speeds = [float(args[i + 1])]
        args = args[:i] + args[i + 2:]
    with open(os.path.join(MODEL_DIR, "model_card.json")) as fh:
        card = json.load(fh)
    names = card["inputs"][:5]
    shape = dict(card["baseline_shape"]) if not args else dict(zip(names, map(float, args)))

    t0 = time.perf_counter()
    drag = joblib.load(os.path.join(MODEL_DIR, card["drag_model_file"]))
    area = joblib.load(os.path.join(MODEL_DIR, card["frontal_area_model_file"]))
    t_load = time.perf_counter() - t0

    print(f"=== Drag prediction with {card['drag_model']} (trained on {sum(card['n_designs'].values())} real CFD runs) ===")
    print("Shape:")
    for n in names:
        lo, hi = card["design_box"][n]
        flag = "" if lo <= shape[n] <= hi else f"   WARNING: outside the CFD range {lo:g}-{hi:g}"
        print(f"   {n:22s} {shape[n]:8.2f}{flag}")
    x5 = np.array([[shape[n] for n in names]])
    a = float(area.predict(x5)[0])
    for s in speeds:
        if not min(card["speeds_kmh"]) <= s <= max(card["speeds_kmh"]):
            print(f"   WARNING: {s:g} km/h is outside the trained speeds {card['speeds_kmh']}")
        t1 = time.perf_counter()
        cd = float(drag.predict(np.hstack([x5, [[s]]]))[0])
        ms = (time.perf_counter() - t1) * 1000
        err = card["cd_error_by_speed"].get(str(int(s)), {}).get("mae")
        unc = f" (expected error about +/- {err * 1000:.0f} drag counts)" if err else ""
        print(f"\nAt {s:g} km/h: Cd = {cd:.4f}{unc}")
        print(f"   frontal area A = {a:.4f} m2,  drag area Cd x A = {cd * a:.4f} m2,  prediction time {ms:.1f} ms")
    base = card["baseline_shape"]
    if args:
        xb = np.array([[base[n] for n in names] + [speeds[0]]])
        cb = float(drag.predict(xb)[0])
        cd0 = float(drag.predict(np.hstack([x5, [[speeds[0]]]]))[0])
        print(f"\nCompared with the baseline sedan at {speeds[0]:g} km/h: {(cd0 - cb) * 1000:+.1f} drag counts "
              f"({(cd0 - cb) / cb * 100:+.1f}%)")
    print(f"\n(model loaded in {t_load * 1000:.0f} ms; one CFD run of a new shape takes hours)")


if __name__ == "__main__":
    main()
