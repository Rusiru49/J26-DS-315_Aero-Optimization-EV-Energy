"""
Sanity check of the saved drag model over the WHOLE design box, not only at the
CFD designs. The optimiser (Workflow 2) will search everywhere in the box, so
the model must not give unbelievable drag anywhere.

Checks:
  1. 10,000 LHS shapes at each CFD speed: predicted Cd must stay within the range
     seen in the CFD data, with a margin of 20 drag counts.
  2. The baseline D000: predicted vs CFD at each speed.
  3. Frontal area over the same shapes: within the range seen in the CFD data
     (with a 2% margin).

Usage:  python scripts/sanity_check_model.py [model_folder] [data_folder]
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import joblib
import numpy as np
import pandas as pd
from scipy.stats import qmc

from component3 import config
from component3.data.loader import read_design_space

COUNTS = 1000
MARGIN_CD = 0.020
MARGIN_AREA = 0.02


def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(config.MODELS_DIR, "member2_sedan_pp1")
    data_dir = sys.argv[2] if len(sys.argv) > 2 else config.REAL_DATA_DIR
    with open(os.path.join(model_dir, "model_card.json")) as fh:
        card = json.load(fh)
    drag = joblib.load(os.path.join(model_dir, card["drag_model_file"]))
    area = joblib.load(os.path.join(model_dir, card["frontal_area_model_file"]))
    space = read_design_space(data_dir)
    if space:
        config.set_design_space(*space)
    runs = pd.read_csv(os.path.join(data_dir, "run_table.csv"))
    label = "Cd_label" if "Cd_label" in runs else "Cd_mean"

    names = config.SHAPE_PARAM_NAMES
    lo = np.array([config.SHAPE_PARAM_BOUNDS[n][0] for n in names])
    hi = np.array([config.SHAPE_PARAM_BOUNDS[n][1] for n in names])
    x = qmc.scale(qmc.LatinHypercube(d=len(names), seed=config.RANDOM_STATE + 7).random(10_000), lo, hi)

    print(f"=== Sanity check of {card['drag_model']} ({model_dir}) over the whole design box ===\n")
    print(f"1. Predicted Cd for 10,000 random shapes inside the box (allowed: CFD range +/- {MARGIN_CD * COUNTS:.0f} counts)")
    ok_all = True
    for s in card["speeds_kmh"]:
        cfd = runs.loc[runs["speed_kmh"] == s, label]
        pred = drag.predict(np.hstack([x, np.full((len(x), 1), float(s))]))
        low, high = cfd.min() - MARGIN_CD, cfd.max() + MARGIN_CD
        outside = float(np.mean((pred < low) | (pred > high)) * 100)
        ok = outside == 0 and np.all(np.isfinite(pred))
        ok_all &= ok
        print(f"   {s} km/h: predicted {pred.min():.4f} to {pred.max():.4f}; CFD {cfd.min():.4f} to {cfd.max():.4f}; "
              f"outside the allowed range: {outside:.2f}%  {'PASS' if ok else 'CHECK'}")

    print("\n2. Baseline D000, predicted vs CFD:")
    base = runs[runs["design_id"] == "D000"]
    for _, r in base.sort_values("speed_kmh").iterrows():
        xb = np.array([[r[n] for n in names] + [float(r["speed_kmh"])]])
        p = float(drag.predict(xb)[0])
        print(f"   {int(r['speed_kmh'])} km/h: predicted {p:.4f}, CFD {r[label]:.4f}, difference {(p - r[label]) * COUNTS:+.1f} counts")

    a = area.predict(x)
    a_lo, a_hi = runs["A_frontal"].min() * (1 - MARGIN_AREA), runs["A_frontal"].max() * (1 + MARGIN_AREA)
    outside_a = float(np.mean((a < a_lo) | (a > a_hi)) * 100)
    ok_all &= outside_a == 0
    print(f"\n3. Frontal area for the same shapes: {a.min():.4f} to {a.max():.4f} m2; CFD {runs['A_frontal'].min():.4f} to "
          f"{runs['A_frontal'].max():.4f} m2; outside (+/- 2%): {outside_a:.2f}%  {'PASS' if outside_a == 0 else 'CHECK'}")
    print(f"\nOverall: {'PASS - no unbelievable values anywhere in the design box' if ok_all else 'CHECK the lines marked CHECK'}")


if __name__ == "__main__":
    main()
