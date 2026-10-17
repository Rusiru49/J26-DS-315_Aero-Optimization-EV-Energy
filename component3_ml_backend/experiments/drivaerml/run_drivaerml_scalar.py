"""
Option A test on REAL CFD: the DrivAerML dataset (Ashton et al., 2024,
arXiv:2408.11969, CC-BY-SA 4.0), using only its small per-run files:
force_mom_<i>.csv (Cd, with each design's own frontal area),
geo_parameters_<i>.csv (16 shape parameters, mm or degrees relative to the
baseline) and geo_ref_<i>.csv (aRef = that design's frontal area in m2).

No pressure fields are used, so this tests the final-surrogate part of
Workflow 1 (XGBoost / Random Forest / ANN trained on real Cd only) and the
frontal-area model. It does NOT test POD or the augmentation.

Questions it answers:
  1. With about 80 real runs (our planned size), how accurate are the three
     candidate models on real high-fidelity CFD? (repeated random splits)
  2. How does accuracy grow with the number of runs (40 / 80 / 160 / 320)?
  3. All 16 parameters vs the 5 closest to ours (windscreen, hood, rear-end
     tapering, ride height, decklid height as the spoiler stand-in), where the
     other 11 still vary and act as unexplained variation.
  4. Can a small ridge model predict the frontal area from the shape values?

Usage:
    python scripts/run_drivaerml_scalar.py
Reads data/drivaerml_small/raw/, writes data/drivaerml_small/drivaerml_scalar_table.csv
and outputs/drivaerml_scalar_results.json.
"""

import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from component3 import config
from component3.workflow1.training import AREA_CANDIDATE_BUILDERS, CANDIDATE_BUILDERS

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "drivaerml_small")
RAW_DIR = os.path.join(DATA_DIR, "raw")
ALL_PARAMS = ["Vehicle_Length", "Vehicle_Width", "Vehicle_Height", "Front_Overhang", "Front_Planview", "Hood_Angle",
              "Approach_Angle", "Windscreen_Angle", "Greenhouse_Tapering", "Backlight_Angle", "Decklid_Height",
              "Rearend_tapering", "Rear_Overhang", "Rear_Diffusor_Angle", "Vehicle_Ride_Height", "Vehicle_Pitch"]
OUR_FIVE = ["Windscreen_Angle", "Hood_Angle", "Rearend_tapering", "Vehicle_Ride_Height", "Decklid_Height"]
TRAIN_SIZES = [40, 80, 160, 320]
N_TEST = 100
N_REPEATS = 10


# ------------------------------------------------------------------ #
# 1. BUILD ONE TABLE FROM THE SMALL FILES
# ------------------------------------------------------------------ #
def build_table():
    rows = []
    for path in sorted(glob.glob(os.path.join(RAW_DIR, "run_*_force_mom.csv"))):
        run = int(re.search(r"run_(\d+)_force_mom\.csv$", path).group(1))
        geo = os.path.join(RAW_DIR, f"run_{run}_geo_parameters.csv")
        ref = os.path.join(RAW_DIR, f"run_{run}_geo_ref.csv")
        if not (os.path.exists(geo) and os.path.exists(ref)):
            continue
        force = pd.read_csv(path, skipinitialspace=True).iloc[0]
        params = pd.read_csv(geo, skipinitialspace=True).iloc[0]
        a_ref = float(pd.read_csv(ref, skipinitialspace=True)["aRef"].iloc[0])
        row = {"run_id": run, "Cd": float(force["Cd"]), "Cl": float(force["Cl"]), "frontal_area": a_ref}
        row.update({p: float(params[p]) for p in ALL_PARAMS})
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("run_id").reset_index(drop=True)
    table.to_csv(os.path.join(DATA_DIR, "drivaerml_scalar_table.csv"), index=False)
    return table


# ------------------------------------------------------------------ #
# 2. REPEATED-SPLIT EVALUATION
# ------------------------------------------------------------------ #
def evaluate(table, inputs, target, builders, train_sizes, n_test=N_TEST, n_repeats=N_REPEATS):
    """For each repeat: draw a random test set of n_test runs, then train each
    model on the first n of the remaining runs (shuffled). Returns one row per
    (model, n, repeat) with the test MAE and worst error."""
    x_all = table[inputs].to_numpy(dtype=float)
    y_all = table[target].to_numpy(dtype=float)
    rows = []
    for rep in range(n_repeats):
        rng = np.random.default_rng(config.RANDOM_STATE + rep)
        order = rng.permutation(len(table))
        test_idx, pool = order[:n_test], order[n_test:]
        for n in train_sizes:
            train_idx = pool[:n]
            for name, build in builders.items():
                model = build().fit(x_all[train_idx], y_all[train_idx])
                err = np.abs(model.predict(x_all[test_idx]) - y_all[test_idx])
                rows.append({"model": name, "n_train": n, "repeat": rep,
                             "mae": float(err.mean()), "max_error": float(err.max())})
    return pd.DataFrame(rows)


def summarise(results, scale, unit):
    s = results.groupby(["model", "n_train"]).agg(mae_mean=("mae", "mean"), mae_sd=("mae", "std"),
                                                  max_mean=("max_error", "mean")).reset_index()
    for c in ("mae_mean", "mae_sd", "max_mean"):
        s[c] = s[c] * scale
    s["unit"] = unit
    return s


def print_block(title, summary, unit):
    print(f"\n{title}  (mean over {N_REPEATS} random splits; {unit}; test set {N_TEST} runs)")
    models = list(dict.fromkeys(summary["model"]))
    header = "  model            " + "".join(f"n={n:<16d}" for n in TRAIN_SIZES)
    print(header)
    for m in models:
        cells = []
        for n in TRAIN_SIZES:
            r = summary[(summary["model"] == m) & (summary["n_train"] == n)].iloc[0]
            cells.append(f"{r['mae_mean']:5.1f} ± {r['mae_sd']:4.1f}     ")
        print(f"  {m:16s} " + "".join(cells))


def main():
    table = build_table()
    print(f"DrivAerML small-file table: {len(table)} runs, Cd {table['Cd'].min():.3f} to {table['Cd'].max():.3f} "
          f"(mean {table['Cd'].mean():.3f}, sd {table['Cd'].std() * 1000:.1f} counts), "
          f"frontal area {table['frontal_area'].min():.3f} to {table['frontal_area'].max():.3f} m2")

    drag_builders = dict(CANDIDATE_BUILDERS)
    drag_builders["predict_the_mean"] = lambda: DummyRegressor(strategy="mean")
    out = {"n_runs": len(table), "cd_mean": float(table["Cd"].mean()), "cd_sd_counts": float(table["Cd"].std() * 1000),
           "n_test": N_TEST, "n_repeats": N_REPEATS, "train_sizes": TRAIN_SIZES}

    for label, inputs in (("all 16 parameters", ALL_PARAMS), ("our 5 closest parameters", OUR_FIVE)):
        res = evaluate(table, inputs, "Cd", drag_builders, TRAIN_SIZES)
        summ = summarise(res, 1000, "drag counts")
        print_block(f"Drag (Cd) error, {label}", summ, "drag counts, mean ± sd of MAE")
        out[f"cd_{'all16' if len(inputs) == 16 else 'five'}"] = summ.to_dict("records")

    area_builders = dict(AREA_CANDIDATE_BUILDERS)
    area_builders["predict_the_mean"] = lambda: DummyRegressor(strategy="mean")
    res = evaluate(table, ALL_PARAMS, "frontal_area", area_builders, TRAIN_SIZES)
    res["mae"] = res["mae"] / table["frontal_area"].mean()
    res["max_error"] = res["max_error"] / table["frontal_area"].mean()
    summ = summarise(res, 100, "% of mean area")
    print_block("Frontal area error, all 16 parameters", summ, "% of the mean area, mean ± sd of MAE")
    out["frontal_area_all16"] = summ.to_dict("records")

    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    with open(os.path.join(config.OUTPUTS_DIR, "drivaerml_scalar_results.json"), "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
