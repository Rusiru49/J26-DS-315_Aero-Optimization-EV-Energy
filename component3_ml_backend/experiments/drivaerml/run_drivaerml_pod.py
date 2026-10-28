"""
Option B test on REAL CFD: the full Workflow 1 method (POD -> amplitude/shape
-> Cd calibration -> Gaussian Process helper -> Latin Hypercube shapes ->
physics labels -> XGBoost / Random Forest / ANN) on DrivAerML surface
pressure (Ashton et al. 2024, arXiv:2408.11969, CC BY-SA 4.0), using the
community subset DrivaerML-PCTR (100,000 sampled surface points per run,
CpMeanTrim; same license). Same algorithms as workflow1/pod.py,
workflow1/helper.py and workflow1/augmentation.py, generalised from 5 to the 16
DrivAerML shape parameters.

Differences from Member 2's data, all stated in the report:
  * each DrivAerML car is meshed separately and the 100,000 points are random,
    so every run's Cp is MAPPED onto one reference point set (nearest
    neighbours in per-run normalised coordinates, inverse-distance weights).
    This is Member 2's fallback path, and it adds a mapping error;
  * no face normals or areas, so no face-sum Cd check;
  * 16 shape parameters (mm / degrees relative to the baseline), one speed.

Design: 80 runs with fields. Each repeat draws 65 of them for training
(POD, calibration, helper, augmentation, real-only models) and 15 for the
label check. The drag models are scored on the ~400 OTHER DrivAerML runs,
which need only Cd, not fields: a much larger and more reliable test set.

Usage:  python scripts/run_drivaerml_pod.py
Needs:  data/drivaerml_pctr/raw/*.pt (option B files) and
        data/drivaerml_small/drivaerml_scalar_table.csv (run_drivaerml_scalar.py)
Writes: data/drivaerml_pctr/mapped_fields.npz, outputs/drivaerml_pod_results.json
"""

import glob
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from component3 import config
from drivaerml_pt_reader import read_pt
from component3.workflow1.pod import _snapshot_pod
from component3.workflow1.training import CANDIDATE_BUILDERS

PCTR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "drivaerml_pctr")
SCALAR_TABLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "drivaerml_small", "drivaerml_scalar_table.csv")
MAPPED = os.path.join(PCTR_DIR, "mapped_fields.npz")

# DrivAerML morph limits (paper, Table 5): used to scale the inputs to 0-1 and as the LHS box.
LIMITS = {"Vehicle_Length": (-150, 200), "Vehicle_Width": (-100, 100), "Vehicle_Height": (-100, 100),
          "Front_Overhang": (-150, 100), "Front_Planview": (-75, 75), "Hood_Angle": (-50, 50),
          "Approach_Angle": (-40, 30), "Windscreen_Angle": (-150, 150), "Greenhouse_Tapering": (-100, 100),
          "Backlight_Angle": (-100, 200), "Decklid_Height": (-50, 50), "Rearend_tapering": (-90, 70),
          "Rear_Overhang": (-150, 100), "Rear_Diffusor_Angle": (-50, 50), "Vehicle_Ride_Height": (-50, 50),
          "Vehicle_Pitch": (-1, 1)}
PARAMS = list(LIMITS)
LO = np.array([LIMITS[p][0] for p in PARAMS], dtype=float)
HI = np.array([LIMITS[p][1] for p in PARAMS], dtype=float)

N_REF_POINTS = 20000
K_NEIGHBOURS = 4
N_MODES = config.N_POD_MODES
N_TRAIN = 65
N_SYNTHETIC = config.N_SYNTHETIC_SHAPES
N_REPEATS = 10


def unit(x):
    return (np.asarray(x, dtype=float) - LO) / (HI - LO)


# ------------------------------------------------------------------ #
# 1. MAP EVERY RUN'S Cp ONTO ONE REFERENCE POINT SET
# ------------------------------------------------------------------ #
def build_mapped_fields(table):
    paths = sorted(glob.glob(os.path.join(PCTR_DIR, "raw", "boundary_*.pt")))
    runs = [int(re.search(r"boundary_(\d+)\.pt$", p).group(1)) for p in paths]
    runs = [r for r in runs if r in set(table["run_id"])]
    params = table.set_index("run_id").loc[runs, PARAMS].to_numpy(dtype=float)
    # Reference = the run closest to the baseline (all morphs near 0).
    ref_run = runs[int(np.argmin(np.linalg.norm(unit(params) - unit(np.zeros(len(PARAMS))), axis=1)))]
    ref = read_pt(os.path.join(PCTR_DIR, "raw", f"boundary_{ref_run}.pt"))
    rng = np.random.default_rng(config.RANDOM_STATE)
    ref_pts = ref["coords"][rng.choice(len(ref["coords"]), N_REF_POINTS, replace=False)].astype(np.float64)

    fields, nn_dist = [], []
    for r in runs:
        d = read_pt(os.path.join(PCTR_DIR, "raw", f"boundary_{r}.pt"))
        dist, idx = cKDTree(d["coords"].astype(np.float64)).query(ref_pts, k=K_NEIGHBOURS)
        w = 1.0 / np.maximum(dist, 1e-9)
        cp = d["targets"][:, 0].astype(np.float64)
        fields.append(np.sum(w * cp[idx], axis=1) / np.sum(w, axis=1))
        nn_dist.append(float(np.median(dist[:, 0])))
    np.savez_compressed(MAPPED, run_ids=np.array(runs), cp=np.array(fields, dtype=np.float32),
                        ref_points=ref_pts.astype(np.float32), ref_run=ref_run, nn_median=np.array(nn_dist))
    return np.load(MAPPED)


# ------------------------------------------------------------------ #
# 2. ONE REPEAT OF THE WORKFLOW 1 METHOD
# ------------------------------------------------------------------ #
def one_repeat(rep, run_ids, fields, table, test_table):
    rng = np.random.default_rng(config.RANDOM_STATE + 100 + rep)
    order = rng.permutation(len(run_ids))
    tr, ho = order[:N_TRAIN], order[N_TRAIN:]
    t = table.set_index("run_id")
    x_all = unit(t.loc[run_ids, PARAMS].to_numpy(dtype=float))
    cd_all = t.loc[run_ids, "Cd"].to_numpy(dtype=float)

    # POD on the training fields
    mean_field = fields[tr].mean(axis=0)
    centred = fields[tr] - mean_field
    modes, _ = _snapshot_pod(centred, N_MODES)
    amps_tr = centred @ modes.T
    var_train = 1 - np.sum((centred - amps_tr @ modes) ** 2) / np.sum(centred ** 2)
    c_ho = fields[ho] - mean_field
    amps_ho = c_ho @ modes.T
    var_holdout = 1 - np.sum((c_ho - amps_ho @ modes) ** 2) / np.sum(c_ho ** 2)

    # Calibration: amplitudes + shape values -> Cd (ridge)
    feats_tr = np.hstack([amps_tr, x_all[tr]])
    scaler = StandardScaler().fit(feats_tr)
    ridge = RidgeCV(alphas=np.logspace(-4, 3, 40)).fit(scaler.transform(feats_tr), cd_all[tr])
    calib = lambda a, x: ridge.predict(scaler.transform(np.hstack([a, x])))

    # Label check 1: true held-out field -> amplitudes -> Cd (tests POD + calibration)
    err_proj = np.abs(calib(amps_ho, x_all[ho]) - cd_all[ho])

    # Helper: Gaussian Process, 16 shape values -> 15 amplitudes
    kernel = ConstantKernel(1.0, (1e-2, 1e3)) * RBF(np.ones(len(PARAMS)), (1e-2, 1e3)) + WhiteKernel(1e-2, (1e-6, 1e1))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3,
                                  random_state=config.RANDOM_STATE).fit(x_all[tr], amps_tr)
    label = lambda x: calib(gp.predict(x), x)

    # Label check 2 (the gate): shape -> helper -> amplitudes -> Cd
    err_full = np.abs(label(x_all[ho]) - cd_all[ho])

    # Augmentation: LHS shapes inside the box, labelled by the physics path
    x_syn = qmc.LatinHypercube(d=len(PARAMS), seed=config.RANDOM_STATE + 10 + rep).random(N_SYNTHETIC)
    y_syn = label(x_syn)

    # Final models: real only vs real + synthetic, scored on the ~400 other runs
    x_test = unit(test_table[PARAMS].to_numpy(dtype=float))
    y_test = test_table["Cd"].to_numpy(dtype=float)
    rows = []
    for name, build in CANDIDATE_BUILDERS.items():
        for mode, (xx, yy) in {"real_only": (x_all[tr], cd_all[tr]),
                               "augmented": (np.vstack([x_all[tr], x_syn]), np.concatenate([cd_all[tr], y_syn]))}.items():
            m = build().fit(xx, yy)
            e = np.abs(m.predict(x_test) - y_test)
            rows.append({"repeat": rep, "model": name, "training": mode, "mae": float(e.mean()), "max": float(e.max())})
    # Reference: the helper path itself used directly as a predictor, and a GP fitted straight to Cd
    e = np.abs(label(x_test) - y_test)
    rows.append({"repeat": rep, "model": "helper_path_direct", "training": "real_only", "mae": float(e.mean()), "max": float(e.max())})
    gp_cd = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3,
                                     random_state=config.RANDOM_STATE).fit(x_all[tr], cd_all[tr])
    e = np.abs(gp_cd.predict(x_test) - y_test)
    rows.append({"repeat": rep, "model": "gp_on_cd_direct", "training": "real_only", "mae": float(e.mean()), "max": float(e.max())})

    checks = {"repeat": rep, "pod_var_train": float(var_train), "pod_var_holdout": float(var_holdout),
              "label_proj_mean": float(err_proj.mean()), "label_proj_max": float(err_proj.max()),
              "label_full_mean": float(err_full.mean()), "label_full_max": float(err_full.max())}
    return rows, checks


def main():
    t0 = time.time()
    table = pd.read_csv(SCALAR_TABLE)
    data = np.load(MAPPED) if os.path.exists(MAPPED) else build_mapped_fields(table)
    run_ids, fields = data["run_ids"], data["cp"].astype(np.float64)
    print(f"{len(run_ids)} runs mapped onto {fields.shape[1]} reference points (reference run {int(data['ref_run'])}); "
          f"median nearest-neighbour distance {np.median(data['nn_median']):.4f} (normalised units)  [{time.time() - t0:.0f} s]")
    test_table = table[~table["run_id"].isin(run_ids)].reset_index(drop=True)
    print(f"Test set for the drag models: {len(test_table)} other DrivAerML runs (Cd only)\n")

    all_rows, all_checks = [], []
    for rep in range(N_REPEATS):
        rows, checks = one_repeat(rep, run_ids, fields, table, test_table)
        all_rows += rows
        all_checks.append(checks)
        print(f"  repeat {rep + 1}/{N_REPEATS} done  [{time.time() - t0:.0f} s]")

    ck = pd.DataFrame(all_checks)
    res = pd.DataFrame(all_rows)
    print(f"\nPOD ({N_MODES} patterns): variance explained, training {ck['pod_var_train'].mean():.1%}, "
          f"held-out runs {ck['pod_var_holdout'].mean():.1%}")
    print("Label check on the 15 held-out runs, drag counts (mean / worst), mean over repeats:")
    print(f"  true field -> POD -> calibration -> Cd       {ck['label_proj_mean'].mean() * 1000:5.1f} / {ck['label_proj_max'].mean() * 1000:5.1f}")
    print(f"  shape -> helper GP -> calibration -> Cd      {ck['label_full_mean'].mean() * 1000:5.1f} / {ck['label_full_max'].mean() * 1000:5.1f}"
          f"   (gate: mean <= {config.LABEL_CHECK_MEAN_TOLERANCE * 1000:.0f})")

    summ = res.groupby(["model", "training"]).agg(mae=("mae", "mean"), sd=("mae", "std"), worst=("max", "mean")).reset_index()
    print(f"\nDrag-model error on {len(test_table)} unseen runs, drag counts, mean ± sd over {N_REPEATS} repeats "
          f"(training: {N_TRAIN} real runs, +{N_SYNTHETIC} synthetic when augmented):")
    for _, r in summ.sort_values(["model", "training"]).iterrows():
        print(f"  {r['model']:20s} {r['training']:10s} {r['mae'] * 1000:5.2f} ± {r['sd'] * 1000:4.2f}   worst {r['worst'] * 1000:5.1f}")
    paired = res.pivot_table(index=["repeat", "model"], columns="training", values="mae").dropna()
    if not paired.empty:
        diff = (paired["augmented"] - paired["real_only"]) * 1000
        print("\nAugmented minus real-only, per model (negative = augmentation helps), drag counts:")
        for m, g in diff.groupby(level="model"):
            print(f"  {m:14s} {g.mean():+5.2f} ± {g.std():4.2f}  (better in {int((g < 0).sum())}/{len(g)} repeats)")

    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    with open(os.path.join(config.OUTPUTS_DIR, "drivaerml_pod_results.json"), "w") as fh:
        json.dump({"n_runs_with_fields": int(len(run_ids)), "n_train": N_TRAIN, "n_test": int(len(test_table)),
                   "n_synthetic": N_SYNTHETIC, "n_repeats": N_REPEATS, "checks": ck.to_dict("records"),
                   "summary": summ.to_dict("records")}, fh, indent=2)
    print(f"\nTotal time {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
