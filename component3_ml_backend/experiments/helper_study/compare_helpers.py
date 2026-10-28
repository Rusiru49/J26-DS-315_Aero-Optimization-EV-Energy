"""
Five versions of the helper (labeller), compared in the same cross-validation
by design as Workflow 1 (same groups, scored on held-out REAL designs only).
The versions and their settings were fixed on 2026-10-21 BEFORE any result was
seen, from the diagnosis (diagnose_helper.py) and the literature:

  H0     today's helper: one GP per pattern (RBF-ARD + noise, constant mean),
         15 patterns, ridge from [pattern amounts, shape values] to Cd
  H1     H0 + a linear trend in each pattern's GP (kernel + C*DotProduct), so it
         does not fall back to the average away from the data
         (Rasmussen & Williams 2006 sec. 2.7; Wilson & Adams 2013; Joseph et al. 2008)
  H3     H0 with the number of patterns chosen by 5-fold cross-validation on the
         fold's training designs, from {2, 4, 6, 8, 10, 15}
         (Hesthaven & Ubbiali 2018; Zancanaro et al. 2021)
  H1+H3  both
  H2     H1+H3 + a correction: a GP with a linear trend fitted to the H1+H3
         label's out-of-fold errors on the training designs, added to the label
         (Drohmann & Carlberg 2015, ROMES; Guo & Hesthaven 2019)

For each version: the label error, the low-drag bias, and XGBoost / Random
Forest / ANN trained on the real training designs + 1,200 shapes labelled by
that version. Real-only models are scored too, for reference.

Usage:  python experiments/helper_study/compare_helpers.py [data_dir]
Writes: experiments/helper_study/results/compare_helpers_<speed>kmh.json / .csv
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, DotProduct, WhiteKernel
from sklearn.model_selection import KFold

from component3 import config
from component3.data.loader import CFDDataset
from component3.workflow1.augmentation import propose_shapes
from component3.workflow1.evaluation import make_folds
from component3.workflow1.helper import PerModeGP
from component3.workflow1.pod import dataset_arrays, fit_pod_arrays
from component3.workflow1.training import CANDIDATE_BUILDERS

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
VARIANTS = ["H0", "H1", "H3", "H1+H3", "H2"]
MODE_GRID = (2, 4, 6, 8, 10, 15)
COUNTS = 1000


def gp(trend):
    kernel = ConstantKernel(1.0, (1e-2, 1e3)) * RBF(np.ones(len(config.SHAPE_PARAM_NAMES)), (1e-2, 1e3)) \
        + WhiteKernel(1e-2, (1e-6, 1e1))
    if trend:
        kernel = kernel + ConstantKernel(1.0, (1e-3, 1e2)) * DotProduct(sigma_0=1.0, sigma_0_bounds="fixed")
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3,
                                    random_state=config.RANDOM_STATE)


class Labeller:
    """shape (0-1 scaled) -> Cd through K pod patterns, per-pattern GPs and the ridge."""

    def __init__(self, a, n_modes, trend):
        self.pod = fit_pod_arrays(a, n_modes)
        amps = (a["fields"] - self.pod.mean_field) @ self.pod.mode_shapes.T
        self.gps = PerModeGP([gp(trend).fit(a["x_unit"], amps[:, k]) for k in range(amps.shape[1])])

    def predict(self, x_unit):
        return self.pod.cd_batch(self.gps.predict(x_unit), x_unit, "ridge")


def inner_oof(a, n_modes, trend, n_inner=5):
    """Out-of-fold predictions of a labeller on the training designs themselves."""
    out = np.zeros(len(a["cd"]))
    for tr, te in KFold(n_inner, shuffle=True, random_state=config.RANDOM_STATE).split(out):
        out[te] = Labeller({k: v[tr] for k, v in a.items()}, n_modes, trend).predict(a["x_unit"][te])
    return out


def choose_modes(a, trend):
    errors = {m: float(np.mean(np.abs(inner_oof(a, m, trend) - a["cd"]))) for m in MODE_GRID}
    best = min(errors, key=errors.get)
    return best, errors


def run_fold(a, tr, te, seed, space):
    config.set_design_space(*space)
    sub = {k: v[tr] for k, v in a.items()}
    shapes = propose_shapes(config.N_SYNTHETIC_SHAPES, seed=seed)
    lo = np.array([config.SHAPE_PARAM_BOUNDS[n][0] for n in config.SHAPE_PARAM_NAMES])
    hi = np.array([config.SHAPE_PARAM_BOUNDS[n][1] for n in config.SHAPE_PARAM_NAMES])
    x_syn = pd.DataFrame(shapes)[config.SHAPE_PARAM_NAMES].to_numpy(dtype=float)
    u_syn, u_test = (x_syn - lo) / (hi - lo), a["x_unit"][te]

    labels = {}   # version -> (labels for test designs, labels for synthetic shapes)
    info = {}
    for name, modes, trend in (("H0", 15, False), ("H1", 15, True)):
        lab = Labeller(sub, modes, trend)
        labels[name] = (lab.predict(u_test), lab.predict(u_syn))
    for name, trend in (("H3", False), ("H1+H3", True)):
        modes, grid = choose_modes(sub, trend)
        lab = Labeller(sub, modes, trend)
        labels[name] = (lab.predict(u_test), lab.predict(u_syn))
        info[f"{name}_modes"] = modes
        if name == "H1+H3":
            resid = sub["cd"] - inner_oof(sub, modes, trend)
            corr = gp(True).fit(sub["x_unit"], resid)
            labels["H2"] = (labels[name][0] + corr.predict(u_test), labels[name][1] + corr.predict(u_syn))

    pred = {}
    x_test = a["x_raw"][te]
    for name in VARIANTS:
        pred[f"label:{name}"] = labels[name][0]
        x_aug = np.vstack([sub["x_raw"], x_syn])
        y_aug = np.concatenate([sub["cd"], labels[name][1]])
        for m, build in CANDIDATE_BUILDERS.items():
            pred[f"{m}+aug:{name}"] = build().fit(x_aug, y_aug).predict(x_test)
    for m, build in CANDIDATE_BUILDERS.items():
        pred[f"{m}+real"] = build().fit(sub["x_raw"], sub["cd"]).predict(x_test)
    return pred, info


def compare(ds, n_jobs=config.N_JOBS):
    from joblib import Parallel, delayed, parallel_config
    runs = ds.runs.reset_index(drop=True)
    a = dataset_arrays(ds, runs)
    ids = runs["run_id"].to_numpy()
    jobs = []
    for rep in range(config.N_CV_REPEATS):
        for f, test_ids in enumerate(make_folds(runs, config.N_CV_FOLDS, config.RANDOM_STATE + 100 * rep)):
            te = np.flatnonzero(np.isin(ids, test_ids))
            tr = np.flatnonzero(~np.isin(ids, test_ids))
            jobs.append((rep, f, tr, te, config.RANDOM_STATE + 1000 * rep + f))
    space = (dict(config.SHAPE_PARAM_BOUNDS), dict(config.BASELINE_SHAPE))
    print(f"   running {len(jobs)} folds, {n_jobs} workers", flush=True)
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=n_jobs, verbose=5)(delayed(run_fold)(a, tr, te, seed, space)
                                                     for _, _, tr, te, seed in jobs)
    rows, infos = [], []
    for (rep, f, _, te, _), (pred, info) in zip(jobs, results):
        infos.append(info)
        for i, idx in enumerate(te):
            rows.append({"repeat": rep, "fold": f, "run_id": ids[idx], "cd": a["cd"][idx],
                         **{k: v[i] for k, v in pred.items()}})
    return pd.DataFrame(rows), infos


def summarise(df, infos, speed):
    def mae(col):
        per_rep = (df[col] - df["cd"]).abs().groupby(df["repeat"]).mean() * COUNTS
        return round(float(per_rep.mean()), 1), round(float(per_rep.std(ddof=0)), 1)

    per = df.groupby("run_id").mean(numeric_only=True)
    low = per["cd"] <= per["cd"].quantile(0.25)
    table = []
    for v in VARIANTS:
        row = {"helper": v, "label": mae(f"label:{v}"),
               "low_drag_bias": round(float(((per[f"label:{v}"] - per["cd"])[low]).mean() * COUNTS), 1)}
        for m in CANDIDATE_BUILDERS:
            row[m] = mae(f"{m}+aug:{v}")
        table.append(row)
    real = {m: mae(f"{m}+real") for m in CANDIDATE_BUILDERS}
    modes = {k: pd.Series([i[k] for i in infos]).value_counts().sort_index().to_dict() for k in ("H3_modes", "H1+H3_modes")}

    print(f"\n=== Helper versions, {speed} km/h: error on held-out REAL designs, drag counts (mean +/- spread over repeats) ===")
    print(f"   {'helper':7s} {'label':>12s} {'low-drag bias':>14s} {'XGBoost+aug':>13s} {'RF+aug':>13s} {'ANN+aug':>13s}")
    for r in table:
        print(f"   {r['helper']:7s} {r['label'][0]:6.1f} +/-{r['label'][1]:4.1f} {r['low_drag_bias']:+13.1f} "
              + " ".join(f"{r[m][0]:6.1f} +/-{r[m][1]:4.1f}" for m in CANDIDATE_BUILDERS))
    print("   real only (reference):       " + "   ".join(f"{m} {v[0]:.1f}" for m, v in real.items()))
    print(f"   patterns chosen inside the folds: H3 {modes['H3_modes']}, H1+H3 {modes['H1+H3_modes']}")
    return {"speed_kmh": speed, "table": table, "real_only": real, "patterns_chosen": modes}


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else config.REAL_DATA_DIR
    os.makedirs(OUT, exist_ok=True)
    for speed in (40, 100):
        print(f"--- {speed} km/h", flush=True)
        df, infos = compare(CFDDataset(data_dir, speed))
        summary = summarise(df, infos, speed)
        df.to_csv(os.path.join(OUT, f"compare_helpers_{speed}kmh.csv"), index=False)
        with open(os.path.join(OUT, f"compare_helpers_{speed}kmh.json"), "w") as fh:
            json.dump(summary, fh, indent=2, default=str)


if __name__ == "__main__":
    main()
