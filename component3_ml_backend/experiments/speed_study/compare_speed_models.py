"""
One drag model per speed (today) or ONE model with speed as a 6th input?
Compared in the same cross-validation by design as Workflow 1 (same groups,
4 x 3), with a held-out design hidden at BOTH speeds, so the combined model
never sees the other speed of a test design. Settings fixed on 2026-10-22
before any result was seen:

  separate   today's method: per speed, models on the 5 shape values
  combined   one model on [5 shape values, speed in km/h], trained on the
             training designs at both speeds

Each of XGBoost, Random Forest and ANN, trained WITH physics-labelled
augmentation (the Workflow 1 labeller, H3, built per speed; the same 1,200 LHS
shapes labelled at each speed) and on the real designs ONLY. Scored on the real
held-out designs, per speed. Rule: the combined model replaces the separate
ones only if it is better at both speeds.

Usage:  python experiments/speed_study/compare_speed_models.py [data_dir]
Writes: experiments/speed_study/results/compare_speed_models.json / .csv
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
import pandas as pd

from component3 import config
from component3.data.loader import CFDDataset
from component3.workflow1.evaluation import augmented_table, build_labeller, make_folds
from component3.workflow1.pod import dataset_arrays
from component3.workflow1.training import CANDIDATE_BUILDERS

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SPEEDS = (40, 100)
COUNTS = 1000


def with_speed(x, speed):
    return np.hstack([x, np.full((len(x), 1), float(speed))])


def run_fold(arrays, tr_ids, te_ids, seed, space):
    config.set_design_space(*space)
    parts = {}
    for s in SPEEDS:
        a, ids = arrays[s]
        tr, te = np.isin(ids, tr_ids), np.isin(ids, te_ids)
        sub = {k: v[tr] for k, v in a.items()}
        pod, helper, (_, method, n_modes) = build_labeller(sub)
        x_aug, y_aug = augmented_table(sub, pod, helper, method, seed)   # same seed -> same synthetic shapes
        parts[s] = {"sub": sub, "x_aug": x_aug, "y_aug": y_aug, "x_test": a["x_raw"][te], "test_ids": ids[te],
                    "n_patterns": n_modes}
    pred = {s: {} for s in SPEEDS}
    for m, build in CANDIDATE_BUILDERS.items():
        for s in SPEEDS:   # separate: one model per speed
            p = parts[s]
            pred[s][f"separate:{m}+aug"] = build().fit(p["x_aug"], p["y_aug"]).predict(p["x_test"])
            pred[s][f"separate:{m}+real"] = build().fit(p["sub"]["x_raw"], p["sub"]["cd"]).predict(p["x_test"])
        x_c_aug = np.vstack([with_speed(parts[s]["x_aug"], s) for s in SPEEDS])
        y_c_aug = np.concatenate([parts[s]["y_aug"] for s in SPEEDS])
        x_c_real = np.vstack([with_speed(parts[s]["sub"]["x_raw"], s) for s in SPEEDS])
        y_c_real = np.concatenate([parts[s]["sub"]["cd"] for s in SPEEDS])
        comb_aug, comb_real = build().fit(x_c_aug, y_c_aug), build().fit(x_c_real, y_c_real)
        for s in SPEEDS:   # combined: one model, asked at each speed
            x_t = with_speed(parts[s]["x_test"], s)
            pred[s][f"combined:{m}+aug"] = comb_aug.predict(x_t)
            pred[s][f"combined:{m}+real"] = comb_real.predict(x_t)
    return pred, {s: parts[s]["test_ids"] for s in SPEEDS}, {s: parts[s]["n_patterns"] for s in SPEEDS}


def main():
    from joblib import Parallel, delayed, parallel_config
    data_dir = sys.argv[1] if len(sys.argv) > 1 else config.REAL_DATA_DIR
    os.makedirs(OUT, exist_ok=True)
    datasets = {s: CFDDataset(data_dir, s) for s in SPEEDS}
    arrays, truth = {}, {}
    for s, ds in datasets.items():
        runs = ds.runs.reset_index(drop=True)
        arrays[s] = (dataset_arrays(ds, runs), runs["run_id"].to_numpy())
        truth[s] = dict(zip(runs["run_id"], runs["Cd"]))
    runs100 = datasets[100].runs.reset_index(drop=True)
    all_ids = runs100["run_id"].to_numpy()
    jobs = []
    for rep in range(config.N_CV_REPEATS):
        for f, test_ids in enumerate(make_folds(runs100, config.N_CV_FOLDS, config.RANDOM_STATE + 100 * rep)):
            jobs.append((rep, f, all_ids[~np.isin(all_ids, test_ids)], np.array(test_ids), config.RANDOM_STATE + 1000 * rep + f))
    space = (dict(config.SHAPE_PARAM_BOUNDS), dict(config.BASELINE_SHAPE))
    print(f"   running {len(jobs)} folds (both speeds each), {config.N_JOBS} workers", flush=True)
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=config.N_JOBS, verbose=5)(
            delayed(run_fold)(arrays, tr, te, seed, space) for _, _, tr, te, seed in jobs)

    rows, patterns = [], []
    for (rep, f, _, _, _), (pred, test_ids, n_pat) in zip(jobs, results):
        patterns.append(n_pat)
        for s in SPEEDS:
            for i, rid in enumerate(test_ids[s]):
                rows.append({"repeat": rep, "fold": f, "speed": s, "run_id": rid, "cd": truth[s][rid],
                             **{k: v[i] for k, v in pred[s].items()}})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "compare_speed_models.csv"), index=False)

    cols = [c for c in df.columns if c.startswith(("separate:", "combined:"))]
    summary = {}
    print("\n=== One model per speed vs one model with speed as input: error on held-out REAL designs, drag counts ===")
    for s in SPEEDS:
        d = df[df["speed"] == s]
        res = {}
        for c in cols:
            per_rep = (d[c] - d["cd"]).abs().groupby(d["repeat"]).mean() * COUNTS
            res[c] = (round(float(per_rep.mean()), 1), round(float(per_rep.std(ddof=0)), 1))
        summary[s] = res
        print(f"\n   {s} km/h            {'separate':>16s} {'combined':>16s}")
        for m in CANDIDATE_BUILDERS:
            for v in ("aug", "real"):
                a, b = res[f"separate:{m}+{v}"], res[f"combined:{m}+{v}"]
                tag = "  <- combined better" if b[0] < a[0] else ""
                print(f"   {m + '+' + v:20s} {a[0]:7.1f} +/-{a[1]:4.1f}  {b[0]:7.1f} +/-{b[1]:4.1f}{tag}")
    best = {s: min(summary[s], key=lambda c: summary[s][c][0]) for s in SPEEDS}
    print(f"\n   best at 40 km/h: {best[40]} {summary[40][best[40]][0]}; at 100 km/h: {best[100]} {summary[100][best[100]][0]}")
    with open(os.path.join(OUT, "compare_speed_models.json"), "w") as fh:
        json.dump({"results_counts": {str(s): v for s, v in summary.items()}, "best": {str(s): v for s, v in best.items()},
                   "patterns_chosen": patterns}, fh, indent=2, default=str)


if __name__ == "__main__":
    main()
