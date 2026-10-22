"""
How many synthetic shapes are enough? The same cross-validation by design as
Workflow 1 (4 groups x 3 repeats, a design hidden at every speed, one model with
speed as input), with the helper built once per fold and speed, then the
drag model trained with N = 0 (real only), 300, 600, 1,200 (today), 2,400 and
4,800 synthetic shapes per speed. Scored on real held-out designs only.

Usage:  python experiments/augmentation_size/size_study.py [data_dir]
Writes: experiments/augmentation_size/results/size_study.json / .csv
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
from component3.workflow1.augmentation import propose_shapes
from component3.workflow1.evaluation import build_labeller, make_folds, shared_designs_to_test, with_speed
from component3.workflow1.helper import label_shapes
from component3.workflow1.pod import dataset_arrays
from component3.workflow1.training import CANDIDATE_BUILDERS

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SIZES = (0, 300, 600, 1200, 2400, 4800)
MODELS = ("xgboost", "ann_mlp")
COUNTS = 1000


def run_fold(arrays, test_ids, seed, space):
    config.set_design_space(*space)
    names = config.SHAPE_PARAM_NAMES
    parts = {}
    for s, (a, ids) in arrays.items():
        te = np.isin(ids, test_ids)
        sub = {k: v[~te] for k, v in a.items()}
        pod, helper, (_, method, _) = build_labeller(sub)
        shapes = propose_shapes(max(SIZES), seed=seed)            # the first N of one large LHS set
        x_syn = pd.DataFrame(shapes)[names].to_numpy(dtype=float)
        parts[s] = {"sub": sub, "x_syn": x_syn, "y_syn": label_shapes(helper, pod, shapes, method),
                    "x_test": a["x_raw"][te], "ids": ids[te], "cd": a["cd"][te]}
    out = {s: {"ids": p["ids"], "cd": p["cd"], "pred": {}} for s, p in parts.items()}
    for n in SIZES:
        # n synthetic shapes = a random subset of one large labelled LHS set (same subset at every speed)
        idx = np.random.default_rng(seed + n).choice(max(SIZES), size=n, replace=False)
        x = np.vstack([np.vstack([with_speed(p["sub"]["x_raw"], s), with_speed(p["x_syn"][idx], s)])
                       for s, p in parts.items()])
        y = np.concatenate([np.concatenate([p["sub"]["cd"], p["y_syn"][idx]]) for p in parts.values()])
        for m in MODELS:
            model = CANDIDATE_BUILDERS[m]().fit(x, y)
            for s, p in parts.items():
                out[s]["pred"][f"{m}:{n}"] = model.predict(with_speed(p["x_test"], s))
    return out


def main():
    from joblib import Parallel, delayed, parallel_config
    data_dir = sys.argv[1] if len(sys.argv) > 1 else config.REAL_DATA_DIR
    os.makedirs(OUT, exist_ok=True)
    datasets = {s: CFDDataset(data_dir, s) for s in (40, 100)}
    arrays = {s: (dataset_arrays(ds, ds.runs.reset_index(drop=True)), ds.runs["run_id"].to_numpy())
              for s, ds in datasets.items()}
    testable = shared_designs_to_test(datasets)
    jobs = [(rep, f, te, config.RANDOM_STATE + 1000 * rep + f) for rep in range(config.N_CV_REPEATS)
            for f, te in enumerate(make_folds(testable, config.N_CV_FOLDS, config.RANDOM_STATE + 100 * rep))]
    space = (dict(config.SHAPE_PARAM_BOUNDS), dict(config.BASELINE_SHAPE))
    print(f"   running {len(jobs)} folds, {config.N_JOBS} workers", flush=True)
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=config.N_JOBS, verbose=5)(delayed(run_fold)(arrays, te, seed, space)
                                                             for _, _, te, seed in jobs)
    rows = []
    for (rep, f, _, _), res in zip(jobs, results):
        for s, r in res.items():
            for i, rid in enumerate(r["ids"]):
                rows.append({"repeat": rep, "speed": s, "run_id": rid, "cd": r["cd"][i],
                             **{k: v[i] for k, v in r["pred"].items()}})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "size_study.csv"), index=False)
    summary = {}
    print("\n=== How many synthetic shapes? Error on held-out REAL designs, drag counts (mean over repeats) ===")
    print(f"   {'shapes per speed':>16s} " + " ".join(f"{m + ' ' + str(s) + 'km/h':>17s}" for m in MODELS for s in (40, 100)))
    for n in SIZES:
        vals = []
        for m in MODELS:
            for s in (40, 100):
                d = df[df["speed"] == s]
                vals.append(float(((d[f"{m}:{n}"] - d["cd"]).abs().groupby(d["repeat"]).mean() * COUNTS).mean()))
        summary[n] = vals
        label = "0 (real only)" if n == 0 else str(n) + ("  <- today" if n == 1200 else "")
        print(f"   {label:>16s} " + " ".join(f"{v:17.1f}" for v in vals))
    with open(os.path.join(OUT, "size_study.json"), "w") as fh:
        json.dump({"columns": [f"{m}_{s}kmh" for m in MODELS for s in (40, 100)], "mae_counts": summary}, fh, indent=2)


if __name__ == "__main__":
    main()
