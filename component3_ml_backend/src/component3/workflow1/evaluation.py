"""
How Workflow 1 tests itself and picks the drag model.

Cross-validation BY DESIGN: the testable designs (not the baseline, not the
extreme corners, not the low-confidence morphs) are split into N_CV_FOLDS
groups. Each group is the exam once while everything else trains, so every
testable design is tested exactly once per repeat. The whole thing is repeated
N_CV_REPEATS times with a different grouping. With ~20 designs a single
hold-out of 4 designs is too noisy to trust; this uses every design.

Inside each fold the full build runs on the training designs only: choose the
labeller (inner cross-validation), POD, helper, 1,200 physics-labelled shapes,
then XGBoost / Random Forest / ANN trained twice, WITH the synthetic shapes and
on the real designs ONLY. Every candidate is scored on real CFD designs only.

Model choice (the rule that replaces a fixed label-check limit): the candidate
with the lowest cross-validated error wins, so augmentation is used exactly
when it beats training on the real designs alone. The winner is then retrained
on ALL designs; its cross-validated error is the uncertainty reported with it.

Diagnostic: the augmented models are also scored on extra synthetic shapes the
helper labelled but the models never saw. A small error there and a large one
on real designs means the helper, not the drag model, is the weak link.
"""

import numpy as np
import pandas as pd

from component3 import config
from component3.workflow1.augmentation import propose_shapes
from component3.workflow1.helper import fit_helper, label_shapes, select_labeller_arrays
from component3.workflow1.pod import dataset_arrays, fit_pod_arrays
from component3.workflow1.training import AREA_CANDIDATE_BUILDERS, CANDIDATE_BUILDERS

N_DIAGNOSTIC_SHAPES = 300   # synthetic shapes kept out of training, for the diagnostic only


def designs_to_test(runs):
    """Designs that may be tested: interior and high-confidence."""
    ok = ~runs["is_corner"].astype(bool) & ~runs["is_baseline"].astype(bool) & ~runs["is_low_confidence"]
    return runs.loc[ok, "run_id"].sort_values().to_numpy()


def make_folds(runs, n_folds, seed):
    """Shuffle the testable designs and cut them into n_folds groups."""
    ids = designs_to_test(runs)
    rng = np.random.default_rng(seed)
    return [list(g) for g in np.array_split(rng.permutation(ids), n_folds)]


def _shapes_table(shapes):
    return pd.DataFrame(shapes)[config.SHAPE_PARAM_NAMES].to_numpy(dtype=np.float64)


def build_labeller(a):
    """Choose the labeller on the given training arrays (inner cross-validation),
    then fit POD + helper on all of them. Returns (pod, helper, kind, method)."""
    (kind, method), _ = select_labeller_arrays(a)
    pod = fit_pod_arrays(a)
    amps = (a["fields"] - pod.mean_field) @ pod.mode_shapes.T
    return pod, fit_helper(a["x_unit"], amps, kind), kind, method


def augmented_table(a, pod, helper, method, seed):
    """Real training rows + N_SYNTHETIC_SHAPES physics-labelled LHS shapes."""
    shapes = propose_shapes(config.N_SYNTHETIC_SHAPES, seed=seed)
    x = np.vstack([a["x_raw"], _shapes_table(shapes)])
    y = np.concatenate([a["cd"], label_shapes(helper, pod, shapes, method)])
    return x, y


def run_fold(a, train_idx, test_idx, seed, settings=None):
    """One fold: build everything on train_idx, predict the designs in test_idx.
    Returns a dict of per-design predictions plus diagnostics. settings carries
    the design box and sample sizes into a parallel worker process, which starts
    with a fresh config (see _settings)."""
    global N_DIAGNOSTIC_SHAPES
    if settings:
        config.set_design_space(*settings["space"])
        config.N_SYNTHETIC_SHAPES = settings["n_synthetic"]
        N_DIAGNOSTIC_SHAPES = settings["n_diagnostic"]
    tr = {k: v[train_idx] for k, v in a.items()}
    x_test, y_test = a["x_raw"][test_idx], a["cd"][test_idx]
    pod, helper, kind, method = build_labeller(tr)
    out = {"kind": kind, "method": method, "pred": {}}
    out["label"] = label_shapes(helper, pod, [dict(zip(config.SHAPE_PARAM_NAMES, r)) for r in x_test], method)

    x_aug, y_aug = augmented_table(tr, pod, helper, method, seed)
    diag_shapes = propose_shapes(N_DIAGNOSTIC_SHAPES, seed=seed + 50_000)
    x_diag, y_diag = _shapes_table(diag_shapes), label_shapes(helper, pod, diag_shapes, method)
    out["diag"] = {}
    for name, build in CANDIDATE_BUILDERS.items():
        model = build().fit(x_aug, y_aug)
        out["pred"][f"{name}+aug"] = model.predict(x_test)
        out["diag"][name] = float(np.mean(np.abs(model.predict(x_diag) - y_diag)))
        out["pred"][f"{name}+real"] = build().fit(tr["x_raw"], tr["cd"]).predict(x_test)
    for name, build in AREA_CANDIDATE_BUILDERS.items():
        out["pred"][f"area:{name}"] = build().fit(tr["x_raw"], tr["aref"]).predict(x_test)
    out["truth"], out["truth_area"] = y_test, a["aref"][test_idx]
    return out


def _settings():
    return {"space": (dict(config.SHAPE_PARAM_BOUNDS), dict(config.BASELINE_SHAPE)),
            "n_synthetic": config.N_SYNTHETIC_SHAPES, "n_diagnostic": N_DIAGNOSTIC_SHAPES}


def cross_validate(ds, n_folds=None, n_repeats=None, progress=print, n_jobs=None):
    """Cross-validation by design on a CFDDataset (one speed). Returns a dict with
    per-repeat errors, out-of-fold predictions and the labeller choices. The folds
    are independent, so they run in parallel (n_jobs, default config.N_JOBS)."""
    from joblib import Parallel, delayed, parallel_config
    n_folds = n_folds or config.N_CV_FOLDS
    n_repeats = n_repeats or config.N_CV_REPEATS
    n_jobs = config.N_JOBS if n_jobs is None else n_jobs
    runs = ds.runs.reset_index(drop=True)
    a = dataset_arrays(ds, runs)
    ids = runs["run_id"].to_numpy()
    jobs = []
    for rep in range(n_repeats):
        for f, test_ids in enumerate(make_folds(runs, n_folds, config.RANDOM_STATE + 100 * rep)):
            test_idx = np.flatnonzero(np.isin(ids, test_ids))
            train_idx = np.flatnonzero(~np.isin(ids, test_ids))
            jobs.append((rep, f, test_idx, train_idx, config.RANDOM_STATE + 1000 * rep + f))
    progress(f"   running {len(jobs)} folds ({n_jobs if n_jobs != 1 else 'no'} parallel workers)")
    if n_jobs == 1:
        folds = [run_fold(a, tr, te, seed) for _, _, te, tr, seed in jobs]
    else:
        # one maths thread per worker: several workers each starting a full set of
        # BLAS threads ran an 8 GB laptop out of memory
        with parallel_config(backend="loky", inner_max_num_threads=1):
            folds = Parallel(n_jobs=n_jobs)(delayed(run_fold)(a, tr, te, seed, _settings()) for _, _, te, tr, seed in jobs)
    records, choices = [], []
    for (rep, f, test_idx, _, _), fold in zip(jobs, folds):
        choices.append(f"{fold['kind']}+{fold['method']}")
        for i, idx in enumerate(test_idx):
            rec = {"repeat": rep, "fold": f, "run_id": ids[idx], "cd": fold["truth"][i],
                   "area": fold["truth_area"][i], "label": fold["label"][i]}
            rec.update({k: v[i] for k, v in fold["pred"].items()})
            rec.update({f"diag:{k}": v for k, v in fold["diag"].items()})
            records.append(rec)
    return {"oof": pd.DataFrame(records), "labeller_choices": choices, "n_folds": n_folds, "n_repeats": n_repeats}


def summarise(cv):
    """Error table (Cd units) of every candidate: mean absolute error per repeat,
    then mean and spread over repeats, RMSE, worst design."""
    oof = cv["oof"]
    cols = [c for c in oof.columns if c.endswith("+aug") or c.endswith("+real")]
    rows = []
    for c in cols + ["label"]:
        err = (oof[c] - oof["cd"]).abs()
        per_rep = err.groupby(oof["repeat"]).mean()
        rows.append({"candidate": c, "mae": float(per_rep.mean()), "mae_spread": float(per_rep.std(ddof=0)),
                     "rmse": float(np.sqrt(np.mean((oof[c] - oof["cd"]) ** 2))), "worst": float(err.max())})
    table = pd.DataFrame(rows).set_index("candidate")
    area_rows = []
    for c in [c for c in oof.columns if c.startswith("area:")]:
        pct = 100 * (oof[c] - oof["area"]).abs() / oof["area"]
        area_rows.append({"model": c.split(":", 1)[1], "mae_percent": float(pct.mean()), "worst_percent": float(pct.max())})
    diag = {c.split(":", 1)[1]: float(oof[c].mean()) for c in oof.columns if c.startswith("diag:")}
    return table, pd.DataFrame(area_rows).sort_values("mae_percent").reset_index(drop=True), diag


def augmentation_wins(cv, model_name):
    """In how many repeats the augmented version beat the real-only version."""
    oof = cv["oof"]
    aug = (oof[f"{model_name}+aug"] - oof["cd"]).abs().groupby(oof["repeat"]).mean()
    real = (oof[f"{model_name}+real"] - oof["cd"]).abs().groupby(oof["repeat"]).mean()
    return int((aug < real).sum()), len(aug)


def train_final(ds, candidate, seed=config.RANDOM_STATE):
    """Retrain the chosen candidate ("<model>+aug" or "<model>+real") on ALL designs.
    Returns (model, info)."""
    name, variant = candidate.split("+")
    runs = ds.runs.reset_index(drop=True)
    a = dataset_arrays(ds, runs)
    info = {"model": name, "uses_augmentation": variant == "aug", "n_designs": len(runs)}
    if variant == "aug":
        pod, helper, kind, method = build_labeller(a)
        x, y = augmented_table(a, pod, helper, method, seed)
        info.update(labeller=f"{kind}+{method}", n_synthetic=config.N_SYNTHETIC_SHAPES)
    else:
        x, y = a["x_raw"], a["cd"]
    return CANDIDATE_BUILDERS[name]().fit(x, y), info


def train_final_area(ds, name):
    runs = ds.runs.reset_index(drop=True)
    x = runs[config.SHAPE_PARAM_NAMES].to_numpy(dtype=np.float64)
    return AREA_CANDIDATE_BUILDERS[name]().fit(x, runs["frontal_area"].to_numpy(dtype=np.float64))
