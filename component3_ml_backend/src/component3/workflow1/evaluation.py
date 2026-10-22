"""
How Workflow 1 tests itself and picks the drag model.

ONE drag model for all speeds: its inputs are the 5 shape values AND the speed
in km/h (decided 2026-10-22). On Member 2's data it was as accurate as one model
per speed (experiments/speed_study), and it scales: a new CFD speed adds rows to
the same model instead of needing a new model with all shapes run again. The
helper (labeller) is still built per speed, as tested; every speed has all
shapes today.

Cross-validation BY DESIGN: the designs testable at every speed (not the
baseline, not the extreme corners, not the low-confidence morphs) are split into
N_CV_FOLDS groups. Each group is the exam once while everything else trains, and
an exam design is hidden at ALL speeds, so the model never sees any speed of
it. Repeated N_CV_REPEATS times with a different grouping.

Inside each fold the full build runs on the training designs only: per speed,
choose the labeller (inner cross-validation: kind, method, number of patterns),
POD, helper, 1,200 physics-labelled shapes; then XGBoost / Random Forest / ANN
trained on all speeds together, WITH the synthetic shapes and on the real
designs ONLY. Two more ANN candidates keep the real designs from being drowned
by the synthetic ones: "aug_weighted" gives real and synthetic rows half the say
each; "pretrain_finetune" learns from real + synthetic, then keeps training on
the real designs only. Every candidate is scored on real CFD designs only, per
speed.

Model choice: the candidate with the lowest cross-validated error, averaged over
the speeds, wins, so augmentation is used exactly when it helps. The winner is
then retrained on ALL designs at all speeds; its cross-validated error per
speed is the uncertainty reported with it.

Diagnostic: the augmented models are also scored on extra synthetic shapes the
helper labelled but the models never saw. A small error there and a large one
on real designs means the helper, not the drag model, is the weak link.

Per-label quality (workflow1/label_quality.py): every synthetic label carries
its own uncertainty, calibrated against held-out real CFD, and must pass
physical plausibility checks and lie in the validated region, or it is left
out of the augmented dataset. The checks run inside every fold, so their
effect is tested fairly too.
"""

import numpy as np
import pandas as pd

from component3 import config
from component3.workflow1.augmentation import propose_shapes
from component3.workflow1.helper import fit_helper, label_shapes, select_labeller_arrays
from component3.workflow1.label_quality import check_labels, predict_labels
from component3.workflow1.pod import dataset_arrays, fit_pod_arrays
from component3.workflow1.training import AREA_CANDIDATE_BUILDERS, CANDIDATE_BUILDERS, FineTunedANN, real_weights

N_DIAGNOSTIC_SHAPES = 300   # synthetic shapes kept out of training, for the diagnostic only
INPUT_NAMES = list(config.SHAPE_PARAM_NAMES) + ["speed_kmh"]


def designs_to_test(runs):
    """Designs that may be tested at this speed: interior and high-confidence."""
    ok = ~runs["is_corner"].astype(bool) & ~runs["is_baseline"].astype(bool) & ~runs["is_low_confidence"]
    return runs.loc[ok, "run_id"].sort_values().to_numpy()


def shared_designs_to_test(datasets):
    """Designs testable at EVERY speed (an exam design is hidden at all speeds)."""
    sets = [set(designs_to_test(ds.runs)) for ds in datasets.values()]
    return np.array(sorted(set.intersection(*sets)))


def make_folds(runs_or_ids, n_folds, seed):
    """Shuffle the testable designs and cut them into n_folds groups. Takes a run
    table (one speed) or an array of design ids already chosen."""
    ids = designs_to_test(runs_or_ids) if isinstance(runs_or_ids, pd.DataFrame) else np.asarray(runs_or_ids)
    rng = np.random.default_rng(seed)
    return [list(g) for g in np.array_split(rng.permutation(ids), n_folds)]


def with_speed(x, speed):
    """Add the speed (km/h) as the 6th input column."""
    return np.hstack([x, np.full((len(x), 1), float(speed))])


def _shapes_table(shapes):
    return pd.DataFrame(shapes)[config.SHAPE_PARAM_NAMES].to_numpy(dtype=np.float64)


def build_labeller(a):
    """Choose the labeller on the given training arrays (inner cross-validation),
    then fit POD (with the chosen number of patterns) + helper on all of them.
    Returns (pod, helper, choice) with choice = (kind, method, n_patterns)."""
    choice, _ = select_labeller_arrays(a)
    kind, _, n_modes = choice
    pod = fit_pod_arrays(a, n_modes)
    amps = (a["fields"] - pod.mean_field) @ pod.mode_shapes.T
    return pod, fit_helper(a["x_unit"], amps, kind), choice


def _unit(x_raw):
    lo = np.array([config.SHAPE_PARAM_BOUNDS[n][0] for n in config.SHAPE_PARAM_NAMES])
    hi = np.array([config.SHAPE_PARAM_BOUNDS[n][1] for n in config.SHAPE_PARAM_NAMES])
    return (np.asarray(x_raw, dtype=np.float64) - lo) / (hi - lo)


def labelled_shapes(a, pod, helper, method, seed):
    """N_SYNTHETIC_SHAPES LHS shapes with their label, raw uncertainty and the
    per-label checks (one row each; "keep" says whether the label is used)."""
    x = _shapes_table(propose_shapes(config.N_SYNTHETIC_SHAPES, seed=seed))
    u = _unit(x)
    amps, cd, sigma = predict_labels(helper, pod, u, method)
    checks = check_labels(pod, amps, u, cd, a["x_unit"], a["cd"])
    table = pd.DataFrame(x, columns=config.SHAPE_PARAM_NAMES)
    table["cd_label"], table["sigma_raw"] = cd, sigma
    return pd.concat([table, checks], axis=1)


def augmented_table(a, pod, helper, method, seed, with_labels=False):
    """Real training rows + the physics-labelled LHS shapes that pass the per-label
    checks (5 shape values; the speed column is added by the caller)."""
    t = labelled_shapes(a, pod, helper, method, seed)
    kept = t[t["keep"]]
    x = np.vstack([a["x_raw"], kept[config.SHAPE_PARAM_NAMES].to_numpy(dtype=np.float64)])
    y = np.concatenate([a["cd"], kept["cd_label"].to_numpy(dtype=np.float64)])
    return (x, y, t) if with_labels else (x, y)


def label_quality_summary(t):
    """Share of synthetic labels kept, and of each check failed."""
    return {"n_proposed": int(len(t)), "kept": float(t["keep"].mean()),
            "out_of_region": float((~t["in_validated_region"]).mean()),
            "cp_above_stagnation": float((~t["cp_below_stagnation"]).mean()),
            "pressure_drag_not_positive": float((~t["pressure_drag_positive"]).mean()),
            "cd_out_of_range": float((~t["cd_in_range"]).mean())}


def build_training_sets(blocks, seed, with_diagnostic=False):
    """Per speed: labeller + augmented table on that speed's training arrays. Then
    stack all speeds with the speed column. blocks = {speed: training arrays}."""
    parts = {}
    for s, a in blocks.items():
        pod, helper, choice = build_labeller(a)
        # same seed: same shapes at every speed
        x_aug, y_aug, labels = augmented_table(a, pod, helper, choice[1], seed, with_labels=True)
        part = {"pod": pod, "helper": helper, "choice": choice, "x_aug": x_aug, "y_aug": y_aug, "a": a, "labels": labels}
        if with_diagnostic:
            diag = propose_shapes(N_DIAGNOSTIC_SHAPES, seed=seed + 50_000)
            part["x_diag"], part["y_diag"] = _shapes_table(diag), label_shapes(helper, pod, diag, choice[1])
        parts[s] = part
    stacked = {
        "x_aug": np.vstack([with_speed(p["x_aug"], s) for s, p in parts.items()]),
        "y_aug": np.concatenate([p["y_aug"] for p in parts.values()]),
        "x_real": np.vstack([with_speed(p["a"]["x_raw"], s) for s, p in parts.items()]),
        "y_real": np.concatenate([p["a"]["cd"] for p in parts.values()]),
        # real rows weighted so real and synthetic each get half the say, speed by speed
        "weights": np.concatenate([real_weights(len(p["a"]["cd"]), len(p["y_aug"]) - len(p["a"]["cd"]))
                                   for p in parts.values()]),
    }
    return parts, stacked


def fit_candidate(candidate, stacked):
    """Fit one drag-model candidate ("<model>+<variant>") on the stacked rows."""
    name, variant = candidate.split("+")
    if variant == "real":
        return CANDIDATE_BUILDERS[name]().fit(stacked["x_real"], stacked["y_real"])
    if variant == "pretrain_finetune":
        return FineTunedANN().fit(stacked["x_aug"], stacked["y_aug"], stacked["x_real"], stacked["y_real"])
    if variant == "aug_weighted":
        return CANDIDATE_BUILDERS[name]().fit(stacked["x_aug"], stacked["y_aug"], mlpregressor__sample_weight=stacked["weights"])
    return CANDIDATE_BUILDERS[name]().fit(stacked["x_aug"], stacked["y_aug"])


CANDIDATES = [f"{m}+{v}" for m in CANDIDATE_BUILDERS for v in ("aug", "real")] + [
    "ann_mlp+aug_weighted", "ann_mlp+pretrain_finetune"]


def run_fold(arrays, test_ids, seed, settings=None):
    """One fold over all speeds. arrays = {speed: (arrays, run ids)}. Builds
    everything without the test designs and predicts them at every speed.
    settings carries the design box and sample sizes into a parallel worker
    process, which starts with a fresh config (see _settings)."""
    global N_DIAGNOSTIC_SHAPES
    if settings:
        config.set_design_space(*settings["space"])
        config.N_SYNTHETIC_SHAPES = settings["n_synthetic"]
        N_DIAGNOSTIC_SHAPES = settings["n_diagnostic"]
    blocks, tests = {}, {}
    for s, (a, ids) in arrays.items():
        te = np.isin(ids, test_ids)
        blocks[s] = {k: v[~te] for k, v in a.items()}
        tests[s] = {"ids": ids[te], "x": a["x_raw"][te], "cd": a["cd"][te], "area": a["aref"][te]}
    parts, stacked = build_training_sets(blocks, seed, with_diagnostic=True)
    out = {s: {"pred": {}, "diag": {}, **tests[s]} for s in arrays}
    for s, p in parts.items():
        kind, method, n_modes = p["choice"]
        out[s]["choice"] = f"{kind}+{method}, {n_modes} patterns"
        _, out[s]["label"], out[s]["label_std"] = predict_labels(p["helper"], p["pod"], _unit(tests[s]["x"]), method)
        out[s]["label_quality"] = label_quality_summary(p["labels"])
    for c in CANDIDATES:
        model = fit_candidate(c, stacked)
        for s in arrays:
            out[s]["pred"][c] = model.predict(with_speed(tests[s]["x"], s))
            if c.endswith("+aug"):
                p = parts[s]
                out[s]["diag"][c.split("+")[0]] = float(np.mean(np.abs(model.predict(with_speed(p["x_diag"], s)) - p["y_diag"])))
    # frontal area is geometry only: one model on the shape values, all speeds' rows
    x_area = np.vstack([b["x_raw"] for b in blocks.values()])
    y_area = np.concatenate([b["aref"] for b in blocks.values()])
    for name, build in AREA_CANDIDATE_BUILDERS.items():
        area_model = build().fit(x_area, y_area)
        for s in arrays:
            out[s]["pred"][f"area:{name}"] = area_model.predict(tests[s]["x"])
    return out


def _settings():
    return {"space": (dict(config.SHAPE_PARAM_BOUNDS), dict(config.BASELINE_SHAPE)),
            "n_synthetic": config.N_SYNTHETIC_SHAPES, "n_diagnostic": N_DIAGNOSTIC_SHAPES}


def cross_validate(datasets, n_folds=None, n_repeats=None, progress=print, n_jobs=None):
    """Cross-validation by design over all speeds. datasets = {speed: CFDDataset}.
    Returns a dict with the out-of-fold predictions (one row per design, speed
    and repeat) and the labeller choices. The folds run in parallel (n_jobs,
    default config.N_JOBS)."""
    from joblib import Parallel, delayed, parallel_config
    n_folds = n_folds or config.N_CV_FOLDS
    n_repeats = n_repeats or config.N_CV_REPEATS
    n_jobs = config.N_JOBS if n_jobs is None else n_jobs
    arrays = {}
    for s, ds in datasets.items():
        runs = ds.runs.reset_index(drop=True)
        arrays[s] = (dataset_arrays(ds, runs), runs["run_id"].to_numpy())
    testable = shared_designs_to_test(datasets)
    jobs = [(rep, f, test_ids, config.RANDOM_STATE + 1000 * rep + f)
            for rep in range(n_repeats)
            for f, test_ids in enumerate(make_folds(testable, n_folds, config.RANDOM_STATE + 100 * rep))]
    progress(f"   running {len(jobs)} folds ({n_jobs if n_jobs != 1 else 'no'} parallel workers)")
    if n_jobs == 1:
        folds = [run_fold(arrays, te, seed) for _, _, te, seed in jobs]
    else:
        # one maths thread per worker: several workers each starting a full set of
        # BLAS threads ran an 8 GB laptop out of memory
        with parallel_config(backend="loky", inner_max_num_threads=1):
            folds = Parallel(n_jobs=n_jobs)(delayed(run_fold)(arrays, te, seed, _settings()) for _, _, te, seed in jobs)
    records, choices, quality = [], {s: [] for s in datasets}, {s: [] for s in datasets}
    for (rep, f, _, _), fold in zip(jobs, folds):
        for s, r in fold.items():
            choices[s].append(r["choice"])
            quality[s].append(r["label_quality"])
            for i, rid in enumerate(r["ids"]):
                rec = {"repeat": rep, "fold": f, "speed": s, "run_id": rid, "cd": r["cd"][i], "area": r["area"][i],
                       "label": r["label"][i], "label_std": r["label_std"][i]}
                rec.update({k: v[i] for k, v in r["pred"].items()})
                rec.update({f"diag:{k}": v for k, v in r["diag"].items()})
                records.append(rec)
    return {"oof": pd.DataFrame(records), "labeller_choices": choices, "n_folds": n_folds, "n_repeats": n_repeats,
            "n_testable": len(testable), "label_quality": {s: pd.DataFrame(q).mean().to_dict() for s, q in quality.items()}}


def for_speed(cv, speed):
    """The cross-validation results of one speed, in the same form."""
    return {**cv, "oof": cv["oof"][cv["oof"]["speed"] == speed].reset_index(drop=True)}


def candidate_columns(oof):
    """Columns holding a drag-model candidate's predictions, e.g. "xgboost+aug"."""
    return [c for c in oof.columns if "+" in c and not c.startswith(("area:", "diag:"))]


def summarise(cv):
    """Error table (Cd units) of every candidate: mean absolute error per repeat,
    then mean and spread over repeats, RMSE, worst design. Use for_speed() first
    for one speed."""
    oof = cv["oof"]
    rows = []
    for c in candidate_columns(oof) + ["label"]:
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


def choose_candidate(cv):
    """The winner: lowest cross-validated error averaged over the speeds."""
    tables = [summarise(for_speed(cv, s))[0] for s in sorted(cv["oof"]["speed"].unique())]
    mean_mae = pd.concat([t["mae"] for t in tables], axis=1).mean(axis=1).drop(index="label")
    return mean_mae.idxmin(), mean_mae.sort_values()


def augmentation_wins(cv, model_name):
    """In how many repeats the augmented version beat the real-only version."""
    oof = cv["oof"]
    aug = (oof[f"{model_name}+aug"] - oof["cd"]).abs().groupby(oof["repeat"]).mean()
    real = (oof[f"{model_name}+real"] - oof["cd"]).abs().groupby(oof["repeat"]).mean()
    return int((aug < real).sum()), len(aug)


def train_final(datasets, candidate, seed=config.RANDOM_STATE):
    """Retrain the chosen candidate on ALL designs at all speeds. Returns (model,
    info); the model's inputs are INPUT_NAMES (5 shape values + speed in km/h)."""
    blocks = {s: dataset_arrays(ds, ds.runs.reset_index(drop=True)) for s, ds in datasets.items()}
    variant = candidate.split("+")[1]
    info = {"model": candidate.split("+")[0], "variant": variant, "uses_augmentation": variant != "real",
            "inputs": INPUT_NAMES, "speeds_kmh": sorted(datasets), "n_designs": {s: len(b["cd"]) for s, b in blocks.items()}}
    if variant == "real":
        stacked = {"x_real": np.vstack([with_speed(b["x_raw"], s) for s, b in blocks.items()]),
                   "y_real": np.concatenate([b["cd"] for b in blocks.values()])}
        return fit_candidate(candidate, stacked), info
    parts, stacked = build_training_sets(blocks, seed)
    info.update(labeller={s: "{}+{}, {} patterns".format(*p["choice"]) for s, p in parts.items()},
                n_synthetic_per_speed=config.N_SYNTHETIC_SHAPES,
                synthetic_kept_per_speed={s: int(p["labels"]["keep"].sum()) for s, p in parts.items()},
                synthetic_labels={s: p["labels"] for s, p in parts.items()})
    return fit_candidate(candidate, stacked), info


def train_final_area(datasets, name):
    """Frontal area from the 5 shape values (geometry only), on all speeds' rows."""
    runs = pd.concat([ds.runs for ds in datasets.values()], ignore_index=True)
    x = runs[config.SHAPE_PARAM_NAMES].to_numpy(dtype=np.float64)
    return AREA_CANDIDATE_BUILDERS[name]().fit(x, runs["frontal_area"].to_numpy(dtype=np.float64))
