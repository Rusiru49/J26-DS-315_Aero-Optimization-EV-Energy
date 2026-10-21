"""
Workflow 1 end to end: data folder -> checked data -> POD -> helper model ->
label check (a real gate) -> augmentation -> model selection -> saved drag
model. Prints the results you would show at PP1.

Usage:
    python scripts/run_workflow1.py                  # the dummy dataset, 100 km/h
    python scripts/run_workflow1.py placeholder 40   # the dummy dataset, 40 km/h
    python scripts/run_workflow1.py real 100         # Member 2's data (config.REAL_DATA_DIR)
    python scripts/run_workflow1.py <folder> 40      # any folder in the contract layout

The models are saved to models/<dataset folder name>_<speed>kmh/, so each
dataset and speed keeps its own drag and frontal-area model.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np
import pandas as pd

from component3 import config
from component3.workflow1.augmentation import build_augmented_dataset
from component3.data.loader import load_cfd_dataset
from component3.data.checker import check_dataset
from component3.workflow1.helper import label_shapes, select_labeller, train_helper_gp
from component3.workflow1.pod import exact_cd_from_faces, explained_variance_ratio, fit_pod, validate_on_holdout
from component3.workflow1.training import real_only_baseline, save_area_model, save_model, train_and_select, train_area_model


def one_split(ds, kind, method):
    """Rebuild the labeller, augmentation and models on one hold-out draw; return errors in drag counts."""
    train, holdout = ds.real_runs(), ds.holdout_runs()
    pod = fit_pod(ds)
    gp = train_helper_gp(ds, pod, kind)
    shapes = [{n: row[n] for n in config.SHAPE_PARAM_NAMES} for _, row in holdout.iterrows()]
    label_err = np.abs(label_shapes(gp, pod, shapes, method) - holdout["Cd"].to_numpy(dtype=float))
    augmented = build_augmented_dataset(ds, pod, gp, method=method)
    _, _, table = train_and_select(augmented, holdout)
    base = real_only_baseline(augmented[augmented["source"] == "real"], holdout)
    out = {"label_mean": label_err.mean() * 1000, "label_max": label_err.max() * 1000}
    for _, r in table.iterrows():
        out[f"aug_{r['model']}"] = r["holdout_mae"] * 1000
    for _, r in base.iterrows():
        out[f"real_{r['model']}"] = r["holdout_mae"] * 1000
    return out


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else "placeholder"
    speed = int(sys.argv[2]) if len(sys.argv) > 2 else config.DEFAULT_SPEED_KMH
    # optional 3rd argument: label-gate limit in drag counts (default: config, a placeholder)
    gate = float(sys.argv[3]) / 1000 if len(sys.argv) > 3 else config.LABEL_CHECK_MEAN_TOLERANCE
    print(f"=== Workflow 1: {source} data, {speed} km/h ===\n")

    dataset = load_cfd_dataset(source, speed)
    results = check_dataset(dataset.data_dir)
    fails = [r for r in results if r[1] == "FAIL"]
    print(f"1. Data check: {len(results) - len(fails)}/{len(results)} checks passed")
    if fails:
        for name, _, detail in fails:
            print(f"   FAIL {name}: {detail}")
        sys.exit("Stopping: the data does not match the contract.")

    train, holdout = dataset.real_runs(), dataset.holdout_runs()
    n_low = int(dataset.runs["is_low_confidence"].sum())
    print(f"2. Split by design: {len(train)} training, {len(holdout)} hold-out "
          f"({dataset.n_failed} failed designs left out; {n_low} low-confidence designs train only)")
    box = ", ".join(f"{n} {lo:g}-{hi:g}" for n, (lo, hi) in config.SHAPE_PARAM_BOUNDS.items())
    print(f"   design box: {box}\n")

    pod = fit_pod(dataset)
    print(f"3. POD: {config.N_POD_MODES} patterns explain {explained_variance_ratio(dataset, pod):.1%} of the pressure variation")

    sanity = np.array([abs(exact_cd_from_faces(dataset, r) - c) for r, c in zip(train["run_id"], train["Cd"])])
    print(f"   exact face integral of the TRUE fields vs recorded Cd (training runs): mean {sanity.mean() * 1000:.2f}, "
          f"worst {sanity.max() * 1000:.2f} drag counts (checks the conventions)")

    if config.LABEL_METHOD == "auto" or config.HELPER_KIND == "auto":
        (kind, method), cv = select_labeller(dataset)
        print("   choosing the labeller by 5-fold cross-validation on the TRAINING designs (drag counts):")
        for (k, m), e in sorted(cv.items(), key=lambda t: t[1]):
            print(f"     helper {k:8s} + {m:5s}: {e * 1000:6.1f}" + ("   <- chosen" if (k, m) == (kind, method) else ""))
    else:
        kind, method, cv = config.HELPER_KIND, config.LABEL_METHOD, {}

    helpers = {k: train_helper_gp(dataset, pod, k) for k in ("per_mode", "shared")}
    gp = helpers[kind]
    shapes = [{n: row[n] for n in config.SHAPE_PARAM_NAMES} for _, row in holdout.iterrows()]
    truth = holdout["Cd"].to_numpy(dtype=float)
    # Run noise: Member 2's standard error of the averaged Cd when given, else the iteration spread.
    noise_col = "Cd_mean_se" if "Cd_mean_se" in dataset.runs else "Cd_std"
    noise_counts = float(dataset.runs[noise_col].mean()) * 1000 if noise_col in dataset.runs else float("nan")
    print("4. Label check on hold-out designs (shape -> helper -> patterns -> Cd), in drag counts:")
    results = {}
    for k in ("per_mode", "shared"):
        for m in ("exact", "ridge"):
            err = np.abs(label_shapes(helpers[k], pod, shapes, m) - truth)
            results[f"{k}+{m}"] = err
            tag = "  <- used" if (k, m) == (kind, method) else ""
            print(f"   helper {k:8s} + {m:5s}: mean {err.mean() * 1000:6.1f}, worst {err.max() * 1000:6.1f}{tag}")
    for m in ("exact", "ridge"):
        _, proj = validate_on_holdout(dataset, pod, m)
        print(f"   ({m} with the TRUE pattern amounts: mean {proj['mean_abs_error'] * 1000:.1f})")
    path_errors = results[f"{kind}+{method}"]
    mean_counts, max_counts = path_errors.mean() * 1000, path_errors.max() * 1000
    print(f"   run noise about {noise_counts:.1f}; gate: mean <= {gate * 1000:.0f}")
    if path_errors.mean() > gate:
        sys.exit("Stopping: the label check failed, so the synthetic labels cannot be trusted.")
    print("   gate passed\n")

    augmented = build_augmented_dataset(dataset, pod, gp, method=method)
    counts = augmented["source"].value_counts().to_dict()
    print(f"5. Augmented data: {counts.get('real', 0)} real + {counts.get('synthetic', 0)} synthetic rows")

    best_name, best_model, table = train_and_select(augmented, holdout)
    baseline = real_only_baseline(augmented[augmented["source"] == "real"], holdout)
    for df in (table, baseline):
        for col in ("holdout_mae", "holdout_rmse", "holdout_max_error"):
            df[col + "_counts"] = (df[col] * 1000).round(2)
    print("\n6. Hold-out error in drag counts (mean / worst), scored on real designs only:")
    print("   trained WITH augmentation:")
    for _, r in table.iterrows():
        print(f"     {r['model']:14s} {r['holdout_mae_counts']:6.2f} / {r['holdout_max_error_counts']:6.2f}")
    print("   trained on the real designs only (no augmentation):")
    for _, r in baseline.iterrows():
        print(f"     {r['model']:14s} {r['holdout_mae_counts']:6.2f} / {r['holdout_max_error_counts']:6.2f}")

    tag = os.path.basename(os.path.normpath(dataset.data_dir))
    model_dir = os.path.join(config.MODELS_DIR, f"{tag}_{speed}kmh")
    path = save_model(best_model, best_name, out_dir=model_dir)
    print(f"\n7. Best model: {best_name}, saved to {path}")

    area_name, area_model, area_table = train_area_model(train, holdout)
    area_path = save_area_model(area_model, out_dir=model_dir)
    best_area = area_table.iloc[0]
    print(f"\n8. Frontal-area model (the energy model needs drag area Cd x A): {area_name}, saved to {area_path}")
    print(f"   hold-out error: mean {best_area['holdout_mae_percent']:.2f}%, worst {best_area['holdout_max_error_percent']:.2f}%")

    repeats = []
    if config.N_REPEAT_SPLITS > 0:
        print(f"\n9. Stability: the whole build repeated on {config.N_REPEAT_SPLITS} different hold-out draws "
              f"(same rules; labeller choice kept: {kind} + {method}). Drag counts, mean +/- spread:")
        for r in range(config.N_REPEAT_SPLITS):
            repeats.append(one_split(dataset.with_split(config.RANDOM_STATE + 1000 + r), kind, method))
            print(f"   draw {r + 1}/{config.N_REPEAT_SPLITS} done", flush=True)
        rep = pd.DataFrame(repeats)
        print(f"   label check: mean {rep['label_mean'].mean():.1f} +/- {rep['label_mean'].std():.1f} "
              f"(worst design, averaged: {rep['label_max'].mean():.1f})")
        for m in ("xgboost", "random_forest", "ann_mlp"):
            a, b = rep[f"aug_{m}"], rep[f"real_{m}"]
            better = int((a < b).sum())
            print(f"   {m:14s} with augmentation {a.mean():6.2f} +/- {a.std():5.2f}   real only {b.mean():6.2f} +/- {b.std():5.2f}"
                  f"   (augmentation better in {better}/{len(rep)} draws)")

    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    summary = {
        "source": source, "speed_kmh": speed, "train_designs": len(train), "holdout_designs": len(holdout),
        "holdout_ids": holdout["run_id"].tolist(), "low_confidence_designs": n_low,
        "design_box": {n: list(b) for n, b in config.SHAPE_PARAM_BOUNDS.items()}, "model_dir": model_dir,
        "label_check_mean_counts": mean_counts, "label_check_max_counts": max_counts,
        "helper_kind": kind, "label_method": method, "label_gate_counts": gate * 1000,
        "labeller_cv_counts": {f"{k}+{m}": e * 1000 for (k, m), e in cv.items()},
        "label_check_by_method": {m: {"mean": float(e.mean() * 1000), "max": float(e.max() * 1000)} for m, e in results.items()},
        "best_model": best_name, "with_augmentation": table.to_dict("records"),
        "real_only": baseline.to_dict("records"),
        "frontal_area_model": area_name, "frontal_area_results": area_table.to_dict("records"),
        "repeated_splits": repeats,
    }
    with open(os.path.join(config.OUTPUTS_DIR, f"workflow1_{tag}_{speed}kmh.json"), "w") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
