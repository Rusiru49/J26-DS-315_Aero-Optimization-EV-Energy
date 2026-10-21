"""
Workflow 1 end to end, for one speed: data folder -> checked data -> POD ->
cross-validated build and test (labeller, augmentation, XGBoost / Random Forest
/ ANN with and without augmentation) -> choose the drag model -> retrain it on
all designs -> save. Prints the results you would show at PP1.

Usage:
    python scripts/run_workflow1.py                  # the dummy dataset, 100 km/h
    python scripts/run_workflow1.py placeholder 40   # the dummy dataset, 40 km/h
    python scripts/run_workflow1.py real 100         # Member 2's data (config.REAL_DATA_DIR)
    python scripts/run_workflow1.py <folder> 40      # any folder in the contract layout

How it tests itself: workflow1/evaluation.py (cross-validation by design, every
testable design is the exam once per repeat). The models are saved to
models/<dataset folder name>_<speed>kmh/, with model_card.json describing them.
"""

import datetime
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import joblib
import numpy as np

from component3 import config
from component3.data.checker import check_dataset
from component3.data.loader import load_cfd_dataset
from component3.workflow1.evaluation import (augmentation_wins, cross_validate, summarise, designs_to_test,
                                             train_final, train_final_area)
from component3.workflow1.pod import dataset_arrays, exact_cd_from_faces, fit_pod_arrays

COUNTS = 1000  # 1 drag count = 0.001 Cd


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else "placeholder"
    speed = int(sys.argv[2]) if len(sys.argv) > 2 else config.DEFAULT_SPEED_KMH
    print(f"=== Workflow 1: {source} data, {speed} km/h ===\n")

    dataset = load_cfd_dataset(source, speed)
    results = check_dataset(dataset.data_dir)
    fails = [r for r in results if r[1] == "FAIL"]
    print(f"1. Data check: {len(results) - len(fails)}/{len(results)} checks passed")
    if fails:
        for name, _, detail in fails:
            print(f"   FAIL {name}: {detail}")
        sys.exit("Stopping: the data does not match the contract.")

    runs = dataset.runs
    n_test = len(designs_to_test(runs))
    n_low = int(runs["is_low_confidence"].sum())
    print(f"2. Designs: {len(runs)} usable ({dataset.n_failed} failed left out); {n_test} can be tested "
          f"(baseline, corners and {n_low} low-confidence designs always train)")
    box = ", ".join(f"{n} {lo:g}-{hi:g}" for n, (lo, hi) in config.SHAPE_PARAM_BOUNDS.items())
    print(f"   design box: {box}\n")

    a = dataset_arrays(dataset, runs.reset_index(drop=True))
    pod_all = fit_pod_arrays(a)
    centered = a["fields"] - pod_all.mean_field
    kept = (centered @ pod_all.mode_shapes.T) @ pod_all.mode_shapes
    explained = 1.0 - np.sum((centered - kept) ** 2) / np.sum(centered ** 2)
    print(f"3. POD on all designs: {config.N_POD_MODES} patterns explain {explained:.1%} of the pressure variation")
    sanity = np.array([abs(exact_cd_from_faces(dataset, r) - c) for r, c in zip(runs["run_id"], runs["Cd"])])
    print(f"   face integral of the TRUE fields vs Cd label: mean {sanity.mean() * COUNTS:.2f}, "
          f"worst {sanity.max() * COUNTS:.2f} drag counts (checks the conventions)\n")

    print(f"4. Cross-validation by design: {config.N_CV_FOLDS} groups x {config.N_CV_REPEATS} repeats; "
          f"every testable design is the exam once per repeat")
    cv = cross_validate(dataset)
    table, area_table, diag = summarise(cv)
    choices = {c: cv["labeller_choices"].count(c) for c in sorted(set(cv["labeller_choices"]))}
    print(f"   labeller chosen inside the folds: {choices}")

    lab = table.loc["label"]
    print("\n5. Results on REAL designs only, drag counts (mean over repeats +/- spread, worst design):")
    print(f"   helper labels (label check)      {lab.mae * COUNTS:6.1f} +/- {lab.mae_spread * COUNTS:4.1f}   worst {lab.worst * COUNTS:5.1f}")
    for m in ("xgboost", "random_forest", "ann_mlp"):
        ra, rr = table.loc[f"{m}+aug"], table.loc[f"{m}+real"]
        wins, n = augmentation_wins(cv, m)
        print(f"   {m:14s} with augmentation {ra.mae * COUNTS:6.1f} +/- {ra.mae_spread * COUNTS:4.1f}   "
              f"real only {rr.mae * COUNTS:6.1f} +/- {rr.mae_spread * COUNTS:4.1f}   "
              f"(augmentation better in {wins}/{n} repeats)")
    print("   diagnostic - augmented models vs synthetic shapes they never saw (how well they learned the helper):")
    print("   " + ", ".join(f"{m} {v * COUNTS:.1f}" for m, v in diag.items()))

    ranked = table.drop(index="label").sort_values("mae")
    winner = ranked.index[0]
    best = ranked.iloc[0]
    print(f"\n6. Chosen drag model: {winner} (lowest cross-validated error). "
          f"Augmentation {'IS' if winner.endswith('+aug') else 'is NOT'} used for this dataset.")
    model, info = train_final(dataset, winner)
    tag = os.path.basename(os.path.normpath(dataset.data_dir))
    model_dir = os.path.join(config.MODELS_DIR, f"{tag}_{speed}kmh")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "drag_model.joblib")
    joblib.dump(model, model_path)
    print(f"   retrained on all {info['n_designs']} designs, saved to {model_path}")
    print(f"   expected error on a new shape: about {best.mae * COUNTS:.1f} drag counts "
          f"(RMSE {best.rmse * COUNTS:.1f}, worst seen {best.worst * COUNTS:.1f})")

    area_name = area_table.iloc[0]["model"]
    area_path = os.path.join(model_dir, "frontal_area_model.joblib")
    joblib.dump(train_final_area(dataset, area_name), area_path)
    print(f"\n7. Frontal-area model: {area_name}, cross-validated error mean "
          f"{area_table.iloc[0]['mae_percent']:.2f}%, worst {area_table.iloc[0]['worst_percent']:.2f}%; saved to {area_path}")

    card = {
        "dataset": tag, "speed_kmh": speed, "created": datetime.date.today().isoformat(),
        "drag_model": winner, **info, "drag_model_file": "drag_model.joblib",
        "frontal_area_model": area_name, "frontal_area_model_file": "frontal_area_model.joblib",
        "inputs": config.SHAPE_PARAM_NAMES, "design_box": {n: list(b) for n, b in config.SHAPE_PARAM_BOUNDS.items()},
        "baseline_shape": dict(config.BASELINE_SHAPE),
        "cd_error_mae": best.mae, "cd_error_rmse": best.rmse, "cd_error_worst": best.worst,
        "frontal_area_error_mae_percent": float(area_table.iloc[0]["mae_percent"]),
        "validation": f"cross-validation by design, {cv['n_folds']} groups x {cv['n_repeats']} repeats, "
                      f"{n_test} testable designs, scored on real CFD only",
    }
    with open(os.path.join(model_dir, "model_card.json"), "w") as fh:
        json.dump(card, fh, indent=2)

    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    summary = {
        **card, "source": source, "n_testable": n_test, "low_confidence_designs": n_low,
        "labeller_choices": choices, "results_counts": (table * COUNTS).round(2).reset_index().to_dict("records"),
        "diagnostic_counts": {k: round(v * COUNTS, 2) for k, v in diag.items()},
        "augmentation_wins": {m: augmentation_wins(cv, m) for m in ("xgboost", "random_forest", "ann_mlp")},
        "frontal_area_results": area_table.to_dict("records"),
    }
    with open(os.path.join(config.OUTPUTS_DIR, f"workflow1_{tag}_{speed}kmh.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    cv["oof"].to_csv(os.path.join(config.OUTPUTS_DIR, f"workflow1_{tag}_{speed}kmh_predictions.csv"), index=False)


if __name__ == "__main__":
    main()
