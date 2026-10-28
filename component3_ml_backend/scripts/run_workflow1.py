"""
Workflow 1 end to end: data folder -> checked data -> POD -> cross-validated
build and test (per-speed labeller, augmentation, XGBoost / Random Forest / ANN
with and without augmentation) -> choose the drag model -> retrain it on all
designs -> save. Prints the results you would show at PP1.

ONE drag model for all speeds: its inputs are the 5 shape values and the speed
in km/h (see workflow1/evaluation.py for why).

Usage:
    python scripts/run_workflow1.py                  # the dummy dataset, all speeds
    python scripts/run_workflow1.py real             # Member 2's data (config.REAL_DATA_DIR)
    python scripts/run_workflow1.py <folder>         # any folder in the contract layout
    python scripts/run_workflow1.py real 100         # only the listed speed(s)

How it tests itself: cross-validation by design, every testable design is the
exam once per repeat, hidden at every speed. The model is saved to
models/<dataset folder name>/, with model_card.json describing it.
"""

import datetime
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import joblib
import numpy as np
import pandas as pd

from component3 import config
from component3.data.checker import check_dataset
from component3.data.loader import load_cfd_dataset
from component3.workflow1.evaluation import (INPUT_NAMES, augmentation_wins, choose_candidate, cross_validate,
                                             for_speed, summarise, train_final, train_final_area)
from component3.workflow1.label_quality import calibrated_sigma, calibration_report
from component3.workflow1.pod import dataset_arrays, exact_cd_from_faces, fit_pod_arrays

COUNTS = 1000  # 1 drag count = 0.001 Cd


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else "placeholder"
    first = load_cfd_dataset(source, config.DEFAULT_SPEED_KMH)
    all_speeds = sorted(int(s) for s in pd.read_csv(os.path.join(first.data_dir, "run_table.csv"))["speed_kmh"].unique())
    speeds = [int(s) for s in sys.argv[2:]] or all_speeds
    datasets = {s: load_cfd_dataset(source, s) for s in speeds}
    print(f"=== Workflow 1: {source} data, speeds {', '.join(f'{s} km/h' for s in speeds)}, one drag model ===\n")

    results = check_dataset(first.data_dir)
    fails = [r for r in results if r[1] == "FAIL"]
    print(f"1. Data check: {len(results) - len(fails)}/{len(results)} checks passed")
    if fails:
        for name, _, detail in fails:
            print(f"   FAIL {name}: {detail}")
        sys.exit("Stopping: the data does not match the contract.")

    runs = datasets[speeds[0]].runs
    n_low = int(runs["is_low_confidence"].sum())
    n_runs = sum(len(ds.runs) for ds in datasets.values())
    print(f"2. Designs: {len(runs)} shapes x {len(speeds)} speeds = {n_runs} runs "
          f"({sum(ds.n_failed for ds in datasets.values())} failed left out); "
          f"baseline, corners and {n_low} low-confidence designs always train")
    box = ", ".join(f"{n} {lo:g}-{hi:g}" for n, (lo, hi) in config.SHAPE_PARAM_BOUNDS.items())
    print(f"   design box: {box}\n")

    print("3. Pressure patterns (POD) per speed, on all designs:")
    for s, ds in datasets.items():
        a = dataset_arrays(ds, ds.runs.reset_index(drop=True))
        pod_all = fit_pod_arrays(a)
        centered = a["fields"] - pod_all.mean_field
        kept = (centered @ pod_all.mode_shapes.T) @ pod_all.mode_shapes
        explained = 1.0 - np.sum((centered - kept) ** 2) / np.sum(centered ** 2)
        sanity = np.array([abs(exact_cd_from_faces(ds, r) - c) for r, c in zip(ds.runs["run_id"], ds.runs["Cd"])])
        print(f"   {s} km/h: {config.N_POD_MODES} patterns explain {explained:.1%}; face integral of the TRUE fields vs "
              f"Cd label: mean {sanity.mean() * COUNTS:.2f}, worst {sanity.max() * COUNTS:.2f} drag counts")

    print(f"\n4. Cross-validation by design: {config.N_CV_FOLDS} groups x {config.N_CV_REPEATS} repeats; every testable "
          f"design is the exam once per repeat, hidden at every speed")
    cv = cross_validate(datasets)
    print(f"   {cv['n_testable']} designs can be tested")
    for s in speeds:
        ch = cv["labeller_choices"][s]
        print(f"   labeller chosen inside the folds, {s} km/h: " + ", ".join(f"{c} x{ch.count(c)}" for c in sorted(set(ch))))

    print("\n5. Results on REAL designs only, drag counts (mean over repeats +/- spread, worst design):")
    per_speed = {}
    for s in speeds:
        cv_s = for_speed(cv, s)
        table, area_table, diag = summarise(cv_s)
        per_speed[s] = (cv_s, table, area_table, diag)
        lab = table.loc["label"]
        print(f"   --- {s} km/h")
        print(f"   helper labels (label check)      {lab.mae * COUNTS:6.1f} +/- {lab.mae_spread * COUNTS:4.1f}   worst {lab.worst * COUNTS:5.1f}")
        for m in ("xgboost", "random_forest", "ann_mlp"):
            ra, rr = table.loc[f"{m}+aug"], table.loc[f"{m}+real"]
            wins, n = augmentation_wins(cv_s, m)
            print(f"   {m:14s} with augmentation {ra.mae * COUNTS:6.1f} +/- {ra.mae_spread * COUNTS:4.1f}   "
                  f"real only {rr.mae * COUNTS:6.1f} +/- {rr.mae_spread * COUNTS:4.1f}   "
                  f"(augmentation better in {wins}/{n} repeats)")
        for c in ("ann_mlp+aug_weighted", "ann_mlp+pretrain_finetune"):
            r = table.loc[c]
            print(f"   {c:26s} {r.mae * COUNTS:6.1f} +/- {r.mae_spread * COUNTS:4.1f}   worst {r.worst * COUNTS:5.1f}")
        print("   diagnostic - augmented models vs synthetic shapes they never saw: "
              + ", ".join(f"{m} {v * COUNTS:.1f}" for m, v in diag.items()))
        q = cv["label_quality"][s]
        print(f"   synthetic labels kept after the per-label checks: {q['kept']:.1%} of {q['n_proposed']:.0f} "
              f"(outside validated region {q['out_of_region']:.1%}, Cp above stagnation {q['cp_above_stagnation']:.1%}, "
              f"pressure drag not positive {q['pressure_drag_not_positive']:.1%}, Cd out of range {q['cd_out_of_range']:.1%})")
        cal = calibration_report(cv_s["oof"])
        per_speed[s] = per_speed[s] + (cal,)
        print(f"   per-label uncertainty, calibrated on held-out real CFD: real Cd inside +/-1 sigma {cal['coverage_1sigma']:.0%} "
              f"(ideal 68%), inside +/-2 sigma {cal['coverage_2sigma']:.0%} (ideal 95%); typical sigma "
              f"{cal['median_sigma'] * COUNTS:.1f} counts")

    winner, ranking = choose_candidate(cv)
    print(f"\n6. Chosen drag model: {winner} (lowest cross-validated error averaged over the speeds). "
          f"Augmentation {'is NOT' if winner.endswith('+real') else 'IS'} used.")
    model, info = train_final(datasets, winner)
    synthetic = info.pop("synthetic_labels", {})
    tag = os.path.basename(os.path.normpath(first.data_dir))
    model_dir = os.path.join(config.MODELS_DIR, tag)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "drag_model.joblib"))
    print(f"   one model, inputs {INPUT_NAMES}; retrained on all designs at all speeds, saved to {model_dir}")
    errors = {}
    for s in speeds:
        best = per_speed[s][1].loc[winner]
        errors[s] = {"mae": best.mae, "rmse": best.rmse, "worst": best.worst}
        print(f"   expected error on a new shape at {s} km/h: about {best.mae * COUNTS:.1f} drag counts "
              f"(RMSE {best.rmse * COUNTS:.1f}, worst seen {best.worst * COUNTS:.1f})")

    area_table = per_speed[speeds[0]][2]
    area_name = area_table.iloc[0]["model"]
    joblib.dump(train_final_area(datasets, area_name), os.path.join(model_dir, "frontal_area_model.joblib"))
    print(f"\n7. Frontal-area model (geometry, no speed): {area_name}, cross-validated error mean "
          f"{area_table.iloc[0]['mae_percent']:.2f}%, worst {area_table.iloc[0]['worst_percent']:.2f}%")

    validation = (f"cross-validation by design, {cv['n_folds']} groups x {cv['n_repeats']} repeats, "
                  f"{cv['n_testable']} testable designs hidden at every speed, scored on real CFD only")
    card = {
        "dataset": tag, "created": datetime.date.today().isoformat(), "drag_model": winner, **info,
        "drag_model_file": "drag_model.joblib", "frontal_area_model": area_name,
        "frontal_area_model_file": "frontal_area_model.joblib", "frontal_area_inputs": config.SHAPE_PARAM_NAMES,
        "design_box": {n: list(b) for n, b in config.SHAPE_PARAM_BOUNDS.items()},
        "baseline_shape": dict(config.BASELINE_SHAPE),
        "cd_error_by_speed": {str(s): e for s, e in errors.items()},
        "frontal_area_error_mae_percent": float(area_table.iloc[0]["mae_percent"]),
        "validation": validation,
        "synthetic_label_checks": {str(k): v for k, v in cv["label_quality"].items()},
        "synthetic_label_uncertainty_calibration": {str(k): per_speed[k][4] for k in speeds},
    }
    with open(os.path.join(model_dir, "model_card.json"), "w") as fh:
        json.dump(card, fh, indent=2, default=float)

    # every synthetic label of the final model, with its calibrated uncertainty and checks
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    for s, t in synthetic.items():
        t = t.copy()
        t.insert(t.columns.get_loc("sigma_raw") + 1, "sigma_calibrated", calibrated_sigma(t["sigma_raw"], (per_speed[s][4]["a"], per_speed[s][4]["b"])))
        t.to_csv(os.path.join(config.OUTPUTS_DIR, f"workflow1_{tag}_{s}kmh_synthetic_labels.csv"), index=False)
    if synthetic:
        print(f"\n   every synthetic label with its own calibrated uncertainty and checks: "
              f"outputs/workflow1_{tag}_<speed>kmh_synthetic_labels.csv")

    # per-speed result files (the PP1 figures read these)
    for s in speeds:
        cv_s, table, area_t, diag, cal = per_speed[s]
        summary = {
            "dataset": tag, "speed_kmh": s, "drag_model": winner, "one_model_all_speeds": True,
            "n_designs": int(len(datasets[s].runs)), "n_testable": cv["n_testable"], "validation": validation,
            "labeller_choices": cv["labeller_choices"][s],
            "results_counts": (table * COUNTS).round(2).reset_index().to_dict("records"),
            "diagnostic_counts": {k: round(v * COUNTS, 2) for k, v in diag.items()},
            "augmentation_wins": {m: augmentation_wins(cv_s, m) for m in ("xgboost", "random_forest", "ann_mlp")},
            "frontal_area_results": area_t.to_dict("records"),
            "ranking_mean_over_speeds_counts": (ranking * COUNTS).round(2).to_dict(),
            "synthetic_label_checks": cv["label_quality"][s], "label_uncertainty_calibration": cal,
        }
        with open(os.path.join(config.OUTPUTS_DIR, f"workflow1_{tag}_{s}kmh.json"), "w") as fh:
            json.dump(summary, fh, indent=2, default=float)
        cv_s["oof"].to_csv(os.path.join(config.OUTPUTS_DIR, f"workflow1_{tag}_{s}kmh_predictions.csv"), index=False)


if __name__ == "__main__":
    main()
