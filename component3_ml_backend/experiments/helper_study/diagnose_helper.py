"""
Where does the helper (labeller) lose its accuracy? A diagnosis on Member 2's
real CFD, with the same cross-validation by design as Workflow 1 (the same
groups, scored on the held-out designs only). Nothing here changes the
pipeline; it only measures.

The helper's path for a shape it has never seen:
    5 shape values --(GP)--> 15 pattern amounts --(ridge or exact integral)--> Cd

Each line below swaps ONE predicted part for the truth, so the drop in error
shows how much that part costs:
    A  true field, true geometry, exact integral    (data sanity: should be ~0)
    B  true field cut to 15 patterns, true geometry  (cost of keeping only 15 patterns)
    C  true pattern amounts, PREDICTED geometry      (+ cost of the geometry model)
    D  true pattern amounts, ridge                   (cost of the ridge amounts -> Cd step)
    E  PREDICTED amounts, exact integral             (the full "exact" helper)
    F  PREDICTED amounts, ridge                      (the full "ridge" helper, used today)
Reference only (not pipeline candidates): a GP and a linear fit straight from
the 5 shape values to Cd, to show how much the shape values alone can explain.

Usage:  python experiments/helper_study/diagnose_helper.py [data_dir]
Writes: experiments/helper_study/results/diagnosis_<speed>kmh.json and .csv
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from component3 import config
from component3.data.loader import CFDDataset
from component3.workflow1.evaluation import make_folds
from component3.workflow1.helper import fit_helper
from component3.workflow1.pod import _small_gp, dataset_arrays, fit_pod_arrays

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
COUNTS = 1000


def diagnose(ds, n_folds=config.N_CV_FOLDS, n_repeats=config.N_CV_REPEATS):
    runs = ds.runs.reset_index(drop=True)
    a = dataset_arrays(ds, runs)
    ids = runs["run_id"].to_numpy()
    rows, amp_rows = [], []
    for rep in range(n_repeats):
        for f, test_ids in enumerate(make_folds(runs, n_folds, config.RANDOM_STATE + 100 * rep)):
            te = np.flatnonzero(np.isin(ids, test_ids))
            tr = np.flatnonzero(~np.isin(ids, test_ids))
            sub = {k: v[tr] for k, v in a.items()}
            pod = fit_pod_arrays(sub)
            amps_train = (sub["fields"] - pod.mean_field) @ pod.mode_shapes.T
            helper = fit_helper(sub["x_unit"], amps_train, "per_mode")
            xu = a["x_unit"][te]
            amp_true = (a["fields"][te] - pod.mean_field) @ pod.mode_shapes.T
            amp_pred = helper.predict(xu)
            field_15 = pod.mean_field + amp_true @ pod.mode_shapes
            w, aref, cdf = a["w"][te], a["aref"][te], a["cdf"][te]
            parts = {
                "A_true_field_true_geometry": -np.sum(a["fields"][te] * w, axis=1) / aref + cdf,
                "B_15_patterns_true_geometry": -np.sum(field_15 * w, axis=1) / aref + cdf,
                "C_true_amounts_predicted_geometry": pod.cd_batch(amp_true, xu, "exact"),
                "D_true_amounts_ridge": pod.cd_batch(amp_true, xu, "ridge"),
                "E_helper_exact": pod.cd_batch(amp_pred, xu, "exact"),
                "F_helper_ridge": pod.cd_batch(amp_pred, xu, "ridge"),
                "ref_gp_on_cd": _small_gp().fit(sub["x_unit"], sub["cd"]).predict(xu),
                "ref_linear_on_cd": LinearRegression().fit(sub["x_unit"], sub["cd"]).predict(xu),
            }
            scale = amps_train.std(axis=0) + 1e-12
            for i, idx in enumerate(te):
                rec = {"repeat": rep, "fold": f, "run_id": ids[idx], "cd": a["cd"][idx]}
                rec.update({k: v[i] for k, v in parts.items()})
                rows.append(rec)
                amp_rows.append({"repeat": rep, "run_id": ids[idx],
                                 **{f"mode{k + 1}": abs(amp_pred[i, k] - amp_true[i, k]) / scale[k]
                                    for k in range(amp_true.shape[1])}})
            print(f"   repeat {rep + 1}, fold {f + 1} done", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(amp_rows)


def report(df, amps, speed):
    cols = [c for c in df.columns if c[0] in "ABCDEF" or c.startswith("ref_")]
    table = {c: float((df[c] - df["cd"]).abs().groupby(df["repeat"]).mean().mean() * COUNTS) for c in cols}
    print(f"\n=== Helper diagnosis, {speed} km/h: mean absolute error on held-out real designs (drag counts) ===")
    for c in cols:
        print(f"   {c:36s} {table[c]:6.1f}")
    per = df.groupby("run_id")[["cd", "F_helper_ridge", "E_helper_exact", "ref_gp_on_cd"]].mean()
    signed = (per["F_helper_ridge"] - per["cd"]) * COUNTS
    low = per["cd"] <= per["cd"].quantile(0.25)
    corr = float(np.corrcoef(per["cd"], signed)[0, 1])
    print(f"\n   bias check (ridge helper): signed error on the 25% lowest-drag designs {signed[low].mean():+.1f}, "
          f"on the rest {signed[~low].mean():+.1f}; correlation of signed error with true Cd {corr:+.2f}")
    print("   (positive error on low-drag designs + negative correlation = the helper pulls extreme shapes toward the average)")
    mode_err = amps.drop(columns=["repeat", "run_id"]).mean()
    print("\n   pattern-amount prediction error (in units of that pattern's spread in training), first 8 patterns:")
    print("   " + ", ".join(f"{m} {v:.2f}" for m, v in mode_err.head(8).items()))
    return {"speed_kmh": speed, "mae_counts": table, "low_drag_signed_error": float(signed[low].mean()),
            "rest_signed_error": float(signed[~low].mean()), "corr_signed_error_vs_cd": corr,
            "amount_error_by_pattern": mode_err.round(3).to_dict()}


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else config.REAL_DATA_DIR
    os.makedirs(OUT, exist_ok=True)
    for speed in (40, 100):
        ds = CFDDataset(data_dir, speed)
        df, amps = diagnose(ds)
        summary = report(df, amps, speed)
        df.to_csv(os.path.join(OUT, f"diagnosis_{speed}kmh.csv"), index=False)
        with open(os.path.join(OUT, f"diagnosis_{speed}kmh.json"), "w") as fh:
            json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
