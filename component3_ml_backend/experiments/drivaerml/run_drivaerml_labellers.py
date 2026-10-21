"""
Labeller comparison on REAL CFD (DrivAerML, CC BY-SA 4.0), same 80 runs,
splits and 404-run test set as run_drivaerml_pod.py.

Question: can the POD pressure patterns add information beyond a Gaussian
Process fitted straight to Cd (which scored 3.9 drag counts and beat every
augmented model)? Labellers compared:

  L0     shape -> GP helper -> 15 POD amounts -> ridge(amounts, shape) -> Cd   (current method)
  L0-30  the same with 30 POD patterns
  L1     GP fitted directly to Cd (no physics; the one to beat)
  L2     L1 + a POD correction: ridge(amounts) predicts L1's error; L1's
         training errors come from 5-fold out-of-fold predictions, so the
         correction never sees L1's own fit to a run
  L3     ridge(amounts, shape, L1 prediction) -> Cd, again with out-of-fold
         L1 predictions on the training runs

For each labeller: (a) its own error on the 404 unseen runs, (b) the label
check on the 15 held-out field runs, (c) XGBoost / Random Forest / ANN
trained on 65 real + 1,200 shapes labelled by it, scored on the 404 runs.

Usage: python scripts/run_drivaerml_labellers.py   (needs run_drivaerml_pod.py's mapped_fields.npz)
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from component3 import config
from component3.workflow1.pod import _snapshot_pod
from run_drivaerml_pod import MAPPED, N_REPEATS, N_SYNTHETIC, N_TRAIN, PARAMS, SCALAR_TABLE, unit
from component3.workflow1.training import CANDIDATE_BUILDERS

ALPHAS = np.logspace(-4, 3, 40)


def gp(n_inputs):
    kernel = ConstantKernel(1.0, (1e-2, 1e3)) * RBF(np.ones(n_inputs), (1e-2, 1e3)) + WhiteKernel(1e-2, (1e-6, 1e1))
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3, random_state=config.RANDOM_STATE)


def ridge_fit(features, target):
    scaler = StandardScaler().fit(features)
    model = RidgeCV(alphas=ALPHAS).fit(scaler.transform(features), target)
    return lambda f: model.predict(scaler.transform(f))


def oof_gp_cd(x, y, folds=5, seed=0):
    """Out-of-fold GP predictions of Cd for the training runs."""
    out = np.zeros_like(y)
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(x):
        out[te] = gp(x.shape[1]).fit(x[tr], y[tr]).predict(x[te])
    return out


def build_labellers(x_tr, cd_tr, fields_tr, rep):
    """Return {name: function(shape array) -> Cd} plus helpers to project true fields."""
    labellers, projectors = {}, {}
    gp_cd = gp(x_tr.shape[1]).fit(x_tr, cd_tr)
    l1 = lambda x: gp_cd.predict(x)
    labellers["L1_gp_direct"] = l1
    cd_oof = oof_gp_cd(x_tr, cd_tr, seed=rep)

    mean_field = fields_tr.mean(axis=0)
    centred = fields_tr - mean_field
    for n_modes in (15, 30):
        modes, _ = _snapshot_pod(centred, n_modes)
        amps = centred @ modes.T
        helper = gp(x_tr.shape[1]).fit(x_tr, amps)
        calib = ridge_fit(np.hstack([amps, x_tr]), cd_tr)
        tag = "L0" if n_modes == 15 else "L0-30"
        labellers[tag] = (lambda h, c: (lambda x: c(np.hstack([h.predict(x), x]))))(helper, calib)
        projectors[tag] = (lambda m, c: (lambda f, x: c(np.hstack([(f - mean_field) @ m.T, x]))))(modes, calib)
        if n_modes == 15:
            resid = ridge_fit(amps, cd_tr - cd_oof)
            labellers["L2_gp_plus_pod_residual"] = (lambda h, r: (lambda x: l1(x) + r(h.predict(x))))(helper, resid)
            projectors["L2_gp_plus_pod_residual"] = (lambda m, r: (lambda f, x: l1(x) + r((f - mean_field) @ m.T)))(modes, resid)
            feat = ridge_fit(np.hstack([amps, x_tr, cd_oof[:, None]]), cd_tr)
            labellers["L3_ridge_with_gp_feature"] = (lambda h, c: (lambda x: c(np.hstack([h.predict(x), x, l1(x)[:, None]]))))(helper, feat)
            projectors["L3_ridge_with_gp_feature"] = (lambda m, c: (lambda f, x: c(np.hstack([(f - mean_field) @ m.T, x, l1(x)[:, None]]))))(modes, feat)
    return labellers, projectors


def main():
    t0 = time.time()
    table = pd.read_csv(SCALAR_TABLE)
    data = np.load(MAPPED)
    run_ids, fields = data["run_ids"], data["cp"].astype(np.float64)
    t = table.set_index("run_id")
    x_all = unit(t.loc[run_ids, PARAMS].to_numpy(dtype=float))
    cd_all = t.loc[run_ids, "Cd"].to_numpy(dtype=float)
    test = table[~table["run_id"].isin(run_ids)]
    x_test, y_test = unit(test[PARAMS].to_numpy(dtype=float)), test["Cd"].to_numpy(dtype=float)

    rows = []
    for rep in range(N_REPEATS):
        order = np.random.default_rng(config.RANDOM_STATE + 100 + rep).permutation(len(run_ids))  # same splits as before
        tr, ho = order[:N_TRAIN], order[N_TRAIN:]
        labellers, projectors = build_labellers(x_all[tr], cd_all[tr], fields[tr], rep)
        x_syn = qmc.LatinHypercube(d=len(PARAMS), seed=config.RANDOM_STATE + 10 + rep).random(N_SYNTHETIC)
        for name, lab in labellers.items():
            e_test = np.abs(lab(x_test) - y_test)
            e_ho = np.abs(lab(x_all[ho]) - cd_all[ho])
            row = {"repeat": rep, "labeller": name, "labeller_test_mae": e_test.mean(), "label_check_mean": e_ho.mean(),
                   "label_check_max": e_ho.max()}
            if name in projectors:
                row["true_field_check_mean"] = np.abs(projectors[name](fields[ho], x_all[ho]) - cd_all[ho]).mean()
            y_syn = lab(x_syn)
            xx, yy = np.vstack([x_all[tr], x_syn]), np.concatenate([cd_all[tr], y_syn])
            for mname, build in CANDIDATE_BUILDERS.items():
                row[f"aug_{mname}"] = np.abs(build().fit(xx, yy).predict(x_test) - y_test).mean()
            rows.append(row)
        print(f"  repeat {rep + 1}/{N_REPEATS} done  [{time.time() - t0:.0f} s]")

    res = pd.DataFrame(rows)
    cols = ["labeller_test_mae", "label_check_mean", "label_check_max", "true_field_check_mean",
            "aug_xgboost", "aug_random_forest", "aug_ann_mlp"]
    mean = res.groupby("labeller")[cols].mean() * 1000
    sd = res.groupby("labeller")[cols].std() * 1000
    print(f"\nDrag counts, mean (± sd) over {N_REPEATS} repeats. Test = {len(test)} unseen runs; label check = 15 held-out field runs.")
    print(f"{'labeller':26s} {'own error':>13s} {'label chk':>10s} {'true-field':>10s} {'aug XGB':>13s} {'aug RF':>13s} {'aug ANN':>13s}")
    for name in ["L0", "L0-30", "L1_gp_direct", "L2_gp_plus_pod_residual", "L3_ridge_with_gp_feature"]:
        m, s = mean.loc[name], sd.loc[name]
        tf = "" if np.isnan(m["true_field_check_mean"]) else f"{m['true_field_check_mean']:.1f}"
        print(f"{name:26s} {m['labeller_test_mae']:6.2f} ± {s['labeller_test_mae']:4.2f} {m['label_check_mean']:10.1f} {tf:>10s} "
              f"{m['aug_xgboost']:6.2f} ± {s['aug_xgboost']:4.2f} {m['aug_random_forest']:6.2f} ± {s['aug_random_forest']:4.2f} "
              f"{m['aug_ann_mlp']:6.2f} ± {s['aug_ann_mlp']:4.2f}")
    piv = res.pivot(index="repeat", columns="labeller", values="labeller_test_mae") * 1000
    for name in ["L2_gp_plus_pod_residual", "L3_ridge_with_gp_feature"]:
        d = piv[name] - piv["L1_gp_direct"]
        print(f"\n{name} minus L1 (own error on the 404 runs): {d.mean():+.2f} ± {d.std():.2f} counts, better in {int((d < 0).sum())}/{len(d)} repeats")

    with open(os.path.join(config.OUTPUTS_DIR, "drivaerml_labeller_results.json"), "w") as fh:
        json.dump(res.to_dict("records"), fh, indent=2)
    print(f"\nTotal time {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
