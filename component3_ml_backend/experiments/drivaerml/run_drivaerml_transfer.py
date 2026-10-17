"""
Test 2: can external CFD (a "source" car family) make the augmentation
labels better than a Gaussian Process trained on the target's own few runs?

Uses only the DrivAerML small-file table (Cd + 16 shape values; CC BY-SA 4.0).
The 484 runs are split into two "car families" by one shape value (below /
above its median): the SOURCE family plays DrivAerML (all its runs known),
the TARGET family plays Member 2's car (only 65 training runs; the rest of the
family is the test set). Splits by Vehicle_Length, Vehicle_Height and
Backlight_Angle, 10 random repeats each.

Methods, all scored on the target's unseen runs:
  gp_direct          GP on the 65 target runs (the one to beat)
  transfer_only      source GP used as-is
  trend_correction   Cd = a + b * source(x) + GP residual, fitted on the 65 runs
  trend_as_input     GP on the 65 runs with source(x) as an extra input
  aug_<model>        XGBoost / RF / ANN on 65 real + 1,200 synthetic target
                     shapes labelled by trend_correction

Caveat: both families come from the same base car (DrivAer), so transfer is
easier here than from DrivAerML to Member 2's shell. A positive result is an
optimistic upper bound; a negative one means the idea does not work.

Usage: python scripts/run_drivaerml_transfer.py   (needs data/drivaerml_small/drivaerml_scalar_table.csv)
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
from sklearn.linear_model import LinearRegression

from component3 import config
from run_drivaerml_pod import PARAMS, SCALAR_TABLE, unit
from component3.workflow1.training import CANDIDATE_BUILDERS

SPLIT_BY = ["Vehicle_Length", "Vehicle_Height", "Backlight_Angle"]
N_TRAIN = 65
N_REPEATS = 10
N_SYNTHETIC = config.N_SYNTHETIC_SHAPES


def gp(n_inputs, restarts=3):
    kernel = ConstantKernel(1.0, (1e-2, 1e3)) * RBF(np.ones(n_inputs), (1e-2, 1e3)) + WhiteKernel(1e-2, (1e-6, 1e1))
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=restarts,
                                    random_state=config.RANDOM_STATE)


def main():
    t0 = time.time()
    table = pd.read_csv(SCALAR_TABLE)
    x_all = unit(table[PARAMS].to_numpy(dtype=float))
    y_all = table["Cd"].to_numpy(dtype=float)
    rows = []

    for split in SPLIT_BY:
        j = PARAMS.index(split)
        cut = np.median(x_all[:, j])
        for side in ("upper", "lower"):  # target = this side, source = the other
            tgt = np.where(x_all[:, j] > cut)[0] if side == "upper" else np.where(x_all[:, j] <= cut)[0]
            src = np.setdiff1d(np.arange(len(table)), tgt)
            source_model = gp(len(PARAMS), restarts=2).fit(x_all[src], y_all[src])
            lo_j, hi_j = x_all[tgt, j].min(), x_all[tgt, j].max()
            gap = y_all[tgt].mean() - y_all[src].mean()
            for rep in range(N_REPEATS):
                rng = np.random.default_rng(config.RANDOM_STATE + 1000 * SPLIT_BY.index(split) + rep)
                order = rng.permutation(tgt)
                tr, te = order[:N_TRAIN], order[N_TRAIN:]
                x_tr, y_tr, x_te, y_te = x_all[tr], y_all[tr], x_all[te], y_all[te]
                s_tr, s_te = source_model.predict(x_tr), source_model.predict(x_te)
                res = {}

                res["gp_direct"] = gp(len(PARAMS)).fit(x_tr, y_tr).predict(x_te)
                res["transfer_only"] = s_te

                lin = LinearRegression().fit(s_tr[:, None], y_tr)
                resid_gp = gp(len(PARAMS)).fit(x_tr, y_tr - lin.predict(s_tr[:, None]))
                label = lambda x: lin.predict(source_model.predict(x)[:, None]) + resid_gp.predict(x)
                res["trend_correction"] = label(x_te)

                res["trend_as_input"] = gp(len(PARAMS) + 1).fit(np.column_stack([x_tr, s_tr]), y_tr).predict(
                    np.column_stack([x_te, s_te]))

                x_syn = qmc.LatinHypercube(d=len(PARAMS), seed=config.RANDOM_STATE + 10 + rep).random(N_SYNTHETIC)
                x_syn[:, j] = lo_j + x_syn[:, j] * (hi_j - lo_j)  # keep synthetic shapes inside the target family
                y_syn = label(x_syn)
                xx, yy = np.vstack([x_tr, x_syn]), np.concatenate([y_tr, y_syn])
                for name, build in CANDIDATE_BUILDERS.items():
                    res[f"aug_{name}"] = build().fit(xx, yy).predict(x_te)
                    res[f"real_only_{name}"] = build().fit(x_tr, y_tr).predict(x_te)

                for method, pred in res.items():
                    rows.append({"split": split, "target_side": side, "repeat": rep, "method": method,
                                 "mae": float(np.mean(np.abs(pred - y_te))), "family_gap_counts": float(gap * 1000)})
            print(f"  split {split} / target {side}: done (Cd gap between families {gap * 1000:+.1f} counts)  [{time.time() - t0:.0f} s]")

    df = pd.DataFrame(rows)
    df["mae"] *= 1000
    order = ["gp_direct", "transfer_only", "trend_correction", "trend_as_input",
             "real_only_xgboost", "aug_xgboost", "real_only_random_forest", "aug_random_forest",
             "real_only_ann_mlp", "aug_ann_mlp"]
    by_split = df.groupby(["split", "target_side", "method"])["mae"].mean().unstack("method")[order]
    print("\nMean error on the target family's unseen runs, drag counts (10 repeats per row):")
    print(by_split.round(2).to_string())
    overall = df.groupby("method")["mae"].agg(["mean", "std"]).loc[order]
    print("\nAll splits together (mean, sd over the 60 runs of each method):")
    print(overall.round(2).to_string())

    piv = df.pivot_table(index=["split", "target_side", "repeat"], columns="method", values="mae")
    print("\nCompared with gp_direct (negative = better than gp_direct):")
    for m in ["trend_correction", "trend_as_input", "aug_xgboost", "aug_random_forest", "aug_ann_mlp"]:
        d = piv[m] - piv["gp_direct"]
        print(f"  {m:18s} {d.mean():+5.2f} ± {d.std():4.2f} counts, better in {int((d < 0).sum())}/{len(d)} cases")

    with open(os.path.join(config.OUTPUTS_DIR, "drivaerml_transfer_results.json"), "w") as fh:
        json.dump(df.to_dict("records"), fh)
    print(f"\nTotal time {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
