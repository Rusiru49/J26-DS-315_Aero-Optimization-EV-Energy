"""
Labeller comparison on the 2D CFD stand-in (real OpenFOAM flow, Member 2's
format, fixed mesh, 5 shape parameters). 100 km/h, usable designs only,
20 random splits by design (about 56 train / 12 test).

Diagnostics (where is accuracy lost?), on the test designs:
  D1  true field + true geometry, exact face integral      (must be ~0: sanity)
  D2  true field projected on 15 POD patterns, true geometry (POD truncation)
  D3  true field projected, current ridge calibration       (calibration alone)
  D4  true field + PREDICTED geometry                        (geometry model alone)
Labellers (shape values only, as for a synthetic shape):
  L0  current: GP helper -> 15 amounts -> ridge(amounts, shape) -> Cd
  LX  exact physics: GP helper -> 15 amounts -> Cp field -> exact integral over the
      PREDICTED face geometry (+ friction drag from a GP on Cd_f)
  L1  GP fitted directly to Cd (no physics)
  LH  L1 + LX's correction: GP on Cd plus a ridge on the POD amounts fitted to L1's
      out-of-fold errors
Then XGBoost / RF / ANN trained on train designs + 1,200 LHS shapes labelled by
LX and by L1, scored on the test designs.

Usage: python scripts/run_cfd2d_labellers.py [data_dir]
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
import numpy as np
import pandas as pd
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from component3 import config
from component3.data.loader import CFDDataset
from component3.workflow1.pod import _snapshot_pod
from component3.workflow1.training import CANDIDATE_BUILDERS

DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(config.COMPONENT_ROOT, "data", "cfd2d_sedan")
SPEED = 100
N_REPEATS = 20
N_TEST = 12
N_MODES = config.N_POD_MODES
DRAG = np.array(config.FLOW_DIRECTION, dtype=float)
NAMES = config.SHAPE_PARAM_NAMES


def gp(d):
    k = ConstantKernel(1.0, (1e-2, 1e3)) * RBF(np.ones(d), (1e-2, 1e3)) + WhiteKernel(1e-2, (1e-6, 1e1))
    return GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=3, random_state=config.RANDOM_STATE)


def main():
    t0 = time.time()
    ds = CFDDataset(DATA, SPEED)
    runs = ds.runs.reset_index(drop=True)
    ids = runs["run_id"].tolist()
    # the box comes from the folder's parameter_ranges.csv, set when the dataset loads
    lo = np.array([config.SHAPE_PARAM_BOUNDS[n][0] for n in NAMES])
    hi = np.array([config.SHAPE_PARAM_BOUNDS[n][1] for n in NAMES])
    x = (runs[NAMES].to_numpy(float) - lo) / (hi - lo)
    cd = runs["Cd"].to_numpy(float)
    cdf = runs["Cd_friction"].to_numpy(float)
    aref = runs["frontal_area"].to_numpy(float)
    cp = np.stack([ds.get_field(r) for r in ids])                       # (n, faces)
    faces = [ds.get_faces(r) for r in ids]
    w = np.stack([(f["normal"] @ DRAG) * f["area"] for f in faces])     # drag projection of each face area
    exact = lambda cpf, wf, a: np.sum(-cpf * wf, axis=1) / a             # pressure part of Cd
    print(f"{len(ids)} usable designs at {SPEED} km/h, {cp.shape[1]} faces; Cd {cd.min():.3f}-{cd.max():.3f} "
          f"(sd {cd.std() * 1000:.0f} counts); friction part {cdf.mean() * 1000:.1f} counts on average")

    rows = []
    for rep in range(N_REPEATS):
        order = np.random.default_rng(config.RANDOM_STATE + 500 + rep).permutation(len(ids))
        te, tr = order[:N_TEST], order[N_TEST:]
        res = {}
        # POD on training Cp
        mean_f = cp[tr].mean(axis=0)
        modes, _ = _snapshot_pod(cp[tr] - mean_f, N_MODES)
        amps_tr = (cp[tr] - mean_f) @ modes.T
        amps_te = (cp[te] - mean_f) @ modes.T
        recon_te = mean_f + amps_te @ modes
        # geometry model: shape values -> drag-projected face areas (and A_ref)
        geo = make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(), Ridge(alpha=1e-3)).fit(x[tr], w[tr])
        aref_model = make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(), Ridge(alpha=1e-3)).fit(x[tr], aref[tr])
        w_pred = lambda xx: geo.predict(xx)
        a_pred = lambda xx: aref_model.predict(xx)
        # friction part from a GP
        gp_f = gp(5).fit(x[tr], cdf[tr])
        # diagnostics
        res["D1_true_field_true_geometry"] = exact(cp[te], w[te], aref[te]) + cdf[te]
        res["D2_pod15_true_geometry"] = exact(recon_te, w[te], aref[te]) + cdf[te]
        feats_tr = np.hstack([amps_tr, x[tr]])
        sc = StandardScaler().fit(feats_tr)
        ridge = RidgeCV(alphas=np.logspace(-4, 3, 40)).fit(sc.transform(feats_tr), cd[tr])
        res["D3_pod15_ridge_calibration"] = ridge.predict(sc.transform(np.hstack([amps_te, x[te]])))
        res["D4_true_field_predicted_geometry"] = exact(cp[te], w_pred(x[te]), a_pred(x[te])) + cdf[te]
        # labellers
        helper = gp(5).fit(x[tr], amps_tr)
        def lab_L0(xx):
            return ridge.predict(sc.transform(np.hstack([helper.predict(xx), xx])))
        def lab_LX(xx):
            field = mean_f + helper.predict(xx) @ modes
            return exact(field, w_pred(xx), a_pred(xx)) + gp_f.predict(xx)
        gp_cd = gp(5).fit(x[tr], cd[tr])
        lab_L1 = gp_cd.predict
        oof = np.zeros(len(tr))
        for a, b in KFold(5, shuffle=True, random_state=rep).split(tr):
            oof[b] = gp(5).fit(x[tr][a], cd[tr][a]).predict(x[tr][b])
        corr = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-4, 3, 40))).fit(amps_tr, cd[tr] - oof)
        def lab_LH(xx):
            return lab_L1(xx) + corr.predict(helper.predict(xx))
        labs = {"L0_current": lab_L0, "LX_exact_physics": lab_LX, "L1_gp_direct": lab_L1, "LH_gp_plus_pod": lab_LH}
        for k, f in labs.items():
            res[k] = f(x[te])
        # augmentation
        x_syn = qmc.LatinHypercube(d=5, seed=config.RANDOM_STATE + 10 + rep).random(config.N_SYNTHETIC_SHAPES)
        for lab_name in ("LX_exact_physics", "L1_gp_direct"):
            y_syn = labs[lab_name](x_syn)
            for mname, build in CANDIDATE_BUILDERS.items():
                m = build().fit(np.vstack([x[tr], x_syn]), np.concatenate([cd[tr], y_syn]))
                res[f"aug[{lab_name.split('_')[0]}]_{mname}"] = m.predict(x[te])
        for mname, build in CANDIDATE_BUILDERS.items():
            res[f"real_only_{mname}"] = build().fit(x[tr], cd[tr]).predict(x[te])
        for k, pred in res.items():
            e = np.abs(pred - cd[te])
            rows.append({"repeat": rep, "method": k, "mae": e.mean() * 1000, "max": e.max() * 1000})
        print(f"  repeat {rep + 1}/{N_REPEATS}  [{time.time() - t0:.0f} s]", flush=True)

    df = pd.DataFrame(rows)
    s = df.groupby("method").agg(mae=("mae", "mean"), sd=("mae", "std"), worst=("max", "mean"))
    order = [m for m in ["D1_true_field_true_geometry", "D2_pod15_true_geometry", "D3_pod15_ridge_calibration",
                         "D4_true_field_predicted_geometry", "L0_current", "LX_exact_physics", "L1_gp_direct",
                         "LH_gp_plus_pod"] if m in s.index]
    order += sorted(m for m in s.index if m.startswith("real_only")) + sorted(m for m in s.index if m.startswith("aug"))
    print(f"\nError on the test designs, drag counts (mean ± sd over {N_REPEATS} splits; worst = mean of each split's worst):")
    for m in order:
        print(f"  {m:42s} {s.loc[m, 'mae']:7.2f} ± {s.loc[m, 'sd']:5.2f}   worst {s.loc[m, 'worst']:7.1f}")
    piv = df.pivot(index="repeat", columns="method", values="mae")
    for m in ["LX_exact_physics", "LH_gp_plus_pod"]:
        d = piv[m] - piv["L1_gp_direct"]
        print(f"\n{m} minus L1_gp_direct: {d.mean():+.2f} ± {d.std():.2f} counts, better in {int((d < 0).sum())}/{len(d)} splits")
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    df.to_json(os.path.join(config.OUTPUTS_DIR, "cfd2d_labeller_results.json"), orient="records")
    print(f"\nTotal time {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
