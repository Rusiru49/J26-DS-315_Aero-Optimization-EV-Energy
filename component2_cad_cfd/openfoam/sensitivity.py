#!/usr/bin/env python3
"""Sensitivity analysis of Cd to the 5 shape parameters (Component 2 data-science layer).

Three views, simple -> advanced:
  1. SRC      standardized regression coefficients (linear effect, with sign)
  2. PermImp  permutation importance of a Gaussian-process surrogate (cross-validated)
  3. Shapley  Shapley effects (variance-based, handles interactions fairly) - the method of the TAF reference [11]

Usage:
  python3 sensitivity.py --demo                       # test on synthetic data (no CFD needed)
  python3 sensitivity.py --table run_table.csv --speed 100 --out sens_out
Needs: numpy pandas scikit-learn scipy matplotlib
"""
import argparse, itertools, math, os, sys
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

PARAMS = ["windshield_angle_deg", "hood_angle_deg", "spoiler_angle_deg", "boat_tail_angle_deg", "ground_clearance_mm"]
SHORT = ["windshield", "hood", "spoiler", "boat-tail", "ground clr"]


def demo_data(n=70, seed=0):
    """Synthetic LHS data with a made-up Cd law (only to test the script)."""
    from scipy.stats import qmc
    lo = np.array([22, 5, 0, 5, 110.]); hi = np.array([34, 15, 20, 25, 190.])
    X = qmc.scale(qmc.LatinHypercube(d=5, seed=seed).random(n), lo, hi)
    z = (X - lo) / (hi - lo)
    cd = 0.34 + 0.05 * z[:, 3] - 0.03 * z[:, 2] + 0.025 * z[:, 4] + 0.015 * z[:, 0] + 0.02 * z[:, 3] * z[:, 4] \
         + 0.002 * z[:, 1] + np.random.default_rng(seed).normal(0, 0.003, n)
    df = pd.DataFrame(X, columns=PARAMS); df["Cd_mean"] = cd; df["design_id"] = ["D%03d" % i for i in range(n)]
    df["speed_kmh"] = 100; df["converged"] = True
    return df


def fit_surrogate(X, y):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.compose import TransformedTargetRegressor
    k = C(1.0) * Matern(length_scale=np.ones(X.shape[1]), nu=2.5) + WhiteKernel(1e-3)
    gp = GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=5, random_state=0)
    return make_pipeline(StandardScaler(), gp)


def shapley_effects(model, lo, hi, d=5, N=20000, seed=1):
    """Exact Shapley effects (all 2^d subsets) of independent-uniform inputs, using a pick-freeze estimate of
    v(S) = Var(E[Y|X_S]) on the surrogate. Returns normalised effects that sum to 1."""
    rng = np.random.default_rng(seed)
    A = lo + (hi - lo) * rng.random((N, d)); B = lo + (hi - lo) * rng.random((N, d))
    yA = model.predict(A); mu = yA.mean(); var = yA.var()
    v = {(): 0.0}
    for r in range(1, d + 1):
        for S in itertools.combinations(range(d), r):
            if r == d: v[S] = var; continue
            M = B.copy(); M[:, list(S)] = A[:, list(S)]       # keep X_S from A, resample the rest
            v[S] = max(0.0, float(np.mean(yA * model.predict(M)) - mu ** 2))
    phi = np.zeros(d)
    for i in range(d):
        others = [j for j in range(d) if j != i]
        for r in range(0, d):
            w = math.factorial(r) * math.factorial(d - r - 1) / math.factorial(d)
            for S in itertools.combinations(others, r):
                S2 = tuple(sorted(S + (i,))); phi[i] += w * (v[S2] - v[tuple(sorted(S))])
    return phi / phi.sum(), var


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table"); ap.add_argument("--speed", type=int, default=100)
    ap.add_argument("--target", default="Cd_mean"); ap.add_argument("--out", default="sens_out")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    if a.demo: df = demo_data()
    else:
        df = pd.read_csv(a.table); df = df[df.speed_kmh == a.speed]
        if "converged" in df: df = df[df.converged.astype(str).str.lower().isin(["true", "1"])]
    X = df[PARAMS].to_numpy(float); y = df[a.target].to_numpy(float)
    print("designs used: %d   target: %s   range %.4f - %.4f" % (len(df), a.target, y.min(), y.max()))

    # 1) SRC: linear effect in standard-deviation units
    from sklearn.linear_model import LinearRegression
    Z = (X - X.mean(0)) / X.std(0); zy = (y - y.mean()) / y.std()
    src = LinearRegression().fit(Z, zy).coef_
    # 2) surrogate quality + permutation importance (on held-out folds)
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.metrics import r2_score
    from sklearn.inspection import permutation_importance
    gp = fit_surrogate(X, y)
    pred = cross_val_predict(gp, X, y, cv=KFold(5, shuffle=True, random_state=0))
    r2cv = r2_score(y, pred); print("surrogate 5-fold CV R2 = %.3f" % r2cv)
    if r2cv < 0.7: print("WARNING: surrogate is weak (R2<0.7) -> sensitivity numbers are unreliable.")
    pi = np.zeros(5); reps = 0
    for tr, te in KFold(5, shuffle=True, random_state=1).split(X):
        m = fit_surrogate(X[tr], y[tr]).fit(X[tr], y[tr])
        pi += permutation_importance(m, X[te], y[te], n_repeats=20, random_state=0).importances_mean; reps += 1
    pi = np.clip(pi / reps, 0, None); pi_n = pi / pi.sum() if pi.sum() > 0 else pi
    # 3) Shapley effects on the surrogate
    gp.fit(X, y)
    sh, var = shapley_effects(gp, X.min(0), X.max(0))
    res = pd.DataFrame({"parameter": PARAMS, "SRC": src, "permutation_importance_norm": pi_n, "shapley_effect": sh})
    res = res.sort_values("shapley_effect", ascending=False)
    res.to_csv(os.path.join(a.out, "sensitivity.csv"), index=False)
    print(res.round(3).to_string(index=False))
    # figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6)); order = res.index
    names = [SHORT[PARAMS.index(p)] for p in res.parameter]
    ax[0].barh(names, res.SRC, color=["#c0392b" if v > 0 else "#2471a3" for v in res.SRC]); ax[0].set_title("SRC (sign = direction)")
    ax[1].barh(names, res.permutation_importance_norm, color="#555"); ax[1].set_title("Permutation importance")
    ax[2].barh(names, res.shapley_effect, color="#1e8449"); ax[2].set_title("Shapley effects (sum = 1)")
    for x in ax: x.invert_yaxis()
    fig.suptitle("Cd sensitivity to shape parameters  (GP surrogate CV R2 = %.2f, n = %d)" % (r2cv, len(df)), fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "sensitivity.png"), dpi=160)
    print("wrote", os.path.join(a.out, "sensitivity.csv"), "and sensitivity.png")


if __name__ == "__main__":
    main()
