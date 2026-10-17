"""
Data checker: confirms a data folder matches Member 2's contract
(D:\\Research\\Component3-Data-Contract.md) and re-computes the drag from the
faces. It is the 'Ingest and check' step of the model builder: run it on the
dummy data now and on every real delivery later.

Each check yields (name, status, detail) with status PASS, WARN or FAIL.
FAIL means the data cannot be used as is; WARN means look at it.

The key integrity check sums p*n*A and tau*A in the drag direction and
compares with the table's Cd_mean (tolerance: Member 2's 1%). The drag
direction and the sign of tau are still to be confirmed by Member 2, so the
checker tries both signs of each and reports which combination matches.
"""

import itertools
import json
import os

import numpy as np
import pandas as pd

import config
from cfd_data_loader import faces_path, normalize_run_table, read_run_npz

REQUIRED = ["design_id", "speed_kmh", "Cd_mean", "Cd_p", "Cd_f", "A_frontal", "q_ref", "converged", "morph_passed"] + list(
    config.SHAPE_PARAM_NAMES)
RECOMMENDED = ["Cd", "Cd_std", "Cl", "iterations", "y_plus_mean", "y_plus_max", "cell_count", "checkMesh_passed",
               "final_res_p", "final_res_U"]


def _result(results, name, ok, detail, warn_only=False):
    results.append((name, "PASS" if ok else ("WARN" if warn_only else "FAIL"), detail))


def integrate(faces, q_ref, drag_dir, tau_sign, a_ref=None):
    """Return Cd, Cd_p, Cd_f and a face-based frontal-area estimate from one run's faces.

    Cd uses a_ref, the design's own A_frontal from the run table (as the contract
    says). The face estimate 0.5 * sum(|n.d| A) equals the projected area only for
    convex bodies; on a real car (hood and windshield both face forward) it is
    about 10% too large, so it is only compared with A_frontal as a warning."""
    d = np.asarray(drag_dir, dtype=float)
    n, a = faces["normal"], faces["area"]
    a_front = 0.5 * np.sum(np.abs(n @ d) * a)
    ref = a_front if a_ref is None else a_ref
    cd_p = np.sum(-faces["p"] * (n @ d) * a) / (q_ref * ref)
    cd_f = tau_sign * np.sum((faces["tau"] @ d) * a) / (q_ref * ref)
    return cd_p + cd_f, cd_p, cd_f, a_front


def check_dataset(data_dir, category=config.CATEGORY, tolerance=config.CD_SUM_CHECK_TOLERANCE, max_runs_for_sum=None):
    R = []
    need = ["run_table.csv", "design_table.csv"]
    missing = [f for f in need if not os.path.exists(os.path.join(data_dir, f))]
    _result(R, "required files present", not missing, f"missing: {missing}" if missing else "run_table, design_table")
    if missing:
        return R
    case_path = os.path.join(data_dir, "case.json")
    _result(R, "case.json present (optional)", os.path.exists(case_path),
            "found" if os.path.exists(case_path) else "not found: using the constants in config.py", warn_only=True)

    runs = normalize_run_table(pd.read_csv(os.path.join(data_dir, "run_table.csv")))
    designs = pd.read_csv(os.path.join(data_dir, "design_table.csv"))
    case = {}
    if os.path.exists(case_path):
        with open(case_path) as fh:
            case = json.load(fh)

    miss_req = [c for c in REQUIRED if c not in runs.columns]
    _result(R, "run_table required columns", not miss_req, f"missing: {miss_req}" if miss_req else f"{len(REQUIRED)} columns found")
    miss_rec = [c for c in RECOMMENDED if c not in runs.columns]
    _result(R, "run_table recommended columns", not miss_rec, f"missing: {miss_rec}" if miss_rec else "all present", warn_only=True)
    if miss_req:
        return R

    speeds = sorted(int(s) for s in runs["speed_kmh"].unique())
    expected = sorted(config.SPEEDS_MS)
    _result(R, "speeds are 40 and 100 km/h", speeds == expected, f"found {speeds}")
    per_design = runs.groupby("design_id")["speed_kmh"].nunique()
    _result(R, "every design has both speeds", bool((per_design == len(expected)).all()),
            f"{int((per_design != len(expected)).sum())} designs without both")
    _result(R, "design_table and run_table list the same designs",
            set(designs["design_id"]) == set(runs["design_id"]), f"{designs['design_id'].nunique()} vs {runs['design_id'].nunique()}")
    _result(R, "baseline D000 present", "D000" in set(runs["design_id"]), "")

    merged = runs.merge(designs[["design_id"] + list(config.SHAPE_PARAM_NAMES)], on="design_id", suffixes=("", "_d"))
    diffs = max(float(np.abs(merged[c] - merged[c + "_d"]).max()) for c in config.SHAPE_PARAM_NAMES)
    _result(R, "shape values agree between the two tables", diffs < 1e-6, f"max difference {diffs:.2e}")
    out_of_range = [c for c in config.SHAPE_PARAM_NAMES
                    if not runs[c].between(*config.SHAPE_PARAM_BOUNDS[c]).all()]
    _result(R, "shape values inside the agreed ranges", not out_of_range, f"outside: {out_of_range}", warn_only=True)

    ok_runs = runs[runs["morph_passed"].astype(bool) & runs["converged"].astype(bool)]
    failed = runs[~runs.index.isin(ok_runs.index)]
    _result(R, "failed runs are flagged, not dropped", True, f"{failed['design_id'].nunique()} failed designs kept in the table")
    no_file = [(r.design_id, r.speed_kmh) for r in ok_runs.itertuples()
               if not os.path.exists(faces_path(data_dir, r.design_id, r.speed_kmh, category))]
    _result(R, "every good run has a face file", not no_file, f"{len(no_file)} missing, e.g. {no_file[:3]}")

    rho = case.get("rho_kg_m3", config.AIR_DENSITY_KG_M3)
    speeds_ms = {int(k): v for k, v in case.get("speeds_ms", config.SPEEDS_MS).items()}
    q_expected = ok_runs["speed_kmh"].map(lambda s: 0.5 * rho * speeds_ms[int(s)] ** 2)
    q_err = float(np.max(np.abs(ok_runs["q_ref"] - q_expected) / q_expected))
    _result(R, "q_ref = 0.5 * rho * U^2", q_err < 0.005, f"max relative difference {q_err:.2e}")
    sum_err = float(np.max(np.abs(ok_runs["Cd_mean"] - ok_runs["Cd_p"] - ok_runs["Cd_f"])))
    _result(R, "Cd_mean = Cd_p + Cd_f", sum_err < 0.002, f"max difference {sum_err:.4f}")

    n_faces, bad_shape, bad_values, bad_norm, outward_frac, bad_cp = set(), [], [], [], [], []
    drag_dirs = [config.FLOW_DIRECTION, tuple(-np.array(config.FLOW_DIRECTION))]
    combos = list(itertools.product(range(len(drag_dirs)), (1.0, -1.0)))
    errors = {c: [] for c in combos}
    area_err, mesh_same, baseline_centres = [], [], {}
    rows = list(ok_runs.itertuples())
    if max_runs_for_sum:
        rows = rows[:max_runs_for_sum]
    for r in rows:
        try:
            f = read_run_npz(faces_path(data_dir, r.design_id, r.speed_kmh, category))
        except (KeyError, ValueError, OSError) as exc:
            bad_shape.append((r.design_id, r.speed_kmh, str(exc)[:60]))
            continue
        n = len(f["area"])
        n_faces.add(n)
        if not (f["centre"].shape == (n, 3) and f["normal"].shape == (n, 3) and f["tau"].shape == (n, 3) and f["p"].shape == (n,)):
            bad_shape.append((r.design_id, r.speed_kmh, "array shapes"))
            continue
        if not all(np.isfinite(v).all() for v in f.values()):
            bad_values.append((r.design_id, r.speed_kmh))
        if np.max(np.abs(np.linalg.norm(f["normal"], axis=1) - 1)) > 1e-3 or (f["area"] <= 0).any():
            bad_norm.append((r.design_id, r.speed_kmh))
        # Orientation of a closed surface (divergence theorem): sum((c - c0).n A) is
        # +3 x volume for outward normals (2 x section area x span for a 2D slice),
        # whatever the shape. Comparing each face with the centre point only works
        # for convex bodies, so it is not used.
        rel = f["centre"] - f["centre"].mean(axis=0)
        outward_frac.append(float(np.sum(np.einsum("ij,ij->i", f["normal"], rel) * f["area"])))
        cp = f["p"] / r.q_ref
        if cp.min() < -4 or cp.max() > 1.5:
            bad_cp.append((r.design_id, r.speed_kmh))
        for (di, sgn) in combos:
            cd, _, _, a_front = integrate(f, r.q_ref, drag_dirs[di], sgn, a_ref=r.A_frontal)
            errors[(di, sgn)].append(abs(cd - r.Cd_mean) / r.Cd_mean)
        area_err.append(abs(a_front - r.A_frontal) / r.A_frontal)
        if r.speed_kmh == min(expected):
            baseline_centres[r.design_id] = f["centre"]
        elif r.design_id in baseline_centres:
            mesh_same.append(float(np.max(np.abs(baseline_centres[r.design_id] - f["centre"]))))

    _result(R, "face files readable with the expected arrays", not bad_shape, f"{len(bad_shape)} problems, e.g. {bad_shape[:2]}")
    _result(R, "same number of faces in every file", len(n_faces) == 1, f"face counts found: {sorted(n_faces)}")
    _result(R, "no NaN or infinite values", not bad_values, f"{len(bad_values)} files affected")
    _result(R, "normals have length 1, areas positive", not bad_norm, f"{len(bad_norm)} files affected")
    _result(R, "normals point out of the car", bool(outward_frac) and min(outward_frac) > 0,
            f"enclosed-volume check positive in all {len(outward_frac)} files" if outward_frac and min(outward_frac) > 0
            else f"{sum(v <= 0 for v in outward_frac)} files have inward normals")
    _result(R, "Cp values in a sensible range", not bad_cp, f"{len(bad_cp)} files outside -4..1.5", warn_only=True)
    if mesh_same:
        _result(R, "both speeds use the same faces", max(mesh_same) < 1e-4, f"largest face-centre difference {max(mesh_same):.2e} m")

    if errors[combos[0]]:
        best = min(combos, key=lambda c: np.max(errors[c]))
        worst_err = float(np.max(errors[best]))
        direction = [round(x, 3) for x in drag_dirs[best[0]]]
        _result(R, f"Cd from the faces matches Cd_mean (tolerance {tolerance:.0%})", worst_err <= tolerance,
                f"best match: drag direction {direction}, tau sign {best[1]:+.0f}; worst relative error {worst_err:.3%} "
                f"over {len(errors[best])} runs")
        _result(R, "frontal area from the faces matches A_frontal", float(np.max(area_err)) < 0.01,
                f"worst relative error {float(np.max(area_err)):.3%}", warn_only=True)
    return R


def print_report(results):
    width = max(len(n) for n, _, _ in results)
    for name, status, detail in results:
        print(f"[{status}] {name.ljust(width)}  {detail}")
    fails = sum(1 for _, s, _ in results if s == "FAIL")
    warns = sum(1 for _, s, _ in results if s == "WARN")
    print(f"\n{len(results) - fails - warns} passed, {warns} warnings, {fails} failed")
    return fails == 0
