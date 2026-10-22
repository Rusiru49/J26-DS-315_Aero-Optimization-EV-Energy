"""
Per-label quality of the synthetic drag values (added 2026-10-22).

The synthetic shapes have no CFD of their own, so a single label cannot be
compared with CFD. Instead every label gets three things:

  1. Its own uncertainty. The helper (a Gaussian Process per pattern) gives an
     uncertainty for each predicted pattern amount; Cd is linear in the amounts,
     so that uncertainty is carried exactly to Cd. This raw value only covers the
     helper's part of the error, so it is CALIBRATED against held-out real CFD:
     sigma = sqrt((a * sigma_raw)^2 + b^2), with a and b fitted by maximum
     likelihood on the cross-validation residuals, and its coverage (how often
     the real Cd falls inside +/-1 and +/-2 sigma) is checked on folds the fit
     did not use.
  2. Physical plausibility checks on the label's own rebuilt pressure field:
       - the pressure coefficient nowhere exceeds the stagnation limit Cp = 1
         (incompressible flow; the real CFD peaks at about 0.98),
       - the pressure drag of that field, integrated over the predicted face
         geometry, is positive,
       - Cd stays within the real CFD range +/- 20 drag counts.
  3. The validated region: the shape is no farther from the real training
     designs than the real designs are from each other (the distances at which
     the labelling method was tested in cross-validation).

A label that fails any check is left out of the augmented dataset.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

CP_STAGNATION_LIMIT = 1.0
CD_RANGE_MARGIN = 0.020


def predict_labels(helper, pod, x_unit, method):
    """Pattern amounts, Cd label and raw Cd uncertainty for shapes scaled to 0-1."""
    amps, std = helper.predict(x_unit, return_std=True)
    amps = np.atleast_2d(amps)
    std = np.asarray(std).reshape(amps.shape)
    cd = pod.cd_batch(amps, x_unit, method)
    sigma = np.sqrt(np.sum((pod.cd_gradient(x_unit, method) * std) ** 2, axis=1))
    return amps, cd, sigma


def region_limit(x_unit_train):
    """Largest distance from a real training design to its nearest other training
    design: the distances at which held-out designs are predicted in validation."""
    d = np.linalg.norm(x_unit_train[:, None, :] - x_unit_train[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return float(d.min(axis=1).max())


def check_labels(pod, amps, x_unit, cd, train_x_unit, train_cd):
    """One row per synthetic shape: the check values and whether the label is kept."""
    dist = np.linalg.norm(x_unit[:, None, :] - train_x_unit[None, :, :], axis=2).min(axis=1)
    max_cp = pod.max_cp_batch(amps)
    cd_p = pod.cd_pressure_batch(amps, x_unit) if pod.geometry is not None else np.full(len(cd), np.nan)
    lo, hi = float(np.min(train_cd)) - CD_RANGE_MARGIN, float(np.max(train_cd)) + CD_RANGE_MARGIN
    t = pd.DataFrame({
        "distance_to_real": dist, "in_validated_region": dist <= region_limit(train_x_unit),
        "max_cp": max_cp, "cp_below_stagnation": max_cp <= CP_STAGNATION_LIMIT,
        "pressure_drag": cd_p, "pressure_drag_positive": ~(cd_p <= 0),
        "cd_in_range": (cd >= lo) & (cd <= hi),
    })
    t["keep"] = t[["in_validated_region", "cp_below_stagnation", "pressure_drag_positive", "cd_in_range"]].all(axis=1)
    return t


def fit_calibration(residuals, sigma_raw):
    """a, b of sigma = sqrt((a sigma_raw)^2 + b^2), by maximum likelihood."""
    r, s = np.asarray(residuals, float), np.asarray(sigma_raw, float)

    def nll(p):
        var = (np.exp(p[0]) * s) ** 2 + np.exp(p[1]) ** 2
        return float(np.sum(0.5 * np.log(var) + 0.5 * r ** 2 / var))

    start = [0.0, np.log(max(np.std(r), 1e-4))]
    p = minimize(nll, start, method="Nelder-Mead").x
    return float(np.exp(p[0])), float(np.exp(p[1]))


def calibrated_sigma(sigma_raw, ab):
    a, b = ab
    return np.sqrt((a * np.asarray(sigma_raw, float)) ** 2 + b ** 2)


def calibration_report(oof):
    """Calibrate on the cross-validation residuals of the labels (one speed) and
    check coverage honestly: within each repeat, each fold is scored with a, b
    fitted on that repeat's other folds (different designs). Returns a dict."""
    oof = oof.dropna(subset=["label_std"])
    inside1, inside2 = [], []
    for _, rep in oof.groupby("repeat"):
        for f, test in rep.groupby("fold"):
            train = rep[rep["fold"] != f]
            ab = fit_calibration(train["label"] - train["cd"], train["label_std"])
            z = np.abs(test["label"] - test["cd"]) / calibrated_sigma(test["label_std"], ab)
            inside1.extend(z <= 1)
            inside2.extend(z <= 2)
    ab = fit_calibration(oof["label"] - oof["cd"], oof["label_std"])
    sig = calibrated_sigma(oof["label_std"], ab)
    return {"a": ab[0], "b": ab[1], "coverage_1sigma": float(np.mean(inside1)), "coverage_2sigma": float(np.mean(inside2)),
            "median_sigma": float(np.median(sig))}
