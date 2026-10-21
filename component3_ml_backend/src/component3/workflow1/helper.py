"""
Internal helper model: 5 shape params -> N_POD_MODES mode amplitudes.

NOT deployed to the frontend — used only inside the augmentation step to
label generator-proposed shapes with a physics-consistent Cd (via
workflow1.pod.PODModel.reconstruct_cd). Trained on the real (non-holdout)
runs only, ~70-100 points with 5 inputs -> 15 outputs — small-sample,
low-dimensional, which is exactly the regime Gaussian Process regression
suits, and GP additionally gives a per-prediction uncertainty that can flag
a generated shape lying far from the training data (label less trustworthy
there). This is a different model-selection decision from the final
surrogate (workflow1/training.py), which trains on the much larger
augmented table where a plain GP would scale badly.
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from component3 import config
def _params_to_array(params):
    """The 5 shape values scaled to 0-1 by the agreed ranges. A GP needs inputs on
    comparable scales: in raw units (degrees vs millimetres) its length-scale fit
    went wrong on the 2D CFD stand-in (61 vs about 12 drag counts)."""
    lo = np.array([config.SHAPE_PARAM_BOUNDS[n][0] for n in config.SHAPE_PARAM_NAMES])
    hi = np.array([config.SHAPE_PARAM_BOUNDS[n][1] for n in config.SHAPE_PARAM_NAMES])
    raw = np.array([params[name] for name in config.SHAPE_PARAM_NAMES], dtype=np.float64)
    return (raw - lo) / (hi - lo)


class PerModeGP:
    """One Gaussian Process per POD pattern, each with its own kernel settings.

    A single scikit-learn GP fitted to all 15 outputs shares ONE set of kernel
    length-scales across every pattern. Tested on the 2D CFD stand-in (real
    OpenFOAM data), that shared fit predicted the pattern amounts so badly that
    the drag labels were off by about 60 drag counts; one GP per pattern brought
    it down to about 12. Same predict() interface as a multi-output GP."""

    def __init__(self, gps):
        self.gps = gps

    def predict(self, x, return_std=False):
        outs = [g.predict(x, return_std=return_std) for g in self.gps]
        if return_std:
            return np.column_stack([o[0] for o in outs]), np.column_stack([o[1] for o in outs])
        return np.column_stack(outs)


def _new_gp():
    kernel = ConstantKernel(1.0, (1e-2, 1e3)) * RBF(
        length_scale=np.ones(len(config.SHAPE_PARAM_NAMES)), length_scale_bounds=(1e-2, 1e3)
    ) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e1))
    return GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, n_restarts_optimizer=3, random_state=config.RANDOM_STATE
    )


def fit_helper(x_unit, amplitudes, kind="per_mode"):
    """kind "per_mode": one GP per pattern (own kernel settings each);
    kind "shared": one multi-output GP, one kernel setting for all patterns."""
    if kind == "shared":
        return _new_gp().fit(x_unit, amplitudes)
    return PerModeGP([_new_gp().fit(x_unit, amplitudes[:, k]) for k in range(amplitudes.shape[1])])


def train_helper_gp(cfd_dataset, pod_model, kind=None):
    """
    Helper on the real training runs: the 5 shape params -> the pattern
    amounts. kind: "per_mode" or "shared" (default config.HELPER_KIND; "auto"
    falls back to "per_mode" here, select_labeller() makes the real choice).
    """
    kind = kind or config.HELPER_KIND
    kind = "per_mode" if kind == "auto" else kind
    real_runs = cfd_dataset.real_runs()
    x = np.stack([_params_to_array(row) for _, row in real_runs.iterrows()])
    y = np.stack([pod_model.project(cfd_dataset.get_field(rid)) for rid in real_runs["run_id"]])
    return fit_helper(x, y, kind)


def select_labeller(cfd_dataset, n_folds=5):
    """
    Choose the helper kind, the amounts -> Cd method AND the number of pressure
    patterns by k-fold cross-validation on the TRAINING designs only (the
    hold-out designs are never used), so each dataset - the dummy, the 2D CFD
    stand-in or Member 2's real data - picks what works best for it. Returns
    ((kind, method, n_modes), table) where table maps that triple -> mean
    absolute Cd error.

    Keeping only the patterns the helper can actually predict from the shape
    matters: on Member 2's data patterns beyond the first two were mostly noise
    that pulled extreme shapes toward the average (experiments/helper_study).
    """
    from component3.workflow1.pod import dataset_arrays
    return select_labeller_arrays(dataset_arrays(cfd_dataset, cfd_dataset.real_runs()), n_folds)


def select_labeller_arrays(a, n_folds=5, mode_grid=None):
    """select_labeller() on arrays already loaded (workflow1.pod.dataset_arrays).
    The per-pattern GPs do not depend on how many patterns are kept, so they are
    fitted once per fold and sliced; the POD ridge and the shared GP are refitted
    for each pattern count."""
    from sklearn.model_selection import KFold
    from component3.workflow1.pod import fit_pod_arrays
    grid = sorted(m for m in (mode_grid or config.POD_MODE_GRID) if m <= config.N_POD_MODES)
    n = len(a["cd"])
    n_folds = min(n_folds, n)
    errors = {}
    for tr, te in KFold(n_folds, shuffle=True, random_state=config.RANDOM_STATE).split(np.arange(n)):
        sub = {k: v[tr] for k, v in a.items()}
        x_te = a["x_unit"][te]
        pods = {m: fit_pod_arrays(sub, m) for m in grid}
        full = pods[grid[-1]]
        amps = (sub["fields"] - full.mean_field) @ full.mode_shapes.T
        per_mode = fit_helper(sub["x_unit"], amps, "per_mode").predict(x_te)
        for m, pod in pods.items():
            k = pod.mode_shapes.shape[0]
            preds = {"per_mode": per_mode[:, :k], "shared": fit_helper(sub["x_unit"], amps[:, :k], "shared").predict(x_te)}
            for kind, pred in preds.items():
                for method in ("exact", "ridge"):
                    err = np.abs(pod.cd_batch(pred, x_te, method) - a["cd"][te])
                    errors.setdefault((kind, method, m), []).append(err)
    table = {k: float(np.concatenate(v).mean()) for k, v in errors.items()}
    return min(table, key=table.get), table


def predict_mode_amplitudes(gp, params, return_std=False):
    """
    Predict the mode amplitudes for a new shape. If return_std, also return
    the per-mode standard deviation — a large value flags a shape far from
    the training data, where the label is less trustworthy (see module
    docstring).
    """
    x = _params_to_array(params).reshape(1, -1)
    if return_std:
        mean, std = gp.predict(x, return_std=True)
        return mean[0], std[0]
    return gp.predict(x)[0]


def label_shape(gp, pod_model, params):
    """Convenience: shape params -> physics-consistent Cd, end to end."""
    amplitudes = predict_mode_amplitudes(gp, params)
    return pod_model.reconstruct_cd(amplitudes, params)


def label_shapes(gp, pod_model, shapes, method=None):
    """Many shapes at once: list of shape-value dicts -> array of Cd labels."""
    from component3.workflow1.pod import shape_features
    raw = np.stack([_params_to_array(p) for p in shapes])
    unit = np.stack([shape_features(p) for p in shapes])
    return pod_model.cd_batch(gp.predict(raw), unit, method)
