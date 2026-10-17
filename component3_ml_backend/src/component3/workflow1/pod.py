"""
Proper Orthogonal Decomposition (POD) on the surface pressure coefficient
(Cp = p / q_ref) of the training designs, plus the step that turns pattern
amounts into a drag coefficient.

Uses SNAPSHOT POD (method of snapshots, Sirovich 1987): with ~60-70 designs
and hundreds of thousands of faces, eigendecomposing the small
designs x designs Gram matrix and lifting the eigenvectors back to spatial
modes is far cheaper than forming a faces x faces covariance matrix.

Pattern amounts -> Cd: two methods (config.LABEL_METHOD picks the default).

  "exact"  (default, the proposal's own design) - drag is CALCULATED, as CFD
           does: rebuild the pressure field from the pattern amounts and add up
           -Cp x (n . d) x A over every face, divided by the frontal area, plus
           the small friction part. A new synthetic shape has no geometry of its
           own, so the face geometry (n . d x A for each face) and the frontal
           area are predicted from the 5 shape values by a smooth quadratic
           ridge fit. This is possible because Member 2's fixed mesh keeps the
           same faces in every design. The friction part (a few % of Cd) comes
           from a small Gaussian Process on the 5 shape values. The pattern sums
           are precomputed, so a label costs microseconds and never builds the
           full field (the "per-pattern drag-integration weights" of the
           proposal, here allowed to vary with the shape).

  "ridge"  (kept for comparison) - a statistical map from the pattern amounts
           plus the shape values to Cd (ridge regression). Not physics; on the
           2D CFD stand-in it was far less accurate (about 67 vs 12 drag
           counts), so it is only an option. Workflow 1 prints both, so the real
           data can decide.

This module never imports anything from data/dummy_generator.py: it only
sees what cfd_data_loader.CFDDataset hands it, as it would with real data.
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from component3 import config
def shape_features(params):
    """The 5 shape values scaled to 0-1 by the agreed ranges (dict or table row)."""
    lo = np.array([config.SHAPE_PARAM_BOUNDS[n][0] for n in config.SHAPE_PARAM_NAMES])
    hi = np.array([config.SHAPE_PARAM_BOUNDS[n][1] for n in config.SHAPE_PARAM_NAMES])
    raw = np.array([params[n] for n in config.SHAPE_PARAM_NAMES], dtype=np.float64)
    return (raw - lo) / (hi - lo)


def drag_projected_areas(faces, drag_dir=config.FLOW_DIRECTION):
    """(n . d) x A for every face: the face's share of the drag integral."""
    return (faces["normal"] @ np.asarray(drag_dir, dtype=float)) * faces["area"]


class GeometryModel:
    """Quadratic ridge fit: normalised shape values -> one value per face (or a
    single value). Exposes its linear form w = feats @ W + b so the drag sums can
    be precomputed."""

    def __init__(self, x, y, alpha=1e-3):
        self.poly = PolynomialFeatures(2, include_bias=False)
        self.scaler = StandardScaler().fit(self.poly.fit_transform(x))
        self.ridge = Ridge(alpha=alpha).fit(self.features(x), y)
        self.W = np.atleast_2d(self.ridge.coef_).T          # (n_features, n_outputs)
        self.b = np.atleast_1d(self.ridge.intercept_)        # (n_outputs,)

    def features(self, x):
        return self.scaler.transform(self.poly.transform(np.atleast_2d(x)))

    def predict(self, x):
        return self.features(x) @ self.W + self.b


class PODModel:
    """
    mean_field: (n_faces,) subtracted before projecting.
    mode_shapes: (n_modes, n_faces) orthonormal spatial modes.
    scaler, ridge: the statistical amounts + shape -> Cd map ("ridge" method).
    geometry, frontal_area, friction: the parts of the "exact" method.
    """

    def __init__(self, mean_field, mode_shapes, scaler, ridge, geometry=None, frontal_area=None, friction=None):
        self.mean_field = mean_field
        self.mode_shapes = mode_shapes
        self.scaler = scaler
        self.ridge = ridge
        self.geometry = geometry
        self.frontal_area = frontal_area
        self.friction = friction
        if geometry is not None:
            # precomputed pattern sums of the drag integral (see module docstring)
            self._mode_w = mode_shapes @ geometry.W.T        # (n_modes, n_features)
            self._mode_b = mode_shapes @ geometry.b          # (n_modes,)
            self._mean_w = geometry.W @ mean_field           # (n_features,)
            self._mean_b = float(geometry.b @ mean_field)

    def project(self, field):
        """Field -> mode amplitudes (mean-centered projection)."""
        centered = field.astype(np.float64) - self.mean_field
        return self.mode_shapes @ centered

    def reconstruct_field(self, amplitudes):
        """Mode amplitudes -> approximate full field (mean added back)."""
        return self.mean_field + amplitudes @ self.mode_shapes

    def cd_batch(self, amplitudes, x_unit, method=None):
        """Cd for many shapes at once. amplitudes (m, n_modes); x_unit (m, 5) shape
        values already scaled to 0-1 (shape_features)."""
        method = method if method not in (None, "auto") else (
            config.LABEL_METHOD if config.LABEL_METHOD != "auto" else "exact")
        amplitudes, x_unit = np.atleast_2d(amplitudes), np.atleast_2d(x_unit)
        if method == "ridge" or self.geometry is None:
            return self.ridge.predict(self.scaler.transform(np.hstack([amplitudes, x_unit])))
        feats = self.geometry.features(x_unit)
        # sum over faces of Cp x (n . d) x A, without building the fields
        mean_part = feats @ self._mean_w + self._mean_b
        mode_part = np.sum(amplitudes * (feats @ self._mode_w.T + self._mode_b), axis=1)
        cd_pressure = -(mean_part + mode_part) / self.frontal_area.predict(x_unit).ravel()
        return cd_pressure + self.friction.predict(x_unit)

    def reconstruct_cd(self, amplitudes, params, method=None):
        """Pattern amounts + the shape's 5 values -> Cd (no CFD needed)."""
        return float(self.cd_batch(amplitudes, shape_features(params), method)[0])


def _snapshot_pod(snapshot_matrix, n_modes):
    """
    snapshot_matrix: (n_runs, n_points), already mean-centered.
    Returns (mode_shapes, singular_values) with mode_shapes: (n_modes, n_points),
    orthonormal rows, ordered by descending explained variance.
    """
    n_runs = snapshot_matrix.shape[0]
    n_modes = min(n_modes, n_runs)

    # Gram matrix is n_runs x n_runs — cheap regardless of n_points.
    gram = snapshot_matrix @ snapshot_matrix.T
    eigvals, eigvecs = np.linalg.eigh(gram)  # ascending order
    order = np.argsort(eigvals)[::-1][:n_modes]
    eigvals = np.clip(eigvals[order], a_min=1e-12, a_max=None)
    eigvecs = eigvecs[:, order]

    # Lift snapshot-space eigenvectors back to spatial modes, normalised.
    mode_shapes = eigvecs.T @ snapshot_matrix  # (n_modes, n_points)
    norms = np.linalg.norm(mode_shapes, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mode_shapes = mode_shapes / norms

    singular_values = np.sqrt(eigvals)
    return mode_shapes, singular_values


def _small_gp():
    kernel = ConstantKernel(1.0, (1e-2, 1e3)) * RBF(np.ones(len(config.SHAPE_PARAM_NAMES)), (1e-2, 1e3)) \
        + WhiteKernel(1e-2, (1e-6, 1e1))
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3,
                                    random_state=config.RANDOM_STATE)


def dataset_arrays(cfd_dataset, runs):
    """Everything the labeller learns from, as arrays, for the given run rows."""
    ids = runs["run_id"].tolist()
    # one face file at a time: only the drag-projected areas are kept (a full face
    # file at Member 2's size is ~25 MB in memory, so keeping all of them is not ok)
    w = np.stack([drag_projected_areas(cfd_dataset.get_faces(r)) for r in ids])
    return {
        "fields": np.stack([cfd_dataset.get_field(r).astype(np.float64) for r in ids]),
        "x_unit": np.stack([shape_features(row) for _, row in runs.iterrows()]),
        "x_raw": runs[config.SHAPE_PARAM_NAMES].to_numpy(dtype=np.float64),
        "cd": runs["Cd"].to_numpy(dtype=np.float64),
        "w": w,
        "aref": runs["frontal_area"].to_numpy(dtype=np.float64),
        "cdf": runs["Cd_friction"].to_numpy(dtype=np.float64),
    }


def fit_pod_arrays(a, n_modes=config.N_POD_MODES):
    """Fit the POD modes and both amounts -> Cd methods on the arrays of training runs."""
    mean_field = a["fields"].mean(axis=0)
    centered = a["fields"] - mean_field
    mode_shapes, _ = _snapshot_pod(centered, n_modes)
    amplitudes = centered @ mode_shapes.T

    # "ridge": statistical map amounts + shape values -> Cd
    features = np.hstack([amplitudes, a["x_unit"]])
    scaler = StandardScaler().fit(features)
    ridge = RidgeCV(alphas=np.logspace(-4, 3, 40)).fit(scaler.transform(features), a["cd"])

    # "exact": face geometry, frontal area and friction drag as smooth functions of shape
    geometry = GeometryModel(a["x_unit"], a["w"])
    frontal = GeometryModel(a["x_unit"], a["aref"])
    friction = _small_gp().fit(a["x_unit"], a["cdf"])
    return PODModel(mean_field, mode_shapes, scaler, ridge, geometry, frontal, friction)


def fit_pod(cfd_dataset, n_modes=config.N_POD_MODES):
    """
    Fit the POD modes and both amounts -> Cd methods on the REAL (non-holdout)
    runs only. Returns a PODModel.
    """
    return fit_pod_arrays(dataset_arrays(cfd_dataset, cfd_dataset.real_runs()), n_modes)


def exact_cd_from_faces(cfd_dataset, run_id):
    """Cd of a real run recomputed from its own true field and geometry (pressure
    integral + the run's friction drag). Should equal the recorded Cd: a check
    that the exact method uses the same conventions as the data."""
    row = cfd_dataset.runs[cfd_dataset.runs["run_id"] == run_id].iloc[0]
    w = drag_projected_areas(cfd_dataset.get_faces(run_id))
    return float(-np.sum(cfd_dataset.get_field(run_id) * w) / row["frontal_area"] + row["Cd_friction"])


def explained_variance_ratio(cfd_dataset, pod_model):
    """How much of the real runs' field variance the kept modes explain —
    useful diagnostic for choosing N_POD_MODES, not required by downstream
    modules."""
    real_runs = cfd_dataset.real_runs()
    fields = np.stack(
        [cfd_dataset.get_field(rid).astype(np.float64) for rid in real_runs["run_id"]]
    )
    centered = fields - pod_model.mean_field
    total_var = np.sum(centered ** 2)
    amplitudes = centered @ pod_model.mode_shapes.T
    reconstructed = amplitudes @ pod_model.mode_shapes
    residual_var = np.sum((centered - reconstructed) ** 2)
    return 1.0 - residual_var / total_var if total_var > 0 else float("nan")


def validate_on_holdout(cfd_dataset, pod_model, method=None):
    """
    Run held-out real CFD runs through the fitted POD model: project their
    TRUE field onto the patterns and turn the amounts into Cd. Tests the
    patterns + the amounts -> Cd step on their own (the full label check, which
    also predicts the amounts from the shape, is in scripts/run_workflow1.py).
    """
    holdout = cfd_dataset.holdout_runs()
    records = []
    for _, row in holdout.iterrows():
        field = cfd_dataset.get_field(row["run_id"])
        amplitudes = pod_model.project(field)
        predicted_cd = pod_model.reconstruct_cd(amplitudes, row, method)
        records.append(
            {
                "run_id": row["run_id"],
                "actual_cd": row["Cd"],
                "predicted_cd": predicted_cd,
                "abs_error": abs(predicted_cd - row["Cd"]),
            }
        )
    errors = np.array([r["abs_error"] for r in records])
    summary = {
        "n_holdout": len(records),
        "mean_abs_error": float(errors.mean()) if len(errors) else float("nan"),
        "max_abs_error": float(errors.max()) if len(errors) else float("nan"),
    }
    return records, summary
