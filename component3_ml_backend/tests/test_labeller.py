"""
The physics labeller: exact face integration, the fast precomputed form, the
per-pattern helper and the cross-validated choice. Uses the dummy dataset.
"""

import os

import numpy as np
import pytest

from component3 import config
from component3.data.loader import load_cfd_dataset
from component3.workflow1.helper import PerModeGP, _params_to_array, fit_helper, label_shapes, select_labeller, train_helper_gp
from component3.workflow1.pod import dataset_arrays, drag_projected_areas, exact_cd_from_faces, fit_pod, shape_features


@pytest.fixture(scope="module")
def dataset():
    if not os.path.exists(os.path.join(config.PLACEHOLDER_DATA_DIR, "run_table.csv")):
        pytest.skip("No dummy dataset - run data/dummy_generator.py first.")
    return load_cfd_dataset("placeholder", 100)


@pytest.fixture(scope="module")
def pod(dataset):
    return fit_pod(dataset)


def test_exact_integral_of_true_fields_reproduces_the_recorded_cd(dataset):
    runs = dataset.real_runs()
    err = [abs(exact_cd_from_faces(dataset, r) - c) for r, c in zip(runs["run_id"][:15], runs["Cd"][:15])]
    assert max(err) < 0.002   # 2 drag counts (the dummy adds a little run noise)


def test_fast_drag_sum_equals_the_face_by_face_integral(dataset, pod):
    rng = np.random.default_rng(0)
    x_unit = rng.random((4, len(config.SHAPE_PARAM_NAMES)))
    amps = rng.normal(0, 1, (4, pod.mode_shapes.shape[0]))
    fast = pod.cd_batch(amps, x_unit, "exact")
    fields = pod.mean_field + amps @ pod.mode_shapes
    w = pod.geometry.predict(x_unit)
    slow = -np.sum(fields * w, axis=1) / pod.frontal_area.predict(x_unit).ravel() + pod.friction.predict(x_unit)
    assert np.allclose(fast, slow, rtol=1e-9, atol=1e-12)


def test_geometry_model_predicts_held_out_face_areas(dataset, pod):
    hold = dataset.holdout_runs()
    rid = hold["run_id"].iloc[0]
    true_w = drag_projected_areas(dataset.get_faces(rid))
    pred_w = pod.geometry.predict(shape_features(hold.iloc[0]))[0]
    assert np.abs(pred_w - true_w).sum() / np.abs(true_w).sum() < 0.1


def test_helper_inputs_are_scaled_to_the_agreed_ranges():
    x = _params_to_array(config.BASELINE_SHAPE)
    assert np.all((x >= 0) & (x <= 1))


def test_per_mode_helper_returns_every_pattern(dataset, pod):
    gp = train_helper_gp(dataset, pod, "per_mode")
    assert isinstance(gp, PerModeGP)
    out = gp.predict(np.full((3, 5), 0.5))
    assert out.shape == (3, pod.mode_shapes.shape[0])
    labels = label_shapes(gp, pod, [config.BASELINE_SHAPE] * 2, "exact")
    assert labels.shape == (2,) and np.all(np.isfinite(labels))


def test_cross_validation_compares_all_four_labellers(dataset):
    (kind, method), table = select_labeller(dataset, n_folds=3)
    assert set(table) == {(k, m) for k in ("per_mode", "shared") for m in ("exact", "ridge")}
    assert table[(kind, method)] == min(table.values())
