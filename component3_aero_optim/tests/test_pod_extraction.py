"""
POD round-trip validation against the placeholder dataset's own known
ground truth. Tolerances are set from the actually-observed error
(mean ~0.0015, max ~0.005 Cd, i.e. a few drag counts) with a safety margin,
not chosen a priori — a regression here means the SVD/projection math
broke, not that the placeholder's synthetic physics changed.
"""

import numpy as np
import pytest

from cfd_data_loader import load_cfd_dataset
from pod_extraction import fit_pod, validate_on_holdout, explained_variance_ratio


@pytest.fixture(scope="module")
def dataset():
    return load_cfd_dataset(source="placeholder")


@pytest.fixture(scope="module")
def pod_model(dataset):
    return fit_pod(dataset)


def test_pod_explains_most_variance(dataset, pod_model):
    """15 modes should recover the vast majority of field variance in the
    placeholder data, since it was constructed from exactly that many
    fixed basis functions — a low value here means the SVD/eigh path is
    broken, not that the data is just hard."""
    evr = explained_variance_ratio(dataset, pod_model)
    assert evr > 0.95


def test_holdout_cd_reconstruction_within_a_few_drag_counts(dataset, pod_model):
    """The reference spec's own acceptance criterion: held-out real runs
    through the POD path should match recorded Cd within a few drag
    counts (1 drag count = 0.001 Cd)."""
    _, summary = validate_on_holdout(dataset, pod_model)
    assert summary["mean_abs_error"] < 0.01
    assert summary["max_abs_error"] < 0.02


def test_mode_shapes_are_orthonormal():
    """Snapshot POD should yield (numerically) orthonormal spatial modes —
    a broken normalisation step would silently corrupt every downstream
    reconstruction without raising an error."""
    dataset = load_cfd_dataset(source="placeholder")
    pod_model = fit_pod(dataset)
    gram = pod_model.mode_shapes @ pod_model.mode_shapes.T
    np.testing.assert_allclose(gram, np.eye(gram.shape[0]), atol=1e-6)


def test_project_then_reconstruct_field_recovers_original(dataset, pod_model):
    """A training run's own field, projected then reconstructed, should be
    close to the original field (up to the variance the kept modes don't
    explain) — catches a transposed axis or a sign flip in project()/
    reconstruct_field() that unit tests on scalars alone might miss."""
    real_runs = dataset.real_runs()
    run_id = real_runs.iloc[0]["run_id"]
    original = dataset.get_field(run_id)

    amplitudes = pod_model.project(original)
    reconstructed = pod_model.reconstruct_field(amplitudes)

    rel_error = np.linalg.norm(reconstructed - original) / np.linalg.norm(original)
    assert rel_error < 0.1
