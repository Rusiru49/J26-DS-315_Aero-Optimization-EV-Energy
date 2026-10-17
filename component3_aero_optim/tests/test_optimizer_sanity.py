"""
Optimiser correctness: it must never report a shape worse than the baseline
as "better". Uses a small n_trials for test speed — the guarantee being
tested comes from enqueueing the baseline shape as a candidate trial (see
optimizer.optimise_shape), not from search luck within a trial budget, so a
small budget is a valid test of that guarantee.
"""

import pytest

import config
from surrogate_training import load_model
import os
from optimizer import compare_to_baseline


@pytest.fixture(scope="module")
def trained_model():
    path = os.path.join(config.MODELS_DIR, "drag_surrogate_xgboost.joblib")
    if not os.path.exists(path):
        pytest.skip("No trained surrogate found — run the Phase 5/6 pipeline first.")
    return load_model(path)


def test_never_reports_worse_than_baseline_as_better(trained_model):
    result = compare_to_baseline(trained_model, "urban", n_trials=15)
    assert result["relative_improvement"] >= -1e-9
    assert result["optimised"]["wh_per_km"] <= result["baseline"]["wh_per_km"] + 1e-9


def test_baseline_check_decision_is_consistent_with_tolerance(trained_model):
    result = compare_to_baseline(trained_model, "urban", n_trials=15)
    if result["relative_improvement"] <= config.BASELINE_OPTIMAL_TOLERANCE:
        assert result["already_near_optimal"] is True
    else:
        assert result["already_near_optimal"] is False
