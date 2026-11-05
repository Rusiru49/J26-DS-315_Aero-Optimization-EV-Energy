"""
The optimiser with Workflow 1's combined drag model, which takes the speed as
a 6th input: each second of a cycle must get the Cd at its own speed, taken
only from the CFD speeds the model was trained at. Needs the model saved by
scripts/run_workflow1.py on Member 2's data (models/member2_sedan_pp1/).
"""

import os

import numpy as np
import pytest

from component3 import config
from component3.workflow1.training import load_model
from component3.workflow2.energy_model import load_driving_cycle, prepare_cycle
from component3.workflow2.optimizer import _cd_per_second, compare_to_baseline

MODEL_DIR = os.path.join(config.MODELS_DIR, "member2_sedan_pp1")


@pytest.fixture(scope="module")
def drag_model():
    path = os.path.join(MODEL_DIR, "drag_model.joblib")
    if not os.path.exists(path):
        pytest.skip("No combined drag model found - run scripts/run_workflow1.py first.")
    return load_model(path)


@pytest.fixture(scope="module")
def area_model():
    path = os.path.join(MODEL_DIR, "frontal_area_model.joblib")
    if not os.path.exists(path):
        pytest.skip("No frontal-area model found - run scripts/run_workflow1.py first.")
    return load_model(path)


def test_cd_follows_the_speed_of_each_second(drag_model):
    cycle = prepare_cycle(load_driving_cycle("expressway"))
    cd, cd_by_speed = _cd_per_second(drag_model, config.BASELINE_SHAPE, cycle)
    low, high = min(config.SPEEDS_MS), max(config.SPEEDS_MS)
    assert set(cd_by_speed) == set(config.SPEEDS_MS)
    assert cd.shape == cycle["speed_kmh"].shape
    v = cycle["speed_kmh"]
    assert np.allclose(cd[v <= low], cd_by_speed[low])
    assert np.allclose(cd[v >= high], cd_by_speed[high])
    lo, hi = sorted(cd_by_speed.values())
    assert np.all((cd >= lo - 1e-12) & (cd <= hi + 1e-12))


def test_combined_model_never_reports_worse_than_baseline(drag_model, area_model):
    result = compare_to_baseline(drag_model, "urban", n_trials=15, area_model=area_model)
    assert result["relative_improvement"] >= -1e-9
    assert result["optimised"]["wh_per_km"] <= result["baseline"]["wh_per_km"] + 1e-9
    for side in ("baseline", "optimised"):
        lo, hi = sorted(result[side]["cd_by_speed"].values())
        assert lo - 1e-12 <= result[side]["cd"] <= hi + 1e-12
