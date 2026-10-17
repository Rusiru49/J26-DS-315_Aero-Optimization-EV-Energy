"""
The frontal-area model and its use in the optimiser (drag area Cd x A).
Needs the dummy dataset and the models saved by scripts/run_workflow1.py.
"""

import os

import pytest

import config
from cfd_data_loader import load_cfd_dataset
from optimizer import compare_to_baseline
from surrogate_training import load_model, predict_frontal_area, train_area_model

DRAG_PATH = os.path.join(config.MODELS_DIR, "drag_surrogate_xgboost.joblib")


@pytest.fixture(scope="module")
def dataset():
    if not os.path.exists(os.path.join(config.PLACEHOLDER_DATA_DIR, "run_table.csv")):
        pytest.skip("No dummy dataset - run placeholder_cfd_generator.py first.")
    return load_cfd_dataset("placeholder", 100)


@pytest.fixture(scope="module")
def area_model(dataset):
    return train_area_model(dataset.real_runs(), dataset.holdout_runs())[1]


def test_area_model_is_accurate_on_held_out_designs(dataset):
    _, _, table = train_area_model(dataset.real_runs(), dataset.holdout_runs())
    assert table.iloc[0]["holdout_max_error_percent"] < 3.0


def test_area_model_follows_the_real_baseline_area(dataset, area_model):
    base = dataset.runs[dataset.runs["run_id"] == "D000"].iloc[0]
    predicted = predict_frontal_area(area_model, config.BASELINE_SHAPE)
    assert abs(predicted - base["frontal_area"]) / base["frontal_area"] < 0.03


@pytest.fixture(scope="module")
def drag_model():
    if not os.path.exists(DRAG_PATH):
        pytest.skip("No trained surrogate found - run scripts/run_workflow1.py first.")
    return load_model(DRAG_PATH)


def test_energy_uses_drag_area_not_the_constant(drag_model, area_model):
    with_area = compare_to_baseline(drag_model, "urban", n_trials=5, area_model=area_model)
    constant = compare_to_baseline(drag_model, "urban", n_trials=5)
    assert constant["baseline"]["frontal_area_m2"] == config.FRONTAL_AREA_M2
    assert with_area["baseline"]["frontal_area_m2"] != config.FRONTAL_AREA_M2
    assert abs(with_area["baseline"]["cd_a_m2"] - with_area["baseline"]["cd"] * with_area["baseline"]["frontal_area_m2"]) < 1e-12


def test_optimiser_with_area_model_never_worse_than_baseline(drag_model, area_model):
    result = compare_to_baseline(drag_model, "expressway", n_trials=15, area_model=area_model)
    assert result["optimised"]["wh_per_km"] <= result["baseline"]["wh_per_km"] + 1e-9


def test_giving_both_area_model_and_fixed_area_is_rejected(drag_model, area_model):
    with pytest.raises(ValueError):
        compare_to_baseline(drag_model, "urban", n_trials=3, area_model=area_model, vehicle_kwargs={"frontal_area_m2": 2.2})
