"""
Tests for how Workflow 1 tests itself (workflow1/evaluation.py): one drag model
for all speeds (5 shape values + speed); every testable design is the exam
exactly once per repeat and hidden at EVERY speed; never a corner / baseline /
low-confidence design; the winner is the candidate with the lowest error on real
designs averaged over the speeds.
"""

import os

import numpy as np
import pandas as pd
import pytest

from component3 import config
from component3.data.dummy_generator import generate_dataset
from component3.data.loader import CFDDataset
from component3.workflow1 import evaluation
from component3.workflow1.evaluation import (CANDIDATES, INPUT_NAMES, choose_candidate, cross_validate, designs_to_test,
                                             for_speed, make_folds, shared_designs_to_test, summarise, train_final)


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="module")
def small_data(tmp_path_factory):
    folder = str(tmp_path_factory.mktemp("cv_sedan"))
    generate_dataset(out_dir=folder, n_designs=14, seed=5)
    table = pd.read_csv(os.path.join(folder, "run_table.csv"))
    table["morph_confidence"] = np.where(table["design_id"] == "D006", "low", "high")
    table.to_csv(os.path.join(folder, "run_table.csv"), index=False)
    return {s: CFDDataset(folder, s) for s in (40, 100)}


@pytest.fixture(scope="module")
def fast(monkeypatch_module):
    monkeypatch_module.setattr(config, "N_SYNTHETIC_SHAPES", 150)
    monkeypatch_module.setattr(evaluation, "N_DIAGNOSTIC_SHAPES", 40)
    # the labeller choice has its own test (test_labeller.py); fix it here for speed
    monkeypatch_module.setattr(evaluation, "select_labeller_arrays", lambda a, n_folds=5: (("per_mode", "ridge", 4), {}))


@pytest.fixture(scope="module")
def cv(small_data, fast):
    return cross_validate(small_data, n_folds=3, n_repeats=2, progress=lambda *_: None, n_jobs=1)


def test_folds_cover_every_shared_testable_design_once(small_data):
    testable = shared_designs_to_test(small_data)
    folds = make_folds(testable, 3, seed=1)
    flat = [d for f in folds for d in f]
    assert sorted(flat) == sorted(testable) and len(flat) == len(set(flat))
    flags = small_data[100].runs.set_index("run_id")
    assert not flags.loc[flat, "is_corner"].any() and not flags.loc[flat, "is_baseline"].any()
    assert "D006" not in flat   # low-confidence: trains only
    assert list(make_folds(small_data[100].runs, 3, 1)) == list(make_folds(designs_to_test(small_data[100].runs), 3, 1))


def test_every_testable_design_is_tested_once_per_repeat_at_every_speed(small_data, cv):
    oof = cv["oof"]
    n = len(shared_designs_to_test(small_data))
    for _, rep in oof.groupby(["repeat", "speed"]):
        assert len(rep) == n and rep["run_id"].is_unique
    # a design is the exam in the same fold at both speeds
    assert (oof.groupby(["repeat", "run_id"])["fold"].nunique() == 1).all()


def test_results_table_and_winner(cv):
    for s in (40, 100):
        table, area_table, diag = summarise(for_speed(cv, s))
        assert set(table.index) == set(CANDIDATES) | {"label"}
        assert (table["mae"] >= 0).all() and set(diag) == {"xgboost", "random_forest", "ann_mlp"}
        assert area_table.iloc[0]["mae_percent"] < 5
    winner, _ = choose_candidate(cv)
    oof = cv["oof"]
    mean_mae = {c: np.mean([(g[c] - g["cd"]).abs().groupby(g["repeat"]).mean().mean() for _, g in oof.groupby("speed")])
                for c in CANDIDATES}
    assert winner == min(mean_mae, key=mean_mae.get)


@pytest.mark.parametrize("candidate", ["random_forest+real", "xgboost+aug", "ann_mlp+aug_weighted", "ann_mlp+pretrain_finetune"])
def test_final_model_uses_shape_and_speed(small_data, fast, candidate):
    model, info = train_final(small_data, candidate)
    assert info["inputs"] == INPUT_NAMES and info["speeds_kmh"] == [40, 100]
    assert info["uses_augmentation"] == (not candidate.endswith("+real"))
    x = small_data[100].runs[config.SHAPE_PARAM_NAMES].to_numpy(dtype=float)
    for s in (40, 100):
        pred = model.predict(np.hstack([x, np.full((len(x), 1), float(s))]))
        assert pred.shape == (len(x),) and np.all(np.isfinite(pred))


def test_parallel_folds_give_the_same_result(small_data, cv):
    """Worker processes start with a fresh config: the design box and sample sizes
    must be passed in, or the parallel run would differ from the serial one."""
    par = cross_validate(small_data, n_folds=3, n_repeats=1, progress=lambda *_: None, n_jobs=2)
    ser = cv["oof"][cv["oof"]["repeat"] == 0]
    key = ["speed", "run_id"]
    p, q = par["oof"].sort_values(key).reset_index(drop=True), ser.sort_values(key).reset_index(drop=True)
    assert list(p["run_id"]) == list(q["run_id"])
    pd.testing.assert_series_equal(p["random_forest+real"], q["random_forest+real"])


def test_real_weights_give_each_side_half_the_say():
    from component3.workflow1.training import real_weights
    w = real_weights(20, 1200)
    assert w[:20].sum() == pytest.approx(w[20:].sum())
