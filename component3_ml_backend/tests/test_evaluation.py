"""
Tests for how Workflow 1 tests itself (workflow1/evaluation.py): every testable
design is the exam exactly once per repeat, never a corner / baseline /
low-confidence design, the training side of a fold never contains its exam
designs, and the winner is the candidate with the lowest error on real designs.
"""

import os

import numpy as np
import pandas as pd
import pytest

from component3 import config
from component3.data.dummy_generator import generate_dataset
from component3.data.loader import CFDDataset
from component3.workflow1 import evaluation
from component3.workflow1.evaluation import cross_validate, make_folds, summarise, designs_to_test, train_final


@pytest.fixture(scope="module")
def small_ds(tmp_path_factory):
    folder = str(tmp_path_factory.mktemp("cv_sedan"))
    generate_dataset(out_dir=folder, n_designs=14, seed=5)
    table = pd.read_csv(os.path.join(folder, "run_table.csv"))
    table["morph_confidence"] = np.where(table["design_id"] == "D006", "low", "high")
    table.to_csv(os.path.join(folder, "run_table.csv"), index=False)
    return CFDDataset(folder, 100)


@pytest.fixture(scope="module")
def cv(small_ds, monkeypatch_module):
    monkeypatch_module.setattr(config, "N_SYNTHETIC_SHAPES", 150)
    monkeypatch_module.setattr(evaluation, "N_DIAGNOSTIC_SHAPES", 40)
    # the labeller choice has its own test (test_labeller.py); fix it here for speed
    monkeypatch_module.setattr(evaluation, "select_labeller_arrays", lambda a, n_folds=5: (("per_mode", "ridge"), {}))
    return cross_validate(small_ds, n_folds=3, n_repeats=2, progress=lambda *_: None, n_jobs=1)


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_folds_cover_every_testable_design_once(small_ds):
    folds = make_folds(small_ds.runs, 3, seed=1)
    flat = [d for f in folds for d in f]
    assert sorted(flat) == sorted(designs_to_test(small_ds.runs))
    assert len(flat) == len(set(flat))
    flags = small_ds.runs.set_index("run_id")
    assert not flags.loc[flat, "is_corner"].any() and not flags.loc[flat, "is_baseline"].any()
    assert "D006" not in flat   # low-confidence: trains only


def test_every_testable_design_is_tested_once_per_repeat(small_ds, cv):
    oof = cv["oof"]
    n = len(designs_to_test(small_ds.runs))
    for _, rep in oof.groupby("repeat"):
        assert len(rep) == n and rep["run_id"].is_unique


def test_results_table_and_winner(cv):
    table, area_table, diag = summarise(cv)
    expected = {f"{m}+{v}" for m in ("xgboost", "random_forest", "ann_mlp") for v in ("aug", "real")} | {"label"}
    assert set(table.index) == expected
    assert (table["mae"] >= 0).all() and set(diag) == {"xgboost", "random_forest", "ann_mlp"}
    oof = cv["oof"]
    mae = {c: (oof[c] - oof["cd"]).abs().groupby(oof["repeat"]).mean().mean() for c in expected - {"label"}}
    assert table.drop(index="label").sort_values("mae").index[0] == min(mae, key=mae.get)
    assert area_table.iloc[0]["mae_percent"] < 5


def test_parallel_folds_give_the_same_result(small_ds, cv):
    """Worker processes start with a fresh config: the design box and sample sizes
    must be passed in, or the parallel run would differ from the serial one."""
    par = cross_validate(small_ds, n_folds=3, n_repeats=1, progress=lambda *_: None, n_jobs=2)
    ser = cv["oof"][cv["oof"]["repeat"] == 0].reset_index(drop=True)
    assert list(par["oof"]["run_id"]) == list(ser["run_id"])
    pd.testing.assert_series_equal(par["oof"]["random_forest+real"], ser["random_forest+real"])
    # the workers choose the labeller themselves (the fix above does not reach them);
    # where they chose the same one, the labels must match exactly
    same = [c == "per_mode+ridge" for c in par["labeller_choices"]]
    rows = par["oof"]["fold"].map(dict(enumerate(same)))
    assert np.allclose(par["oof"].loc[rows, "label"], ser.loc[rows, "label"])


def test_final_model_is_trained_on_all_designs(small_ds, monkeypatch_module):
    model, info = train_final(small_ds, "random_forest+real")
    assert info["n_designs"] == len(small_ds.runs) and not info["uses_augmentation"]
    x = small_ds.runs[config.SHAPE_PARAM_NAMES].to_numpy(dtype=float)
    assert model.predict(x).shape == (len(x),)
