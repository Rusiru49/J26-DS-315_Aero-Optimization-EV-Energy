"""
Tests for the data contract tooling: the generator, the checker (it must pass
clean data AND catch broken data) and the loader's by-design split. Uses a
small dataset written to a temporary folder, so it does not depend on the
full dummy dataset.
"""

import os
import shutil

import numpy as np
import pandas as pd
import pytest

import config
from cfd_data_loader import CFDDataset, faces_path
from data_checker import check_dataset
from placeholder_cfd_generator import generate_dataset


@pytest.fixture(scope="module")
def small_data(tmp_path_factory):
    folder = str(tmp_path_factory.mktemp("small_sedan"))
    generate_dataset(out_dir=folder, n_designs=12, seed=3)
    return folder


def _copy(small_data, tmp_path):
    target = str(tmp_path / "copy")
    shutil.copytree(small_data, target)
    return target


def _status(results, name_part):
    return [s for n, s, _ in results if name_part in n][0]


def test_clean_data_passes_every_check(small_data):
    results = check_dataset(small_data)
    assert [r for r in results if r[1] == "FAIL"] == []


def test_checker_finds_the_drag_direction_and_shear_sign(small_data):
    detail = [d for n, _, d in check_dataset(small_data) if n.startswith("Cd from the faces")][0]
    assert "[0.0, 0.0, -1.0]" in detail and "tau sign +1" in detail


def test_checker_catches_a_wrong_q_ref(small_data, tmp_path):
    folder = _copy(small_data, tmp_path)
    table = pd.read_csv(os.path.join(folder, "run_table.csv"))
    table["q_ref"] = table["q_ref"] * 1.1
    table.to_csv(os.path.join(folder, "run_table.csv"), index=False)
    assert _status(check_dataset(folder), "q_ref") == "FAIL"


def test_checker_catches_pressure_that_does_not_add_up_to_cd(small_data, tmp_path):
    folder = _copy(small_data, tmp_path)
    path = faces_path(folder, "D000", 100)
    data = dict(np.load(path))
    data["p"] = data["p"] * 1.05
    np.savez_compressed(path, **data)
    assert _status(check_dataset(folder), "Cd from the faces") == "FAIL"


def test_checker_catches_a_missing_face_file(small_data, tmp_path):
    folder = _copy(small_data, tmp_path)
    os.remove(faces_path(folder, "D000", 40))
    assert _status(check_dataset(folder), "every good run has a face file") == "FAIL"


def test_checker_catches_a_changed_face_count(small_data, tmp_path):
    folder = _copy(small_data, tmp_path)
    path = faces_path(folder, "D000", 100)
    data = {k: v[:-5] for k, v in dict(np.load(path)).items()}
    np.savez_compressed(path, **data)
    assert _status(check_dataset(folder), "same number of faces") == "FAIL"


def test_split_is_by_design_and_keeps_corners_and_baseline_in_training(small_data):
    ds100, ds40 = CFDDataset(small_data, 100), CFDDataset(small_data, 40)
    hold100, hold40 = set(ds100.holdout_runs()["run_id"]), set(ds40.holdout_runs()["run_id"])
    assert hold100 == hold40, "both speeds of a design must land on the same side"
    assert not ds100.holdout_runs()["is_corner"].any()
    assert not ds100.holdout_runs()["is_baseline"].any()
    assert ds100.real_runs()["is_baseline"].any()
    assert hold100.isdisjoint(set(ds100.real_runs()["run_id"]))


def test_failed_designs_are_left_out_and_counted(small_data):
    ds = CFDDataset(small_data, 100)
    assert ds.n_failed == 2
    assert len(ds.runs) == 11  # 12 designs + baseline - 2 failed


def test_model_target_is_the_averaged_cd_and_field_is_cp(small_data):
    ds = CFDDataset(small_data, 100)
    table = pd.read_csv(os.path.join(small_data, "run_table.csv"))
    row = table[(table.design_id == "D000") & (table.speed_kmh == 100)].iloc[0]
    assert ds.runs.loc[ds.runs.run_id == "D000", "Cd"].iloc[0] == pytest.approx(row.Cd_mean)
    cp = ds.get_field("D000")
    assert cp.shape == (8640,) and -4 < cp.min() and cp.max() < 1.5


def test_loader_works_when_design_table_has_only_the_agreed_columns(small_data, tmp_path):
    folder = _copy(small_data, tmp_path)
    path = os.path.join(folder, "design_table.csv")
    table = pd.read_csv(path)
    table[["design_id"] + config.SHAPE_PARAM_NAMES].to_csv(path, index=False)
    ds = CFDDataset(folder, 100)
    assert 3 <= ds.runs["is_corner"].sum() <= 5  # 5 flagged, minus any that failed
    assert bool(ds.runs.loc[ds.runs.run_id == "D000", "is_baseline"].iloc[0])
    assert not ds.holdout_runs()["is_corner"].any()


def test_a_delivery_without_case_json_still_loads_and_passes(small_data, tmp_path):
    """case.json is not part of the agreed contract, so a real delivery may not have it."""
    folder = _copy(small_data, tmp_path)
    os.remove(os.path.join(folder, "case.json"))
    results = check_dataset(folder)
    assert [r for r in results if r[1] == "FAIL"] == []
    assert _status(results, "case.json") == "WARN"
    dataset = CFDDataset(folder, 100)
    assert len(dataset.real_runs()) > 0 and dataset.case == {}


def test_a_new_holdout_draw_follows_the_same_rules(small_data):
    ds = CFDDataset(small_data, 100)
    other = ds.with_split(12345)
    hold_a, hold_b = set(ds.holdout_runs()["run_id"]), set(other.holdout_runs()["run_id"])
    assert len(hold_a) == len(hold_b) and hold_a != hold_b          # same size, different designs
    flags = other.runs.set_index("run_id")
    assert not flags.loc[list(hold_b), "is_corner"].any() and not flags.loc[list(hold_b), "is_baseline"].any()
    assert set(ds.holdout_runs()["run_id"]) == hold_a               # the original split is unchanged
