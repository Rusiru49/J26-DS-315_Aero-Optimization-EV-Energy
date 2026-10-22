"""
Tests for the per-label quality of the synthetic drag values
(workflow1/label_quality.py): the uncertainty calibration recovers a known
error level, the coverage check is honest, the Cd uncertainty is carried
through exactly, and the checks reject what they should.
"""

import numpy as np
import pandas as pd
import pytest

from component3.data.dummy_generator import generate_dataset
from component3.data.loader import CFDDataset
from component3.workflow1.helper import fit_helper
from component3.workflow1.label_quality import (CP_STAGNATION_LIMIT, calibrated_sigma, calibration_report,
                                                check_labels, fit_calibration, predict_labels, region_limit)
from component3.workflow1.pod import dataset_arrays, fit_pod_arrays


@pytest.fixture(scope="module")
def labeller(tmp_path_factory):
    folder = str(tmp_path_factory.mktemp("lq_sedan"))
    generate_dataset(out_dir=folder, n_designs=14, seed=4)
    ds = CFDDataset(folder, 100)
    a = dataset_arrays(ds, ds.runs.reset_index(drop=True))
    pod = fit_pod_arrays(a, 4)
    amps = (a["fields"] - pod.mean_field) @ pod.mode_shapes.T
    return a, pod, fit_helper(a["x_unit"], amps, "per_mode")


def test_calibration_recovers_a_known_error_level():
    rng = np.random.default_rng(0)
    raw = rng.uniform(0.001, 0.004, 4000)
    resid = rng.normal(0, np.sqrt((3 * raw) ** 2 + 0.002 ** 2))
    a, b = fit_calibration(resid, raw)
    assert a == pytest.approx(3, rel=0.15) and b == pytest.approx(0.002, rel=0.25)
    z = np.abs(resid) / calibrated_sigma(raw, (a, b))
    assert np.mean(z <= 1) == pytest.approx(0.68, abs=0.04) and np.mean(z <= 2) == pytest.approx(0.95, abs=0.03)


def test_calibration_report_scores_each_fold_with_a_fit_from_other_folds():
    rng = np.random.default_rng(1)
    rows = []
    for rep in range(2):
        for i in range(40):
            s = rng.uniform(0.002, 0.005)
            rows.append({"repeat": rep, "fold": i % 4, "cd": 0.35, "label": 0.35 + rng.normal(0, 2 * s), "label_std": s})
    rep = calibration_report(pd.DataFrame(rows))
    assert 0.5 < rep["coverage_1sigma"] < 0.85 and rep["coverage_2sigma"] > 0.85


def test_uncertainty_is_carried_exactly_through_the_linear_cd_step(labeller):
    a, pod, helper = labeller
    x = a["x_unit"][:3]
    amps, cd, sigma = predict_labels(helper, pod, x, "ridge")
    _, std = helper.predict(x, return_std=True)
    k = 1   # move one pattern amount by its std: Cd moves by gradient x std
    bumped = amps.copy(); bumped[:, k] += std[:, k]
    step = pod.cd_batch(bumped, x, "ridge") - cd
    assert np.allclose(step, pod.cd_gradient(x, "ridge")[:, k] * std[:, k])
    assert np.all(sigma >= np.abs(step) - 1e-12)


def test_checks_reject_far_shapes_impossible_pressure_and_out_of_range_cd(labeller):
    a, pod, helper = labeller
    x = np.vstack([a["x_unit"][:2], [[3.0, 3.0, 3.0, 3.0, 3.0]]])     # last one far outside the data
    amps, cd, _ = predict_labels(helper, pod, x, "ridge")
    t = check_labels(pod, amps, x, cd, a["x_unit"], a["cd"])
    assert bool(t["in_validated_region"].iloc[0]) and not bool(t["in_validated_region"].iloc[2])
    assert not bool(t["keep"].iloc[2])
    big = amps.copy(); big[0] *= 1e4                                   # a field far beyond the stagnation limit
    t2 = check_labels(pod, big[:1], x[:1], np.array([10.0]), a["x_unit"], a["cd"])
    assert not bool(t2["cp_below_stagnation"].iloc[0]) or t2["max_cp"].iloc[0] > CP_STAGNATION_LIMIT
    assert not bool(t2["cd_in_range"].iloc[0]) and not bool(t2["keep"].iloc[0])
    assert region_limit(a["x_unit"]) > 0
