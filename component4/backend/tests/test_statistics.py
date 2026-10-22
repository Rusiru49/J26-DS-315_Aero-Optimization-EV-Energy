"""Statistics engine checked against hand calculations."""
import math

import pandas as pd
import pytest
from scipy import stats

from app.services import statistics as st


def test_paired_ttest_matches_hand_calculation():
    before, after = [10, 12, 14, 16], [9, 10, 13, 13]
    # differences 1,2,1,3 -> mean 1.75, sd = sqrt(2.75/3) = 0.957427, se = 0.478714
    r = st.paired_ttest(before, after)
    assert r["n_pairs"] == 4 and r["df"] == 3
    assert r["mean_difference"] == pytest.approx(1.75)
    assert r["sd_difference"] == pytest.approx(math.sqrt(2.75 / 3))
    assert r["t_statistic"] == pytest.approx(1.75 / (math.sqrt(2.75 / 3) / 2), rel=1e-9)  # 3.6556
    assert r["cohens_dz"] == pytest.approx(1.75 / math.sqrt(2.75 / 3))                       # 1.8277
    ref = stats.ttest_rel(before, after)  # independent implementation
    assert r["t_statistic"] == pytest.approx(ref.statistic) and r["p_value"] == pytest.approx(ref.pvalue)
    assert r["ci_low"] < 1.75 < r["ci_high"]
    assert "caution" not in r


def test_paired_ttest_identical_differences_and_bad_input():
    r = st.paired_ttest([5, 6, 7], [4, 5, 6])  # every difference is exactly 1
    assert r["t_statistic"] is None and r["significant"] is True and r["note"]
    with pytest.raises(ValueError):
        st.paired_ttest([1], [1])
    with pytest.raises(ValueError):
        st.paired_ttest([1, 2], [1, 2, 3])


def test_anova_matches_hand_calculation():
    groups = {"a": [1, 2, 3], "b": [2, 3, 4], "c": [5, 6, 7]}
    # means 2,3,6; SSB = 26, SSW = 6; F = (26/2)/(6/6) = 13; eta^2 = 26/32
    r = st.anova_tukey(groups)
    assert r["df_between"] == 2 and r["df_within"] == 6
    assert r["f_statistic"] == pytest.approx(13.0)
    assert r["eta_squared"] == pytest.approx(26 / 32)
    assert r["p_value"] == pytest.approx(stats.f.sf(13, 2, 6))
    assert r["significant"] is True
    tk = {(p["group1"], p["group2"]): p["reject"] for p in r["tukey_hsd"]}
    # q = diff / sqrt(MSE/n) with MSE = 1, n = 3 -> a-b 1.73 (no), a-c 6.93 (yes), b-c 5.20 (yes); q_crit ~ 4.34
    assert tk[("a", "b")] is False and tk[("a", "c")] is True and tk[("b", "c")] is True


def test_anova_needs_replicates():
    with pytest.raises(ValueError):
        st.anova_tukey({"a": [1.0], "b": [2.0], "c": [3.0]})


def test_correlation_perfect_and_constant():
    r = st.correlation([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
    assert r["pearson_r"] == pytest.approx(1.0) and r["spearman_rho"] == pytest.approx(1.0)
    r = st.correlation([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
    assert r["pearson_r"] == pytest.approx(-1.0)
    r = st.correlation([1, 2, 3], [4, 4, 4])
    assert r["pearson_r"] is None and "constant" in r["note"]


def test_regression_matches_hand_calculation():
    # slope = Sxy/Sxx = 19.9/10 = 1.99 ; intercept = 6.02 - 1.99*3 = 0.05
    # SST = 39.708, SSE = 0.107 -> R2 = 0.99731, adj R2 = 1-(1-R2)*4/3 = 0.99641
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2.1, 3.9, 6.2, 7.8, 10.1]})
    r = st.regression(df, "y", ["x"])
    coefs = {c["name"]: c["coef"] for c in r["coefficients"]}
    assert coefs["x"] == pytest.approx(1.99) and coefs["const"] == pytest.approx(0.05)
    assert r["r_squared"] == pytest.approx(1 - 0.107 / 39.708, abs=1e-4)
    assert r["adj_r_squared"] == pytest.approx(0.996407, abs=1e-4)
    assert r["n"] == 5


def test_regression_needs_enough_rows():
    df = pd.DataFrame({"x": [1, 2], "y": [1, 2]})
    with pytest.raises(ValueError):
        st.regression(df, "y", ["x"])


def test_descriptive():
    d = st.descriptive([2, 4, 6])
    assert d["n"] == 3 and d["mean"] == 4 and d["median"] == 4 and d["std"] == pytest.approx(2.0)


def test_live_statistics_on_current_dummy_data(store, settings):
    """3 pairs, 1 value per condition: t-test, correlation, regression meet their method minimum
    and show n; ANOVA cannot run (needs at least 2 values per condition)."""
    r = st.live_statistics(store, settings)
    assert r["n_pairs"] == 3
    cards = {t["id"]: t for t in r["tests"]}
    assert cards["paired_t"]["status"] == "ready" and cards["paired_t"]["result"]["n_pairs"] == 3
    assert cards["correlation"]["status"] == "ready" and cards["correlation"]["result"]["n"] == 3
    assert cards["regression"]["status"] == "ready" and cards["regression"]["result"]["n"] == 3
    assert cards["regression"]["result"]["predictors"] == ["avg_speed_kmh"]  # Cd reduction does not vary
    assert cards["anova"]["status"] == "locked" and cards["anova"]["result"] is None
    assert cards["anova"]["reason"] and "values per condition" in cards["anova"]["have"]


def test_live_statistics_locks_when_below_method_minimum(make_store, settings):
    def one_condition(p):  # keep only Urban results -> a single pair
        keep = {"SLDC_Urban"}
        p["results"] = [r for r in p["results"] if r["condition_id"] in keep]
        p["comparisons"] = [c for c in p["comparisons"] if c["condition_id"] in keep]
    r = st.live_statistics(make_store(one_condition), settings)
    assert {t["id"]: t["status"] for t in r["tests"]} == {
        "paired_t": "locked", "anova": "locked", "correlation": "locked", "regression": "locked"}


def test_sample_statistics_has_no_labels_and_is_repeatable(settings):
    a, b = st.sample_statistics(settings), st.sample_statistics(settings)
    assert "synthetic" not in str(a).lower() and "banner" not in a
    assert a["paired_t"]["n_pairs"] == 48 and len(a["anova"]["groups"]) == 6
    assert a["paired_t"]["t_statistic"] == b["paired_t"]["t_statistic"]  # same seed, same result
    assert a["regression"]["n"] == 48