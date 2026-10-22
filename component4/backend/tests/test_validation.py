"""The data validation module must accept good data and reject bad data (FR4, NFR4)."""
import pytest

from app.config import Settings
from app.services.validation import run_validation

S = Settings()


def _report(store):
    r = run_validation(store, S)
    return r, {c["id"]: c["status"] for c in r["checks"]}


def test_dummy_package_passes(store):
    r, by = _report(store)
    assert r["status"] == "pass" and r["counts"]["fail"] == 0
    assert by["range_check"] == by["cost_check"] == by["c3_crosscheck"] == "pass"


def test_wrong_range_is_caught(make_store):
    def bad(p):
        p["results"][1]["range_km"] = 500.0
    r, by = _report(make_store(bad))
    assert r["status"] == "fail" and by["range_check"] == "fail"


def test_wrong_cost_is_caught(make_store):
    def bad(p):
        p["results"][0]["cost_lkr_per_100km"] = 99.0
    _, by = _report(make_store(bad))
    assert by["cost_check"] == "fail"


def test_wrong_unit_is_caught(make_store):
    def bad(p):
        p["units"]["energy_wh_per_km"] = "kWh/100km"
    _, by = _report(make_store(bad))
    assert by["units"] == "fail"


def test_broken_link_between_designs_is_caught(make_store):
    def bad(p):
        p["evs"][0]["configurations"][1]["optimized_from"] = "SOMETHING_ELSE"
    _, by = _report(make_store(bad))
    assert by["structure"] == "fail"


def test_shape_parameter_outside_limits_is_caught(make_store):
    def bad(p):
        p["evs"][0]["configurations"][1]["shape_parameters"]["spoiler_angle_deg"] = 45
    _, by = _report(make_store(bad))
    assert by["shape_params"] == "fail"


def test_missing_result_row_is_caught(make_store):
    def bad(p):  # drop the optimized Urban row
        p["results"] = [r for r in p["results"]
                        if not (r["config_id"] == "EV01_OPT1" and r["condition_id"] == "SLDC_Urban")]
    _, by = _report(make_store(bad))
    assert by["completeness"] == "fail"


def test_impossible_value_is_caught(make_store):
    def bad(p):
        p["results"][0]["energy_wh_per_km"] = -5
    _, by = _report(make_store(bad))
    assert by["ranges"] == "fail"


def test_disagreement_with_c3_savings_is_caught(make_store):
    def bad(p):
        p["comparisons"][0]["energy_saving_wh_per_km"] = 50.0
    _, by = _report(make_store(bad))
    assert by["c3_crosscheck"] == "fail"


def test_missing_tariff_is_a_warning_not_a_failure(make_store):
    def bad(p):
        p["model_notes"]["tariff_lkr_per_kwh"] = None
    r, by = _report(make_store(bad))
    assert by["cost_check"] == "warn" and r["status"] == "warn"


def test_optimized_worse_than_baseline_warns(make_store):
    def bad(p):  # make optimized Urban worse (keep range/cost consistent so only this check fires)
        for r in p["results"]:
            if r["config_id"] == "EV01_OPT1" and r["condition_id"] == "SLDC_Urban":
                r["energy_wh_per_km"] = 150.0
                r["range_km"] = round(80000 / 150.0, 1)
                r["cost_lkr_per_100km"] = round(150.0 / 1000 * 100 * 45 / 0.9, 2)
    _, by = _report(make_store(bad))
    assert by["not_worse"] == "warn"

def test_standard_cycle_pointing_to_unknown_condition_is_caught(make_store):
    def bad(p):
        p["conditions"][0]["benchmark_of"] = "SLDC_Nowhere"
    _, by = _report(make_store(bad))
    assert by["conditions"] == "fail"