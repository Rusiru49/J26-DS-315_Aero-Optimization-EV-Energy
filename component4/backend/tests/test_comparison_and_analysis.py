"""Function 1 and Function 2 calculations checked against hand-calculated reference cases."""
import pytest

from app.services import analysis, comparison
from app.services.store import NotFound


def test_expressway_savings_match_hand_calculation(store):
    # Reference case worked out by hand from the dummy rows (Expressway design = EV01_OPT2):
    #   baseline 205.0 Wh/km, 390.2 km, 1025.00 LKR/100km
    #   optimized 189.1 Wh/km, 423.1 km, 945.50 LKR/100km
    r = comparison.compare(store, "EV01", "SLDC_Expressway")
    s = r["savings"]
    assert r["optimized_config_id"] == "EV01_OPT2"
    assert r["baseline"]["energy_wh_per_km"] == 205.0 and r["optimized"]["energy_wh_per_km"] == 189.1
    assert s["energy_saving_wh_per_km"] == pytest.approx(15.9)
    assert s["energy_saving_pct"] == pytest.approx(15.9 / 205.0 * 100)
    assert s["range_gain_km"] == pytest.approx(32.9)
    assert s["cost_saving_lkr_per_100km"] == pytest.approx(79.5)
    assert s["cost_saving_pct"] == pytest.approx(79.5 / 1025.0 * 100)


def test_savings_function_is_plain_arithmetic():
    s = comparison.savings(150, 144, 360, 375, 6.67, 6.40)
    assert s["energy_saving_wh_per_km"] == 6 and s["energy_saving_pct"] == pytest.approx(4.0)
    assert s["range_gain_km"] == 15 and s["range_gain_pct"] == pytest.approx(15 / 360 * 100)
    assert s["cost_saving_lkr_per_100km"] == pytest.approx(0.27)


def test_all_conditions_with_a_design_agree_with_formulas(store):
    for cid in ("SLDC_Urban", "SLDC_Expressway"):
        r = comparison.compare(store, "EV01", cid)
        assert r["available"] and r["optimized_for"] == cid
        b, o, s = r["baseline"], r["optimized"], r["savings"]
        assert s["energy_saving_wh_per_km"] == pytest.approx(b["energy_wh_per_km"] - o["energy_wh_per_km"])
        assert s["range_gain_km"] == pytest.approx(o["range_km"] - b["range_km"])
        assert s["cost_saving_lkr_per_100km"] == pytest.approx(b["cost_lkr_per_100km"] - o["cost_lkr_per_100km"])
        assert s["energy_saving_wh_per_km"] > 0
        # C4's own calculation agrees with C3's comparison row
        assert abs(r["c3_cross_check"]["energy_saving_wh_per_km"]) < 0.06


def test_aerodynamics_and_shape_changes(store):
    r = comparison.compare(store, "EV01", "SLDC_Urban")
    assert r["aerodynamics"]["cd_change"] == pytest.approx(-0.022)
    assert r["aerodynamics"]["saving_within_uncertainty"] is False
    names = {c["name"]: c for c in r["shape_changes"]}
    assert len(names) == 5 and names["spoiler_angle_deg"]["delta"] == 4
    assert names["spoiler_angle_deg"]["unit"] == "deg"


def test_coming_soon_and_unknown(store):
    r = comparison.compare(store, "EV01", "SLDC_Rural")
    assert r["available"] is False and r["status"] == "coming_soon"
    with pytest.raises(NotFound):
        comparison.compare(store, "EV99", "WLTP")
    with pytest.raises(NotFound):
        comparison.compare(store, "EV01", "SLDC_Nope")


def test_same_input_same_output(store):
    """NFR3: repeated runs with the same input give identical results."""
    a = comparison.compare(store, "EV01", "SLDC_Urban")
    b = comparison.compare(store, "EV01", "SLDC_Urban")
    assert a == b


# ------------------------------------------------------------------ Function 2
def test_condition_comparison_each_condition_with_its_own_design(store):
    r = analysis.condition_comparison(store, "EV01", "WLTP")
    by = {c["condition_id"]: c for c in r["conditions"]}
    assert set(by) == {"SLDC_Urban", "SLDC_Expressway"}            # WLTP is the benchmark, not a row
    assert by["SLDC_Urban"]["design_id"] == "EV01_OPT1" and by["SLDC_Expressway"]["design_id"] == "EV01_OPT2"
    assert r["benchmark_condition"] == "WLTP" and r["benchmark_baseline_energy_wh_per_km"] == 152.0
    # baseline under the condition vs baseline under WLTP, by hand: (138.0 - 152.0) / 152.0
    assert by["SLDC_Urban"]["baseline_vs_benchmark_pct"] == pytest.approx(-14.0 / 152.0 * 100)
    assert by["SLDC_Expressway"]["baseline_vs_benchmark_pct"] == pytest.approx(53.0 / 152.0 * 100)
    assert r["ranking"] == ["SLDC_Expressway", "SLDC_Urban"]
    assert {c["condition_id"] for c in r["coming_soon"]} == {"SLDC_Suburban", "SLDC_Intercity", "SLDC_Rural"}
    text = " ".join(r["insights"])
    assert "WLTP" in text and "predicted" in text and "own optimized design" in text
    assert "consistent with" not in text and "proof" not in text  # no explanations of cause


def test_user_can_pick_which_conditions_to_compare(store):
    only_urban = analysis.condition_comparison(store, "EV01", "WLTP", ["SLDC_Urban"])
    assert [c["condition_id"] for c in only_urban["conditions"]] == ["SLDC_Urban"]
    both = analysis.condition_comparison(store, "EV01", "WLTP", ["SLDC_Expressway", "SLDC_Urban"])
    assert len(both["conditions"]) == 2
    # a condition without results is left out (it is listed under coming_soon)
    assert len(analysis.condition_comparison(store, "EV01", "WLTP", ["SLDC_Urban", "SLDC_Rural"])["conditions"]) == 1
    with pytest.raises(NotFound):
        analysis.condition_comparison(store, "EV01", "WLTP", ["SLDC_Nope"])


def test_cd_check_uses_c3_uncertainty(store):
    s = analysis.sensitivity(store, "EV01", "SLDC_Urban")
    cd = s["cd_check"]
    assert cd["cd_change"] == pytest.approx(-0.022)
    assert cd["optimized_cd_uncertainty"] == 0.004
    assert cd["change_in_uncertainties"] == pytest.approx(0.022 / 0.004)  # 5.5
    assert cd["within_uncertainty"] is False
    assert "assum" not in str(s).lower()


def test_tariff_sensitivity_is_proportional(store):
    base = comparison.compare(store, "EV01", "SLDC_Expressway")
    t0 = analysis.sensitivity(store, "EV01", "SLDC_Expressway")["tariff"]
    t10 = analysis.sensitivity(store, "EV01", "SLDC_Expressway", tariff_change_pct=10)["tariff"]
    assert t0["selected"]["cost_saving_lkr_per_100km"] == pytest.approx(
        base["savings"]["cost_saving_lkr_per_100km"])                    # 79.5
    assert t10["selected"]["cost_saving_lkr_per_100km"] == pytest.approx(79.5 * 1.10)
    assert t10["selected"]["tariff_lkr_per_kwh"] == pytest.approx(45 * 1.10)
    sweep = {r["tariff_change_pct"]: r["cost_saving_lkr_per_100km"] for r in t0["sweep"]}
    assert sweep[-20] == pytest.approx(79.5 * 0.8) and sweep[20] == pytest.approx(79.5 * 1.2)
    # cost of the design by hand: 189.1 Wh/km -> 18.91 kWh/100km x 45 / 0.9 = 945.5 LKR; +10 % tariff
    assert t10["selected"]["optimized_cost_lkr_per_100km"] == pytest.approx(945.5 * 1.10)


def test_sensitivity_coming_soon(store):
    assert analysis.sensitivity(store, "EV01", "SLDC_Rural")["status"] == "coming_soon"