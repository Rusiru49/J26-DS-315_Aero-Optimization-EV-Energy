"""Best design: each optimized design is compared with the baseline under its own condition."""
import pytest

from app.services import comparison, designs
from app.services.store import NotFound


def test_each_condition_uses_its_own_design(store):
    assert comparison.compare(store, "EV01", "SLDC_Urban")["optimized_config_id"] == "EV01_OPT1"
    assert comparison.compare(store, "EV01", "SLDC_Expressway")["optimized_config_id"] == "EV01_OPT2"
    with pytest.raises(NotFound):
        comparison.compare(store, "EV01", "SLDC_Urban", "EV01_NOPE")


def test_wltp_has_no_optimized_result(store):
    r = comparison.compare(store, "EV01", "WLTP")
    assert r["available"] is False          # only the baseline has a WLTP result


def test_ranking_matches_hand_calculation(store):
    r = designs.ranking(store, "EV01")
    by = {d["config_id"]: d for d in r["designs"]}
    assert set(by) == {"EV01_OPT1", "EV01_OPT2"}
    # OPT1 (Urban): baseline 138.0 -> 134.1 Wh/km. OPT2 (Expressway): 205.0 -> 189.1 Wh/km.
    assert by["EV01_OPT1"]["mean_energy_saving_pct"] == pytest.approx(3.9 / 138 * 100)
    assert by["EV01_OPT2"]["mean_energy_saving_pct"] == pytest.approx(15.9 / 205 * 100)
    d = by["EV01_OPT1"]
    assert d["overall_score_pct"] == pytest.approx(
        (d["mean_energy_saving_pct"] + d["mean_range_gain_pct"] + d["mean_cost_saving_pct"]) / 3)
    assert d["n_conditions"] == 1 and d["per_condition"][0]["condition_id"] == "SLDC_Urban"
    scores = [x["overall_score_pct"] for x in r["designs"]]
    assert scores == sorted(scores, reverse=True)
    assert [x["rank"] for x in r["designs"]] == [1, 2]
    assert r["best_config_id"] == r["designs"][0]["config_id"] == "EV01_OPT2"
    assert r["tie_for_first"] is False
    assert "equal weight" in r["rule"]


def test_ranking_by_single_result_and_winner_per_condition(store):
    for key in ("energy", "range", "cost"):
        r = designs.ranking(store, "EV01", key)
        assert r["rank_by"] == key and r["best_config_id"] == "EV01_OPT2"
    r = designs.ranking(store, "EV01")
    assert {w["condition_id"]: w["best_config_id"] for w in r["best_by_condition"]} == {
        "SLDC_Urban": "EV01_OPT1", "SLDC_Expressway": "EV01_OPT2"}
    with pytest.raises(ValueError):
        designs.ranking(store, "EV01", "speed")


def test_designs_endpoint(client):
    j = client.get("/api/designs", params={"ev_id": "EV01"}).json()
    assert j["best_config_id"] == "EV01_OPT2" and len(j["designs"]) == 2
    assert client.get("/api/designs", params={"ev_id": "EV01", "rank_by": "speed"}).status_code == 422
    assert client.get("/api/designs", params={"ev_id": "EV99"}).status_code == 404


def test_comparison_also_gives_the_baseline_under_wltp(store):
    r = comparison.compare(store, "EV01", "SLDC_Urban")
    b = r["benchmark"]
    assert b["condition_id"] == "WLTP"
    # by hand: baseline 152.0 Wh/km under WLTP, 138.0 under Urban
    assert b["baseline"]["energy_wh_per_km"] == 152.0
    assert b["difference"]["baseline_energy_wh_per_km"] == pytest.approx(138.0 - 152.0)
    assert b["difference"]["baseline_energy_pct"] == pytest.approx(-14.0 / 152.0 * 100)
    assert "optimized" not in b