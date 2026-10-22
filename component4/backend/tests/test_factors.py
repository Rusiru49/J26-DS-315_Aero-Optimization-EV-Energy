"""What affects the saving: tables, 'not tested yet' list and plain statements."""
import pytest

from app.services import factors


def test_tables_by_condition_and_design(store):
    r = factors.factor_analysis(store)
    assert r["n_pairs"] == 2 and r["n_conditions"] == 2 and r["n_designs"] == 2
    speeds = [c["avg_speed_kmh"] for c in r["by_condition"]]
    assert speeds == sorted(speeds) and speeds == [21.8, 94.5]
    assert all(d["n"] == 1 for d in r["by_design"])
    cda = [d["cda_reduction_pct"] for d in r["by_design"]]
    assert cda == sorted(cda)
    # by hand: OPT1 CdA = 0.208 x 2.28, baseline = 0.230 x 2.30
    opt1 = next(d for d in r["by_design"] if d["config_id"] == "EV01_OPT1")
    assert opt1["cda_reduction_pct"] == pytest.approx((0.230 * 2.30 - 0.208 * 2.28) / (0.230 * 2.30) * 100)


def test_variation_cannot_be_split_when_each_design_has_one_condition(store):
    r = factors.factor_analysis(store)
    assert r["variation"] is None and "one condition only" in r["variation_note"]
    text = " ".join(r["insights"])
    assert "cannot be separated" in text and "does not prove the cause" in text
    # no claim that the larger drag reduction gives the larger saving (only 1 value per condition)
    assert "larger reduction" not in text


def test_untested_factors_are_listed_with_reason(store):
    r = factors.factor_analysis(store)
    assert all(not f["tested"] and "EV" in f["reason"] for f in r["vehicle_factors"])
    assert {f["factor"] for f in r["vehicle_factors"]} == {"Vehicle mass", "Battery size"}
    assert "synthetic" not in str(r).lower()


def test_endpoint(client):
    j = client.get("/api/factors").json()
    assert j["available"] and len(j["insights"]) >= 3