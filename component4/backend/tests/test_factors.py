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


def test_layer_1_states_facts_without_negative_labels(store):
    r = factors.factor_analysis(store)
    assert r["variation"] is None                     # still returned for the API; the page shows a table instead
    text = " ".join(r["insights"])
    assert "highest under Expressway" in text and "lowest under Urban" in text
    assert "does not prove the cause" in text
    # only 2 conditions: no speed link is claimed
    assert "average speed" not in text
    # no claim that the larger drag reduction gives the larger saving (only 1 value per condition)
    assert "larger reduction" not in text
    assert "cannot be separated" not in text and "told apart" not in text


def test_untested_factors_are_listed_with_reason(store):
    r = factors.factor_analysis(store)
    assert all(not f["tested"] and "EV" in f["reason"] for f in r["vehicle_factors"])
    assert {f["factor"] for f in r["vehicle_factors"]} == {"Vehicle mass", "Battery size"}
    assert "synthetic" not in str(r).lower()


def test_endpoint(client):
    j = client.get("/api/factors").json()
    assert j["available"] and len(j["insights"]) >= 2

def test_explain_layer_uses_only_data_and_c3_notes(store):
    j = factors.factor_analysis(store)
    ex = {e["factor"]: e for e in j["explain"]}
    # drag area: both designs listed with their numbers, and not separable yet (one condition per design)
    d = ex["Cd and frontal area (drag area)"]
    assert d["status"] == "not_separable" and "EV01_OPT1" in d["text"] and "EV01_OPT2" in d["text"]
    # driving condition: both conditions listed
    c = ex["Driving condition (speed, idle share, distance)"]
    assert "Urban" in c["text"] and "Expressway" in c["text"]
    # C3 settings come straight from the model notes of the data file
    assert ex["Regenerative braking"]["status"] == "fixed" and "included" in ex["Regenerative braking"]["text"]
    assert "not included" in ex["Auxiliary load"]["text"]
    assert ex["Road gradient"]["status"] == "fixed"
    # one EV only: vehicle factors are not tested, and the reason is given
    assert ex["Vehicle mass"]["status"] == "needs_data" and "EV" in ex["Vehicle mass"]["text"]
    assert ex["Electricity tariff"]["status"] == "cost_only"
    assert "Not available yet" in j["explain_later"]

def test_layer_1_speed_link_appears_with_three_conditions(make_store):
    def add_suburban(p):
        for c in p["conditions"]:
            if c["condition_id"] == "SLDC_Suburban":
                c.update(avg_speed_kmh=50.0, max_speed_kmh=80.0, idle_fraction=0.1, distance_km=15.0)
        ev = p["evs"][0]
        ev["configurations"].append({
            "config_id": "EV01_OPT3", "type": "optimized", "optimized_from": "EV01_BASE",
            "optimized_for": "SLDC_Suburban", "frontal_area_m2": 2.28, "cd": 0.205, "cd_uncertainty": 0.004,
            "shape_parameters": dict(ev["configurations"][1]["shape_parameters"]), "geometry_file": "EV01_OPT3.glb"})
        for cfg, e in (("EV01_BASE", 160.0), ("EV01_OPT3", 150.0)):
            p["results"].append({"ev_id": "EV01", "config_id": cfg, "condition_id": "SLDC_Suburban",
                                 "energy_wh_per_km": e, "range_km": round(80000 / e, 1),
                                 "cost_lkr_per_100km": round(e / 1000 * 100 * 45 / 0.9, 2)})
    r = factors.factor_analysis(make_store(add_suburban))
    assert r["n_conditions"] == 3
    text = " ".join(r["insights"])
    assert "Across the 3 driving conditions" in text and "average speed" in text