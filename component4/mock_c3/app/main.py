"""Mock C3 service (DUMMY).

Behaves like the real C3 prediction service will behave, using the dummy package.
C4 backend talks to this over HTTP, so going live later = change C3_API_URL only.

Endpoints
  GET  /v1/health
  GET  /v1/catalog   static part: EVs, configurations, shape definitions, conditions, model notes, units
  POST /v1/predict   {"ev_id": "...", "condition_id": "..."} -> results for baseline + optimized

Run (from c4-platform/mock_c3):
  uvicorn app.main:app --port 8001
"""
import datetime
import json
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

DEFAULT_FILE = Path(__file__).resolve().parents[2] / "data" / "c3_package_dummy.json"
DATA_FILE = Path(os.getenv("C3_MOCK_DATA_FILE", str(DEFAULT_FILE)))
LATENCY_MS = int(os.getenv("C3_MOCK_LATENCY_MS", "0"))

app = FastAPI(title="Mock C3 service (dummy data)", version="0.1")
_PKG = json.loads(DATA_FILE.read_text(encoding="utf-8"))


class PredictRequest(BaseModel):
    ev_id: str
    condition_id: str


@app.get("/v1/health")
def health():
    return {"status": "ok", "service": "mock-c3", "package_version": _PKG.get("package_version"),
            "dummy": True}


@app.get("/v1/catalog")
def catalog():
    """Everything that does not depend on a selection."""
    return {k: v for k, v in _PKG.items() if k not in ("results", "comparisons")}


@app.post("/v1/predict")
def predict(req: PredictRequest):
    if LATENCY_MS:
        time.sleep(LATENCY_MS / 1000)
    ev = next((e for e in _PKG["evs"] if e["ev_id"] == req.ev_id), None)
    if ev is None:
        raise HTTPException(404, f"Unknown ev_id '{req.ev_id}'")
    if not any(c["condition_id"] == req.condition_id for c in _PKG["conditions"]):
        raise HTTPException(404, f"Unknown condition_id '{req.condition_id}'")

    cfg_by_id = {c["config_id"]: c for c in ev["configurations"]}
    rows = [r for r in _PKG["results"]
            if r["ev_id"] == req.ev_id and r["condition_id"] == req.condition_id]
    results = []
    for r in rows:
        cfg = cfg_by_id[r["config_id"]]
        results.append({
            "config_id": r["config_id"],
            "cd": cfg["cd"],
            "frontal_area_m2": cfg["frontal_area_m2"],
            "energy_wh_per_km": r["energy_wh_per_km"],
            "range_km": r["range_km"],
            "cost_lkr_per_100km": r["cost_lkr_per_100km"],
        })
    comps = [c for c in _PKG.get("comparisons", [])
             if c["ev_id"] == req.ev_id and c["condition_id"] == req.condition_id]
    return {
        "ev_id": req.ev_id,
        "condition_id": req.condition_id,
        "available": bool(results),   # False = condition exists but C3 has no results yet
        "computed_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "results": results,
        "comparisons": comps,
    }