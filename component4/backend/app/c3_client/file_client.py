"""Reads a C3 package file directly. Fallback for demos when no C3 service is running.

Gives exactly the same answers as the mock service (same package, same format).
"""
import datetime
import json
from pathlib import Path

from pydantic import ValidationError

from ..schemas import Catalog, PredictResponse
from .base import C3Client, C3Error


class FileC3Client(C3Client):
    mode = "file"

    def __init__(self, path: Path):
        self.path = Path(path)

    def _load(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise C3Error(f"Cannot read C3 package file {self.path}: {e}")

    def get_catalog(self) -> Catalog:
        pkg = self._load()
        data = {k: v for k, v in pkg.items() if k not in ("results", "comparisons")}
        try:
            return Catalog.model_validate(data)
        except ValidationError as e:
            raise C3Error(f"C3 package does not match the agreed format: {e}")

    def predict(self, ev_id: str, condition_id: str) -> PredictResponse:
        pkg = self._load()
        ev = next((e for e in pkg.get("evs", []) if e["ev_id"] == ev_id), None)
        if ev is None:
            raise C3Error(f"Unknown ev_id '{ev_id}'")
        if not any(c["condition_id"] == condition_id for c in pkg.get("conditions", [])):
            raise C3Error(f"Unknown condition_id '{condition_id}'")
        cfg = {c["config_id"]: c for c in ev["configurations"]}
        results = []
        for r in pkg.get("results", []):
            if r["ev_id"] == ev_id and r["condition_id"] == condition_id:
                c = cfg[r["config_id"]]
                results.append({
                    "config_id": r["config_id"], "cd": c["cd"],
                    "frontal_area_m2": c["frontal_area_m2"],
                    "energy_wh_per_km": r["energy_wh_per_km"], "range_km": r["range_km"],
                    "cost_lkr_per_100km": r["cost_lkr_per_100km"],
                })
        comps = [c for c in pkg.get("comparisons", [])
                 if c["ev_id"] == ev_id and c["condition_id"] == condition_id]
        try:
            return PredictResponse.model_validate({
                "ev_id": ev_id, "condition_id": condition_id, "available": bool(results),
                "computed_at": datetime.datetime.now().isoformat(timespec="seconds"),
                "results": results, "comparisons": comps,
            })
        except ValidationError as e:
            raise C3Error(f"C3 package rows do not match the agreed format: {e}")