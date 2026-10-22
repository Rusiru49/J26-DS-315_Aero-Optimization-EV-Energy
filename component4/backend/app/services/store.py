"""In-memory store of everything C3 sent.

At start (and on refresh) we ask C3 for the catalog and for every EV x condition,
and keep the answers in memory. Single selections are answered from memory (fast),
and the statistics can see the whole grid at once (they need all combinations).
No database is used (see report: results are processed at runtime).
"""
import threading
import time
from datetime import datetime
from typing import Optional

from ..c3_client import C3Client, C3Error
from ..schemas import Catalog, Condition, Configuration, EV, PredictResponse, ResultRow

CONDITION_LABELS = {
    "WLTP": "WLTP (standard)",
    "WLTP_Urban": "WLTP (for Urban)",
    "WLTP_Expressway": "WLTP (for Expressway)",
    "SLDC_Urban": "Urban",
    "SLDC_Suburban": "Sub Urban",
    "SLDC_Intercity": "Intercity",
    "SLDC_Expressway": "Expressway (Highway)",
    "SLDC_Rural": "Rural",
}


class NotFound(Exception):
    pass


class DataStore:
    def __init__(self, client: C3Client):
        self.client = client
        self._lock = threading.Lock()
        self.catalog: Optional[Catalog] = None
        self.predictions: dict[tuple[str, str], PredictResponse] = {}
        self.loaded_at: Optional[str] = None
        self.load_seconds: Optional[float] = None
        self.last_error: Optional[str] = None

    # ---------- loading ----------
    def load(self) -> None:
        """Fetch catalog + all EV x condition predictions. Raises C3Error if C3 fails."""
        t0 = time.perf_counter()
        try:
            catalog = self.client.get_catalog()
            predictions = {}
            for ev in catalog.evs:
                for cond in catalog.conditions:
                    predictions[(ev.ev_id, cond.condition_id)] = \
                        self.client.predict(ev.ev_id, cond.condition_id)
        except C3Error as e:
            self.last_error = str(e)
            raise
        with self._lock:
            self.catalog, self.predictions = catalog, predictions
            self.loaded_at = datetime.now().isoformat(timespec="seconds")
            self.load_seconds = round(time.perf_counter() - t0, 3)
            self.last_error = None

    def ensure_loaded(self) -> None:
        if self.catalog is None:
            self.load()

    @property
    def is_loaded(self) -> bool:
        return self.catalog is not None

    # ---------- lookups ----------
    def get_ev(self, ev_id: str) -> EV:
        self.ensure_loaded()
        for ev in self.catalog.evs:
            if ev.ev_id == ev_id:
                return ev
        raise NotFound(f"Unknown EV '{ev_id}'")

    def get_condition(self, condition_id: str) -> Condition:
        self.ensure_loaded()
        for c in self.catalog.conditions:
            if c.condition_id == condition_id:
                return c
        raise NotFound(f"Unknown driving condition '{condition_id}'")

    def baseline_config(self, ev_id: str) -> Configuration:
        ev = self.get_ev(ev_id)
        for c in ev.configurations:
            if c.type == "baseline":
                return c
        raise NotFound(f"EV '{ev_id}' has no baseline configuration")

    def optimized_configs(self, ev_id: str) -> list[Configuration]:
        return [c for c in self.get_ev(ev_id).configurations if c.type == "optimized"]

    def optimized_config(self, ev_id: str, config_id: Optional[str] = None,
                         for_condition: Optional[str] = None) -> Configuration:
        """The optimized design asked for. With no id: the design found for `for_condition`
        (if there is one), otherwise the first optimized design."""
        opts = self.optimized_configs(ev_id)
        if not opts:
            raise NotFound(f"EV '{ev_id}' has no optimized configuration")
        if config_id is None:
            if for_condition:
                for c in opts:
                    if c.optimized_for == for_condition:
                        return c
            return opts[0]
        for c in opts:
            if c.config_id == config_id:
                return c
        raise NotFound(f"Unknown optimized configuration '{config_id}' for EV '{ev_id}'")

    def prediction(self, ev_id: str, condition_id: str) -> PredictResponse:
        self.get_ev(ev_id)
        self.get_condition(condition_id)
        return self.predictions[(ev_id, condition_id)]

    def row(self, ev_id: str, condition_id: str, config_id: str) -> Optional[ResultRow]:
        for r in self.prediction(ev_id, condition_id).results:
            if r.config_id == config_id:
                return r
        return None

    def is_available(self, condition_id: str) -> bool:
        """A condition is available when C3 sent results for it (for any EV)."""
        self.ensure_loaded()
        return any(p.available and p.results
                   for (_, cid), p in self.predictions.items() if cid == condition_id)

    def benchmark_for(self, condition_id: str) -> Optional[Condition]:
        """The standard-cycle (WLTP) entry to compare the baseline with, for a driving condition.
        First choice: the entry made for this condition (`benchmark_of`). Otherwise a single
        standard entry that is for all conditions. A standard entry itself has no benchmark."""
        cond = self.get_condition(condition_id)
        if cond.group == "standard":
            return None
        standard = [c for c in self.available_conditions() if c.group == "standard"]
        for c in standard:
            if c.benchmark_of == condition_id:
                return c
        for c in standard:
            if not c.benchmark_of:
                return c
        return None

    def available_conditions(self) -> list[Condition]:
        self.ensure_loaded()
        return [c for c in self.catalog.conditions if self.is_available(c.condition_id)]

    def coming_soon_conditions(self) -> list[Condition]:
        self.ensure_loaded()
        return [c for c in self.catalog.conditions if not self.is_available(c.condition_id)]

    def pairs(self) -> list[dict]:
        """All baseline/optimized pairs: EV x optimized config x available condition."""
        self.ensure_loaded()
        out = []
        for ev in self.catalog.evs:
            base = self.baseline_config(ev.ev_id)
            for opt in self.optimized_configs(ev.ev_id):
                for cond in self.available_conditions():
                    rb = self.row(ev.ev_id, cond.condition_id, base.config_id)
                    ro = self.row(ev.ev_id, cond.condition_id, opt.config_id)
                    if rb is None or ro is None:
                        continue
                    out.append({"ev_id": ev.ev_id, "optimized_config_id": opt.config_id,
                                "condition_id": cond.condition_id, "base": rb, "opt": ro})
        return out

    def meta(self) -> dict:
        self.ensure_loaded()
        notes = self.catalog.model_notes
        return {
            "package_version": self.catalog.package_version,
            "generated_on": self.catalog.generated_on,
            "package_status": self.catalog.status,
            "is_dummy": bool(notes.is_dummy),
            "data_origin": notes.data_origin or ("DUMMY" if notes.is_dummy else "C3"),
            "predicted_only": not notes.experimentally_validated,
            "source_mode": self.client.mode,
            "loaded_at": self.loaded_at,
            "load_seconds": self.load_seconds,
        }