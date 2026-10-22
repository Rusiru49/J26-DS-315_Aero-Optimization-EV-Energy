"""All API routes (prefix /api). The frontend only talks to these."""
from typing import Optional

from fastapi import APIRouter, Depends, Query

from ..config import Settings
from ..deps import get_settings, get_store
from ..services import analysis, comparison, statistics, validation
from ..services.store import CONDITION_LABELS, DataStore

router = APIRouter(prefix="/api")


@router.get("/health")
def health(store: DataStore = Depends(get_store)):
    return {"status": "ok" if store.is_loaded else "c3_unavailable",
            "c3_mode": store.client.mode, "data_loaded": store.is_loaded,
            "loaded_at": store.loaded_at, "last_error": store.last_error}


@router.post("/refresh")
def refresh(store: DataStore = Depends(get_store)):
    """Re-fetch everything from C3 (use when C3 data changed)."""
    store.load()
    return {"status": "reloaded", **store.meta()}


@router.get("/catalog")
def catalog(store: DataStore = Depends(get_store)):
    """What the selectors and panels need: EVs, conditions (with availability), definitions, notes."""
    store.ensure_loaded()
    cat = store.catalog
    conditions = []
    for c in cat.conditions:
        avail = store.is_available(c.condition_id)
        conditions.append({**c.model_dump(),
                           "label": CONDITION_LABELS.get(c.condition_id, c.condition_id),
                           "available": avail,
                           "status": "available" if avail else "coming_soon"})
    return {
        "meta": store.meta(),
        "evs": [ev.model_dump() for ev in cat.evs],
        "conditions": conditions,
        "shape_parameter_definitions": [d.model_dump() for d in cat.shape_parameter_definitions],
        "model_notes": cat.model_notes.model_dump(),
        "units": cat.units,
    }


# ---- Function 1 -------------------------------------------------------------
@router.get("/comparison")
def get_comparison(ev_id: str, condition_id: str, optimized_config_id: Optional[str] = None,
                   store: DataStore = Depends(get_store)):
    return comparison.compare(store, ev_id, condition_id, optimized_config_id)


# ---- Function 2 -------------------------------------------------------------
@router.get("/conditions/compare")
def conditions_compare(ev_id: str, optimized_config_id: Optional[str] = None,
                       store: DataStore = Depends(get_store),
                       settings: Settings = Depends(get_settings)):
    return analysis.condition_comparison(store, ev_id, settings.benchmark_condition,
                                         optimized_config_id)


@router.get("/sensitivity")
def get_sensitivity(ev_id: str, condition_id: str, optimized_config_id: Optional[str] = None,
                    tariff_change_pct: float = Query(0.0, ge=-50, le=100),
                    store: DataStore = Depends(get_store)):
    return analysis.sensitivity(store, ev_id, condition_id, optimized_config_id,
                                tariff_change_pct)


@router.get("/mix")
def get_mix(ev_id: str, condition_a: str = "SLDC_Urban", condition_b: str = "SLDC_Expressway",
            share_a_pct: float = Query(50.0, ge=0, le=100),
            distance_km: float = Query(100.0, ge=0, le=1_000_000),
            optimized_config_id: Optional[str] = None,
            store: DataStore = Depends(get_store)):
    return analysis.mix(store, ev_id, condition_a, condition_b, share_a_pct,
                        distance_km, optimized_config_id)


# ---- data quality and statistics -------------------------------------------
@router.get("/data-quality")
def data_quality(store: DataStore = Depends(get_store),
                 settings: Settings = Depends(get_settings)):
    return {"meta": store.meta(), **validation.run_validation(store, settings)}


@router.get("/statistics/live")
def statistics_live(store: DataStore = Depends(get_store),
                    settings: Settings = Depends(get_settings)):
    return {"meta": store.meta(), **statistics.live_statistics(store, settings)}


@router.get("/statistics/sample")
def statistics_sample(settings: Settings = Depends(get_settings)):
    """The four methods run on a generated sample (6 conditions x 8 designs)."""
    return statistics.sample_statistics(settings)