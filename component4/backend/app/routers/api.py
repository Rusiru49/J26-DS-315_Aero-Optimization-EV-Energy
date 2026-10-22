"""All API routes (prefix /api). The frontend only talks to these."""
from typing import Optional

from fastapi import APIRouter, Depends, Query

from ..config import Settings
from ..deps import get_settings, get_store
from ..services import analysis, comparison, designs, factors, statistics, validation
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
def conditions_compare(ev_id: str, conditions: Optional[str] = None,
                       store: DataStore = Depends(get_store),
                       settings: Settings = Depends(get_settings)):
    """`conditions` = comma-separated condition ids to compare (default: all that have results)."""
    ids = [c for c in conditions.split(",") if c] if conditions else None
    return analysis.condition_comparison(store, ev_id, settings.benchmark_condition, ids)


@router.get("/sensitivity")
def get_sensitivity(ev_id: str, condition_id: str, optimized_config_id: Optional[str] = None,
                    tariff_change_pct: float = Query(0.0, ge=-50, le=100),
                    store: DataStore = Depends(get_store)):
    return analysis.sensitivity(store, ev_id, condition_id, optimized_config_id,
                                tariff_change_pct)


# ---- best design (ranking of the optimized designs) --------------------------
@router.get("/designs")
def get_designs(ev_id: str, rank_by: str = "overall", store: DataStore = Depends(get_store)):
    return designs.ranking(store, ev_id, rank_by)


# ---- what affects the saving (plain view for engineers) ----------------------
@router.get("/factors")
def get_factors(store: DataStore = Depends(get_store)):
    return factors.factor_analysis(store)


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