"""Data models for what C3 sends (the agreed data contract).

Parsing C3 data through these models is the first data-integrity gate (NFR4):
wrong types or missing required fields are rejected before any analysis.
Extra fields are allowed so C3 can add information without breaking C4.
"""
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(extra="allow")


class Configuration(_Base):
    config_id: str
    type: Literal["baseline", "optimized"]
    optimized_from: Optional[str] = None
    optimized_for: Optional[str] = None   # the driving condition this design was found for
    frontal_area_m2: float
    cd: float
    cd_uncertainty: Optional[float] = None
    shape_parameters: dict[str, float] = Field(default_factory=dict)
    geometry_file: Optional[str] = None


class EV(_Base):
    ev_id: str
    ev_model: str
    mass_kg: float
    usable_battery_kwh: float
    configurations: list[Configuration]


class ShapeParamDef(_Base):
    name: str
    description: str = ""
    unit: str = ""
    min: float
    max: float
    baseline: Optional[float] = None
    effect_on_geometry: str = ""


class Condition(_Base):
    condition_id: str
    group: str
    avg_speed_kmh: Optional[float] = None
    max_speed_kmh: Optional[float] = None
    idle_fraction: Optional[float] = None
    distance_km: Optional[float] = None
    # For a standard-cycle entry (group "standard", e.g. WLTP): the driving condition it is the
    # benchmark of (e.g. "SLDC_Urban"). Empty = one benchmark for all conditions.
    benchmark_of: Optional[str] = None


class ModelNotes(_Base):
    is_dummy: bool = False
    data_origin: Optional[str] = None
    energy_definition: Optional[str] = None
    range_definition: Optional[str] = None
    includes_regen_braking: Optional[bool] = None
    includes_auxiliary_load: Optional[bool] = None
    road_gradient: Optional[str] = None
    cd_source: Optional[str] = None
    optimized_cd_method: Optional[str] = None
    tariff_lkr_per_kwh: Optional[float] = None
    tariff_source: Optional[str] = None
    charging_efficiency: Optional[float] = None
    experimentally_validated: bool = False


class Catalog(_Base):
    package_version: str
    generated_on: Optional[str] = None
    status: str = "draft"
    evs: list[EV]
    shape_parameter_definitions: list[ShapeParamDef] = Field(default_factory=list)
    conditions: list[Condition]
    model_notes: ModelNotes = Field(default_factory=ModelNotes)
    units: dict[str, str] = Field(default_factory=dict)


class ResultRow(_Base):
    config_id: str
    cd: Optional[float] = None
    frontal_area_m2: Optional[float] = None
    energy_wh_per_km: float
    range_km: float
    cost_lkr_per_100km: float


class C3Comparison(_Base):
    ev_id: str
    baseline_config_id: str
    optimized_config_id: str
    condition_id: str
    energy_saving_wh_per_km: float
    energy_saving_pct: float
    range_gain_km: float
    cost_saving_lkr_per_100km: float
    saving_within_uncertainty: Optional[bool] = None


class PredictResponse(_Base):
    ev_id: str
    condition_id: str
    available: bool = True
    computed_at: Optional[str] = None
    results: list[ResultRow] = Field(default_factory=list)
    comparisons: list[C3Comparison] = Field(default_factory=list)