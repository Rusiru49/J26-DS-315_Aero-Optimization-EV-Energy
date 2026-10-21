"""Backend settings, read from environment variables (see .env.example)."""
import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_list(name: str, default: str) -> list[str]:
    return [x.strip() for x in os.getenv(name, default).split(",") if x.strip()]


@dataclass
class Settings:
    # "api"  -> call the C3 service over HTTP (mock now, real later: only the URL changes)
    # "file" -> read a C3 package file directly (fallback if no service is running)
    c3_mode: str = field(default_factory=lambda: os.getenv("C3_MODE", "api").lower())
    c3_api_url: str = field(default_factory=lambda: os.getenv("C3_API_URL", "http://127.0.0.1:8001"))
    c3_timeout_s: float = field(default_factory=lambda: float(os.getenv("C3_TIMEOUT_S", "5")))
    data_file: Path = field(default_factory=lambda: Path(os.getenv(
        "DATA_FILE", str(PROJECT_ROOT / "data" / "c3_package_dummy.json"))))
    cors_origins: list[str] = field(default_factory=lambda: _env_list(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"))

    # Benchmark condition used for "WLTP vs local" comparison
    benchmark_condition: str = "WLTP"

    # Minimum data for each test = the method's own mathematical minimum
    # (regression: needs more rows than predictors + 1, checked in the code)
    min_pairs_ttest: int = 2
    min_groups_anova: int = 2
    min_per_group_anova: int = 2
    min_points_correlation: int = 3
    alpha: float = 0.05

    # Validation tolerances
    # Tolerances only allow for rounding in C3's numbers (values are rounded to 1-2 decimals)
    range_tolerance: float = 0.01      # relative, reported range vs battery x 1000 / energy
    cost_tolerance: float = 0.01       # relative, reported cost vs energy x tariff / efficiency
    crosscheck_tolerance: float = 0.06  # absolute, C4's savings vs C3's own savings


def get_settings() -> Settings:
    return Settings()