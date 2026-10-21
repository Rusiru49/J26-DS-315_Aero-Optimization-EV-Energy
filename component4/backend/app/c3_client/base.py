"""One interface for talking to C3. The rest of the backend only uses this."""
from abc import ABC, abstractmethod

from ..schemas import Catalog, PredictResponse


class C3Error(Exception):
    """C3 could not be reached, or sent data that does not match the agreed format."""


class C3Client(ABC):
    mode: str = "unknown"

    @abstractmethod
    def get_catalog(self) -> Catalog:
        """EVs, configurations, shape definitions, conditions, model notes, units."""

    @abstractmethod
    def predict(self, ev_id: str, condition_id: str) -> PredictResponse:
        """Baseline + optimized results for one EV under one condition."""