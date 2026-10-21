"""Calls the C3 service over HTTP (mock service now, real C3 later)."""
from typing import Optional

import httpx
from pydantic import ValidationError

from ..schemas import Catalog, PredictResponse
from .base import C3Client, C3Error


class ApiC3Client(C3Client):
    mode = "api"

    def __init__(self, base_url: str, timeout_s: float = 5.0,
                 client: Optional[httpx.Client] = None):
        # `client` lets tests inject an in-process client (no real network)
        self._client = client or httpx.Client(base_url=base_url, timeout=timeout_s)
        self.base_url = base_url

    def _request(self, method: str, path: str, **kw):
        last = None
        for _ in range(2):  # one retry for a short network problem
            try:
                resp = self._client.request(method, path, **kw)
                break
            except httpx.TransportError as e:
                last = e
        else:
            raise C3Error(f"Cannot reach C3 service at {self.base_url}: {last}")
        if resp.status_code >= 400:
            raise C3Error(f"C3 service returned {resp.status_code} for {path}: {resp.text[:200]}")
        try:
            return resp.json()
        except ValueError as e:
            raise C3Error(f"C3 service sent invalid JSON for {path}: {e}")

    def get_catalog(self) -> Catalog:
        data = self._request("GET", "/v1/catalog")
        try:
            return Catalog.model_validate(data)
        except ValidationError as e:
            raise C3Error(f"C3 catalog does not match the agreed format: {e}")

    def predict(self, ev_id: str, condition_id: str) -> PredictResponse:
        data = self._request("POST", "/v1/predict",
                             json={"ev_id": ev_id, "condition_id": condition_id})
        try:
            return PredictResponse.model_validate(data)
        except ValidationError as e:
            raise C3Error(f"C3 prediction does not match the agreed format: {e}")