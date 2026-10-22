from fastapi import Request

from .config import Settings
from .services.store import DataStore


def get_store(request: Request) -> DataStore:
    return request.app.state.store


def get_settings(request: Request) -> Settings:
    return request.app.state.settings