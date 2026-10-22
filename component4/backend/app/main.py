"""C4 backend entry point.

Run (from c4-platform/backend):
  uvicorn app.main:app --port 8000 --reload
"""
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .c3_client import C3Client, C3Error, make_client
from .config import Settings, get_settings
from .routers import api
from .services.store import DataStore, NotFound

log = logging.getLogger("c4")


def create_app(settings: Optional[Settings] = None, client: Optional[C3Client] = None) -> FastAPI:
    settings = settings or get_settings()
    store = DataStore(client or make_client(settings))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            store.load()
            log.info("Loaded C3 data (%s mode) in %ss", store.client.mode, store.load_seconds)
        except C3Error as e:  # keep running; endpoints retry and report a clear error
            log.warning("C3 not available at start: %s", e)
        yield

    app = FastAPI(title="C4 - EV aerodynamic impact analysis backend", version="0.1",
                  lifespan=lifespan)
    app.state.settings, app.state.store = settings, store

    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins,
                       allow_methods=["*"], allow_headers=["*"],
                       expose_headers=["X-Process-Time-ms"])

    @app.middleware("http")
    async def timing(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-ms"] = f"{(time.perf_counter() - t0) * 1000:.1f}"
        return response

    @app.exception_handler(NotFound)
    async def _not_found(_, exc: NotFound):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(C3Error)
    async def _c3(_, exc: C3Error):
        return JSONResponse(status_code=502, content={
            "detail": f"Prediction service (C3) problem: {exc}", "code": "c3_error"})

    @app.exception_handler(ValueError)
    async def _value(_, exc: ValueError):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    app.include_router(api.router)
    return app


app = create_app()