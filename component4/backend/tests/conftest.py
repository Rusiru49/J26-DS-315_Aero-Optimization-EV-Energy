import copy
import importlib.util
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.c3_client import ApiC3Client, FileC3Client
from app.config import Settings
from app.main import create_app
from app.services.store import DataStore

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "c3_package_dummy.json"


@pytest.fixture(scope="session")
def raw_package() -> dict:
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))


@pytest.fixture()
def settings() -> Settings:
    return Settings(c3_mode="file", data_file=DATA_FILE)


@pytest.fixture()
def store(settings) -> DataStore:
    s = DataStore(FileC3Client(DATA_FILE))
    s.load()
    return s


@pytest.fixture()
def make_store(tmp_path, raw_package):
    """Build a store from a modified copy of the dummy package: make_store(lambda pkg: ...)."""
    def _make(mutate) -> DataStore:
        pkg = copy.deepcopy(raw_package)
        mutate(pkg)
        f = tmp_path / "pkg.json"
        f.write_text(json.dumps(pkg), encoding="utf-8")
        s = DataStore(FileC3Client(f))
        s.load()
        return s
    return _make


@pytest.fixture()
def client(settings):
    with TestClient(create_app(settings)) as c:
        yield c


def load_mock_app():
    """Import the mock C3 service under another module name (it is also called 'app')."""
    path = ROOT / "mock_c3" / "app" / "main.py"
    spec = importlib.util.spec_from_file_location("mock_c3_main", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.app


@pytest.fixture()
def api_mode_client(settings):
    """C4 backend whose C3 client talks HTTP (in-process) to the mock C3 service."""
    api_client = ApiC3Client("http://mock-c3", client=TestClient(load_mock_app(),
                                                                 base_url="http://mock-c3"))
    s = Settings(c3_mode="api", data_file=DATA_FILE)
    with TestClient(create_app(s, client=api_client)) as c:
        yield c