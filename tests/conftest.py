"""Shared pytest fixtures.

Loads .env so the integration tests can pick up CLOVA_API_KEY without the
caller exporting it explicitly. Unit tests stub network calls and don't need
a key — but `model_client._get_client` reads the env var at construction
time, so we still want it available when an integration run is requested.
"""
from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

load_dotenv()


def _has_clova_key() -> bool:
    key = os.environ.get("CLOVA_API_KEY", "")
    return bool(key) and not key.startswith("nv-xxxx")


@pytest.fixture
def has_clova_key() -> bool:
    return _has_clova_key()


@pytest.fixture(autouse=True)
def _reset_model_client_cache():
    """Clients are cached per-process; reset between tests so respx mocks
    on `httpx.AsyncClient` aren't shared across cases."""
    from core import model_client as mc
    mc._CLIENT_CACHE.clear()
    yield
    mc._CLIENT_CACHE.clear()


def pytest_collection_modifyitems(config, items):
    """Auto-skip integration tests when CLOVA_API_KEY isn't usable."""
    if _has_clova_key():
        return
    skip = pytest.mark.skip(
        reason="CLOVA_API_KEY not set (or placeholder); skipping integration"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
