"""Shared pytest fixtures.

Loads .env so the integration tests can pick up CLOVA_API_KEY / UPSTAGE_API_KEY
without the caller exporting them explicitly. Unit tests stub network calls
and don't need a key — but `model_client._get_client` reads the env var at
construction time, so we still want it available when an integration run is
requested.

Integration tests are tagged with provider-specific markers so a missing
key for one provider doesn't skip the other:

    pytest -m "integration and naver"
    pytest -m "integration and upstage"
"""
from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

load_dotenv()


def _has_clova_key() -> bool:
    key = os.environ.get("CLOVA_API_KEY", "")
    return bool(key) and not key.startswith("nv-xxxx")


def _has_upstage_key() -> bool:
    key = os.environ.get("UPSTAGE_API_KEY", "")
    return bool(key) and not key.startswith("up-xxxx")


@pytest.fixture
def has_clova_key() -> bool:
    return _has_clova_key()


@pytest.fixture
def has_upstage_key() -> bool:
    return _has_upstage_key()


@pytest.fixture(autouse=True)
def _reset_model_client_cache():
    """Clients are cached per-process; reset between tests so respx mocks
    on `httpx.AsyncClient` aren't shared across cases."""
    from core import model_client as mc
    mc._CLIENT_CACHE.clear()
    yield
    mc._CLIENT_CACHE.clear()


def pytest_collection_modifyitems(config, items):
    """Auto-skip integration tests when their provider's key is missing."""
    has_clova = _has_clova_key()
    has_upstage = _has_upstage_key()

    skip_clova = pytest.mark.skip(
        reason="CLOVA_API_KEY not set (or placeholder)"
    )
    skip_upstage = pytest.mark.skip(
        reason="UPSTAGE_API_KEY not set (or placeholder)"
    )

    for item in items:
        if "integration" not in item.keywords:
            continue
        # A test can opt in to either provider via its own pytest marker.
        if "naver" in item.keywords and not has_clova:
            item.add_marker(skip_clova)
        if "upstage" in item.keywords and not has_upstage:
            item.add_marker(skip_upstage)
        # Backward compat: integration tests with no provider tag fall back
        # to requiring CLOVA (the original behavior).
        if "naver" not in item.keywords and "upstage" not in item.keywords:
            if not has_clova:
                item.add_marker(skip_clova)
