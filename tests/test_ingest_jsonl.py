"""Unit tests for ingest.ingest_jsonl helpers.

Network and Qdrant are stubbed out: the goal is to exercise the retry,
pagination, and error-handling logic without external dependencies.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from ingest import ingest_jsonl as ij


def _http_error(status: int, *, retry_after: str | None = None) -> httpx.HTTPStatusError:
    req = httpx.Request("POST", "https://example/embed")
    headers = {"retry-after": retry_after} if retry_after else {}
    resp = httpx.Response(status, request=req, headers=headers)
    return httpx.HTTPStatusError("boom", request=req, response=resp)


# ─── _embed_with_retry ──────────────────────────────────────────────────────

async def test_embed_with_retry_succeeds_first_try(monkeypatch):
    calls = 0

    async def fake_model_call(provider, task, **kw):
        nonlocal calls
        calls += 1
        return [0.1] * 1024

    monkeypatch.setattr(ij, "model_call", fake_model_call)

    vec = await ij._embed_with_retry("naver", "hello")
    assert len(vec) == 1024
    assert calls == 1


async def test_embed_with_retry_recovers_from_429(monkeypatch):
    seq = [_http_error(429), _http_error(429), [0.5] * 1024]

    async def fake_model_call(provider, task, **kw):
        item = seq.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    # Skip real backoff sleeps to keep the test fast.
    async def no_sleep(_s):
        return None

    monkeypatch.setattr(ij, "model_call", fake_model_call)
    monkeypatch.setattr(ij.asyncio, "sleep", no_sleep)

    vec = await ij._embed_with_retry("naver", "hello")
    assert len(vec) == 1024
    assert seq == []  # all three responses consumed


async def test_embed_with_retry_honors_retry_after_header(monkeypatch):
    waits: list[float] = []
    seq = [_http_error(429, retry_after="2"), [0.0] * 4]

    async def fake_model_call(provider, task, **kw):
        item = seq.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def capture_sleep(s):
        waits.append(float(s))

    monkeypatch.setattr(ij, "model_call", fake_model_call)
    monkeypatch.setattr(ij.asyncio, "sleep", capture_sleep)

    await ij._embed_with_retry("naver", "x")
    assert waits == [2.0]


async def test_embed_with_retry_exhausts_retries(monkeypatch):
    async def always_429(provider, task, **kw):
        raise _http_error(429)

    async def no_sleep(_s):
        return None

    monkeypatch.setattr(ij, "model_call", always_429)
    monkeypatch.setattr(ij.asyncio, "sleep", no_sleep)

    with pytest.raises(httpx.HTTPStatusError) as exc:
        await ij._embed_with_retry("naver", "x")
    assert exc.value.response.status_code == 429


async def test_embed_with_retry_propagates_4xx_other_than_429(monkeypatch):
    async def fake(provider, task, **kw):
        raise _http_error(400)

    monkeypatch.setattr(ij, "model_call", fake)

    with pytest.raises(httpx.HTTPStatusError) as exc:
        await ij._embed_with_retry("naver", "x")
    assert exc.value.response.status_code == 400


# ─── _existing_ids ──────────────────────────────────────────────────────────

async def test_existing_ids_paginates_until_offset_none():
    page1 = [SimpleNamespace(id="a"), SimpleNamespace(id="b")]
    page2 = [SimpleNamespace(id="c")]

    qdrant = SimpleNamespace()
    qdrant.scroll = AsyncMock(side_effect=[(page1, "cursor-1"), (page2, None)])

    ids = await ij._existing_ids(qdrant, "msds_corpus_naver")
    assert ids == {"a", "b", "c"}
    assert qdrant.scroll.await_count == 2


async def test_existing_ids_handles_empty_collection():
    qdrant = SimpleNamespace()
    qdrant.scroll = AsyncMock(return_value=([], None))

    ids = await ij._existing_ids(qdrant, "empty")
    assert ids == set()
    assert qdrant.scroll.await_count == 1
