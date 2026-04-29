"""Single-call live tests against Upstage Solar.

Skipped automatically when UPSTAGE_API_KEY is missing or still the placeholder.
Each test makes one API call to keep the rate-limit blast radius tiny.

Run with:
    pytest -m "integration and upstage"
"""
from __future__ import annotations

import pytest

from core.clients.upstage import UpstageClient

pytestmark = [pytest.mark.integration, pytest.mark.upstage]


@pytest.fixture
async def client():
    c = UpstageClient()
    try:
        yield c
    finally:
        await c.aclose()


async def test_embed_passage_returns_4096_dim_vector(client: UpstageClient):
    vec = await client.embed("에탄올의 인화점", role="passage")
    assert isinstance(vec, list)
    assert len(vec) == 4096, f"expected 4096-dim, got {len(vec)}"
    # JSON decodes 0 as int; the rest are floats. Accept both.
    assert all(isinstance(x, (int, float)) for x in vec)


async def test_embed_query_returns_4096_dim_vector(client: UpstageClient):
    # Query model produces vectors compatible with the passage collection
    # (same dim, same space) — that's the whole point of the role split.
    vec = await client.embed("에탄올 독성", role="query")
    assert len(vec) == 4096


async def test_rerank_is_local_identity_passthrough(client: UpstageClient):
    # Upstage doesn't expose a public rerank endpoint; the client implements
    # rerank as a no-op passthrough on top of ANN ordering. This test pins
    # that — no network call, deterministic result.
    docs = ["a", "b", "c"]
    out = await client.rerank("anything", docs, top_n=2)
    assert out == [
        {"index": 0, "score": 1.0},
        {"index": 1, "score": 0.99},
    ]


async def test_chat_returns_nonempty_string(client: UpstageClient):
    answer = await client.chat(
        system="당신은 간결하게 답하는 화학 전문가입니다.",
        user="에탄올의 화학식을 한 단어로 답하세요.",
        max_tokens=64,
    )
    assert isinstance(answer, str)
    assert len(answer.strip()) > 0
