"""HTTP-level unit tests for UpstageClient.

Mock Upstage's OpenAI-compatible endpoints with respx so we can lock down
the *single-call* contract of each method (URL, headers, JSON shape,
response parsing) without needing a live UPSTAGE_API_KEY. Mirrors
test_naver_client_unit.py for parity across providers.
"""
from __future__ import annotations

import json as _json

import httpx
import pytest
import respx

from core.clients.upstage import UpstageClient


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("UPSTAGE_API_KEY", "up-test-key")
    return UpstageClient()


@respx.mock
async def test_embed_passage_uses_passage_alias(client: UpstageClient):
    route = respx.post("https://api.upstage.ai/v1/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "index": 0,
                          "embedding": [0.1] * 4096}],
                "model": "embedding-passage",
                "usage": {"prompt_tokens": 4, "total_tokens": 4},
            },
        )
    )

    vec = await client.embed("에탄올", role="passage")

    assert route.called
    sent = route.calls.last.request
    assert sent.headers["authorization"] == "Bearer up-test-key"
    body = _json.loads(sent.content)
    assert body == {"model": "embedding-passage", "input": "에탄올"}

    assert isinstance(vec, list) and len(vec) == 4096

    await client.aclose()


@respx.mock
async def test_embed_query_uses_query_alias(client: UpstageClient):
    route = respx.post("https://api.upstage.ai/v1/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={"data": [{"embedding": [0.0] * 4096}]},
        )
    )

    await client.embed("질의", role="query")
    body = _json.loads(route.calls.last.request.content)
    assert body["model"] == "embedding-query"

    await client.aclose()


@respx.mock
async def test_rerank_is_identity_passthrough_with_no_http_call(
    client: UpstageClient,
):
    # Upstage has no public rerank endpoint; Qdrant ANN already does the
    # cosine ranking on normalized solar-embedding vectors. So rerank is a
    # local op — we assert that NO HTTP request goes out.
    rerank_route = respx.post("https://api.upstage.ai/v1/rerank")

    out = await client.rerank("에탄올 독성", ["a", "b", "c", "d"], top_n=2)

    assert rerank_route.called is False
    assert out == [
        {"index": 0, "score": 1.0},
        {"index": 1, "score": 0.99},
    ]

    await client.aclose()


@respx.mock
async def test_rerank_caps_at_top_n_and_at_input_length(
    client: UpstageClient,
):
    # top_n larger than documents → cap to len(documents).
    out = await client.rerank("q", ["d1", "d2"], top_n=10)
    assert [d["index"] for d in out] == [0, 1]

    out = await client.rerank("q", [], top_n=5)
    assert out == []

    await client.aclose()


@respx.mock
async def test_chat_defaults_to_solar_pro2_and_returns_content(
    client: UpstageClient,
):
    route = respx.post("https://api.upstage.ai/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "1",
                "object": "chat.completion",
                "created": 0,
                "model": "solar-pro2",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "C2H6O"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3,
                          "total_tokens": 13},
            },
        )
    )

    answer = await client.chat(
        system="간결하게 답해.", user="에탄올 화학식?", max_tokens=64
    )

    assert answer == "C2H6O"
    body = _json.loads(route.calls.last.request.content)
    assert body["model"] == "solar-pro2"
    assert body["messages"] == [
        {"role": "system", "content": "간결하게 답해."},
        {"role": "user", "content": "에탄올 화학식?"},
    ]
    assert body["max_tokens"] == 64
    assert body["temperature"] == 0.3
    assert body["top_p"] == 0.8
    # reasoning_effort is opt-in only — must not be present by default.
    assert "reasoning_effort" not in body

    await client.aclose()


@respx.mock
async def test_chat_forwards_reasoning_effort_when_requested(
    client: UpstageClient,
):
    route = respx.post("https://api.upstage.ai/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}]},
        )
    )

    await client.chat(system="s", user="u", reasoning_effort="high")
    body = _json.loads(route.calls.last.request.content)
    assert body["reasoning_effort"] == "high"

    await client.aclose()


@respx.mock
async def test_chat_uses_custom_model_when_provided(client: UpstageClient):
    route = respx.post("https://api.upstage.ai/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, json={"choices": [{"message": {"content": "ok"}}]}
        )
    )

    await client.chat(system="s", user="u", model="solar-pro3")
    body = _json.loads(route.calls.last.request.content)
    assert body["model"] == "solar-pro3"

    await client.aclose()


@respx.mock
async def test_embed_propagates_http_error(client: UpstageClient):
    respx.post("https://api.upstage.ai/v1/embeddings").mock(
        return_value=httpx.Response(429, headers={"retry-after": "1"})
    )

    with pytest.raises(httpx.HTTPStatusError) as exc:
        await client.embed("x", role="passage")
    assert exc.value.response.status_code == 429

    await client.aclose()


def test_init_requires_api_key(monkeypatch):
    monkeypatch.delenv("UPSTAGE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="UPSTAGE_API_KEY"):
        UpstageClient()
