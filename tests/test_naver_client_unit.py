"""HTTP-level unit tests for NaverClovaClient.

These mock the CLOVA endpoints with respx so we can verify the *single-call*
contract of each method (request URL, payload shape, response parsing) without
needing a live CLOVA_API_KEY. Complements
test_naver_client_integration.py, which exercises the same surface against
the real API.
"""
from __future__ import annotations

import httpx
import pytest
import respx

from core.clients.naver import NaverClovaClient


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("CLOVA_API_KEY", "nv-test-key")
    c = NaverClovaClient()
    yield c
    # aclose is awaited by the test that needs it; pytest will GC the loop.


@respx.mock
async def test_embed_posts_text_and_returns_vector(client: NaverClovaClient):
    route = respx.post(
        "https://clovastudio.stream.ntruss.com/v1/api-tools/embedding/v2"
    ).mock(
        return_value=httpx.Response(
            200, json={"result": {"embedding": [0.1] * 1024}}
        )
    )

    vec = await client.embed("에탄올", role="passage")

    assert route.called
    sent = route.calls.last.request
    assert sent.headers["authorization"] == "Bearer nv-test-key"
    # Body is `{"text": "에탄올"}` — role is intentionally not forwarded.
    import json as _json
    body = _json.loads(sent.content)
    assert body == {"text": "에탄올"}

    assert isinstance(vec, list)
    assert len(vec) == 1024

    await client.aclose()


@respx.mock
async def test_rerank_adapts_cited_documents_to_index_score(client: NaverClovaClient):
    # CLOVA's reranker returns citedDocuments (no scalar relevance score).
    # The client should translate that back into the {index, score} shape
    # Retriever expects, with a monotonically descending pseudo-score.
    respx.post(
        "https://clovastudio.stream.ntruss.com/v1/api-tools/reranker"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "result": {
                    "citedDocuments": [
                        {"id": "2", "doc": "doc-C"},
                        {"id": "0", "doc": "doc-A"},
                    ]
                }
            },
        )
    )

    out = await client.rerank(
        "에탄올 독성", ["doc-A", "doc-B", "doc-C"], top_n=5
    )

    assert [d["index"] for d in out] == [2, 0]
    # Pseudo-score: descending, first item is exactly 1.0.
    assert out[0]["score"] == 1.0
    assert out[1]["score"] < out[0]["score"]

    await client.aclose()


@respx.mock
async def test_rerank_truncates_to_top_n(client: NaverClovaClient):
    respx.post(
        "https://clovastudio.stream.ntruss.com/v1/api-tools/reranker"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "result": {
                    "citedDocuments": [
                        {"id": str(i), "doc": f"d{i}"} for i in range(5)
                    ]
                }
            },
        )
    )

    out = await client.rerank("q", ["d0", "d1", "d2", "d3", "d4"], top_n=2)
    assert len(out) == 2
    assert [d["index"] for d in out] == [0, 1]

    await client.aclose()


@respx.mock
async def test_rerank_handles_empty_citations(client: NaverClovaClient):
    respx.post(
        "https://clovastudio.stream.ntruss.com/v1/api-tools/reranker"
    ).mock(return_value=httpx.Response(200, json={"result": {}}))

    out = await client.rerank("q", ["a", "b"], top_n=3)
    assert out == []

    await client.aclose()


@respx.mock
async def test_chat_posts_messages_and_returns_content(client: NaverClovaClient):
    # Default model is HCX-007 (reasoning); it expects `maxCompletionTokens`.
    route = respx.post(
        f"https://clovastudio.stream.ntruss.com/v3/chat-completions/{NaverClovaClient.CHAT_MODEL_DEFAULT}"
    ).mock(
        return_value=httpx.Response(
            200, json={"result": {"message": {"content": "C2H6O"}}}
        )
    )

    answer = await client.chat(
        system="간결하게 답해.", user="에탄올 화학식?", max_tokens=32
    )

    assert answer == "C2H6O"
    sent = route.calls.last.request
    import json as _json
    body = _json.loads(sent.content)
    assert body["messages"] == [
        {"role": "system", "content": "간결하게 답해."},
        {"role": "user", "content": "에탄올 화학식?"},
    ]
    # HCX-007 renamed maxTokens → maxCompletionTokens.
    assert body["maxCompletionTokens"] == 32
    assert "maxTokens" not in body
    # Defaults that the pipeline relies on:
    assert body["temperature"] == 0.3
    assert body["topP"] == 0.8

    await client.aclose()


@respx.mock
async def test_chat_legacy_hcx005_keeps_maxTokens_field(client: NaverClovaClient):
    """Caller can still pin model='HCX-005' for the older non-reasoning
    endpoint, which uses the legacy `maxTokens` field name."""
    respx.post(
        "https://clovastudio.stream.ntruss.com/v3/chat-completions/HCX-005"
    ).mock(
        return_value=httpx.Response(
            200, json={"result": {"message": {"content": "ok"}}}
        )
    )

    await client.chat(system="s", user="u", model="HCX-005", max_tokens=128)
    import json as _json
    body = _json.loads(respx.calls.last.request.content)
    assert body["maxTokens"] == 128
    assert "maxCompletionTokens" not in body

    await client.aclose()


@respx.mock
async def test_chat_falls_back_to_thinking_when_content_empty(
    client: NaverClovaClient,
):
    """HCX-007 can return empty `content` when the budget was consumed by
    reasoning. We surface `thinkingContent` so the caller sees *something*
    instead of an empty string."""
    respx.post(
        f"https://clovastudio.stream.ntruss.com/v3/chat-completions/"
        f"{NaverClovaClient.CHAT_MODEL_DEFAULT}"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "result": {
                    "message": {
                        "content": "",
                        "thinkingContent": "에탄올의 화학식을 떠올려보면…",
                    }
                }
            },
        )
    )

    out = await client.chat(system="s", user="u")
    assert out.startswith("에탄올의 화학식")

    await client.aclose()


@respx.mock
async def test_chat_uses_custom_model_when_provided(client: NaverClovaClient):
    route = respx.post(
        "https://clovastudio.stream.ntruss.com/v3/chat-completions/HCX-tuned-001"
    ).mock(
        return_value=httpx.Response(
            200, json={"result": {"message": {"content": "ok"}}}
        )
    )

    await client.chat(system="s", user="u", model="HCX-tuned-001")
    assert route.called

    await client.aclose()


@respx.mock
async def test_embed_propagates_http_error(client: NaverClovaClient):
    respx.post(
        "https://clovastudio.stream.ntruss.com/v1/api-tools/embedding/v2"
    ).mock(return_value=httpx.Response(429, headers={"retry-after": "1"}))

    with pytest.raises(httpx.HTTPStatusError) as exc:
        await client.embed("x")
    # 429 is what the ingest retry loop keys on.
    assert exc.value.response.status_code == 429

    await client.aclose()


def test_init_requires_api_key(monkeypatch):
    monkeypatch.delenv("CLOVA_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="CLOVA_API_KEY"):
        NaverClovaClient()
