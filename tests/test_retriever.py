"""Unit tests for core.retriever."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from core import retriever as r


def _scored(payload: dict):
    """ScoredPoint stand-in — Retriever only reads `.payload`."""
    return SimpleNamespace(payload=payload)


def test_collection_name_is_provider_namespaced():
    assert r.collection_name_for("naver") == "msds_corpus_naver"
    assert r.collection_name_for("upstage") == "msds_corpus_upstage"


async def test_search_passes_qvec_and_maps_rerank_back_to_payloads(monkeypatch):
    # 1) Stub the embed + rerank calls. embed returns a fake vector; rerank
    #    cites the 2nd and 0th candidate (in that order) so we can verify
    #    the index mapping and rank-score preservation.
    async def fake_model_call(provider, task, **kw):
        if task == "embed":
            return [0.42] * 8
        if task == "rerank":
            assert kw["query"] == "에탄올 독성"
            assert len(kw["documents"]) == 3
            return [
                {"index": 2, "score": 1.0},
                {"index": 0, "score": 0.99},
            ]
        raise AssertionError(f"unexpected task: {task}")

    monkeypatch.setattr(r, "model_call", fake_model_call)

    retr = r.Retriever("naver")
    # Replace the real Qdrant client with a stub.
    payloads = [
        {"text": "doc-A", "chem_id": "001", "section": 11},
        {"text": "doc-B", "chem_id": "002", "section": 11},
        {"text": "doc-C", "chem_id": "003", "section": 11},
    ]
    qresp = SimpleNamespace(points=[_scored(p) for p in payloads])
    retr.qdrant = SimpleNamespace(
        query_points=AsyncMock(return_value=qresp),
        close=AsyncMock(),
    )

    out = await retr.search("에탄올 독성", top_k_first=3, top_k_final=2)

    # Order follows the rerank cite order: payload[2], then payload[0].
    assert [d["text"] for d in out] == ["doc-C", "doc-A"]
    assert out[0]["rerank_score"] == 1.0
    assert out[1]["rerank_score"] == 0.99

    # query_points was invoked with the embedding vector and the recall limit.
    retr.qdrant.query_points.assert_awaited_once()
    kwargs = retr.qdrant.query_points.await_args.kwargs
    assert kwargs["collection_name"] == "msds_corpus_naver"
    assert kwargs["query"] == [0.42] * 8
    assert kwargs["limit"] == 3

    await retr.aclose()


async def test_search_returns_empty_list_when_no_hits(monkeypatch):
    async def fake_model_call(provider, task, **kw):
        # rerank should NOT be called when there are no ANN hits.
        assert task != "rerank", "rerank skipped when no candidates"
        return [0.0] * 4

    monkeypatch.setattr(r, "model_call", fake_model_call)

    retr = r.Retriever("naver")
    retr.qdrant = SimpleNamespace(
        query_points=AsyncMock(return_value=SimpleNamespace(points=[])),
        close=AsyncMock(),
    )

    out = await retr.search("nothing matches")
    assert out == []
    await retr.aclose()
