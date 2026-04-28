"""Unit tests for core.pipeline.generate_msds_section."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core import pipeline as p
from core.prompts import get_system_prompt


def _retriever_stub(docs: list[dict]) -> SimpleNamespace:
    return SimpleNamespace(
        search=AsyncMock(return_value=docs),
        aclose=AsyncMock(),
    )


async def test_generate_msds_section_assembles_query_and_context(monkeypatch):
    captured = {}

    async def fake_model_call(provider, task, **kw):
        if task == "chat":
            captured["system"] = kw["system"]
            captured["user"] = kw["user"]
            captured["provider"] = provider
            return "<<MSDS-OUTPUT>>"
        raise AssertionError(f"unexpected task: {task}")

    docs = [
        {
            "text": "에탄올 인화점 13°C",
            "source": "chunks.jsonl#000001#sec9",
        },
        {
            "text": "에탄올 LD50(rat, oral) 7060 mg/kg",
            "source": "chunks.jsonl#000001#sec11",
        },
    ]
    retr = _retriever_stub(docs)
    captured["search_query"] = None

    async def search(query, **kw):
        captured["search_query"] = query
        captured["top_k_first"] = kw.get("top_k_first")
        captured["top_k_final"] = kw.get("top_k_final")
        return docs

    retr.search = search

    monkeypatch.setattr(p, "model_call", fake_model_call)
    monkeypatch.setattr(p, "Retriever", lambda **kw: retr)

    out = await p.generate_msds_section(
        product_name="HW-Cleaner 200",
        components=[
            {"name": "Ethanol", "casNumber": "64-17-5", "weightPercent": 45}
        ],
        use_description="정밀세정제",
        physical_form="액체",
        section_number=11,
        provider="naver",
    )

    assert out == "<<MSDS-OUTPUT>>"

    # Search query should include product name + every CAS + the section tag.
    assert "HW-Cleaner 200" in captured["search_query"]
    assert "64-17-5" in captured["search_query"]
    assert "항목11" in captured["search_query"]

    # Default ANN/recall sizes.
    assert captured["top_k_first"] == 20
    assert captured["top_k_final"] == 5

    # System prompt is the section-specific template.
    assert captured["system"] == get_system_prompt(11)

    # User message must surface evidence verbatim and include component info.
    assert "에탄올 인화점 13°C" in captured["user"]
    assert "에탄올 LD50(rat, oral) 7060 mg/kg" in captured["user"]
    assert "Ethanol" in captured["user"]
    assert "정밀세정제" in captured["user"]
    assert "액체" in captured["user"]
    assert captured["provider"] == "naver"

    retr.aclose.assert_awaited_once()


async def test_generate_msds_section_marks_no_evidence_when_search_empty(monkeypatch):
    captured = {}

    async def fake_model_call(provider, task, **kw):
        captured["user"] = kw["user"]
        return "ok"

    retr = _retriever_stub([])
    monkeypatch.setattr(p, "model_call", fake_model_call)
    monkeypatch.setattr(p, "Retriever", lambda **kw: retr)

    await p.generate_msds_section(
        product_name="X",
        components=[{"name": "Y", "casNumber": "1-2-3", "weightPercent": 10}],
        use_description="u",
        physical_form="고체",
        section_number=2,
        provider="naver",
    )

    assert "(검색된 근거 없음)" in captured["user"]


async def test_default_provider_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "naver")
    assert p._default_provider() == "naver"

    monkeypatch.setenv("MODEL_PROVIDER", "upstage")
    assert p._default_provider() == "upstage"

    monkeypatch.setenv("MODEL_PROVIDER", "bogus")
    with pytest.raises(ValueError):
        p._default_provider()
