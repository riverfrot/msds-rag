"""Single-call live tests against CLOVA Studio.

Skipped automatically when CLOVA_API_KEY is missing or still the placeholder.
Each test makes one API call to keep the rate-limit blast radius tiny.

Run with:
    pytest -m integration
"""
from __future__ import annotations

import pytest

from core.clients.naver import NaverClovaClient

pytestmark = [pytest.mark.integration, pytest.mark.naver]


@pytest.fixture
async def client():
    c = NaverClovaClient()
    try:
        yield c
    finally:
        await c.aclose()


async def test_embed_returns_1024_float_vector(client: NaverClovaClient):
    vec = await client.embed("에탄올의 인화점", role="passage")
    assert isinstance(vec, list)
    assert len(vec) == 1024, f"expected 1024-dim, got {len(vec)}"
    assert all(isinstance(x, float) for x in vec)


async def test_rerank_returns_index_score_pairs(client: NaverClovaClient):
    docs = [
        "에탄올은 인화성 액체이며 흡입 시 두통을 유발할 수 있다.",
        "벤젠은 1군 발암물질로 알려져 있다.",
        "아세톤은 휘발성이 강한 유기용매이다.",
    ]
    out = await client.rerank("에탄올 독성", docs, top_n=3)

    assert isinstance(out, list)
    # Reranker may cite a subset; require at least one cited document.
    assert len(out) >= 1
    for item in out:
        assert set(item.keys()) == {"index", "score"}
        assert 0 <= item["index"] < len(docs)
        assert isinstance(item["score"], float)
    # Ensure the relevant doc (ethanol) is among the cited results.
    cited_idx = {item["index"] for item in out}
    assert 0 in cited_idx, "ethanol passage should be cited for ethanol query"


async def test_chat_returns_nonempty_string(client: NaverClovaClient):
    answer = await client.chat(
        system="당신은 간결하게 답하는 화학 전문가입니다.",
        user="에탄올의 화학식을 한 단어로 답하세요.",
        max_tokens=64,
    )
    assert isinstance(answer, str)
    assert len(answer.strip()) > 0
