"""Unit tests for the multi-provider dispatcher in core.model_client.

The dispatcher translates `(provider, task)` into a single client method call
and is the single seam every other module routes through. These tests pin
its argument-forwarding contract and provider caching behavior without
hitting any network.
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from core import model_client as mc


class _FakeClient:
    """Stand-in for BaseModelClient with AsyncMock methods we can assert on."""

    def __init__(self) -> None:
        self.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
        self.rerank = AsyncMock(
            return_value=[{"index": 0, "score": 0.9}]
        )
        self.chat = AsyncMock(return_value="hello")
        self.aclose = AsyncMock()


@pytest.fixture
def fake(monkeypatch):
    f = _FakeClient()
    # Bypass real client construction; conftest auto-clears the cache.
    mc._CLIENT_CACHE["naver"] = f
    return f


async def test_model_call_embed_forwards_text_and_role(fake):
    out = await mc.model_call("naver", "embed", text="에탄올", role="query")
    fake.embed.assert_awaited_once_with("에탄올", role="query")
    assert out == [0.1, 0.2, 0.3]


async def test_model_call_embed_defaults_role_to_passage(fake):
    await mc.model_call("naver", "embed", text="x")
    fake.embed.assert_awaited_once_with("x", role="passage")


async def test_model_call_rerank_forwards_query_documents_topn(fake):
    await mc.model_call(
        "naver", "rerank", query="q", documents=["a", "b"], top_n=7
    )
    fake.rerank.assert_awaited_once_with("q", ["a", "b"], top_n=7)


async def test_model_call_chat_separates_system_user_from_kwargs(fake):
    await mc.model_call(
        "naver",
        "chat",
        system="S",
        user="U",
        max_tokens=128,
        temperature=0.0,
    )
    fake.chat.assert_awaited_once_with(
        "S", "U", max_tokens=128, temperature=0.0
    )


async def test_model_call_unknown_task_raises(fake):
    with pytest.raises(ValueError, match="unknown task"):
        await mc.model_call("naver", "translate", text="x")  # type: ignore[arg-type]


async def test_get_client_caches_by_provider(monkeypatch):
    constructed: list[str] = []

    class Stub(_FakeClient):
        def __init__(self, tag: str) -> None:
            super().__init__()
            constructed.append(tag)

    # Patch the lazy import paths used inside _get_client.
    import core.clients.naver as naver_mod
    monkeypatch.setattr(
        naver_mod, "NaverClovaClient", lambda: Stub("naver")
    )

    c1 = mc._get_client("naver")
    c2 = mc._get_client("naver")
    assert c1 is c2
    assert constructed == ["naver"]


async def test_get_client_unknown_provider_raises():
    with pytest.raises(ValueError, match="unknown provider"):
        mc._get_client("anthropic")  # type: ignore[arg-type]


async def test_aclose_all_closes_and_clears_cache(fake):
    other = _FakeClient()
    mc._CLIENT_CACHE["upstage"] = other

    await mc.aclose_all()

    fake.aclose.assert_awaited_once()
    other.aclose.assert_awaited_once()
    assert mc._CLIENT_CACHE == {}
