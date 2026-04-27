"""Multi-provider model client. See document/ModelClient.md."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal


Provider = Literal["naver", "upstage"]
Task = Literal["embed", "rerank", "chat"]


class BaseModelClient(ABC):
    @abstractmethod
    async def embed(
        self,
        text: str,
        *,
        role: Literal["query", "passage"] = "passage",
    ) -> list[float]: ...

    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int = 5,
    ) -> list[dict]: ...

    @abstractmethod
    async def chat(self, system: str, user: str, **kwargs) -> str: ...

    async def aclose(self) -> None:
        """Override if the client owns network resources."""


_CLIENT_CACHE: dict[Provider, BaseModelClient] = {}


def _get_client(provider: Provider) -> BaseModelClient:
    if provider in _CLIENT_CACHE:
        return _CLIENT_CACHE[provider]

    # Lazy import so a missing API key for one provider doesn't break the other.
    if provider == "naver":
        from .clients.naver import NaverClovaClient
        client: BaseModelClient = NaverClovaClient()
    elif provider == "upstage":
        from .clients.upstage import UpstageClient
        client = UpstageClient()
    else:
        raise ValueError(f"unknown provider: {provider}")

    _CLIENT_CACHE[provider] = client
    return client


async def model_call(provider: Provider, task: Task, **kwargs):
    """Unified entry point.

    Examples:
        await model_call("naver",   "chat",   system="...", user="...")
        await model_call("upstage", "embed",  text="...", role="query")
        await model_call("naver",   "rerank", query="...", documents=[...])
    """
    client = _get_client(provider)

    if task == "embed":
        return await client.embed(
            kwargs["text"],
            role=kwargs.get("role", "passage"),
        )
    if task == "rerank":
        return await client.rerank(
            kwargs["query"],
            kwargs["documents"],
            top_n=kwargs.get("top_n", 5),
        )
    if task == "chat":
        chat_kwargs = {
            k: v for k, v in kwargs.items() if k not in ("system", "user")
        }
        return await client.chat(kwargs["system"], kwargs["user"], **chat_kwargs)

    raise ValueError(f"unknown task: {task}")


async def aclose_all() -> None:
    for client in _CLIENT_CACHE.values():
        await client.aclose()
    _CLIENT_CACHE.clear()
