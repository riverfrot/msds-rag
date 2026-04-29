"""Upstage (Solar) client."""
from __future__ import annotations

import os
from typing import Literal

import httpx

from ..model_client import BaseModelClient


class UpstageClient(BaseModelClient):
    BASE = "https://api.upstage.ai/v1"
    # Solar Pro 2 is the current generation; supports `reasoning_effort`
    # for multi-step reasoning. Tracked alias = "solar-pro2".
    CHAT_MODEL_DEFAULT = "solar-pro2"
    RERANK_MODEL_DEFAULT = "solar-reranker"

    # Upstage splits embedding into separate query/passage models; mismatch
    # destroys retrieval quality, so callers must pass the correct role.
    # Aliases (preferred) auto-route to the current `solar-embedding-1-large-*`
    # versions, so we don't have to bump model names on every minor release.
    EMBED_MODEL = {
        "query":   "embedding-query",
        "passage": "embedding-passage",
    }

    def __init__(self) -> None:
        api_key = os.environ.get("UPSTAGE_API_KEY")
        if not api_key:
            raise RuntimeError("UPSTAGE_API_KEY is not set")
        self._client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )

    async def embed(
        self,
        text: str,
        *,
        role: Literal["query", "passage"] = "passage",
    ) -> list[float]:
        r = await self._client.post(
            f"{self.BASE}/embeddings",
            json={"model": self.EMBED_MODEL[role], "input": text},
        )
        r.raise_for_status()
        return r.json()["data"][0]["embedding"]

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int = 5,
    ) -> list[dict]:
        # Upstage does not currently expose a public rerank API. Since
        # solar-embedding-1-large produces normalized vectors (cosine ==
        # dot product) and Qdrant ANN already returns hits sorted by
        # cosine similarity, the ANN ordering *is* the rerank result.
        #
        # We therefore implement rerank as an identity passthrough that
        # preserves the input order, caps at top_n, and emits a
        # monotonically descending pseudo-score so the Retriever contract
        # ({"index": int, "score": float}) is unchanged.
        del query  # ANN already used it; we don't re-rank here.
        return [
            {"index": i, "score": 1.0 - i * 0.01}
            for i in range(min(top_n, len(documents)))
        ]

    async def chat(self, system: str, user: str, **kwargs) -> str:
        # OpenAI-compatible schema.
        body = {
            "model": kwargs.get("model", self.CHAT_MODEL_DEFAULT),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": kwargs.get("temperature", 0.3),
            "top_p": kwargs.get("top_p", 0.8),
            "max_tokens": kwargs.get("max_tokens", 2048),
        }
        # Solar-Pro2 specific: opt-in chain-of-thought budget.
        # Forwarded only when the caller explicitly asks for it so we don't
        # pay reasoning latency on routine MSDS sections that just need
        # template filling.
        if "reasoning_effort" in kwargs:
            body["reasoning_effort"] = kwargs["reasoning_effort"]
        r = await self._client.post(
            f"{self.BASE}/chat/completions", json=body
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    async def aclose(self) -> None:
        await self._client.aclose()
