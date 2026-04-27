"""Upstage (Solar) client."""
from __future__ import annotations

import os
from typing import Literal

import httpx

from ..model_client import BaseModelClient


class UpstageClient(BaseModelClient):
    BASE = "https://api.upstage.ai/v1"
    CHAT_MODEL_DEFAULT = "solar-pro"
    RERANK_MODEL_DEFAULT = "solar-reranker"

    # Upstage splits embedding into separate query/passage models; mismatch
    # destroys retrieval quality, so callers must pass the correct role.
    EMBED_MODEL = {
        "query":   "solar-embedding-1-large-query",
        "passage": "solar-embedding-1-large-passage",
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
        r = await self._client.post(
            f"{self.BASE}/rerank",
            json={
                "model": self.RERANK_MODEL_DEFAULT,
                "query": query,
                "documents": documents,
                "top_n": top_n,
            },
        )
        r.raise_for_status()
        return [
            {"index": d["index"], "score": d["relevance_score"]}
            for d in r.json()["results"]
        ]

    async def chat(self, system: str, user: str, **kwargs) -> str:
        # OpenAI-compatible schema.
        r = await self._client.post(
            f"{self.BASE}/chat/completions",
            json={
                "model": kwargs.get("model", self.CHAT_MODEL_DEFAULT),
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": kwargs.get("temperature", 0.3),
                "top_p": kwargs.get("top_p", 0.8),
                "max_tokens": kwargs.get("max_tokens", 2048),
            },
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    async def aclose(self) -> None:
        await self._client.aclose()
