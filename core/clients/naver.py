"""Naver Cloud (CLOVA Studio) client."""
from __future__ import annotations

import os
from typing import Literal

import httpx

from ..model_client import BaseModelClient


class NaverClovaClient(BaseModelClient):
    BASE = "https://clovastudio.stream.ntruss.com"
    CHAT_MODEL_DEFAULT = "HCX-005"

    def __init__(self) -> None:
        api_key = os.environ.get("CLOVA_API_KEY")
        if not api_key:
            raise RuntimeError("CLOVA_API_KEY is not set")
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
        role: Literal["query", "passage"] = "passage",  # noqa: ARG002 - kept for interface parity
    ) -> list[float]:
        # CLOVA does not split query/passage models; role is accepted but ignored.
        r = await self._client.post(
            f"{self.BASE}/v1/api-tools/embedding/v2",
            json={"text": text},
        )
        r.raise_for_status()
        return r.json()["result"]["embedding"]

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int = 5,
    ) -> list[dict]:
        r = await self._client.post(
            f"{self.BASE}/v1/api-tools/reranker",
            json={"query": query, "documents": documents, "topN": top_n},
        )
        r.raise_for_status()
        return [
            {"index": d["index"], "score": d["score"]}
            for d in r.json()["result"]["citedDocuments"]
        ]

    async def chat(self, system: str, user: str, **kwargs) -> str:
        # Tuned model id can be passed via kwargs["model"].
        model = kwargs.get("model", self.CHAT_MODEL_DEFAULT)
        r = await self._client.post(
            f"{self.BASE}/v3/chat-completions/{model}",
            json={
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "topP": kwargs.get("top_p", 0.8),
                "topK": kwargs.get("top_k", 0),
                "maxTokens": kwargs.get("max_tokens", 2048),
                "temperature": kwargs.get("temperature", 0.3),
                "repeatPenalty": kwargs.get("repeat_penalty", 1.1),
            },
        )
        r.raise_for_status()
        return r.json()["result"]["message"]["content"]

    async def aclose(self) -> None:
        await self._client.aclose()
