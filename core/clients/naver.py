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
        # CLOVA's "reranker" actually does RAG-style citation: it ranks the
        # input docs and returns only the cited subset (no scalar scores).
        # Schema docs: https://api.ncloud-docs.com/docs/clovastudio-reranker
        # We adapt back to the {index, score} contract Retriever expects by
        # using the original list position as `index` and descending rank as
        # `score`.
        r = await self._client.post(
            f"{self.BASE}/v1/api-tools/reranker",
            json={
                "query": query,
                "documents": [
                    {"id": str(i), "doc": d} for i, d in enumerate(documents)
                ],
            },
        )
        r.raise_for_status()
        cited = r.json()["result"].get("citedDocuments") or []
        out: list[dict] = []
        for rank, c in enumerate(cited[:top_n]):
            try:
                idx = int(c["id"])
            except (KeyError, ValueError, TypeError):
                continue
            # Monotonically decreasing pseudo-score; downstream only uses
            # ordering and a numeric placeholder.
            out.append({"index": idx, "score": 1.0 - rank * 0.01})
        return out

    async def chat(self, system: str, user: str, **kwargs) -> str:
        # Tuned model id can be passed via kwargs["model"].
        model = kwargs.get("model", self.CHAT_MODEL_DEFAULT)

        # HCX-007 is a reasoning-style model: it renamed `maxTokens` →
        # `maxCompletionTokens` (which now includes thinking tokens) and
        # surfaces chain-of-thought in `result.message.thinkingContent`.
        # Older HCX-005 keeps the legacy `maxTokens` field.
        is_reasoning = model.startswith("HCX-007")
        # Reasoning eats budget; bump the default so `content` actually has
        # room after the model has finished thinking.
        default_max = 4096 if is_reasoning else 2048
        max_field = "maxCompletionTokens" if is_reasoning else "maxTokens"

        body = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "topP": kwargs.get("top_p", 0.8),
            "topK": kwargs.get("top_k", 0),
            max_field: kwargs.get("max_tokens", default_max),
            "temperature": kwargs.get("temperature", 0.3),
            "repeatPenalty": kwargs.get("repeat_penalty", 1.1),
        }
        r = await self._client.post(
            f"{self.BASE}/v3/chat-completions/{model}", json=body
        )
        r.raise_for_status()
        msg = r.json()["result"]["message"]
        # On reasoning models, when the budget is consumed mid-think,
        # `content` can come back empty. Fall back to the thinking trace
        # so the caller still sees *something* instead of "" — and bump
        # max_tokens next call.
        return msg.get("content") or msg.get("thinkingContent", "")

    async def aclose(self) -> None:
        await self._client.aclose()
