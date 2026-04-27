"""Qdrant retrieval + provider-aware reranking."""
from __future__ import annotations

from qdrant_client import AsyncQdrantClient

from .model_client import Provider, model_call


def collection_name_for(provider: Provider) -> str:
    # Embedding dim differs per provider (CLOVA 1024 / Upstage 4096), so
    # collections must be physically separate.
    return f"msds_corpus_{provider}"


class Retriever:
    def __init__(
        self,
        provider: Provider,
        qdrant_url: str = "http://localhost:6333",
    ) -> None:
        self.provider = provider
        self.collection = collection_name_for(provider)
        self.qdrant = AsyncQdrantClient(url=qdrant_url)

    async def search(
        self,
        query: str,
        top_k_first: int = 20,
        top_k_final: int = 5,
    ) -> list[dict]:
        # 1) Embed the query. Upstage uses a query-specific model; CLOVA ignores role.
        qvec = await model_call(self.provider, "embed", text=query, role="query")

        # 2) ANN recall.
        hits = await self.qdrant.search(
            collection_name=self.collection,
            query_vector=qvec,
            limit=top_k_first,
        )
        if not hits:
            return []

        candidates = [h.payload["text"] for h in hits]
        meta = [h.payload for h in hits]

        # 3) Rerank for precision; cap at top_k_final to keep token cost bounded.
        reranked = await model_call(
            self.provider,
            "rerank",
            query=query,
            documents=candidates,
            top_n=top_k_final,
        )

        return [
            {**meta[r["index"]], "rerank_score": r["score"]}
            for r in reranked
        ]

    async def aclose(self) -> None:
        await self.qdrant.close()
