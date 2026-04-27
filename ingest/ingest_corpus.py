"""Ingest a corpus directory into Qdrant for the chosen provider.

Usage:
    python -m ingest.ingest_corpus --corpus ./corpus --provider naver
"""
from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path
from typing import cast

import click
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from core.model_client import Provider, aclose_all, model_call
from core.retriever import collection_name_for

load_dotenv()


# Embedding dimensions per provider; mismatched dim crashes Qdrant inserts,
# so the lookup is the source of truth here, not the docs.
EMBED_DIM: dict[Provider, int] = {
    "naver":   1024,
    "upstage": 4096,
}

CHUNK_SIZE = 1000
CHUNK_STRIDE = 800
BATCH_SIZE = 64


def _chunk(text: str) -> list[str]:
    chunks = []
    for i in range(0, len(text), CHUNK_STRIDE):
        piece = text[i : i + CHUNK_SIZE]
        if piece.strip():
            chunks.append(piece)
    return chunks


async def _ensure_collection(
    qdrant: AsyncQdrantClient, name: str, dim: int
) -> None:
    existing = {c.name for c in (await qdrant.get_collections()).collections}
    if name in existing:
        return
    await qdrant.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )


async def ingest(corpus_dir: str, provider: Provider, qdrant_url: str) -> None:
    qdrant = AsyncQdrantClient(url=qdrant_url)
    collection = collection_name_for(provider)

    try:
        await _ensure_collection(qdrant, collection, EMBED_DIM[provider])

        files = sorted(Path(corpus_dir).rglob("*.txt"))
        click.echo(
            f"[ingest] provider={provider} collection={collection} "
            f"files={len(files)}"
        )

        buffer: list[PointStruct] = []
        total = 0

        for path in files:
            text = path.read_text(encoding="utf-8")
            chunks = _chunk(text)

            for chunk_idx, chunk in enumerate(chunks):
                vec = await model_call(
                    provider, "embed", text=chunk, role="passage"
                )
                point_id = str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL, f"{path.name}#{chunk_idx}"
                    )
                )
                buffer.append(
                    PointStruct(
                        id=point_id,
                        vector=vec,
                        payload={
                            "text": chunk,
                            "source": path.name,
                            "chunk_idx": chunk_idx,
                        },
                    )
                )
                if len(buffer) >= BATCH_SIZE:
                    await qdrant.upsert(
                        collection_name=collection, points=buffer
                    )
                    total += len(buffer)
                    click.echo(f"  upserted {total} pts so far")
                    buffer = []

        if buffer:
            await qdrant.upsert(collection_name=collection, points=buffer)
            total += len(buffer)

        click.echo(f"[ingest] done. total points={total}")
    finally:
        await qdrant.close()
        await aclose_all()


@click.command(help="MSDS 코퍼스를 Qdrant에 임베딩하여 적재.")
@click.option(
    "--corpus",
    required=True,
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="*.txt 파일들이 있는 디렉토리",
)
@click.option(
    "--provider",
    required=True,
    type=click.Choice(["naver", "upstage"]),
    help="임베딩 provider",
)
@click.option("--qdrant-url", default=None, help="기본: $QDRANT_URL or http://localhost:6333")
def main(corpus: str, provider: str, qdrant_url: str | None) -> None:
    url = qdrant_url or os.environ.get("QDRANT_URL", "http://localhost:6333")
    asyncio.run(ingest(corpus, cast(Provider, provider), url))


if __name__ == "__main__":
    main()
