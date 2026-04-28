"""Ingest a pre-chunked JSONL file into Qdrant.


Unlike ingest_corpus (which slides a 1000-char window over raw .txt files),
each line in chunks.jsonl is already a semantically meaningful chunk
(one MSDS section per chem_id). Re-chunking would shred those boundaries
and discard the chem_id/section metadata, so we ingest verbatim.

Usage:
    python -m ingest.ingest_jsonl \\
        --jsonl ./sample/msds_data/chunks.jsonl \\
        --provider naver
"""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import cast

import click
import httpx
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from core.model_client import Provider, aclose_all, model_call
from core.retriever import collection_name_for

load_dotenv()


EMBED_DIM: dict[Provider, int] = {
    "naver":   1024,
    "upstage": 4096,
}

BATCH_SIZE = 64
EMBED_CHAR_CAP = 500  # CLOVA embed endpoint caps input around 500 chars.

# CLOVA's embed endpoint rate-limits aggressively (HTTP 429). A small sleep
# between calls + exponential backoff on 429 keeps us under the cap without
# manual tuning.
PER_CALL_SLEEP = 0.25
MAX_RETRIES = 6


async def _embed_with_retry(provider: Provider, text: str) -> list[float]:
    """Call the embedding API with backoff on 429/5xx."""
    delay = 1.0
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            return await model_call(
                provider, "embed", text=text, role="passage"
            )
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status == 429 or 500 <= status < 600:
                last_exc = e
                # Honor Retry-After when present, else exponential backoff.
                ra = e.response.headers.get("retry-after")
                wait = float(ra) if ra and ra.replace(".", "", 1).isdigit() else delay
                await asyncio.sleep(wait)
                delay = min(delay * 2, 30.0)
                continue
            raise
        except (httpx.TransportError, httpx.TimeoutException) as e:
            last_exc = e
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30.0)
    assert last_exc is not None
    raise last_exc


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


async def _existing_ids(qdrant: AsyncQdrantClient, collection: str) -> set[str]:
    """Return all point IDs currently in the collection."""
    ids: set[str] = set()
    next_offset = None
    while True:
        points, next_offset = await qdrant.scroll(
            collection_name=collection,
            limit=512,
            offset=next_offset,
            with_payload=False,
            with_vectors=False,
        )
        ids.update(str(p.id) for p in points)
        if next_offset is None:
            break
    return ids


async def ingest(jsonl_path: str, provider: Provider, qdrant_url: str) -> None:
    qdrant = AsyncQdrantClient(url=qdrant_url)
    collection = collection_name_for(provider)
    src_name = Path(jsonl_path).name

    try:
        await _ensure_collection(qdrant, collection, EMBED_DIM[provider])
        already = await _existing_ids(qdrant, collection)

        with open(jsonl_path, encoding="utf-8") as f:
            lines = [ln for ln in f if ln.strip()]

        click.echo(
            f"[ingest_jsonl] provider={provider} collection={collection} "
            f"chunks={len(lines)} already_in_qdrant={len(already)}"
        )

        buffer: list[PointStruct] = []
        total = 0
        skipped = 0
        already_skipped = 0

        for line_idx, line in enumerate(lines):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                click.echo(f"  [skip] line {line_idx}: bad JSON ({e})")
                skipped += 1
                continue

            text = (rec.get("text") or "").strip()
            if not text:
                skipped += 1
                continue

            chem_id = rec.get("chem_id", "")
            section = rec.get("section", 0)
            section_name = rec.get("section_name", "")

            point_id = str(
                uuid.uuid5(uuid.NAMESPACE_URL, f"{chem_id}#{section}")
            )
            if point_id in already:
                already_skipped += 1
                continue

            try:
                vec = await _embed_with_retry(provider, text[:EMBED_CHAR_CAP])
            except Exception as e:
                click.echo(
                    f"  [skip] embed failed @ chem={chem_id} sec={section}: {e}"
                )
                skipped += 1
                continue
            await asyncio.sleep(PER_CALL_SLEEP)

            buffer.append(
                PointStruct(
                    id=point_id,
                    vector=vec,
                    payload={
                        "text":         text,
                        "chem_id":      chem_id,
                        "section":      section,
                        "section_name": section_name,
                        # `source` is read by core.pipeline when formatting
                        # retrieved evidence; keep it human-readable.
                        "source":       f"{src_name}#{chem_id}#sec{section}",
                    },
                )
            )

            if len(buffer) >= BATCH_SIZE:
                await qdrant.upsert(collection_name=collection, points=buffer)
                total += len(buffer)
                click.echo(f"  upserted {total} pts so far")
                buffer = []

        if buffer:
            await qdrant.upsert(collection_name=collection, points=buffer)
            total += len(buffer)

        click.echo(
            f"[ingest_jsonl] done. upserted={total} "
            f"already_in_qdrant={already_skipped} skipped={skipped}"
        )
    finally:
        await qdrant.close()
        await aclose_all()


@click.command(help="이미 청킹된 chunks.jsonl을 Qdrant에 적재.")
@click.option(
    "--jsonl",
    "jsonl_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    help="청크 JSONL 파일 경로 (한 줄에 한 청크)",
)
@click.option(
    "--provider",
    required=True,
    type=click.Choice(["naver", "upstage"]),
    help="임베딩 provider",
)
@click.option(
    "--qdrant-url",
    default=None,
    help="기본: $QDRANT_URL or http://localhost:6333",
)
def main(jsonl_path: str, provider: str, qdrant_url: str | None) -> None:
    url = qdrant_url or os.environ.get("QDRANT_URL", "http://localhost:6333")
    asyncio.run(ingest(jsonl_path, cast(Provider, provider), url))


if __name__ == "__main__":
    main()
