"""End-to-end RAG pipeline for a single MSDS section."""
from __future__ import annotations

import os
from typing import cast

from .model_client import Provider, model_call
from .prompts import get_system_prompt
from .retriever import Retriever


def _default_provider() -> Provider:
    val = os.environ.get("MODEL_PROVIDER", "naver")
    if val not in ("naver", "upstage"):
        raise ValueError(
            f"MODEL_PROVIDER must be 'naver' or 'upstage', got: {val!r}"
        )
    return cast(Provider, val)


async def generate_msds_section(
    product_name: str,
    components: list[dict],
    use_description: str,
    physical_form: str,
    section_number: int,
    *,
    provider: Provider | None = None,
    qdrant_url: str | None = None,
    top_k_first: int = 20,
    top_k_final: int = 5,
) -> str:
    provider = provider or _default_provider()
    qdrant_url = qdrant_url or os.environ.get("QDRANT_URL", "http://localhost:6333")

    retriever = Retriever(provider=provider, qdrant_url=qdrant_url)
    try:
        # CAS-based retrieval gives more precise hits than the product name alone.
        cas_terms = " ".join(
            c["casNumber"] for c in components if c.get("casNumber")
        )
        search_query = f"{product_name} {cas_terms} 항목{section_number}".strip()

        docs = await retriever.search(
            search_query,
            top_k_first=top_k_first,
            top_k_final=top_k_final,
        )
        context = "\n\n".join(
            f"[근거 {i + 1}: {d.get('source', 'unknown')}]\n{d['text']}"
            for i, d in enumerate(docs)
        ) or "(검색된 근거 없음)"

        system = get_system_prompt(section_number)
        user = (
            "[제품 정보]\n"
            f"- 제품명: {product_name}\n"
            f"- 성분: {components}\n"
            f"- 용도: {use_description}\n"
            f"- 물리적 형태: {physical_form}\n\n"
            "[검색된 근거]\n"
            f"{context}\n\n"
            f"위 정보를 근거로 MSDS {section_number}번 항목을 산업안전보건법 고시 양식에 맞춰 작성하세요. "
            "근거에 없는 내용은 추측하지 말고 '제품별 시험 결과 참조'로 표기하세요."
        )

        return await model_call(
            provider, "chat", system=system, user=user
        )
    finally:
        await retriever.aclose()
