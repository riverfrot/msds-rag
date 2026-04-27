"""CLI entry point for single-section MSDS generation."""
from __future__ import annotations

import asyncio
import json
from typing import cast

import click
from dotenv import load_dotenv

from core.model_client import Provider, aclose_all
from core.pipeline import generate_msds_section

load_dotenv()


@click.command(help="MSDS 단일 항목을 RAG로 생성합니다.")
@click.option("--product", required=True, help="제품명")
@click.option(
    "--components",
    required=True,
    help='성분 JSON. 예: \'[{"name":"Ethanol","casNumber":"64-17-5","weightPercent":45}]\'',
)
@click.option("--use", "use_description", required=True, help="제품 용도")
@click.option("--form", "physical_form", required=True, help="물리적 형태 (예: 액체)")
@click.option(
    "--section",
    required=True,
    type=click.IntRange(1, 16),
    help="MSDS 항목 번호 (1-16)",
)
@click.option(
    "--provider",
    type=click.Choice(["naver", "upstage"]),
    default=None,
    help="모델 provider. 미지정 시 $MODEL_PROVIDER, 그것도 없으면 naver.",
)
def gen(
    product: str,
    components: str,
    use_description: str,
    physical_form: str,
    section: int,
    provider: str | None,
) -> None:
    try:
        components_obj = json.loads(components)
    except json.JSONDecodeError as e:
        raise click.BadParameter(f"--components must be valid JSON: {e}") from e

    async def _run() -> str:
        try:
            return await generate_msds_section(
                product_name=product,
                components=components_obj,
                use_description=use_description,
                physical_form=physical_form,
                section_number=section,
                provider=cast("Provider | None", provider),
            )
        finally:
            await aclose_all()

    click.echo(asyncio.run(_run()))


if __name__ == "__main__":
    gen()
