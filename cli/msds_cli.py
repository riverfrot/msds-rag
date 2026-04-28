"""CLI entry point for single-section MSDS generation."""
from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from typing import cast

import click
from dotenv import load_dotenv

from core.model_client import Provider, aclose_all
from core.pipeline import generate_msds_section

load_dotenv()


_FILENAME_SAFE = re.compile(r"[^A-Za-z0-9가-힣._-]+")


def _slugify(value: str) -> str:
    """Make a string safe for use as a filename component."""
    cleaned = _FILENAME_SAFE.sub("_", value).strip("_")
    return cleaned or "msds"


def _default_output_path(product: str, section: int, provider: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{_slugify(product)}_section{section:02d}_{provider}_{ts}.md"
    return Path("output") / name


def _build_document(
    *,
    product: str,
    components: list[dict],
    use_description: str,
    physical_form: str,
    section: int,
    provider: str,
    body: str,
) -> str:
    """Wrap the model output with a small header so the saved file is
    self-describing (you can hand it to a reviewer without explaining
    what was generated and how)."""
    generated_at = datetime.now().isoformat(timespec="seconds")
    header = (
        f"# MSDS Section {section} — {product}\n\n"
        f"- Provider: `{provider}`\n"
        f"- Generated at: `{generated_at}`\n"
        f"- Use: {use_description}\n"
        f"- Physical form: {physical_form}\n"
        f"- Components: `{json.dumps(components, ensure_ascii=False)}`\n\n"
        "---\n\n"
    )
    return header + body.rstrip() + "\n"


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
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=None,
    help="결과 저장 경로. 미지정 시 ./output/<제품>_section<NN>_<provider>_<timestamp>.md",
)
@click.option(
    "--no-save",
    is_flag=True,
    default=False,
    help="파일 저장을 끄고 stdout 으로만 출력합니다.",
)
@click.option(
    "--quiet",
    is_flag=True,
    default=False,
    help="stdout 으로 본문 출력을 생략하고 저장 경로만 보고합니다.",
)
def gen(
    product: str,
    components: str,
    use_description: str,
    physical_form: str,
    section: int,
    provider: str | None,
    output_path: Path | None,
    no_save: bool,
    quiet: bool,
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

    body = asyncio.run(_run())

    if not quiet:
        click.echo(body)

    if no_save:
        return

    resolved_provider = provider or "naver"
    target = output_path or _default_output_path(product, section, resolved_provider)
    target.parent.mkdir(parents=True, exist_ok=True)

    document = _build_document(
        product=product,
        components=components_obj,
        use_description=use_description,
        physical_form=physical_form,
        section=section,
        provider=resolved_provider,
        body=body,
    )
    target.write_text(document, encoding="utf-8")
    # Always report the save path on stderr so it survives `> file.md`.
    click.echo(f"[saved] {target}", err=True)


if __name__ == "__main__":
    gen()
