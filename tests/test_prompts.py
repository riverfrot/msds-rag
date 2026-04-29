"""Unit tests for core.prompts.

These prompts are the contract between the pipeline and the LLM. A typo or
silently dropped section breaks generation for that section only, which is
hard to spot otherwise.
"""
from __future__ import annotations

import pytest

from core.prompts import SYSTEM_PROMPTS, get_system_prompt


@pytest.mark.parametrize("section", list(range(1, 17)))
def test_get_system_prompt_covers_all_16_sections(section: int):
    prompt = get_system_prompt(section)
    assert isinstance(prompt, str) and prompt.strip()
    # Every section header is bracketed and starts with the section number.
    assert f"[{section}." in prompt


def test_get_system_prompt_includes_base_principles():
    prompt = get_system_prompt(11)
    # Anti-hallucination rule must always be present.
    assert "추측하지" in prompt
    assert "MSDS" in prompt


@pytest.mark.parametrize("section", list(range(1, 17)))
def test_base_requires_source_citation_on_every_section(section: int):
    """Every section inherits the base block that mandates source tags
    next to numeric values (closes the 'Naver = no sources' gap)."""
    prompt = get_system_prompt(section)
    assert "출처 표기" in prompt
    assert "모델 내부 지식만으로 채운 값은 적지 않습니다" in prompt


def test_section9_pins_flash_point_measurement_method():
    """§9 must force closed-cup vs open-cup tagging on flash points and
    forbid mixing methods within the same answer."""
    prompt = get_system_prompt(9)
    assert "폐쇄식" in prompt
    assert "개방식" in prompt
    assert "측정법을 섞지 말 것" in prompt
    # Boiling-point / azeotrope confusion guard.
    assert "공비점" in prompt
    # pH semantic guard for non-aqueous mixes.
    assert "non-aqueous" in prompt


@pytest.mark.parametrize("section,marker", [
    (2, "H-codes"),
    (3, "CAS번호"),
    (9, "인화점"),
    (11, "급성독성"),
    (14, "UN No."),
])
def test_section_specific_markers(section: int, marker: str):
    """Spot-check that section-specific structure survived edits."""
    assert marker in get_system_prompt(section)


def test_get_system_prompt_rejects_out_of_range():
    with pytest.raises(ValueError):
        get_system_prompt(0)
    with pytest.raises(ValueError):
        get_system_prompt(17)


def test_system_prompts_alias_matches_function():
    # The dict alias is built once at import; ensure it stays in sync.
    for n in range(1, 17):
        assert SYSTEM_PROMPTS[n] == get_system_prompt(n)
