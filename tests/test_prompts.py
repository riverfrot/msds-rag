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
