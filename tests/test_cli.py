"""CLI integration tests for cli.msds_cli.

The pipeline itself is mocked — we only care about argument parsing,
file-saving behavior, and the various output flags.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from cli import msds_cli


@pytest.fixture(autouse=True)
def _stub_pipeline(monkeypatch):
    """Replace the async pipeline + aclose with deterministic stubs.

    `asyncio.run` runs them, so they must remain coroutine functions.
    """

    async def fake_generate(**kwargs):
        return "[9. 물리·화학적 특성]\n가. 외관: 투명한 액체"

    async def fake_aclose():
        return None

    monkeypatch.setattr(msds_cli, "generate_msds_section", fake_generate)
    monkeypatch.setattr(msds_cli, "aclose_all", fake_aclose)


def _common_args(extra: list[str] | None = None) -> list[str]:
    args = [
        "--product", "HW-Cleaner 200",
        "--components",
        '[{"name":"Ethanol","casNumber":"64-17-5","weightPercent":45}]',
        "--use", "정밀세정제",
        "--form", "액체",
        "--section", "9",
        "--provider", "naver",
    ]
    if extra:
        args.extend(extra)
    return args


def test_default_run_writes_file_to_output_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(msds_cli.gen, _common_args())

    assert result.exit_code == 0, result.output
    # Body shows up on stdout.
    assert "[9. 물리·화학적 특성]" in result.output

    # Default location: ./output/<slug>_section09_naver_<ts>.md
    out_dir = tmp_path / "output"
    files = list(out_dir.glob("*.md"))
    assert len(files) == 1
    saved = files[0]
    assert "section09" in saved.name
    assert "naver" in saved.name
    assert "HW-Cleaner_200" in saved.name

    contents = saved.read_text(encoding="utf-8")
    # Self-describing header survives.
    assert "MSDS Section 9 — HW-Cleaner 200" in contents
    assert "Provider: `naver`" in contents
    assert "[9. 물리·화학적 특성]" in contents


def test_explicit_output_overrides_default(tmp_path):
    runner = CliRunner()
    target = tmp_path / "nested" / "result.md"
    result = runner.invoke(
        msds_cli.gen, _common_args(["--output", str(target)])
    )

    assert result.exit_code == 0, result.output
    assert target.exists()
    # Parent created on demand.
    assert target.parent.is_dir()
    assert "[9." in target.read_text(encoding="utf-8")


def test_no_save_skips_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(msds_cli.gen, _common_args(["--no-save"]))

    assert result.exit_code == 0, result.output
    assert "[9." in result.output
    # No output directory created when saving is suppressed.
    assert not (tmp_path / "output").exists()


def test_quiet_suppresses_stdout_but_still_saves(tmp_path):
    # Click 8.2 separates stdout/stderr by default — no `mix_stderr` flag.
    runner = CliRunner()
    target = tmp_path / "out.md"
    result = runner.invoke(
        msds_cli.gen, _common_args(["--quiet", "--output", str(target)])
    )

    assert result.exit_code == 0
    # Body should not be on stdout under --quiet.
    assert "[9." not in result.stdout
    # The "[saved] ..." notice goes to stderr so a `> file.md` redirect
    # of stdout still gets a clean body.
    assert "[saved]" in result.stderr
    assert target.exists()


def test_invalid_components_json_is_rejected():
    runner = CliRunner()
    bad = ["--product", "X", "--components", "not-json",
           "--use", "u", "--form", "액체",
           "--section", "1", "--provider", "naver"]
    result = runner.invoke(msds_cli.gen, bad)

    # Click reports BadParameter as exit code 2.
    assert result.exit_code == 2
    assert "must be valid JSON" in result.output


def test_section_out_of_range_is_rejected():
    runner = CliRunner()
    result = runner.invoke(msds_cli.gen, _common_args(["--section", "0"]))
    # Re-evaluate: --section appears later in the list, so the original
    # "9" is still present. Use a fresh list instead.
    args = [
        "--product", "X",
        "--components", "[]",
        "--use", "u",
        "--form", "액체",
        "--section", "17",
        "--provider", "naver",
    ]
    result = runner.invoke(msds_cli.gen, args)
    assert result.exit_code == 2  # IntRange validation


def test_slugify_handles_special_chars():
    assert msds_cli._slugify("HW-Cleaner 200!!") == "HW-Cleaner_200"
    assert msds_cli._slugify("///") == "msds"
    assert msds_cli._slugify("정밀세정제 v1.2") == "정밀세정제_v1.2"


def test_default_output_path_shape(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = msds_cli._default_output_path("HW-Cleaner 200", 9, "naver")
    assert isinstance(p, Path)
    assert p.parts[0] == "output"
    assert p.suffix == ".md"
    assert "section09" in p.name
    assert "naver" in p.name
