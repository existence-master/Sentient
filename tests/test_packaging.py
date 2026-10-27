"""The frozen engine's entry point and self-check (packaging/engine_entry.py, `sentient selfcheck`).

v3.0.0-beta.1's first installer build crashed on start without numpy, broke every model call without tokenizers,
and could not load Sentient's tools for the Claude Code backend because `claude-code-tools` was missing from the
entry point's command list. These tests keep the cheap parts of that from coming back; the build itself runs
`selfcheck` and a real `serve` on the frozen engine (packaging/build_engine.py).
"""

import importlib.util
from pathlib import Path

import typer.main
from typer.testing import CliRunner

from sentient import cli

ROOT = Path(__file__).resolve().parents[1]


def _entry_commands() -> set[str]:
    spec = importlib.util.spec_from_file_location("engine_entry", ROOT / "packaging" / "engine_entry.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._COMMANDS


def test_the_frozen_entry_point_knows_every_cli_command():
    commands = set(typer.main.get_command(cli.app).commands)
    assert commands == _entry_commands()


def test_selfcheck_imports_what_the_engine_needs():
    result = CliRunner().invoke(cli.app, ["selfcheck"])
    assert result.exit_code == 0, result.output
    assert "selfcheck ok" in result.output
    assert "database version" in result.output


def test_selfcheck_fails_loudly_on_a_missing_module(monkeypatch):
    monkeypatch.setattr(cli, "_SELFCHECK_MODULES", ("sentient.app", "no_such_module_for_sentient"))
    result = CliRunner().invoke(cli.app, ["selfcheck"])
    assert result.exit_code == 1
    assert "no_such_module_for_sentient" in result.output


def test_the_engine_spec_keeps_what_litellm_imports():
    spec = (ROOT / "packaging" / "sentient-engine.spec").read_text(encoding="utf-8")
    block = spec[spec.index("EXCLUDES = ["):]
    block = block[: block.index("\n]")]
    excluded = {line.strip().strip(",").strip('"') for line in block.splitlines()[1:]
                if line.strip() and not line.strip().startswith("#")}
    assert "torch" in excluded  # the parse found the list
    assert "tokenizers" not in excluded
