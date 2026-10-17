"""Changelog entries in changes/ become a version section of CHANGELOG.md (packaging/release_changelog.py)."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "release_changelog", ROOT / "packaging" / "release_changelog.py"
)
release = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(release)

CHANGELOG = """# Changelog

## [Unreleased]

<!-- towncrier release notes start -->

### Added
- **Older thing.** Written by hand (#1).

### Fixed
- **Older fix.** Also by hand.

## [0.1.0] - 2026-01-01

### Added
- First.

[Unreleased]: https://github.com/existence-master/Sentient/compare/v0.1.0...main
[0.1.0]: https://github.com/existence-master/Sentient/releases/tag/v0.1.0
"""


def test_fold_merges_repeated_headings_and_links_the_version():
    built = CHANGELOG.replace(
        "<!-- towncrier release notes start -->\n\n",
        "<!-- towncrier release notes start -->\n\n## [0.2.0] - 2026-02-01\n\n"
        "### Fixed\n- **New fix.** From a fragment (#9).\n\n### Added\n- **New thing.** Two\n  lines (#8).\n\n",
    )
    text = release.fold_release(built, "0.2.0")
    section = release.section(text, "0.2.0")
    assert section == (
        "## [0.2.0] - 2026-02-01\n\n"
        "### Added\n- **New thing.** Two\n  lines (#8).\n- **Older thing.** Written by hand (#1).\n\n"
        "### Fixed\n- **New fix.** From a fragment (#9).\n- **Older fix.** Also by hand.\n"
    )
    assert "## [0.1.0] - 2026-01-01\n\n### Added\n- First.\n" in text
    assert (
        "[Unreleased]: https://github.com/existence-master/Sentient/compare/v0.2.0...main\n" in text
    )
    assert "[0.2.0]: https://github.com/existence-master/Sentient/releases/tag/v0.2.0\n" in text
    # folding again changes nothing
    assert release.fold_release(text, "0.2.0") == text


def test_towncrier_renders_fragments_in_the_changelog_style(tmp_path):
    shutil.copy(ROOT / "pyproject.toml", tmp_path)
    (tmp_path / "changes").mkdir()
    shutil.copy(ROOT / "changes" / "template.md", tmp_path / "changes")
    (tmp_path / "CHANGELOG.md").write_text(CHANGELOG, encoding="utf-8")
    (tmp_path / "changes" / "12.fixed.md").write_text(
        "**Paused tasks stay paused.** After a restart.\n"
    )
    (tmp_path / "changes" / "15.added.md").write_text("**Alarms can snooze.** From the banner?\n")
    (tmp_path / "changes" / "+tidy.changed.md").write_text("Settings are tidier.\n")

    section = release.draft(tmp_path, "0.2.0", "2026-02-01")

    assert section == (
        "## [0.2.0] - 2026-02-01\n\n"
        "### Added\n- **Alarms can snooze.** From the banner (#15)?\n"
        "- **Older thing.** Written by hand (#1).\n\n"
        "### Changed\n- Settings are tidier.\n\n"
        "### Fixed\n- **Paused tasks stay paused.** After a restart (#12).\n"
        "- **Older fix.** Also by hand.\n"
    )
    # the draft builds a copy: the changelog and the entries are untouched
    assert (tmp_path / "CHANGELOG.md").read_text(encoding="utf-8") == CHANGELOG
    assert (tmp_path / "changes" / "12.fixed.md").exists()
