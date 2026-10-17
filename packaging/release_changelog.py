"""Turn the changelog entries in changes/ into a version section of CHANGELOG.md.

    python packaging/release_changelog.py 3.0.0-alpha.1 --draft   # print the section, change nothing
    python packaging/release_changelog.py 3.0.0-alpha.1           # write CHANGELOG.md, remove the entries

towncrier writes the entries under the "towncrier release notes start" marker as a new version. Entries written
straight into CHANGELOG.md (everything before towncrier was set up) sit under the same marker with their own
"### Added", "### Fixed" ... headings, so after towncrier the new version would list each heading twice. This
runs towncrier and then folds each heading into one list, newest entries first, and adds the version's link.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_URL = "https://github.com/existence-master/Sentient"
# Keep a Changelog order; anything else keeps its place after these
ORDER = ["Added", "Changed", "Deprecated", "Removed", "Fixed", "Security"]
# An entry that already ends with its own "(#123)" gets a second one from towncrier: "(#123) (#123)".
_DUP_REF = re.compile(r"(\(#\d+\))(?:\s*)+")


def _trim(lines: list[str]) -> list[str]:
    start, end = 0, len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return lines[start:end]


def fold_release(changelog: str, version: str) -> str:
    """Merge repeated "### Heading" lists inside the section of `version` and link its heading."""
    title = f"## [{version}]"
    start = changelog.index(title)
    nxt = changelog.find("\n## ", start + len(title))
    end = len(changelog) if nxt == -1 else nxt + 1
    head, _, body = changelog[start:end].partition("\n")

    intro: list[str] = []
    groups: dict[str, list[list[str]]] = {}
    current = intro
    for line in body.splitlines():
        if line.startswith("### "):
            current = []
            groups.setdefault(line[4:].strip(), []).append(current)
        else:
            current.append(line)

    out = [head, ""]
    if _trim(intro):
        out += [*_trim(intro), ""]
    rank = {name: i for i, name in enumerate(ORDER)}
    for name in sorted(groups, key=lambda n: rank.get(n, len(ORDER))):
        items = [_DUP_REF.sub(r"", line) for chunk in groups[name] for line in _trim(chunk)]
        if items:
            out += [f"### {name}", *items, ""]
    text = changelog[:start] + "\n".join(out) + "\n" + changelog[end:]

    tag = f"v{version}"
    if f"[{version}]: " not in text:
        text = re.sub(
            r"^\[Unreleased\]: .*$",
            f"[Unreleased]: {REPO_URL}/compare/{tag}...main\n[{version}]: {REPO_URL}/releases/tag/{tag}",
            text,
            count=1,
            flags=re.MULTILINE,
        )
    return text


def _towncrier(root: Path, version: str, date: str, *extra: str) -> None:
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "towncrier",
            "build",
            "--version",
            version,
            "--date",
            date,
            "--dir",
            str(root),
            "--config",
            str(root / "pyproject.toml"),
            *extra,
        ],
        cwd=root,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    sys.stderr.write(done.stdout + done.stderr)  # progress lines; stdout is for the draft
    done.check_returncode()


def section(changelog: str, version: str) -> str:
    start = changelog.index(f"## [{version}]")
    nxt = changelog.find("\n## ", start + 1)
    return changelog[start : None if nxt == -1 else nxt + 1].rstrip() + "\n"


def draft(root: Path, version: str, date: str) -> str:
    """The section a release would write: a real towncrier build on a copy of `root`."""
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp)
        shutil.copy(root / "pyproject.toml", copy)
        shutil.copy(root / "CHANGELOG.md", copy)
        shutil.copytree(root / "changes", copy / "changes")
        # its own repository, so towncrier stages the copy there and never in a repository around it
        subprocess.run(["git", "init", "-q"], cwd=copy, check=True)
        _towncrier(copy, version, date, "--keep")
        return section(
            fold_release((copy / "CHANGELOG.md").read_text(encoding="utf-8"), version), version
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("version", help="the version without a leading v, e.g. 3.0.0-alpha.1")
    parser.add_argument("--date", default=dt.date.today().isoformat())
    parser.add_argument(
        "--draft", action="store_true", help="print the new section, change nothing"
    )
    args = parser.parse_args(argv)

    if args.draft:
        sys.stdout.buffer.write(draft(ROOT, args.version, args.date).encode("utf-8"))
        return 0

    _towncrier(ROOT, args.version, args.date, "--yes")
    path = ROOT / "CHANGELOG.md"
    path.write_text(fold_release(path.read_text(encoding="utf-8"), args.version), encoding="utf-8")
    subprocess.run(["git", "add", "CHANGELOG.md"], check=True, cwd=ROOT)
    print(f"CHANGELOG.md has a {args.version} section now. Check it, then commit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
