"""Freeze the Sentient engine and stage it for electron-builder.

    .venv/Scripts/python.exe packaging/build_engine.py          # Windows
    .venv/bin/python packaging/build_engine.py                  # macOS / Linux

Runs PyInstaller against ``packaging/sentient-engine.spec`` (onedir), smoke-tests
the produced executable, then copies it to ``desktop/build/engine`` where
``desktop/electron-builder.yml`` picks it up as ``extraResources`` (installed
next to the app as ``resources/engine/sentient-engine.exe``).

``--skip-if-fresh`` reuses an existing staged engine when nothing under
``sentient/`` or ``packaging/`` has changed since it was built.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SPEC = REPO / "packaging" / "sentient-engine.spec"
WORK = REPO / "packaging" / "build"
DIST = REPO / "packaging" / "dist"
BUNDLE = DIST / "sentient-engine"
STAGE = REPO / "desktop" / "build" / "engine"
EXE_NAME = "sentient-engine.exe" if os.name == "nt" else "sentient-engine"


def log(message: str) -> None:
    print(f"[engine] {message}", flush=True)


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def newest_source_mtime() -> float:
    newest = 0.0
    for root in (REPO / "sentient", REPO / "packaging"):
        for f in root.rglob("*"):
            if not f.is_file() or "__pycache__" in f.parts or f.suffix == ".pyc":
                continue
            if WORK in f.parents or DIST in f.parents:
                continue
            newest = max(newest, f.stat().st_mtime)
    return newest


def is_fresh() -> bool:
    exe = STAGE / EXE_NAME
    return exe.exists() and exe.stat().st_mtime >= newest_source_mtime()


def freeze(clean: bool) -> None:
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--distpath", str(DIST),
           "--workpath", str(WORK), "--log-level", "WARN"]
    if clean:
        cmd.append("--clean")
    cmd.append(str(SPEC))
    log(" ".join(cmd))
    result = subprocess.run(cmd, cwd=REPO)
    if result.returncode != 0:
        raise SystemExit(f"PyInstaller failed with exit code {result.returncode}")


def verify() -> None:
    """The frozen engine must at least import and answer `version`."""
    exe = BUNDLE / EXE_NAME
    if not exe.exists():
        raise SystemExit(f"expected {exe} to exist")
    out = subprocess.run([str(exe), "version"], capture_output=True, text=True, timeout=180)
    if out.returncode != 0 or "sentient" not in out.stdout.lower():
        raise SystemExit(f"frozen engine failed `version`:\n{out.stdout}\n{out.stderr}")
    log(f"frozen engine says: {out.stdout.strip()}")


def stage() -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(BUNDLE, STAGE)
    log(f"staged -> {STAGE} ({dir_size(STAGE) / 1e6:.0f} MB)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", action="store_true", help="discard PyInstaller's cache first")
    parser.add_argument("--skip-if-fresh", action="store_true",
                        help="reuse the staged engine when nothing changed")
    args = parser.parse_args()

    if args.skip_if_fresh and is_fresh():
        log(f"up to date, reusing {STAGE}")
        return 0

    started = time.perf_counter()
    freeze(args.clean)
    verify()
    stage()
    log(f"done in {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
