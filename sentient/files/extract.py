"""Turn a file on disk into text the model can read.

Shared by chat attachments, the file_read tool, and memory import. Heavy
parsers are imported lazily so a missing optional package only disables that
format.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

TEXT_SUFFIXES = {
    ".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".yaml", ".yml", ".xml", ".html", ".htm",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".cpp", ".h", ".cs", ".go", ".rs", ".rb",
    ".php", ".sh", ".ps1", ".sql", ".ini", ".toml", ".cfg", ".log", ".ics", ".eml", ".srt", ".vtt",
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}


def is_text_like(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def extract_text(path: Path, max_chars: int = 200_000) -> str:
    """Best-effort text extraction. Raises ValueError for unsupported formats."""
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        text = path.read_text(encoding="utf-8", errors="replace")
    elif suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        parts = []
        total = 0
        for i, page in enumerate(reader.pages):
            t = page.extract_text() or ""
            parts.append(f"[page {i + 1}]\n{t}")
            total += len(t)
            if total > max_chars:
                break
        text = "\n\n".join(parts)
    elif suffix == ".docx":
        import docx

        d = docx.Document(str(path))
        text = "\n".join(p.text for p in d.paragraphs)
    else:
        raise ValueError(f"cannot read {suffix or 'this'} files as text")
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[... truncated ...]"
    return text
