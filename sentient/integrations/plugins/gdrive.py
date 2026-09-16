"""Google Drive: search, read (export Google formats / extract others), upload from Sentient's files folder."""

from __future__ import annotations

import asyncio
import json
import mimetypes
import secrets as pysecrets
import tempfile
from pathlib import Path

from sentient import paths
from sentient.files.extract import extract_text
from sentient.integrations.base import IntegrationError, itool
from sentient.integrations.common import truncate
from sentient.integrations.google import gapi
from sentient.integrations.plugins._google_base import GooglePlugin
from sentient.tools.base import Risk, ToolContext

API = "https://www.googleapis.com/drive/v3"
UPLOAD = "https://www.googleapis.com/upload/drive/v3/files"
PID = "gdrive"
FIELDS = "id,name,mimeType,modifiedTime,size,webViewLink,owners(displayName,emailAddress)"
EXPORTS = {
    "application/vnd.google-apps.document": ("text/plain", ".txt"),
    "application/vnd.google-apps.spreadsheet": ("text/csv", ".csv"),
    "application/vnd.google-apps.presentation": ("text/plain", ".txt"),
}
CONVERT = {
    ".docx": "application/vnd.google-apps.document", ".doc": "application/vnd.google-apps.document",
    ".txt": "application/vnd.google-apps.document", ".md": "application/vnd.google-apps.document",
    ".xlsx": "application/vnd.google-apps.spreadsheet", ".csv": "application/vnd.google-apps.spreadsheet",
    ".pptx": "application/vnd.google-apps.presentation",
}
MAX_DOWNLOAD = 25 * 1024 * 1024


def _simplify(f: dict) -> dict:
    return {"id": f.get("id"), "name": f.get("name"), "mime_type": f.get("mimeType"), "modified": f.get("modifiedTime"),
            "size": int(f["size"]) if f.get("size") else None, "url": f.get("webViewLink"),
            "owners": [o.get("emailAddress") for o in f.get("owners") or []]}


def drive_query(query: str) -> str:
    q = query.strip()
    if any(tok in q for tok in ("=", " contains ", " in parents", "mimeType", "trashed")):
        return q
    esc = q.replace("\\", "\\\\").replace("'", "\\'")
    return f"(name contains '{esc}' or fullText contains '{esc}') and trashed = false"


@itool(PID, "gdrive_search")
async def gdrive_search(ctx: ToolContext, query: str, max_results: int = 10) -> dict:
    """Search Google Drive files by words in the name or content (e.g. "Q3 budget"), or with Drive query syntax
    (e.g. "mimeType='application/pdf' and modifiedTime > '2026-09-01'"). Returns ids, names, types and links."""
    res = await gapi(ctx, PID, "GET", f"{API}/files", params={
        "q": drive_query(query), "pageSize": max(1, min(int(max_results or 10), 100)),
        "fields": f"files({FIELDS})", "orderBy": "modifiedTime desc", "supportsAllDrives": "true",
        "includeItemsFromAllDrives": "true"})
    files = [_simplify(f) for f in res.get("files") or []]
    return {"query": query, "count": len(files), "files": files}


@itool(PID, "gdrive_read_file")
async def gdrive_read_file(ctx: ToolContext, file_id: str, max_chars: int = 20000) -> dict:
    """Read a Drive file as text. Google Docs/Slides become plain text, Sheets become CSV; PDFs, Word, text and
    code files are extracted. Use gdrive_search first to get the file id."""
    meta = await gapi(ctx, PID, "GET", f"{API}/files/{file_id}", params={"fields": FIELDS, "supportsAllDrives": "true"})
    mime = meta.get("mimeType", "")
    limit = max(1000, min(int(max_chars or 20000), 200000))
    info = _simplify(meta)
    if mime in EXPORTS:
        export_mime, _ = EXPORTS[mime]
        r = await gapi(ctx, PID, "GET", f"{API}/files/{file_id}/export", params={"mimeType": export_mime}, raw=True)
        text = r.content.decode("utf-8", errors="replace")
    elif mime.startswith("application/vnd.google-apps"):
        raise IntegrationError(f"'{meta.get('name')}' is a Google file type ({mime.split('.')[-1]}) that can't be read as text.")
    else:
        if info["size"] and info["size"] > MAX_DOWNLOAD:
            raise IntegrationError(f"'{meta.get('name')}' is too large to read ({info['size'] // 1_000_000} MB).")
        r = await gapi(ctx, PID, "GET", f"{API}/files/{file_id}", params={"alt": "media", "supportsAllDrives": "true"}, raw=True)
        suffix = Path(meta.get("name") or "").suffix or (mimetypes.guess_extension(mime) or "")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / f"file{suffix.lower()}"
            p.write_bytes(r.content)
            try:
                text = await asyncio.to_thread(extract_text, p, limit + 1000)
            except ValueError as exc:
                raise IntegrationError(f"'{meta.get('name')}' ({mime}) can't be read as text: {exc}.") from exc
    text, cut = truncate(text, limit)
    return {**info, "content": text, "truncated": cut}


def _safe_local(name: str) -> Path:
    root = paths.files_dir().resolve()
    target = (root / name).resolve()
    if root not in target.parents:
        raise IntegrationError("That file isn't inside Sentient's files folder.")
    if not target.is_file():
        raise IntegrationError(f"There's no file called '{name}' in Sentient's files folder. Use file_list to check.")
    return target


@itool(PID, "gdrive_upload_file", risk=Risk.write)
async def gdrive_upload_file(ctx: ToolContext, name: str, folder_id: str | None = None,
                             convert_to_google_format: bool = False, drive_name: str | None = None) -> dict:
    """Upload a file from Sentient's files folder (e.g. "charts/sales.png" or "uploads/report.docx") to Google Drive.
    Set convert_to_google_format=true to turn Word/Excel/PowerPoint/CSV/text into Google Docs/Sheets/Slides."""
    path = _safe_local(name)
    if path.stat().st_size > MAX_DOWNLOAD:
        raise IntegrationError("That file is too large to upload from here (25 MB limit).")
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    metadata: dict = {"name": drive_name or path.name}
    if folder_id:
        metadata["parents"] = [folder_id]
    if convert_to_google_format and path.suffix.lower() in CONVERT:
        metadata["mimeType"] = CONVERT[path.suffix.lower()]
        if metadata["name"].lower().endswith(path.suffix.lower()):
            metadata["name"] = metadata["name"][: -len(path.suffix)]
    boundary = f"sentient{pysecrets.token_hex(12)}"
    body = (
        f"--{boundary}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n{json.dumps(metadata)}\r\n"
        f"--{boundary}\r\nContent-Type: {mime}\r\n\r\n"
    ).encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    res = await gapi(ctx, PID, "POST", UPLOAD, params={"uploadType": "multipart", "fields": FIELDS,
                                                       "supportsAllDrives": "true"},
                     content=body, headers={"Content-Type": f"multipart/related; boundary={boundary}"})
    return {"uploaded": True, **_simplify(res)}


class GDrivePlugin(GooglePlugin):
    id = PID
    display_name = "Google Drive"
    description = (
        "Find and read the files in your Google Drive, including Docs, Sheets, Slides, PDFs and Word files, and "
        "upload files Sentient made (like charts or reports) back to Drive."
    )
    category = "productivity"
    icon = "google-drive"
    api_name = "Google Drive API"
    api_slug = "drive.googleapis.com"
    selection_hint = "Use to search for, read, or upload files in Google Drive."
    tools = [gdrive_search, gdrive_read_file, gdrive_upload_file]


PLUGIN = GDrivePlugin()
