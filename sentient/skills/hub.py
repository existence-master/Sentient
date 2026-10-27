"""Skills hub (#210): find skills other people wrote, from a few sources, and install them for review.

A source is either

- a public **GitHub repository** of AgentSkills folders (``<path>/<skill>/SKILL.md`` plus its ``scripts/``,
  ``references/`` and ``assets/``; https://agentskills.io/specification). This is the layout of
  ``anthropics/skills`` and ``openai/skills``, and the format Hermes and OpenClaw read; or
- a site that publishes a **skills index** at ``/.well-known/skills/index.json`` (``{"skills": [{"name",
  "description", "files": [...]}]}``, the Hermes ``well-known`` source), with each file at ``<base>/<name>/<file>``.

Sentient lists a small default set (``DEFAULT_SOURCES``, trust ``trusted``). Sources the user adds are ``community``
and kept in ``skills/.hub/sources.json``; installing from one needs an explicit extra yes (``confirm_source``).

Everything fetched is outside content (ADR 0018). Nothing here shows it to the model: a skill is downloaded into a
quarantine folder, checked with the safety check of skills from outside (``vet.py``, ADR 0027), and only then staged
in ``pending/`` with its ``skill_imports`` record (source ``hub``). A ``high`` verdict blocks the install unless the
user says "install anyway" (``accept_risk``), and approving it still needs "approve anyway". The exact version (the
commit, or the files' hashes for an index source) is pinned in the lockfile (``lockfile.py``); an update is fetched
the same way and waits in review again, with the list of files it changes.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

import yaml

from sentient.skills import vet
from sentient.skills.loader import slugify, valid_name
from sentient.skills.lockfile import HubLock, file_changes
from sentient.store.db import now_iso

log = logging.getLogger(__name__)

API = "https://api.github.com"
RAW = "https://raw.githubusercontent.com"
TIMEOUT_S = 20.0
LISTING_TTL_S = 15 * 60
MAX_SKILLS = 150         # skills listed from one source
MAX_DEPTH = 4            # how deep below the source's folder a SKILL.md may be (Hermes groups skills by category)
MAX_FILES = 120          # files one skill may bring
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_BUNDLE_BYTES = 12 * 1024 * 1024
MAX_INDEX_BYTES = 1024 * 1024
CONCURRENCY = 8
SECRET_NAMES = {"auth.json", "id_rsa", "id_ed25519", "credentials", "wallet.dat"}
_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
_SENTIENT_AUTHORS = {"user", "assistant", "community"}


class HubError(Exception):
    """A plain sentence for the user, with the HTTP status the gateway answers with."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


class HubConfirm(Exception):
    """The install needs an extra yes: ``confirm_source`` (a source the user added) or ``high_risk`` (the safety check
    found high-risk problems). ``vetting`` is the report for ``high_risk``."""

    def __init__(self, code: str, message: str, vetting: dict | None = None):
        super().__init__(message)
        self.code = code
        self.vetting = vetting


# ---------------------------------------------------------------------------- sources
@dataclass
class Source:
    id: str
    name: str
    kind: str                 # github | index
    url: str                  # shown to the user: the repository or the site
    trust: str = "community"  # trusted (listed by Sentient) | community (added by the user)
    default: bool = False
    description: str = ""
    owner: str = ""
    repo: str = ""
    ref: str = "HEAD"
    path: str = ""            # folder inside the repository that holds the skills
    index_url: str = ""
    added_at: str | None = None

    @property
    def publisher(self) -> str:
        """Who made a skill that doesn't say: the publisher of a default source, else the repository owner or site."""
        if self.default:
            return self.name
        return self.owner or urlsplit(self.index_url or self.url).netloc

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v not in ("", None) or k in {"path", "added_at"}}


DEFAULT_SOURCES = [
    Source(
        id="anthropic", name="Anthropic", kind="github", url="https://github.com/anthropics/skills", trust="trusted",
        default=True, owner="anthropics", repo="skills", ref="main", path="skills",
        description="Example skills from Anthropic, the makers of Claude, published with the Agent Skills format.",
    ),
    Source(
        id="openai", name="OpenAI", kind="github", url="https://github.com/openai/skills", trust="trusted",
        default=True, owner="openai", repo="skills", ref="main", path="skills/.curated",
        description="The curated skills catalog OpenAI publishes for Codex, in the same format.",
    ),
]


def parse_source_url(url: str, name: str | None = None) -> Source:
    """A source from what the user pasted: a GitHub repository or folder link, or a site with a skills index."""
    url = (url or "").strip()
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if parts.scheme not in {"http", "https"} or not host:
        raise HubError("Use a web address that starts with https://, such as a GitHub repository.")
    if parts.scheme == "http" and host not in _LOCAL_HOSTS:
        raise HubError("Use an https:// address. Plain http:// only works for a source on this computer.")
    label = " ".join(str(name or "").split())[:60]
    if host in {"github.com", "www.github.com"}:
        segs = [s for s in parts.path.split("/") if s]
        if len(segs) < 2:
            raise HubError("Use the address of a GitHub repository, such as https://github.com/owner/repo.")
        owner, repo = segs[0], segs[1].removesuffix(".git")
        ref, rest = "HEAD", []
        if len(segs) >= 4 and segs[2] in {"tree", "blob"}:
            ref, rest = segs[3], segs[4:]
        if rest and rest[-1] == "SKILL.md":
            rest = rest[:-1]
        path = "/".join(rest)
        if not _safe_rel(path, allow_empty=True) or not re.fullmatch(r"[\w.-]+", owner) or not re.fullmatch(r"[\w.-]+", repo):
            raise HubError("That GitHub address has a path Sentient can't use.")
        sid = slugify(f"{owner}-{repo}-{path}" if path else f"{owner}-{repo}")[:48].strip("-")
        shown = f"https://github.com/{owner}/{repo}" + (f"/tree/{ref}/{path}" if path or ref != "HEAD" else "")
        return Source(id=sid, name=label or f"{owner}/{repo}" + (f"/{path}" if path else ""), kind="github", url=shown,
                      owner=owner, repo=repo, ref=ref, path=path, added_at=now_iso())
    base = url.split("#", 1)[0].split("?", 1)[0].rstrip("/")
    index_url = base if base.endswith("/index.json") else f"{base}/.well-known/skills/index.json"
    sid = slugify(f"{parts.netloc}{parts.path.removesuffix('/index.json')}".replace(".well-known/skills", ""))[:48].strip("-")
    return Source(id=sid or "site", name=label or parts.netloc, kind="index", url=base, index_url=index_url,
                  added_at=now_iso())


def _safe_rel(rel: str, *, allow_empty: bool = False) -> bool:
    """A relative path inside a skill or repository: no ``..``, no drive or absolute path, no backslashes."""
    if not rel:
        return allow_empty
    if rel.startswith("/") or "\\" in rel or ":" in rel or "\0" in rel:
        return False
    return all(p not in {"", ".", ".."} for p in rel.split("/"))


def _keep(rel: str) -> bool:
    """Files a skill brings: no dotfiles or dot folders (never downloaded, like vet.py never reads them), no secrets."""
    parts = rel.split("/")
    return _safe_rel(rel) and not any(p.startswith(".") for p in parts) and parts[-1] not in SECRET_NAMES \
        and "__pycache__" not in parts


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()


def fingerprint(files: dict[str, str]) -> str:
    """One hash for a version of a skill: SHA-256 over each file's path and git blob hash, in order. GitHub's tree
    listing gives the same value without downloading anything."""
    h = hashlib.sha256()
    for rel in sorted(files):
        h.update(f"{rel}\0{files[rel]}\n".encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------- what a skill asks for
def _strs(value: Any) -> list[str]:
    if isinstance(value, str):
        return [s for s in re.split(r"[\s,]+", value) if s]
    if isinstance(value, list):
        out = []
        for v in value:
            if isinstance(v, dict):
                v = v.get("name") or v.get("key")
            if v not in (None, ""):
                out.append(str(v))
        return out
    return []


def _metadata(meta: dict, key: str) -> dict:
    m = meta.get("metadata")
    if not isinstance(m, dict):
        return {}
    v = m.get(key)
    if isinstance(v, str):  # OpenClaw allows a JSON object written as a string
        with contextlib.suppress(ValueError):
            v = json.loads(v)
    return v if isinstance(v, dict) else {}


def _dedupe(items: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for i in items:
        seen.setdefault(i.strip(), None)
    return [i for i in seen if i][:20]


def asks_of(meta: dict, files: list[str]) -> list[dict]:
    """What a skill asks for, in plain words, from its header and its files. Reads the Agent Skills fields
    (``allowed-tools``, ``compatibility``), Hermes's (``platforms``, ``required_environment_variables``,
    ``metadata.hermes.requires_tools``) and OpenClaw's (``metadata.openclaw.requires.bins`` / ``env``, ``os``)."""
    hermes, claw = _metadata(meta, "hermes"), _metadata(meta, "openclaw")
    requires = claw.get("requires") if isinstance(claw.get("requires"), dict) else {}
    out: list[dict] = []

    def add(kind: str, label: str, items: list[str]) -> None:
        items = _dedupe(items)
        if items:
            out.append({"kind": kind, "label": label, "items": items})

    add("tools", "Tools it wants to use", [*_strs(meta.get("allowed-tools")), *_strs(meta.get("requires_tools")),
                                           *_strs(hermes.get("requires_tools")), *_strs(hermes.get("requires_toolsets"))])
    add("apps", "Programs it needs on this computer", [*_strs(requires.get("bins")), *_strs(requires.get("anyBins"))])
    add("keys", "Keys or settings it asks for", [*_strs(meta.get("required_environment_variables")),
                                                 *_strs(requires.get("env")), *_strs(claw.get("primaryEnv"))])
    add("platform", "Works on", [*_strs(meta.get("platforms")), *_strs(claw.get("os"))])
    compat = " ".join(str(meta.get("compatibility") or "").split())[:500]
    if compat:
        out.append({"kind": "needs", "label": "What it needs", "items": [compat]})
    programs = [f for f in files if Path(f).suffix.lower() in vet._PROGRAMS]
    if programs:
        out.append({"kind": "programs", "label": "Scripts it brings", "items": programs[:20],
                    "more": max(0, len(programs) - 20)})
    return out


def _author(meta: dict, fallback: str) -> str:
    m = meta.get("metadata") if isinstance(meta.get("metadata"), dict) else {}
    for cand in (m.get("author"), meta.get("author")):
        if isinstance(cand, str) and cand.strip() and cand.strip().lower() not in _SENTIENT_AUTHORS:
            return " ".join(cand.split())[:80]
    return fallback


def _version(meta: dict) -> str | None:
    m = meta.get("metadata") if isinstance(meta.get("metadata"), dict) else {}
    v = meta.get("version", m.get("version"))
    return str(v)[:40] if v not in (None, "") else None


def parse_header(text: str) -> tuple[dict, str] | None:
    m = _FRONTMATTER.match(text.replace("\r\n", "\n").lstrip("﻿"))
    if not m:
        return None
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        return None
    return (meta, m.group(2)) if isinstance(meta, dict) else None


def restamp(text: str, name: str) -> str:
    """The SKILL.md as installed: its header gets Sentient's name and ``author: community`` (the maker it named is
    kept as ``metadata.author``), and the steps stay exactly as downloaded, so hidden text stays visible to the
    safety check."""
    parsed = parse_header(text)
    if parsed is None:
        raise HubError("This skill's header can't be read, so it can't be installed.")
    meta, body = parsed
    made_by = meta.get("author")
    metadata = meta.get("metadata") if isinstance(meta.get("metadata"), dict) else None
    if isinstance(made_by, str) and made_by.strip().lower() not in _SENTIENT_AUTHORS:
        metadata = {**(metadata or {}), "author": metadata.get("author", made_by) if metadata else made_by}
    meta = {**meta, "name": name, "author": "community"}
    if metadata is not None:
        meta["metadata"] = metadata
    meta["description"] = " ".join(str(meta.get("description") or "").split())
    front = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True).strip()
    return f"---\n{front}\n---\n\n{body.strip()}\n"


# ---------------------------------------------------------------------------- the hub
@dataclass
class _Entry:
    """One skill in a source listing, with what install needs to fetch it."""
    source: str
    path: str                   # its folder below the source's folder ("." when the source is one skill)
    name: str
    description: str
    author: str
    license: str | None
    version: str | None
    asks: list[dict]
    files: list[str]
    url: str
    fingerprint: str | None     # known from the listing for GitHub sources
    problem: str | None = None  # why it can't be installed
    blobs: dict[str, str] = field(default_factory=dict)
    sizes: dict[str, int] = field(default_factory=dict)
    repo_path: str = ""
    skill_md: str = ""

    def to_dict(self) -> dict:
        return {"source": self.source, "path": self.path, "name": self.name, "description": self.description,
                "author": self.author, "license": self.license, "version": self.version, "asks": self.asks,
                "files": self.files, "url": self.url, "fingerprint": self.fingerprint, "problem": self.problem}


class SkillsHub:
    def __init__(self, library: Any, *, transport: Any = None):
        self.lib = library
        self.transport = transport  # tests pass an httpx transport; respx works on the default one too
        self._listings: dict[str, tuple[float, dict, dict[str, _Entry]]] = {}
        self._md_cache: dict[str, str] = {}
        self._busy = asyncio.Lock()

    # ------------------------------------------------------------------ files
    @property
    def state_dir(self) -> Path:
        return self.lib.root / ".hub"

    @property
    def lock(self) -> HubLock:
        return HubLock(self.state_dir / "lock.json")

    def _user_sources(self) -> list[Source]:
        try:
            raw = json.loads((self.state_dir / "sources.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        out = []
        for item in raw if isinstance(raw, list) else []:
            with contextlib.suppress(TypeError):
                if isinstance(item, dict):
                    out.append(Source(**{**item, "trust": "community", "default": False}))
        return out

    def _save_sources(self, sources: list[Source]) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        path = self.state_dir / "sources.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps([asdict(s) for s in sources], indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)

    def sources(self) -> list[Source]:
        return [*DEFAULT_SOURCES, *self._user_sources()]

    def source(self, sid: str) -> Source:
        for s in self.sources():
            if s.id == sid:
                return s
        raise HubError("There is no such source. It may have been removed.", 404)

    def add_source(self, url: str, name: str | None = None) -> Source:
        src = parse_source_url(url, name)
        existing = self.sources()
        for s in existing:
            same = (s.kind == src.kind == "github" and (s.owner.lower(), s.repo.lower(), s.path) ==
                    (src.owner.lower(), src.repo.lower(), src.path)) or (src.kind == "index" and s.index_url == src.index_url)
            if same:
                raise HubError(f"{s.name} is already in your sources.", 409)
        ids = {s.id for s in existing}
        base, n = src.id, 1
        while src.id in ids:
            n += 1
            src.id = f"{base[:44]}-{n}"
        self._save_sources([*self._user_sources(), src])
        return src

    def remove_source(self, sid: str) -> None:
        src = self.source(sid)
        if src.default:
            raise HubError("Sources Sentient lists can't be removed.", 409)
        self._save_sources([s for s in self._user_sources() if s.id != sid])
        self._listings.pop(sid, None)

    # ------------------------------------------------------------------ HTTP
    def _client(self):
        import httpx

        return httpx.AsyncClient(timeout=TIMEOUT_S, follow_redirects=False, transport=self.transport,
                                 headers={"User-Agent": "Sentient-skills-hub"})

    async def _get(self, client, url: str, *, limit: int, what: str, accept: str = "*/*") -> bytes:
        import httpx

        try:
            async with client.stream("GET", url, headers={"Accept": accept}) as resp:
                if resp.status_code in {301, 302, 307, 308} and urlsplit(url).netloc == "api.github.com":
                    raise HubError("That repository has moved. Add it again with its new address.")
                if resp.status_code in {403, 429} and resp.headers.get("x-ratelimit-remaining") == "0":
                    reset = int(resp.headers.get("x-ratelimit-reset") or 0)
                    mins = max(1, round((reset - time.time()) / 60)) if reset else 60
                    raise HubError(f"GitHub limits how often it can be asked without an account. Try again in about "
                                   f"{mins} minute{'s' if mins != 1 else ''}.", 503)
                if resp.status_code == 404:
                    raise HubError(f"Couldn't find {what}. Check the address, and that it is public.", 404)
                if 300 <= resp.status_code < 400:
                    raise HubError(f"{what.capitalize()} sent Sentient somewhere else, so it wasn't followed.", 502)
                if resp.status_code >= 400:
                    raise HubError(f"The site answered with an error ({resp.status_code}) for {what}.", 502)
                body = b""
                async for chunk in resp.aiter_bytes():
                    body += chunk
                    if len(body) > limit:
                        raise HubError(f"{what.capitalize()} is too large.", 413)
                return body
        except httpx.HTTPError as exc:
            raise HubError(f"Couldn't reach {urlsplit(url).netloc}: {type(exc).__name__}.", 502) from exc

    async def _json(self, client, url: str, *, what: str, limit: int = 8 * 1024 * 1024) -> Any:
        body = await self._get(client, url, limit=limit, what=what, accept="application/vnd.github+json, application/json")
        try:
            return json.loads(body.decode("utf-8", errors="replace"))
        except ValueError as exc:
            raise HubError(f"{what.capitalize()} isn't readable.", 502) from exc

    # ------------------------------------------------------------------ browsing
    async def browse(self, sid: str, *, refresh: bool = False) -> dict:
        """``{source, commit, fetched_at, skills: [...], truncated}``; each skill carries ``installed`` when it is in
        the lockfile (``{name, status, update_available}``)."""
        src = self.source(sid)
        listing, _ = await self._listing(src, refresh=refresh)
        locked = self.lock.all()
        by_path = {(e.get("source"), e.get("path")): (n, e) for n, e in locked.items()}
        skills = []
        for item in listing["skills"]:
            found = by_path.get((sid, item["path"]))
            installed = None
            if found:
                name, entry = found
                newest = item.get("fingerprint")
                installed = {"name": name, "status": entry.get("status"),
                             "update_available": bool(newest) and newest != entry.get("fingerprint")}
            skills.append({**item, "installed": installed})
        return {**listing, "skills": skills}

    async def _listing(self, src: Source, *, refresh: bool = False) -> tuple[dict, dict[str, _Entry]]:
        cached = self._listings.get(src.id)
        if cached and not refresh and time.monotonic() - cached[0] < LISTING_TTL_S:
            return cached[1], cached[2]
        async with self._client() as client:
            entries, commit, truncated = await (self._github(client, src) if src.kind == "github" else self._index(client, src))
        listing = {"source": src.to_dict(), "commit": commit, "fetched_at": now_iso(), "truncated": truncated,
                   "skills": [e.to_dict() for e in sorted(entries.values(), key=lambda e: e.name)]}
        self._listings[src.id] = (time.monotonic(), listing, entries)
        return listing, entries

    async def _github(self, client, src: Source) -> tuple[dict[str, _Entry], str, bool]:
        o, r = src.owner, src.repo
        what = f"the GitHub repository {o}/{r}"
        commit = await self._json(client, f"{API}/repos/{o}/{r}/commits/{quote(src.ref, safe='')}", what=what)
        sha = str((commit or {}).get("sha") or "")
        if not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise HubError(f"Couldn't read the latest version of {o}/{r}.", 502)
        tree_ref = f"{sha}:{src.path}" if src.path else sha
        tree = await self._json(client, f"{API}/repos/{o}/{r}/git/trees/{quote(tree_ref, safe=':/')}?recursive=1",
                                what=f"the folder {src.path or '/'} in {o}/{r}")
        blobs: dict[str, tuple[str, int]] = {}
        for e in (tree or {}).get("tree") or []:
            if isinstance(e, dict) and e.get("type") == "blob" and e.get("mode") != "120000" and _safe_rel(str(e.get("path") or "")):
                blobs[str(e["path"])] = (str(e.get("sha") or ""), int(e.get("size") or 0))
        folders: list[str] = []
        for p in sorted({p.rpartition("/")[0] for p in blobs if p == "SKILL.md" or p.endswith("/SKILL.md")},
                        key=lambda f: (f.count("/"), f)):
            if (p.count("/") + 1 if p else 0) > MAX_DEPTH:
                continue
            if any(f == "" or p.startswith(f + "/") for f in folders):
                continue  # a skill inside another skill's folder belongs to that skill
            folders.append(p)
        truncated = bool((tree or {}).get("truncated")) or len(folders) > MAX_SKILLS
        folders = folders[:MAX_SKILLS]

        sem = asyncio.Semaphore(CONCURRENCY)

        async def one(folder: str) -> _Entry:
            prefix = f"{folder}/" if folder else ""
            files = {p[len(prefix):]: blobs[p] for p in blobs if p.startswith(prefix) and _keep(p[len(prefix):])}
            repo_path = "/".join(x for x in (src.path, folder) if x)
            md_sha = files["SKILL.md"][0]
            text = self._md_cache.get(md_sha)
            if text is None:
                async with sem:
                    raw = await self._get(client, f"{RAW}/{o}/{r}/{sha}/{quote(repo_path + '/SKILL.md' if repo_path else 'SKILL.md')}",
                                          limit=vet.MAX_FETCH_BYTES, what="a SKILL.md")
                text = raw.decode("utf-8", errors="replace")
                self._md_cache[md_sha] = text
            entry = self._entry(src, folder or ".", text, sorted(files), fallback=folder.rpartition("/")[2] or r,
                                url=f"https://github.com/{o}/{r}/tree/{sha}/{repo_path}".rstrip("/"))
            entry.blobs = {k: v[0] for k, v in files.items()}
            entry.sizes = {k: v[1] for k, v in files.items()}
            entry.fingerprint = fingerprint(entry.blobs)
            entry.repo_path = repo_path
            if len(files) > MAX_FILES or sum(entry.sizes.values()) > MAX_BUNDLE_BYTES:
                entry.problem = entry.problem or (f"It is too large to install here ({len(files)} files, "
                                                  f"{sum(entry.sizes.values()) // (1024 * 1024) + 1} MB).")
            return entry

        results = await asyncio.gather(*(one(f) for f in folders), return_exceptions=True)
        entries = {}
        for folder, res in zip(folders, results, strict=True):
            if isinstance(res, _Entry):
                entries[res.path] = res
            else:
                log.info("skills hub: skipped %s/%s/%s: %s", o, r, folder, res)
        return entries, sha, truncated

    async def _index(self, client, src: Source) -> tuple[dict[str, _Entry], str | None, bool]:
        host = urlsplit(src.index_url).netloc
        data = await self._json(client, src.index_url, what=f"the skills index on {host}", limit=MAX_INDEX_BYTES)
        raw = data.get("skills") if isinstance(data, dict) else None
        if not isinstance(raw, list):
            raise HubError(f"{host} doesn't publish a skills index Sentient can read.", 502)
        base = src.index_url.removesuffix("/index.json")
        items = [i for i in raw if isinstance(i, dict) and isinstance(i.get("name"), str) and valid_name(i["name"])]
        truncated = len(items) > MAX_SKILLS
        sem = asyncio.Semaphore(CONCURRENCY)

        async def one(item: dict) -> _Entry:
            name = item["name"]
            files = [f for f in item.get("files") or ["SKILL.md"] if isinstance(f, str)]
            if "SKILL.md" not in files:
                files.append("SKILL.md")
            async with sem:
                raw_md = await self._get(client, f"{base}/{name}/SKILL.md", limit=vet.MAX_FETCH_BYTES, what="a SKILL.md")
            text = raw_md.decode("utf-8", errors="replace")
            kept = sorted(f for f in files if _keep(f))
            entry = self._entry(src, name, text, kept, fallback=name, url=f"{base}/{name}/")
            if item.get("version") not in (None, ""):
                entry.version = str(item["version"])[:40]
            if not entry.description:
                entry.description = " ".join(str(item.get("description") or "").split())[:1024]
            if len(kept) > MAX_FILES:
                entry.problem = f"It is too large to install here ({len(kept)} files)."
            entry.skill_md = text
            return entry

        results = await asyncio.gather(*(one(i) for i in items[:MAX_SKILLS]), return_exceptions=True)
        entries = {r.path: r for r in results if isinstance(r, _Entry)}
        return entries, None, truncated

    def _entry(self, src: Source, path: str, text: str, files: list[str], *, fallback: str, url: str) -> _Entry:
        parsed = parse_header(text)
        meta, problem = (parsed[0], None) if parsed else ({}, "Its header can't be read, so it can't be installed.")
        name = slugify(str(meta.get("name") or fallback))
        if not valid_name(name):
            name = slugify(fallback) if valid_name(slugify(fallback)) else "hub-skill"
        lic = meta.get("license")
        return _Entry(
            source=src.id, path=path, name=name,
            description=" ".join(str(meta.get("description") or "").split())[:1024],
            author=_author(meta, src.publisher), license=" ".join(str(lic).split())[:120] if lic else None,
            version=_version(meta), asks=asks_of(meta, files), files=files, url=url, fingerprint=None, problem=problem,
        )

    async def _find(self, sid: str, path: str, *, refresh: bool = False) -> tuple[Source, dict, _Entry]:
        src = self.source(sid)
        listing, entries = await self._listing(src, refresh=refresh)
        entry = entries.get(path)
        if entry is None and not refresh:
            listing, entries = await self._listing(src, refresh=True)
            entry = entries.get(path)
        if entry is None:
            raise HubError(f"{src.name} no longer has that skill.", 404)
        return src, listing, entry

    # ------------------------------------------------------------------ fetching one skill
    async def _download(self, src: Source, listing: dict, entry: _Entry, dest: Path) -> dict[str, str]:
        """Download every file of ``entry`` into ``dest``; returns ``{relative path: git blob sha}``. GitHub files are
        fetched at the pinned commit and must match the tree's hashes."""
        dest.mkdir(parents=True, exist_ok=True)
        root = dest.resolve()
        total = 0
        out: dict[str, str] = {}
        sem = asyncio.Semaphore(CONCURRENCY)
        if entry.problem:
            raise HubError(entry.problem, 422)

        async with self._client() as client:
            async def one(rel: str) -> tuple[str, bytes]:
                if src.kind == "github":
                    repo_rel = "/".join(x for x in (entry.repo_path, rel) if x)
                    url = f"{RAW}/{src.owner}/{src.repo}/{listing['commit']}/{quote(repo_rel)}"
                else:
                    url = f"{src.index_url.removesuffix('/index.json')}/{entry.path}/{quote(rel)}"
                async with sem:
                    return rel, await self._get(client, url, limit=MAX_FILE_BYTES, what=f"the file {rel}")

            results = await asyncio.gather(*(one(rel) for rel in entry.files))
        for rel, data in results:
            total += len(data)
            if total > MAX_BUNDLE_BYTES:
                raise HubError("This skill is too large to install here.", 413)
            sha = git_blob_sha(data)
            if src.kind == "github" and entry.blobs.get(rel) and entry.blobs[rel] != sha:
                raise HubError(f"{rel} didn't match the version GitHub listed, so nothing was installed.", 502)
            target = (dest / rel).resolve()
            if not target.is_relative_to(root):
                raise HubError("This skill has a file path Sentient can't use.", 422)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            out[rel] = sha
        if "SKILL.md" not in out:
            raise HubError("That skill has no SKILL.md.", 422)
        return out

    @contextlib.contextmanager
    def _quarantine(self):
        folder = self.state_dir / "quarantine" / uuid.uuid4().hex[:12]
        try:
            yield folder
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    async def preview(self, sid: str, path: str) -> dict:
        """Everything the preview shows, without installing: the listing entry, the SKILL.md as written, its files
        and the safety check of the downloaded folder."""
        src, listing, entry = await self._find(sid, path)
        out: dict = {**entry.to_dict(), "source_info": src.to_dict(), "commit": listing.get("commit")}
        if entry.problem:
            return {**out, "skill_md": entry.skill_md or None, "vetting": None}
        with self._quarantine() as q:
            blobs = await self._download(src, listing, entry, q / "skill")
            text = (q / "skill" / "SKILL.md").read_text(encoding="utf-8", errors="replace")
            report = vet.vet_folder(q / "skill", source="hub", origin=self._origin(src, listing, entry))
        installed = self.lock.find(sid, entry.path)
        return {**out, "fingerprint": fingerprint(blobs), "skill_md": text, "vetting": report,
                "sizes": entry.sizes or None,
                "installed": {"name": installed[0], "status": installed[1].get("status")} if installed else None}

    @staticmethod
    def _origin(src: Source, listing: dict, entry: _Entry) -> str:
        return entry.url if src.kind == "github" else f"{src.index_url.removesuffix('/index.json')}/{entry.path}/"

    # ------------------------------------------------------------------ installing and updating
    def _target_name(self, src: Source, entry: _Entry) -> str:
        taken = lambda n: self.lib.get_any(n) is not None or (self.lib.pending_dir / n).exists() or self.lock.get(n)  # noqa: E731
        if not taken(entry.name):
            return entry.name
        alt = f"{entry.name[:63 - len(src.id)]}-{src.id}"[:64].strip("-")
        if valid_name(alt) and not taken(alt):
            return alt
        raise HubError(f"You already have a skill named {entry.name}. Rename or remove it first.", 409)

    async def install(self, app: Any, sid: str, path: str, *, confirm_source: bool = False,
                      accept_risk: bool = False) -> str:
        """Fetch, check and stage a skill in pending review, and pin it in the lockfile. Returns its name."""
        async with self._busy:
            src, listing, entry = await self._find(sid, path)
            if (found := self.lock.find(sid, entry.path)) is not None:
                raise HubError(f"{entry.name} is already installed as {found[0]}. Check it for updates instead.", 409)
            if not src.default and confirm_source is not True:
                raise HubConfirm("confirm_source", f"{entry.name} comes from {src.name}, a source you added, not one "
                                 "Sentient lists. Install it only if you trust who runs it.")
            name = self._target_name(src, entry)
            with self._quarantine() as q:
                staged = q / name
                blobs = await self._download(src, listing, entry, staged)
                report = vet.vet_folder(staged, source="hub", origin=self._origin(src, listing, entry))
                if report["verdict"] == "high" and accept_risk is not True:
                    raise HubConfirm("high_risk", "The safety check found high-risk problems in this skill. Install "
                                     "it anyway only if you trust where it came from.", report)
                md = staged / "SKILL.md"
                md.write_text(restamp(md.read_text(encoding="utf-8", errors="replace"), name), encoding="utf-8")
                dst = self.lib.pending_dir / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(staged), str(dst))
            try:
                await self._staged(app, name, dst, src, listing, entry, blobs, previous=None)
            except BaseException:
                shutil.rmtree(dst, ignore_errors=True)  # never leave a skill in review without its safety record
                with contextlib.suppress(Exception):
                    await vet.forget(app.store, name)
                self.lock.remove(name)
                raise
            return name

    async def _staged(self, app: Any, name: str, folder: Path, src: Source, listing: dict, entry: _Entry,
                      blobs: dict[str, str], *, previous: dict | None) -> None:
        from sentient.evolution.log import log_event
        from sentient.skills import checks as skill_checks
        from sentient.skills.loader import parse_skill_file

        origin = self._origin(src, listing, entry)
        report = vet.vet_folder(folder, source="hub", origin=origin)
        await vet.record_import(app.store, name, report)
        skill = parse_skill_file(folder / "SKILL.md")
        if skill is not None and skill.validate:  # its file checks now; a code check from outside never runs
            check = await skill_checks.run(app, name, skill.validate, [folder], run_code=False, outside=True)
            await app.skills.record_check(name, check, app.store, pending=True)
        version = {"commit": listing.get("commit"), "version": entry.version, "fingerprint": fingerprint(blobs),
                   "files": blobs, "url": origin, "made_by": entry.author}
        if previous is None:
            lock_entry = {"source": src.id, "source_name": src.name, "source_url": src.url, "trust": src.trust,
                          "kind": src.kind, "path": entry.path, "repo_path": entry.repo_path or None,
                          "upstream_name": entry.name, "installed_at": now_iso(), "status": "pending",
                          "active": None, "changes": None, **version}
            reason = f"Installed from {src.name} in the Skills hub"
        else:
            changes = file_changes((previous.get("active") or previous).get("files"), blobs)
            lock_entry = {**previous, **version, "status": "pending", "changes": changes, "updated_at": now_iso()}
            n = sum(len(v) for v in changes.values())
            reason = f"Update from {src.name}: {n} file{'s' if n != 1 else ''} changed"
        self.lock.put(name, lock_entry)
        await app.skills.record_patch(name, "pending_review", app.store)
        await log_event(app.store, "skill_imported", {
            "name": name, "source": "hub", "hub_source": src.id, "origin": origin, "verdict": report["verdict"],
            "commit": listing.get("commit"), "pending": True, "reason": reason, "update": previous is not None,
        })
        app.bus.publish("skill.updated", {"name": name, "state": "pending_review"})

    async def check_updates(self, *, refresh: bool = True) -> list[dict]:
        """Every installed hub skill with whether its source has a newer version (``update_available``)."""
        out = []
        seen: set[str] = set()
        for name, entry in sorted(self.lock.all().items()):
            item = self.summary(name, entry)
            try:
                src = self.source(str(entry.get("source")))
                _, entries = await self._listing(src, refresh=refresh and src.id not in seen)
                seen.add(src.id)
                latest = entries.get(str(entry.get("path")))
                if latest is None:
                    item.update(update_available=False, problem=f"{src.name} no longer has this skill.")
                else:
                    newest = latest.fingerprint
                    if newest is None:  # an index source: the files have to be fetched to compare them
                        with self._quarantine() as q:
                            newest = fingerprint(await self._download(src, {"commit": None}, latest, q / "skill"))
                    item.update(update_available=newest != entry.get("fingerprint"), latest_version=latest.version,
                                latest_fingerprint=newest)
            except HubError as exc:
                item.update(update_available=False, problem=str(exc))
            out.append(item)
        return out

    async def update(self, app: Any, name: str, *, accept_risk: bool = False) -> dict:
        """Fetch the newest version of an installed skill into pending review. ``{changed: false}`` when it is the
        same version; otherwise ``{changed: true, name, changes}``."""
        async with self._busy:
            name = name.strip().lower()
            current = self.lock.get(name)
            if current is None:
                raise HubError(f"{name} wasn't installed from the Skills hub.", 404)
            if current.get("status") == "pending" or (self.lib.pending_dir / name).exists():
                raise HubError(f"A version of {name} is already waiting for your review. Approve or decline it "
                               "first.", 409)
            if self.lib.get_active_file(name) is None:
                raise HubError(f"{name} isn't active, so there is nothing to update.", 409)
            src, listing, entry = await self._find(str(current.get("source")), str(current.get("path")), refresh=True)
            if entry.fingerprint is not None and entry.fingerprint == current.get("fingerprint"):
                return {"changed": False, "name": name, "commit": current.get("commit")}
            with self._quarantine() as q:
                staged = q / name
                blobs = await self._download(src, listing, entry, staged)
                if fingerprint(blobs) == current.get("fingerprint"):
                    return {"changed": False, "name": name, "commit": current.get("commit")}
                report = vet.vet_folder(staged, source="hub", origin=self._origin(src, listing, entry))
                if report["verdict"] == "high" and accept_risk is not True:
                    raise HubConfirm("high_risk", "The safety check found high-risk problems in the new version. Get "
                                     "it anyway only if you trust where it came from.", report)
                md = staged / "SKILL.md"
                md.write_text(restamp(md.read_text(encoding="utf-8", errors="replace"), name), encoding="utf-8")
                dst = self.lib.pending_dir / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(staged), str(dst))
            try:
                await self._staged(app, name, dst, src, listing, entry, blobs, previous=current)
            except BaseException:
                shutil.rmtree(dst, ignore_errors=True)
                self.lock.put(name, current)
                raise
            return {"changed": True, "name": name, "changes": self.lock.get(name).get("changes"),
                    "from_commit": current.get("commit"), "commit": listing.get("commit")}

    # ------------------------------------------------------------------ review decisions and the Skills screen
    def approved(self, name: str) -> None:
        self.lock.approved(name)

    def declined(self, name: str) -> None:
        self.lock.declined(name)

    def deleted(self, name: str) -> None:
        self.lock.remove(name)

    def is_pending_bundle(self, name: str) -> bool:
        """The pending copy is a whole folder from the hub (approving must not keep files the update removed)."""
        entry = self.lock.get(name)
        return bool(entry) and entry.get("status") == "pending"

    def summary(self, name: str, entry: dict | None = None) -> dict | None:
        """What the Skills screen shows about where a hub skill came from (``hub`` on a skill)."""
        entry = entry if entry is not None else self.lock.get(name)
        if not entry:
            return None
        active = entry.get("active") if isinstance(entry.get("active"), dict) else None
        return {
            "name": name, "source": entry.get("source"), "source_name": entry.get("source_name"),
            "source_url": entry.get("source_url"), "trust": entry.get("trust"), "path": entry.get("path"),
            "url": entry.get("url"), "commit": entry.get("commit"), "version": entry.get("version"),
            "made_by": entry.get("made_by"), "status": entry.get("status"), "installed_at": entry.get("installed_at"),
            "approved_commit": (active or {}).get("commit"), "approved_at": (active or {}).get("approved_at"),
            "changes": entry.get("changes"),
        }
