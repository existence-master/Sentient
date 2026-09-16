"""GitHub: personal access token (default) or OAuth device flow when a client id is configured."""

from __future__ import annotations

import asyncio
import base64
import logging
import secrets as pysecrets
import time
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from sentient.integrations.base import IntegrationError, IntegrationPlugin, SetupField, creds, itool
from sentient.integrations.common import http_client, truncate
from sentient.tools.base import Risk, ToolContext

if TYPE_CHECKING:  # pragma: no cover
    from sentient.integrations.service import IntegrationManager

log = logging.getLogger(__name__)

API = "https://api.github.com"
PID = "github"
DEVICE_CODE_URL = "https://github.com/login/device/code"
DEVICE_TOKEN_URL = "https://github.com/login/oauth/access_token"


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"}


async def gh(ctx: ToolContext, method: str, path: str, *, params: dict | None = None, json: Any = None) -> Any:
    c = await creds(ctx, PID)
    async with http_client(headers=_headers(c["token"])) as http:
        r = await http.request(method, f"{API}{path}", params=params, json=json)
    if r.status_code == 401:
        raise IntegrationError("GitHub rejected the saved token (it may have expired). Reconnect GitHub.")
    if r.status_code == 404:
        raise IntegrationError("GitHub couldn't find that (or the token has no access to it).")
    r.raise_for_status()
    return r.json() if r.content else {}


def _repo(repo: str) -> str:
    repo = repo.strip().removeprefix("https://github.com/").strip("/")
    if repo.count("/") != 1:
        raise IntegrationError("Repositories are written as owner/name, e.g. octocat/hello-world.")
    return repo


def _issue(i: dict) -> dict:
    return {"number": i.get("number"), "title": i.get("title"), "state": i.get("state"),
            "author": (i.get("user") or {}).get("login"), "labels": [lab.get("name") for lab in i.get("labels") or []],
            "assignees": [a.get("login") for a in i.get("assignees") or []], "comments": i.get("comments"),
            "created_at": i.get("created_at"), "updated_at": i.get("updated_at"), "url": i.get("html_url"),
            "is_pull_request": "pull_request" in i}


def _repo_info(r: dict) -> dict:
    return {"full_name": r.get("full_name"), "description": r.get("description"), "private": r.get("private"),
            "language": r.get("language"), "stars": r.get("stargazers_count"), "forks": r.get("forks_count"),
            "open_issues": r.get("open_issues_count"), "default_branch": r.get("default_branch"),
            "updated_at": r.get("updated_at"), "url": r.get("html_url")}


@itool(PID, "github_list_repos")
async def github_list_repos(ctx: ToolContext, owner: str | None = None, max_results: int = 30) -> dict:
    """List repositories: the user's own (most recently updated first), or those of another user/organisation."""
    n = max(1, min(int(max_results or 30), 100))
    path = f"/users/{quote(owner)}/repos" if owner else "/user/repos"
    rows = await gh(ctx, "GET", path, params={"sort": "updated", "per_page": n})
    return {"repositories": [_repo_info(r) for r in rows]}


@itool(PID, "github_repo_info")
async def github_repo_info(ctx: ToolContext, repo: str) -> dict:
    """Get details about a repository (owner/name): description, language, stars, default branch, open issues."""
    return _repo_info(await gh(ctx, "GET", f"/repos/{_repo(repo)}"))


@itool(PID, "github_list_issues")
async def github_list_issues(ctx: ToolContext, repo: str, state: str = "open", labels: str | None = None,
                             max_results: int = 30) -> dict:
    """List issues in a repository (not pull requests). `state`: open, closed or all; `labels`: comma-separated."""
    params: dict[str, Any] = {"state": state, "per_page": max(1, min(int(max_results or 30), 100))}
    if labels:
        params["labels"] = labels
    rows = await gh(ctx, "GET", f"/repos/{_repo(repo)}/issues", params=params)
    return {"repo": repo, "issues": [_issue(i) for i in rows if "pull_request" not in i]}


@itool(PID, "github_get_issue")
async def github_get_issue(ctx: ToolContext, repo: str, number: int) -> dict:
    """Read an issue or pull request with its description and the latest comments."""
    r = _repo(repo)
    issue = await gh(ctx, "GET", f"/repos/{r}/issues/{int(number)}")
    comments = await gh(ctx, "GET", f"/repos/{r}/issues/{int(number)}/comments", params={"per_page": 30})
    body, _ = truncate(issue.get("body") or "", 10000)
    return {**_issue(issue), "body": body,
            "comment_list": [{"author": (c.get("user") or {}).get("login"), "body": truncate(c.get("body") or "", 2000)[0],
                              "created_at": c.get("created_at")} for c in comments]}


@itool(PID, "github_create_issue", risk=Risk.send)
async def github_create_issue(ctx: ToolContext, repo: str, title: str, body: str = "", labels: list[str] | None = None,
                              assignees: list[str] | None = None) -> dict:
    """Open a new issue in a repository."""
    payload: dict[str, Any] = {"title": title, "body": body}
    if labels:
        payload["labels"] = labels
    if assignees:
        payload["assignees"] = assignees
    return {"created": True, **_issue(await gh(ctx, "POST", f"/repos/{_repo(repo)}/issues", json=payload))}


@itool(PID, "github_update_issue", risk=Risk.write)
async def github_update_issue(ctx: ToolContext, repo: str, number: int, title: str | None = None,
                              body: str | None = None, state: str | None = None, labels: list[str] | None = None,
                              assignees: list[str] | None = None) -> dict:
    """Edit an issue: title, body, state ("open"/"closed"), labels or assignees (only the fields you pass)."""
    payload = {k: v for k, v in {"title": title, "body": body, "state": state, "labels": labels,
                                 "assignees": assignees}.items() if v is not None}
    if not payload:
        raise IntegrationError("Nothing to change: pass at least one field.")
    return {"updated": True, **_issue(await gh(ctx, "PATCH", f"/repos/{_repo(repo)}/issues/{int(number)}", json=payload))}


@itool(PID, "github_comment_issue", risk=Risk.send)
async def github_comment_issue(ctx: ToolContext, repo: str, number: int, body: str) -> dict:
    """Post a comment on an issue or pull request."""
    c = await gh(ctx, "POST", f"/repos/{_repo(repo)}/issues/{int(number)}/comments", json={"body": body})
    return {"commented": True, "url": c.get("html_url")}


@itool(PID, "github_list_pull_requests")
async def github_list_pull_requests(ctx: ToolContext, repo: str, state: str = "open", max_results: int = 30) -> dict:
    """List pull requests in a repository. `state`: open, closed or all."""
    rows = await gh(ctx, "GET", f"/repos/{_repo(repo)}/pulls",
                    params={"state": state, "per_page": max(1, min(int(max_results or 30), 100))})
    return {"repo": repo, "pull_requests": [{
        "number": p.get("number"), "title": p.get("title"), "state": p.get("state"), "draft": p.get("draft"),
        "author": (p.get("user") or {}).get("login"), "head": (p.get("head") or {}).get("ref"),
        "base": (p.get("base") or {}).get("ref"), "updated_at": p.get("updated_at"), "url": p.get("html_url")}
        for p in rows]}


@itool(PID, "github_create_pull_request", risk=Risk.send)
async def github_create_pull_request(ctx: ToolContext, repo: str, title: str, head: str, base: str | None = None,
                                     body: str = "", draft: bool = False) -> dict:
    """Open a pull request from branch `head` into `base` (default: the repository's default branch)."""
    r = _repo(repo)
    if not base:
        base = (await gh(ctx, "GET", f"/repos/{r}")).get("default_branch", "main")
    p = await gh(ctx, "POST", f"/repos/{r}/pulls", json={"title": title, "head": head, "base": base, "body": body,
                                                         "draft": bool(draft)})
    return {"created": True, "number": p.get("number"), "url": p.get("html_url")}


@itool(PID, "github_read_file")
async def github_read_file(ctx: ToolContext, repo: str, path: str = "", ref: str | None = None) -> dict:
    """Read a file from a repository, or list a folder when `path` is a directory ("" = repository root)."""
    params = {"ref": ref} if ref else None
    data = await gh(ctx, "GET", f"/repos/{_repo(repo)}/contents/{quote(path.strip('/'))}", params=params)
    if isinstance(data, list):
        return {"repo": repo, "path": path, "type": "dir",
                "entries": [{"name": e.get("name"), "path": e.get("path"), "type": e.get("type"), "size": e.get("size")}
                            for e in data]}
    if data.get("encoding") == "base64" and data.get("content") is not None:
        raw = base64.b64decode(data["content"])
        if b"\x00" in raw[:2000]:
            return {"repo": repo, "path": data.get("path"), "type": "file", "binary": True, "size": data.get("size"),
                    "url": data.get("html_url")}
        text, cut = truncate(raw.decode("utf-8", errors="replace"), 40000)
        return {"repo": repo, "path": data.get("path"), "type": "file", "content": text, "truncated": cut,
                "url": data.get("html_url")}
    return {"repo": repo, "path": data.get("path"), "type": data.get("type"), "size": data.get("size"),
            "url": data.get("html_url"), "note": "File is too large to read through the API."}


@itool(PID, "github_search")
async def github_search(ctx: ToolContext, query: str, kind: str = "issues", max_results: int = 10) -> dict:
    """Search GitHub. `kind`: "issues" (issues and PRs, e.g. "repo:owner/name is:open bug"), "code"
    (e.g. "def main repo:owner/name"), or "repositories" (e.g. "language:rust stars:>1000")."""
    kind = kind if kind in {"issues", "code", "repositories"} else "issues"
    n = max(1, min(int(max_results or 10), 50))
    res = await gh(ctx, "GET", f"/search/{kind}", params={"q": query, "per_page": n})
    items = res.get("items") or []
    if kind == "issues":
        rows = [_issue(i) for i in items]
    elif kind == "repositories":
        rows = [_repo_info(i) for i in items]
    else:
        rows = [{"repo": (i.get("repository") or {}).get("full_name"), "path": i.get("path"), "url": i.get("html_url")}
                for i in items]
    return {"query": query, "kind": kind, "total": res.get("total_count"), "results": rows}


class GitHubPlugin(IntegrationPlugin):
    id = PID
    display_name = "GitHub"
    description = (
        "Work with your repositories: list repos, read files, triage and update issues, comment, open pull "
        "requests and search code across GitHub."
    )
    category = "development"
    icon = "github"
    auth_type = "api_key"
    selection_hint = "Use for GitHub repositories, issues, pull requests, code search or reading repo files."
    setup_fields = [SetupField("token", "Personal access token", secret=True, required=True,
                               help="A fine-grained or classic token from github.com/settings/tokens.",
                               placeholder="github_pat_...")]
    docs_url = "https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens"
    instructions_md = (
        "1. Sign in at https://github.com and open https://github.com/settings/personal-access-tokens/new "
        "(Settings → Developer settings → Personal access tokens → Fine-grained tokens → Generate new token).\n"
        "2. Name it `Sentient` and pick an expiration (for example 90 days).\n"
        "3. Under **Repository access**, choose **All repositories** (or only the ones Sentient should see).\n"
        "4. Under **Permissions → Repository permissions**, set **Contents**, **Issues**, **Pull requests** and "
        "**Metadata** to **Read and write** (Metadata is read-only).\n"
        "5. Click **Generate token** and copy it right away (GitHub shows it only once).\n"
        "6. Paste it here and click **Connect**.\n"
    )
    tools = [github_list_repos, github_repo_info, github_list_issues, github_get_issue, github_create_issue,
             github_update_issue, github_comment_issue, github_list_pull_requests, github_create_pull_request,
             github_read_file, github_search]

    async def validate(self, fields: dict[str, str], mgr: IntegrationManager) -> tuple[dict, str | None]:
        token = str(fields.get("token", "")).strip()
        if not token:
            raise IntegrationError("Please paste your GitHub personal access token.")
        async with http_client(headers=_headers(token)) as http:
            r = await http.get(f"{API}/user")
        if r.status_code in (401, 403):
            raise IntegrationError("GitHub didn't accept that token. Check it was copied completely and hasn't expired.")
        r.raise_for_status()
        return {"token": token}, r.json().get("login")

    async def begin_oauth(self, fields: dict[str, str], mgr: IntegrationManager) -> dict | None:
        client_id = mgr.app.config.integrations.github_oauth_client_id.strip()
        if str(fields.get("token", "")).strip() or not client_id:
            return None
        async with http_client(headers={"Accept": "application/json"}) as http:
            r = await http.post(DEVICE_CODE_URL, data={"client_id": client_id, "scope": "repo read:user"})
        r.raise_for_status()
        data = r.json()
        if "device_code" not in data:
            raise IntegrationError(f"GitHub sign-in couldn't start: {data.get('error_description') or data.get('error')}")
        state = pysecrets.token_urlsafe(16)
        await mgr._save_state(PID, status="connecting", error=None)
        await mgr.publish(PID)
        mgr.spawn(self._poll_device(mgr, client_id, data))
        return {"auth_url": data.get("verification_uri", "https://github.com/login/device"), "state": state,
                "user_code": data.get("user_code")}

    @staticmethod
    async def _wait_for_device_token(client_id: str, data: dict) -> str:
        interval = float(data.get("interval", 5))
        deadline = time.time() + float(data.get("expires_in", 900))
        while time.time() < deadline:
            await asyncio.sleep(interval)
            async with http_client(headers={"Accept": "application/json"}) as http:
                r = await http.post(DEVICE_TOKEN_URL, data={
                    "client_id": client_id, "device_code": data["device_code"],
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code"})
            body = r.json()
            if body.get("access_token"):
                return body["access_token"]
            err = body.get("error")
            if err == "slow_down":
                interval += 5
            elif err != "authorization_pending":
                raise IntegrationError(body.get("error_description") or err or "GitHub sign-in failed.")
        raise IntegrationError("GitHub sign-in timed out. Try again.")

    async def _poll_device(self, mgr: IntegrationManager, client_id: str, data: dict) -> None:
        try:
            token = await self._wait_for_device_token(client_id, data)
            creds_, label = await self.validate({"token": token}, mgr)
            await mgr.store_credentials(PID, creds_)
            await mgr.mark_connected(PID, label)
        except Exception as exc:
            log.warning("github device flow failed: %s", exc)
            if not mgr._connected_sync(PID):
                await mgr.set_error(PID, str(exc))


PLUGIN = GitHubPlugin()
