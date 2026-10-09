"""IntegrationManager: the integrations package's service and public API.

Other packages use only these methods (signatures are stable):

- ``await connected_plugins() -> set[str]``   builtin + connected integration ids (+ connected MCP servers)
- ``await is_connected(plugin_id) -> bool``
- ``await get_credentials(plugin_id) -> dict | None``
- ``await poll_source(source, since) -> list[dict]``  new, normalized, privacy-filtered items
- ``await list_integrations() -> list[dict]``          Integration objects (docs/API.md section 5)
- ``feed_active(source) -> bool``                      a change feed / push watcher keeps ``source`` current
- ``await feed_status() -> list[dict]``                change feed status per feed-capable source
- ``await emit_items(source, origin, items, event=None) -> list[dict]``  shared seen record + ``source.items``

Every new item from poll, feed or webhook is published once as ``source.items`` (docs/API.md section 16).

Credentials live in the OS keychain as JSON under ``integration:<id>``. Connection
state (connected, account label, status, error, privacy filters) lives in the core
``integrations`` table. Every state change publishes ``integration.updated``.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import secrets as pysecrets
import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from sentient.integrations import base as _base
from sentient.integrations import google
from sentient.integrations.base import IntegrationError, IntegrationPlugin
from sentient.integrations.common import (
    delete_secret,
    email_blocked,
    event_blocked,
    load_secret_json,
    normalize_filters,
    store_secret_json,
)
from sentient.integrations.feeds import FeedEngine
from sentient.integrations.hooks import HookStore
from sentient.integrations.plugins import PLUGIN_MODULES
from sentient.services import Service
from sentient.store.db import now_iso

if TYPE_CHECKING:  # pragma: no cover
    from sentient.app import SentientApp

log = logging.getLogger(__name__)

OAUTH_FLOW_TTL_S = 15 * 60
DEFAULT_POLL_LOOKBACK = timedelta(hours=1)
SEEN_CAP = 500


def secret_name(plugin_id: str) -> str:
    return f"integration:{plugin_id}"


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


class IntegrationManager(Service):
    name = "integrations"

    def __init__(self, app: SentientApp):
        super().__init__(app)
        self._plugins: dict[str, IntegrationPlugin] = {}
        self._state: dict[str, dict] = {}
        self._creds: dict[str, dict | None] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._pending_oauth: dict[str, dict] = {}
        self._background: set[asyncio.Task] = set()
        self.listener = google.LoopbackListener(self._oauth_callback)
        self.feeds = FeedEngine(self)
        self.hooks = HookStore(self)
        from sentient.integrations.mcp import MCPManager

        self.mcp = MCPManager(self)

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        _base._CURRENT_MANAGER = self
        store = self.app.store
        await store.ensure_column("integrations", "status", "TEXT")
        await store.ensure_column("integrations", "error", "TEXT")
        await store.db.commit()
        self._load_plugins()
        for row in await store.fetchall("SELECT * FROM integrations"):
            self._state[row["id"]] = self._row_to_state(dict(row))
        for p in self._plugins.values():
            try:
                self.app.registry.register(p)
            except ValueError:
                log.exception("integration %s could not be registered", p.id)
        self.sync_registry()
        await self.mcp.start()
        await self.feeds.load()
        if getattr(self.app, "enable_background", False):
            self.feeds.start_loop()
            for p in self._plugins.values():
                if getattr(p, "watch", None) is not None and self._connected_sync(p.id):
                    self.feeds.start_watch(p.id)

    async def stop(self) -> None:
        await self.feeds.stop()
        await self.mcp.stop()
        await self.listener.stop()
        for t in list(self._background):
            t.cancel()
        await super().stop()
        if _base._CURRENT_MANAGER is self:
            _base._CURRENT_MANAGER = None

    def _load_plugins(self) -> None:
        if self._plugins:
            return
        for mod_name in PLUGIN_MODULES:
            try:
                module = importlib.import_module(f"sentient.integrations.plugins.{mod_name}")
            except Exception:
                log.exception("integration module %s failed to import", mod_name)
                continue
            found = list(getattr(module, "PLUGINS", []) or [])
            if getattr(module, "PLUGIN", None) is not None:
                found.append(module.PLUGIN)
            for p in found:
                if isinstance(p, IntegrationPlugin):
                    self._plugins[p.id] = p

    def spawn(self, coro: Any) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        return task

    def lock(self, key: str) -> asyncio.Lock:
        return self._locks.setdefault(key, asyncio.Lock())

    # ------------------------------------------------------------------ plugins & registry
    def plugin(self, plugin_id: str) -> IntegrationPlugin | None:
        return self._plugins.get(plugin_id)

    def plugins(self) -> list[IntegrationPlugin]:
        return list(self._plugins.values())

    def sync_registry(self) -> None:
        """Hide tools of disconnected integrations from the model (config: hide_disconnected_tools)."""
        reg = self.app.registry
        hide = self.app.config.integrations.hide_disconnected_tools
        for p in self._plugins.values():
            if reg.plugin(p.id) is not p:
                continue
            visible = p.is_builtin or not hide or self._connected_sync(p.id)
            reg.set_hidden(p.id, not visible)

    # ------------------------------------------------------------------ state
    @staticmethod
    def _row_to_state(row: dict) -> dict:
        settings: dict = {}
        if row.get("settings"):
            try:
                settings = json.loads(row["settings"]) or {}
            except json.JSONDecodeError:
                settings = {}
        connected = bool(row.get("connected"))
        return {
            "connected": connected,
            "auth_type": row.get("auth_type"),
            "account_label": row.get("account_label"),
            "settings": settings,
            "status": row.get("status") or ("connected" if connected else "disconnected"),
            "error": row.get("error"),
            "connected_at": row.get("connected_at"),
        }

    def _get_state(self, plugin_id: str) -> dict:
        return self._state.get(plugin_id) or {
            "connected": False, "auth_type": None, "account_label": None, "settings": {},
            "status": "disconnected", "error": None, "connected_at": None,
        }

    async def _save_state(self, plugin_id: str, **changes: Any) -> dict:
        st = {**self._get_state(plugin_id), **changes}
        p = self.plugin(plugin_id)
        st["auth_type"] = st.get("auth_type") or (p.auth_type if p else None)
        self._state[plugin_id] = st
        await self.app.store.execute(
            "INSERT INTO integrations(id, connected, auth_type, account_label, settings, connected_at, updated_at, status, error)"
            " VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET connected=excluded.connected,"
            " auth_type=excluded.auth_type, account_label=excluded.account_label, settings=excluded.settings,"
            " connected_at=excluded.connected_at, updated_at=excluded.updated_at, status=excluded.status, error=excluded.error",
            (plugin_id, int(bool(st["connected"])), st["auth_type"], st["account_label"], json.dumps(st["settings"]),
             st.get("connected_at"), now_iso(), st["status"], st["error"]),
        )
        return st

    def _connected_sync(self, plugin_id: str) -> bool:
        p = self.plugin(plugin_id)
        if p is not None and p.is_builtin:
            return True
        return bool(self._get_state(plugin_id)["connected"])

    async def set_error(self, plugin_id: str, message: str) -> None:
        await self._save_state(plugin_id, status="error", error=message)
        await self.publish(plugin_id)

    async def publish(self, plugin_id: str) -> None:
        if self.plugin(plugin_id) is not None:
            self.app.bus.publish("integration.updated", await self.integration(plugin_id))

    # ------------------------------------------------------------------ public API
    async def connected_plugins(self) -> set[str]:
        ids = {pid for pid in self._plugins if self._connected_sync(pid)}
        ids |= self.mcp.connected_plugin_ids()
        return ids

    async def is_connected(self, plugin_id: str) -> bool:
        if plugin_id.startswith("mcp_") and plugin_id not in self._plugins:
            return plugin_id in self.mcp.connected_plugin_ids()
        return self._connected_sync(plugin_id)

    async def get_credentials(self, plugin_id: str) -> dict | None:
        if plugin_id not in self._creds:
            self._creds[plugin_id] = load_secret_json(secret_name(plugin_id))
        c = self._creds[plugin_id]
        return dict(c) if c else None

    async def store_credentials(self, plugin_id: str, data: dict) -> None:
        if not store_secret_json(secret_name(plugin_id), data):
            raise IntegrationError("Your system keychain is unavailable, so the credentials can't be saved safely.")
        self._creds[plugin_id] = dict(data)

    def secret_names(self) -> list[str]:
        """Keychain entry names this package uses (for the core secrets listing)."""
        names = [google.CLIENT_SECRET_NAME]
        names += [secret_name(p.id) for p in self._plugins.values() if not p.is_builtin]
        return names

    async def list_integrations(self) -> list[dict]:
        return [await self.integration(pid) for pid in self._plugins]

    async def integration(self, plugin_id: str) -> dict:
        p = self.plugin(plugin_id)
        if p is None:
            raise KeyError(plugin_id)
        st = self._get_state(plugin_id)
        connected = self._connected_sync(plugin_id)
        fields = [f.to_dict() for f in p.setup_fields]
        if plugin_id in google.SCOPES and google.get_client() is not None:
            for f in fields:
                f["required"] = False
        status = "connected" if connected and st["status"] != "error" else st["status"]
        if p.is_builtin:
            status = "connected"
        elif not connected and status == "connected":
            status = "disconnected"
        return {
            "id": p.id,
            "display_name": p.display_name,
            "description": p.description,
            "category": p.category,
            "icon": p.icon,
            "auth_type": p.auth_type,
            "connected": connected,
            "account_label": st["account_label"] if connected else None,
            "status": status,
            "error": st["error"],
            "setup": {"fields": fields, "instructions_md": p.instructions_md, "docs_url": p.docs_url},
            "privacy_filters": {"supported": bool(p.privacy_fields), "fields": list(p.privacy_fields)},
            "triggers": await self._triggers(p),
            "alternative_for": p.optional_alternative_for,
            "tools": [{"name": t.name, "description": t.description, "risk": t.risk.name} for t in p.tools],
        }

    # ------------------------------------------------------------------ connect / disconnect / test
    async def connect(self, plugin_id: str, fields: dict[str, Any] | None = None) -> dict:
        p = self.plugin(plugin_id)
        if p is None:
            raise KeyError(plugin_id)
        fields = {k: ("" if v is None else str(v)) for k, v in (fields or {}).items()}
        if p.is_builtin:
            return await self.integration(plugin_id)
        started = await p.begin_oauth(fields, self)
        if started is not None:
            return started
        try:
            credentials, label = await p.validate(fields, self)
        except IntegrationError as exc:
            await self._save_state(plugin_id, status="error" if not self._connected_sync(plugin_id) else "connected",
                                   error=str(exc))
            await self.publish(plugin_id)
            raise
        await self.store_credentials(plugin_id, credentials)
        await self.mark_connected(plugin_id, label)
        return await self.integration(plugin_id)

    async def mark_connected(self, plugin_id: str, account_label: str | None) -> None:
        previous = self._get_state(plugin_id)
        await self._save_state(plugin_id, connected=True, account_label=account_label, status="connected",
                               error=None, connected_at=now_iso())
        self.sync_registry()
        await self._restart_feed(plugin_id, account_changed=previous.get("account_label") != account_label
                                 or not previous.get("connected"))
        await self.publish(plugin_id)

    async def _restart_feed(self, plugin_id: str, *, account_changed: bool) -> None:
        p = self.plugin(plugin_id)
        if p is None or (getattr(p, "change_feed", None) is None and getattr(p, "watch", None) is None):
            return
        await self.feeds.stop_watch(plugin_id)
        if account_changed:
            await self.feeds.reset(plugin_id)  # a different mailbox/calendar: re-baseline from now
        if getattr(self.app, "enable_background", False):
            self.feeds.start_watch(plugin_id)

    async def disconnect(self, plugin_id: str) -> dict:
        p = self.plugin(plugin_id)
        if p is None:
            raise KeyError(plugin_id)
        if p.is_builtin:
            return await self.integration(plugin_id)
        delete_secret(secret_name(plugin_id))
        self._creds.pop(plugin_id, None)
        for state, flow in list(self._pending_oauth.items()):
            if flow["plugin_id"] == plugin_id:
                self._pending_oauth.pop(state, None)
        await self._save_state(plugin_id, connected=False, account_label=None, status="disconnected", error=None,
                               connected_at=None)
        self.sync_registry()
        await self.feeds.stop_watch(plugin_id)
        await self.feeds.reset(plugin_id)
        disable = getattr(getattr(self.app, "tasks", None), "disable_tasks_for_plugin", None)
        if callable(disable):
            try:
                await disable(plugin_id)
            except Exception:
                log.exception("disabling tasks for %s failed", plugin_id)
        await self.publish(plugin_id)
        return await self.integration(plugin_id)

    async def cancel_connect(self, plugin_id: str) -> dict:
        """Abandon a pending browser sign-in (the user closed the dialog or gave up)."""
        p = self.plugin(plugin_id)
        if p is None:
            raise KeyError(plugin_id)
        for state, flow in list(self._pending_oauth.items()):
            if flow["plugin_id"] == plugin_id:
                self._pending_oauth.pop(state, None)
        if not self._connected_sync(plugin_id) and self._get_state(plugin_id)["status"] == "connecting":
            await self._save_state(plugin_id, status="disconnected", error=None)
            await self.publish(plugin_id)
        return await self.integration(plugin_id)

    async def test(self, plugin_id: str) -> dict:
        p = self.plugin(plugin_id)
        if p is None:
            raise KeyError(plugin_id)
        if not p.is_builtin and not self._connected_sync(plugin_id):
            return {"ok": False, "detail": f"{p.display_name} isn't connected yet."}
        try:
            detail = await p.test(await self.get_credentials(plugin_id), self)
        except Exception as exc:
            msg = str(exc) or type(exc).__name__
            if not p.is_builtin:
                await self.set_error(plugin_id, msg)
            return {"ok": False, "detail": msg}
        st = self._get_state(plugin_id)
        if not p.is_builtin and st["status"] == "error":
            await self._save_state(plugin_id, status="connected", error=None)
            await self.publish(plugin_id)
        return {"ok": True, "detail": detail}

    # ------------------------------------------------------------------ Google OAuth
    async def start_google_flow(self, plugin_id: str, fields: dict[str, str]) -> dict:
        cid, secret = fields.get("client_id", "").strip(), fields.get("client_secret", "").strip()
        if cid and secret:
            google.save_client(cid, secret)
        elif cid or secret:
            raise IntegrationError("Enter both the OAuth Client ID and the Client secret.")
        client = google.get_client()
        if client is None:
            raise IntegrationError("Enter your Google OAuth Client ID and Client secret first (see the steps).")
        now = time.time()
        for s, flow in list(self._pending_oauth.items()):
            if now - flow["created"] > OAUTH_FLOW_TTL_S:
                self._pending_oauth.pop(s, None)
        await self.listener.start(self.app.config.integrations.oauth_redirect_port)
        verifier, challenge = google.pkce_pair()
        state = pysecrets.token_urlsafe(24)
        redirect_uri = self.listener.redirect_uri()
        self._pending_oauth[state] = {"plugin_id": plugin_id, "verifier": verifier, "redirect_uri": redirect_uri,
                                      "created": now, "kind": "google"}
        auth_url = google.build_auth_url(client["client_id"], redirect_uri, google.SCOPES[plugin_id], state, challenge)
        if not self._connected_sync(plugin_id):
            await self._save_state(plugin_id, status="connecting", error=None)
            await self.publish(plugin_id)
        return {"auth_url": auth_url, "state": state}

    async def _oauth_callback(self, params: dict[str, str]) -> tuple[bool, str]:
        flow = self._pending_oauth.pop(params.get("state", ""), None)
        if flow is None or time.time() - flow["created"] > OAUTH_FLOW_TTL_S:
            return False, "This sign-in link has expired."
        pid = flow["plugin_id"]
        p = self.plugin(pid)
        name = p.display_name if p else pid
        if params.get("error") or not params.get("code"):
            reason = "You cancelled the sign-in." if params.get("error") == "access_denied" else (
                f"Google reported: {params.get('error') or 'no authorization code'}.")
            await self._save_state(pid, status="connected" if self._connected_sync(pid) else "error", error=reason)
            await self.publish(pid)
            return False, reason
        try:
            token = await google.exchange_code(params["code"], flow["verifier"], flow["redirect_uri"])
            previous = await self.get_credentials(pid) or {}
            if not token.get("refresh_token") and previous.get("refresh_token"):
                token["refresh_token"] = previous["refresh_token"]
            email = await google.fetch_email(token["access_token"])
            await self.store_credentials(pid, token)
            await self.mark_connected(pid, email)
        except IntegrationError as exc:
            await self._save_state(pid, status="error", error=str(exc))
            await self.publish(pid)
            return False, str(exc)
        except Exception as exc:
            log.exception("google token exchange failed")
            await self._save_state(pid, status="error", error=f"Sign-in failed: {exc}")
            await self.publish(pid)
            return False, "Sign-in failed."
        return True, f"{name} is connected{f' as {email}' if email else ''}."

    # ------------------------------------------------------------------ privacy filters
    async def get_privacy_filters(self, plugin_id: str) -> dict[str, list[str]]:
        if self.plugin(plugin_id) is None:
            raise KeyError(plugin_id)
        return normalize_filters(self._get_state(plugin_id)["settings"].get("privacy_filters"))

    async def set_privacy_filters(self, plugin_id: str, filters: dict) -> None:
        if self.plugin(plugin_id) is None:
            raise KeyError(plugin_id)
        settings = dict(self._get_state(plugin_id)["settings"])
        settings["privacy_filters"] = normalize_filters(filters)
        await self._save_state(plugin_id, settings=settings)
        await self.publish(plugin_id)

    async def filter_items(self, plugin_id: str, items: list[dict], kind: str) -> list[dict]:
        filters = await self.get_privacy_filters(plugin_id)
        if not any(filters.values()):
            return items
        blocked = email_blocked if kind == "email" else event_blocked
        return [i for i in items if not blocked(i, filters)]

    # ------------------------------------------------------------------ polling (proactivity / triggered tasks)
    async def poll_state(self, source: str) -> dict:
        row = await self.app.store.fetchone("SELECT * FROM integration_poll_state WHERE source = ?", (source,))
        if row is None:
            d = {"source": source, "cursor": None, "seen": [], "last_error": None, "updated_at": None}
        else:
            d = dict(row)
            d["seen"] = json.loads(d["seen"]) if d.get("seen") else []
        feed = await self.feeds.state(source)
        d["feed"] = {k: feed[k] for k in ("status", "failures", "last_sync_at", "last_success_at", "last_error",
                                          "note", "next_attempt_at")}
        return d

    async def _save_poll_state(self, source: str, cursor: str | None, seen: list[str], error: str | None) -> None:
        await self.app.store.execute(
            "INSERT INTO integration_poll_state(source, cursor, seen, last_error, updated_at) VALUES(?,?,?,?,?)"
            " ON CONFLICT(source) DO UPDATE SET cursor=excluded.cursor, seen=excluded.seen,"
            " last_error=excluded.last_error, updated_at=excluded.updated_at",
            (source, cursor, json.dumps(seen[:SEEN_CAP]), error, now_iso()),
        )

    async def poll_source(self, source: str, since: str | None = None) -> list[dict]:
        """New items from ``gmail`` or ``gcalendar`` since ``since`` (ISO) or the last poll.

        Items are normalized (see docs/API.md section 5), privacy filters are applied, and an
        item is returned at most once. Returns [] when the source isn't connected. Raises
        IntegrationError (after recording ``last_error``) when the service call fails.
        """
        p = self.plugin(source)
        poll = getattr(p, "poll", None)
        if p is None or poll is None:
            raise ValueError(f"{source} cannot be polled")
        if not self._connected_sync(source):
            return []
        async with self.lock(f"poll:{source}"):
            st = await self.poll_state(source)
            started = datetime.now(UTC)
            since_dt = _parse_ts(since) or _parse_ts(st["cursor"]) or (started - DEFAULT_POLL_LOOKBACK)
            try:
                items = await poll(self, since_dt)
            except Exception as exc:
                if isinstance(exc, IntegrationError):
                    msg = str(exc) or f"{p.display_name} couldn't be checked."
                else:
                    # never surface a bare repr like 'gcalendar' (e.g. a KeyError) to the user
                    detail = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
                    msg = f"{p.display_name} couldn't be checked ({detail})."
                await self._save_poll_state(source, st["cursor"], st["seen"], msg)
                if isinstance(exc, IntegrationError):
                    raise
                raise IntegrationError(msg) from exc
            legacy_seen = {str(k) for k in st["seen"]}  # returned by builds before the shared seen record
            items = [i for i in items if isinstance(i, dict) and str(i.get("_key", i.get("id"))) not in legacy_seen]
            keys = [str(i.get("_key", i.get("id"))) for i in items]
            # claims each item in the shared seen record, applies privacy filters, publishes source.items (poll)
            kept = await self.feeds.emit(source, "poll", items)
            await self._save_poll_state(source, started.isoformat(), list(dict.fromkeys(keys + list(st["seen"]))), None)
            return kept

    async def recent_threads(self, source: str, *, newer_than_days: int, idle_days: int, limit: int = 40) -> dict:
        """Recent mail conversations of ``gmail`` or ``email_imap`` for follow-ups (docs/API.md section 5).

        Returns ``{addresses, threads}``; a thread is dropped whole when any of its messages, senders or
        recipients is hidden by the privacy filters. ``{addresses: [], threads: []}`` when not connected."""
        p = self.plugin(source)
        fn = getattr(p, "recent_threads", None)
        if p is None or fn is None:
            raise ValueError(f"{source} has no mail threads")
        if not self._connected_sync(source):
            return {"addresses": [], "threads": []}
        try:
            data = await fn(self, newer_than_days=newer_than_days, idle_days=idle_days, limit=limit)
        except IntegrationError:
            raise
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
            raise IntegrationError(f"{p.display_name} couldn't be checked ({detail}).") from exc
        filters = await self.get_privacy_filters(source)
        if any(filters.values()):
            emails = [e.lower() for e in filters.get("emails", [])]

            def blocked(thread: dict) -> bool:
                for m in thread.get("messages") or []:
                    if email_blocked(m, filters):
                        return True
                    people = f"{m.get('to') or ''} {m.get('cc') or ''}".lower()
                    if any(e in people for e in emails):
                        return True
                return False

            data = {**data, "threads": [t for t in data.get("threads") or [] if not blocked(t)]}
        return data

    # ------------------------------------------------------------------ change feeds, push watchers, source.items
    def feed_active(self, source: str) -> bool:
        """True when a change feed (gmail, gcalendar) or push watcher (email_imap) keeps ``source`` current,
        so timer polling can skip it. Falls back to False after repeated feed failures."""
        return self.feeds.active(source)

    async def feed_status(self) -> list[dict]:
        return await self.feeds.status()

    async def sync_feeds_now(self) -> int:
        """Run every due change feed once. Returns the number of new items published."""
        return await self.feeds.sync_all()

    async def emit_items(self, source: str, origin: str, items: list[dict], *, event: str | None = None) -> list[dict]:
        """Publish new items once (shared seen record, privacy filters, ``source.items``)."""
        return await self.feeds.emit(source, origin, items, event=event)

    async def _triggers(self, p: IntegrationPlugin) -> list[dict]:
        dynamic = getattr(p, "dynamic_triggers", None)
        if dynamic is None:
            return list(p.triggers)
        try:
            return list(await dynamic(self))
        except Exception:
            log.exception("triggers of %s could not be listed", p.id)
            return list(p.triggers)

