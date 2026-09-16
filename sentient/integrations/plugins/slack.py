"""Slack Web API with a user (xoxp-) or bot (xoxb-) token."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sentient.integrations.base import IntegrationError, IntegrationPlugin, SetupField, creds, itool
from sentient.integrations.common import http_client
from sentient.tools.base import Risk, ToolContext

if TYPE_CHECKING:  # pragma: no cover
    from sentient.integrations.service import IntegrationManager

API = "https://slack.com/api"
PID = "slack"
FRIENDLY = {
    "not_in_channel": "Sentient's Slack app isn't in that channel. Invite it to the channel first.",
    "channel_not_found": "That Slack channel doesn't exist or isn't visible to the token.",
    "missing_scope": "The Slack token is missing a permission this needs. Add the scope and reinstall the app.",
    "invalid_auth": "Slack rejected the saved token. Reconnect Slack.",
    "token_revoked": "The Slack token was revoked. Reconnect Slack.",
    "ratelimited": "Slack is rate limiting requests. Try again in a minute.",
}


async def slack_call(token: str, method: str, *, params: dict | None = None, json: dict | None = None) -> dict:
    async with http_client(headers={"Authorization": f"Bearer {token}"}) as http:
        if json is not None:
            r = await http.post(f"{API}/{method}", json=json, headers={"Content-Type": "application/json; charset=utf-8"})
        else:
            r = await http.get(f"{API}/{method}", params=params)
    r.raise_for_status()
    data = r.json()
    if not data.get("ok"):
        err = data.get("error", "unknown_error")
        extra = f" (needs {data.get('needed')})" if data.get("needed") else ""
        raise IntegrationError(FRIENDLY.get(err, f"Slack error: {err}") + extra)
    return data


async def _call(ctx: ToolContext, method: str, **kw: Any) -> dict:
    c = await creds(ctx, PID)
    return await slack_call(c["token"], method, **kw)


async def _channel_id(ctx: ToolContext, channel: str) -> str:
    ch = channel.strip()
    if ch and ch[0] in "CGD" and ch[1:].isalnum() and ch.upper() == ch:
        return ch
    name = ch.lstrip("#").lower()
    cursor = None
    for _ in range(20):
        params = {"limit": 1000, "exclude_archived": "true", "types": "public_channel,private_channel"}
        if cursor:
            params["cursor"] = cursor
        data = await _call(ctx, "conversations.list", params=params)
        for c in data.get("channels") or []:
            if (c.get("name") or "").lower() == name:
                return c["id"]
        cursor = (data.get("response_metadata") or {}).get("next_cursor")
        if not cursor:
            break
    raise IntegrationError(f"Couldn't find a Slack channel called '{channel}'.")


def _msg(m: dict) -> dict:
    return {"ts": m.get("ts"), "user": m.get("user") or m.get("username") or m.get("bot_id"), "text": m.get("text"),
            "thread_ts": m.get("thread_ts"), "reply_count": m.get("reply_count")}


@itool(PID, "slack_list_channels")
async def slack_list_channels(ctx: ToolContext, include_private: bool = False, limit: int = 200) -> dict:
    """List Slack channels (names and ids) in the workspace."""
    types = "public_channel,private_channel" if include_private else "public_channel"
    data = await _call(ctx, "conversations.list", params={"limit": max(1, min(int(limit or 200), 1000)),
                                                          "exclude_archived": "true", "types": types})
    return {"channels": [{"id": c.get("id"), "name": c.get("name"), "is_private": c.get("is_private"),
                          "is_member": c.get("is_member"), "members": c.get("num_members"),
                          "topic": (c.get("topic") or {}).get("value")} for c in data.get("channels") or []]}


@itool(PID, "slack_post_message", risk=Risk.send)
async def slack_post_message(ctx: ToolContext, channel: str, text: str) -> dict:
    """Post a message to a Slack channel (channel id like C0123 or a name like #general)."""
    data = await _call(ctx, "chat.postMessage", json={"channel": await _channel_id(ctx, channel), "text": text})
    return {"sent": True, "channel": data.get("channel"), "ts": data.get("ts")}


@itool(PID, "slack_channel_history")
async def slack_channel_history(ctx: ToolContext, channel: str, limit: int = 20) -> dict:
    """Read the most recent messages in a Slack channel (newest first)."""
    cid = await _channel_id(ctx, channel)
    data = await _call(ctx, "conversations.history", params={"channel": cid, "limit": max(1, min(int(limit or 20), 200))})
    return {"channel": cid, "messages": [_msg(m) for m in data.get("messages") or []]}


@itool(PID, "slack_thread_replies")
async def slack_thread_replies(ctx: ToolContext, channel: str, thread_ts: str, limit: int = 50) -> dict:
    """Read the replies in a Slack thread (thread_ts is the parent message's ts)."""
    cid = await _channel_id(ctx, channel)
    data = await _call(ctx, "conversations.replies", params={"channel": cid, "ts": thread_ts,
                                                             "limit": max(1, min(int(limit or 50), 200))})
    return {"channel": cid, "thread_ts": thread_ts, "messages": [_msg(m) for m in data.get("messages") or []]}


@itool(PID, "slack_reply_in_thread", risk=Risk.send)
async def slack_reply_in_thread(ctx: ToolContext, channel: str, thread_ts: str, text: str) -> dict:
    """Reply inside a Slack thread."""
    data = await _call(ctx, "chat.postMessage", json={"channel": await _channel_id(ctx, channel), "text": text,
                                                      "thread_ts": thread_ts})
    return {"sent": True, "channel": data.get("channel"), "ts": data.get("ts")}


@itool(PID, "slack_list_users")
async def slack_list_users(ctx: ToolContext, limit: int = 200) -> dict:
    """List people in the Slack workspace (ids, names, titles) - bots and deactivated accounts are skipped."""
    data = await _call(ctx, "users.list", params={"limit": max(1, min(int(limit or 200), 1000))})
    return {"users": [{"id": u.get("id"), "name": u.get("name"), "real_name": u.get("real_name"),
                       "title": (u.get("profile") or {}).get("title"), "email": (u.get("profile") or {}).get("email")}
                      for u in data.get("members") or [] if not u.get("is_bot") and not u.get("deleted")
                      and u.get("id") != "USLACKBOT"]}


class SlackPlugin(IntegrationPlugin):
    id = PID
    display_name = "Slack"
    description = (
        "Catch up on and take part in your Slack workspace: list channels, read channel history and threads, "
        "post messages and reply in threads, and look up teammates."
    )
    category = "communication"
    icon = "slack"
    auth_type = "api_key"
    selection_hint = "Use to read or post Slack messages, channels, threads, or find Slack users."
    setup_fields = [SetupField("token", "Slack OAuth token", secret=True, required=True,
                               help="The User OAuth Token (xoxp-...) or Bot token (xoxb-...) of your Slack app.",
                               placeholder="xoxp-...")]
    docs_url = "https://api.slack.com/authentication/token-types"
    instructions_md = (
        "1. Open https://api.slack.com/apps and click **Create New App → From scratch**. Name it `Sentient` and "
        "pick your workspace.\n"
        "2. In the left menu open **OAuth & Permissions**.\n"
        "3. Under **User Token Scopes**, click **Add an OAuth Scope** and add: `channels:read`, `channels:history`, "
        "`groups:read`, `groups:history`, `chat:write`, `users:read`, `users:read.email`.\n"
        "4. Scroll up and click **Install to Workspace**, then **Allow**.\n"
        "5. Copy the **User OAuth Token** (it starts with `xoxp-`).\n"
        "6. Paste it here and click **Connect**. Messages Sentient sends will appear as you.\n"
    )
    tools = [slack_list_channels, slack_post_message, slack_channel_history, slack_thread_replies,
             slack_reply_in_thread, slack_list_users]

    async def validate(self, fields: dict[str, str], mgr: IntegrationManager) -> tuple[dict, str | None]:
        token = str(fields.get("token", "")).strip()
        if not token.startswith(("xoxp-", "xoxb-", "xoxe.")):
            raise IntegrationError("That doesn't look like a Slack token. It should start with xoxp- or xoxb-.")
        data = await slack_call(token, "auth.test")
        label = " @ ".join(x for x in (data.get("user"), data.get("team")) if x) or None
        return {"token": token, "team_id": data.get("team_id"), "user_id": data.get("user_id")}, label


PLUGIN = SlackPlugin()
