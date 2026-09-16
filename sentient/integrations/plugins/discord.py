"""Discord via a bot token (the bot must be invited to the servers it should use)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sentient.integrations.base import IntegrationError, IntegrationPlugin, SetupField, creds, itool
from sentient.integrations.common import http_client
from sentient.tools.base import Risk, ToolContext

if TYPE_CHECKING:  # pragma: no cover
    from sentient.integrations.service import IntegrationManager

API = "https://discord.com/api/v10"
PID = "discord"
CHANNEL_TYPES = {0: "text", 2: "voice", 4: "category", 5: "announcement", 13: "stage", 15: "forum"}


async def dapi(ctx: ToolContext, method: str, path: str, *, params: dict | None = None, json: Any = None) -> Any:
    token = (await creds(ctx, PID))["token"]
    async with http_client(headers={"Authorization": f"Bot {token}"}) as http:
        r = await http.request(method, f"{API}{path}", params=params, json=json)
    if r.status_code == 401:
        raise IntegrationError("Discord rejected the saved bot token. Reconnect Discord.")
    if r.status_code == 403:
        raise IntegrationError("The Sentient bot doesn't have permission for that channel. Check its role permissions.")
    r.raise_for_status()
    return r.json() if r.content else {}


@itool(PID, "discord_list_guilds")
async def discord_list_guilds(ctx: ToolContext) -> dict:
    """List the Discord servers the Sentient bot has been added to."""
    rows = await dapi(ctx, "GET", "/users/@me/guilds")
    return {"guilds": [{"id": g.get("id"), "name": g.get("name")} for g in rows]}


@itool(PID, "discord_list_channels")
async def discord_list_channels(ctx: ToolContext, guild_id: str) -> dict:
    """List the channels in a Discord server (ids, names, types)."""
    rows = await dapi(ctx, "GET", f"/guilds/{guild_id}/channels")
    return {"guild_id": guild_id, "channels": [
        {"id": c.get("id"), "name": c.get("name"), "type": CHANNEL_TYPES.get(c.get("type"), str(c.get("type"))),
         "topic": c.get("topic")} for c in sorted(rows, key=lambda c: c.get("position", 0))]}


@itool(PID, "discord_read_messages")
async def discord_read_messages(ctx: ToolContext, channel_id: str, limit: int = 20) -> dict:
    """Read recent messages in a Discord channel (message text needs the bot's Message Content intent)."""
    rows = await dapi(ctx, "GET", f"/channels/{channel_id}/messages", params={"limit": max(1, min(int(limit or 20), 100))})
    return {"channel_id": channel_id, "messages": [
        {"id": m.get("id"), "author": (m.get("author") or {}).get("username"), "content": m.get("content"),
         "timestamp": m.get("timestamp")} for m in rows]}


@itool(PID, "discord_send_message", risk=Risk.send)
async def discord_send_message(ctx: ToolContext, channel_id: str, content: str) -> dict:
    """Send a message to a Discord channel as the Sentient bot."""
    if not content.strip():
        raise IntegrationError("The message is empty.")
    m = await dapi(ctx, "POST", f"/channels/{channel_id}/messages", json={"content": content[:2000]})
    return {"sent": True, "id": m.get("id"), "channel_id": channel_id}


class DiscordPlugin(IntegrationPlugin):
    id = PID
    display_name = "Discord"
    description = (
        "Let Sentient act in your Discord servers through its own bot: list servers and channels, read recent "
        "messages and send messages to channels."
    )
    category = "communication"
    icon = "discord"
    auth_type = "api_key"
    selection_hint = "Use to list Discord servers/channels, read Discord messages, or send a message to a Discord channel."
    setup_fields = [SetupField("token", "Bot token", secret=True, required=True,
                               help="Discord Developer Portal → your application → Bot → Reset Token.")]
    docs_url = "https://discord.com/developers/docs/quick-start/getting-started"
    instructions_md = (
        "1. Open https://discord.com/developers/applications and click **New Application**. Name it `Sentient`.\n"
        "2. Open the **Bot** tab. Click **Reset Token**, confirm, and copy the token.\n"
        "3. On the same tab, turn on **Message Content Intent** (so Sentient can read messages) and save.\n"
        "4. Open **OAuth2 → URL Generator**. Tick **bot**, then under Bot Permissions tick **View Channels**, "
        "**Send Messages** and **Read Message History**.\n"
        "5. Copy the generated URL at the bottom, open it in your browser, pick your server and click **Authorize**.\n"
        "6. Paste the bot token here and click **Connect**.\n"
    )
    tools = [discord_list_guilds, discord_list_channels, discord_read_messages, discord_send_message]

    async def validate(self, fields: dict[str, str], mgr: IntegrationManager) -> tuple[dict, str | None]:
        token = str(fields.get("token", "")).strip().removeprefix("Bot ").strip()
        if not token:
            raise IntegrationError("Please paste the Discord bot token.")
        async with http_client(headers={"Authorization": f"Bot {token}"}) as http:
            r = await http.get(f"{API}/users/@me")
        if r.status_code in (401, 403):
            raise IntegrationError("Discord didn't accept that bot token. Reset it in the Developer Portal and paste the new one.")
        r.raise_for_status()
        return {"token": token}, r.json().get("username")


PLUGIN = DiscordPlugin()
