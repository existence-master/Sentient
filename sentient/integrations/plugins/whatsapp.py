"""WhatsApp via a self-hosted WAHA server. Sends only to the user's own number (v2 behaviour)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import httpx

from sentient.integrations.base import IntegrationError, IntegrationPlugin, SetupField, creds, itool
from sentient.integrations.common import http_client
from sentient.tools.base import Risk, ToolContext

if TYPE_CHECKING:  # pragma: no cover
    from sentient.integrations.service import IntegrationManager

PID = "whatsapp"


def _headers(api_key: str) -> dict:
    return {"X-Api-Key": api_key} if api_key else {}


def normalize_number(raw: str) -> str:
    digits = re.sub(r"\D", "", raw or "")
    if len(digits) < 8:
        raise IntegrationError("Enter your WhatsApp number with the country code, e.g. +91 98765 43210.")
    return digits


def _check_session(resp: httpx.Response, session: str) -> None:
    if resp.status_code in (401, 403):
        raise IntegrationError("The WAHA server rejected the API key.")
    if resp.status_code == 404:
        raise IntegrationError(f"WAHA has no session called '{session}'. Start it in the WAHA dashboard.")
    if resp.status_code >= 400:
        raise IntegrationError(f"The WAHA server returned an error (HTTP {resp.status_code}).")
    status = (resp.json() or {}).get("status")
    if status and status != "WORKING":
        raise IntegrationError(f"The WAHA session is {status}. Scan the QR code in the WAHA dashboard first.")


@itool(PID, "whatsapp_send_message", risk=Risk.send)
async def whatsapp_send_message(ctx: ToolContext, message: str) -> dict:
    """Send a WhatsApp message to the USER'S OWN number (reminders, alerts, results of a task). It cannot message anyone else."""
    c = await creds(ctx, PID)
    if not message.strip():
        raise IntegrationError("The message is empty.")
    async with http_client(headers=_headers(c.get("api_key", ""))) as http:
        r = await http.post(f"{c['base_url']}/api/sendText",
                            json={"session": c.get("session") or "default", "chatId": c["chat_id"], "text": message})
    if r.status_code in (401, 403):
        raise IntegrationError("The WAHA server rejected the API key. Reconnect WhatsApp.")
    r.raise_for_status()
    data = r.json() if r.content else {}
    msg_id = data.get("id")
    if isinstance(msg_id, dict):
        msg_id = msg_id.get("_serialized") or msg_id.get("id")
    return {"sent": True, "id": msg_id}


class WhatsAppPlugin(IntegrationPlugin):
    id = PID
    display_name = "WhatsApp"
    description = (
        "Let Sentient send WhatsApp messages to your own number, for reminders, alerts and task results. "
        "Needs a WAHA (WhatsApp HTTP API) server linked to a WhatsApp account; this is optional and meant for "
        "technical users."
    )
    category = "communication"
    icon = "whatsapp"
    auth_type = "manual"
    selection_hint = "Use only to send a WhatsApp message to the user themself (notifications, reminders, results)."
    setup_fields = [
        SetupField("base_url", "WAHA server URL", secret=False, required=True, help="Where WAHA runs.",
                   placeholder="http://localhost:3000"),
        SetupField("api_key", "WAHA API key", secret=True, required=False, help="The WAHA_API_KEY you set, if any."),
        SetupField("phone_number", "Your WhatsApp number", secret=False, required=True,
                   help="With country code. Messages go only to this number.", placeholder="+91 98765 43210"),
        SetupField("session", "WAHA session name", secret=False, required=False, help="Usually 'default'.",
                   placeholder="default"),
    ]
    docs_url = "https://waha.devlike.pro/docs/overview/quick-start/"
    instructions_md = (
        "WhatsApp has no official personal API, so Sentient uses WAHA, a small server you run yourself.\n\n"
        "1. Install Docker Desktop from https://www.docker.com/products/docker-desktop/ and start it.\n"
        "2. Open a terminal and run: `docker run -it -p 3000:3000 -e WAHA_API_KEY=choose-a-secret devlikeapro/waha`\n"
        "3. Open http://localhost:3000/dashboard, start the **default** session, and scan the QR code with "
        "WhatsApp on your phone (**Settings → Linked devices → Link a device**).\n"
        "4. When the session shows **WORKING**, come back here.\n"
        "5. Enter `http://localhost:3000` as the server URL, the API key you chose, and your own WhatsApp number "
        "with country code.\n"
        "6. Click **Connect**.\n"
    )
    tools = [whatsapp_send_message]

    async def validate(self, fields: dict[str, str], mgr: IntegrationManager) -> tuple[dict, str | None]:
        base = str(fields.get("base_url", "")).strip().rstrip("/")
        if not re.match(r"^https?://", base):
            raise IntegrationError("Enter the WAHA server URL, e.g. http://localhost:3000.")
        api_key = str(fields.get("api_key", "")).strip()
        session = str(fields.get("session", "")).strip() or "default"
        digits = normalize_number(str(fields.get("phone_number", "")))
        try:
            async with http_client(timeout=15, headers=_headers(api_key)) as http:
                s = await http.get(f"{base}/api/sessions/{session}")
                chk = await http.get(f"{base}/api/contacts/check-exists", params={"phone": digits, "session": session})
        except httpx.HTTPError as exc:
            raise IntegrationError(f"Couldn't reach the WAHA server at {base}. Is it running?") from exc
        _check_session(s, session)
        chat_id = f"{digits}@c.us"
        if chk.status_code < 400:
            data = chk.json() or {}
            if data.get("numberExists") is False:
                raise IntegrationError("That number isn't registered on WhatsApp. Check the country code.")
            chat_id = data.get("chatId") or chat_id
        return {"base_url": base, "api_key": api_key, "session": session, "phone_number": digits, "chat_id": chat_id}, f"+{digits}"


PLUGIN = WhatsAppPlugin()
