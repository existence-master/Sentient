"""Trello with an API key + user token."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sentient.integrations.base import IntegrationError, IntegrationPlugin, SetupField, creds, itool
from sentient.integrations.common import http_client
from sentient.tools.base import Risk, ToolContext

if TYPE_CHECKING:  # pragma: no cover
    from sentient.integrations.service import IntegrationManager

API = "https://api.trello.com/1"
PID = "trello"


async def tapi(ctx: ToolContext, method: str, path: str, *, params: dict | None = None, c: dict | None = None) -> Any:
    c = c or await creds(ctx, PID)
    auth = {"key": c["api_key"], "token": c["token"]}
    async with http_client() as http:
        r = await http.request(method, f"{API}{path}", params={**auth, **(params or {})})
    if r.status_code == 401:
        raise IntegrationError("Trello rejected the saved key or token. Reconnect Trello.")
    r.raise_for_status()
    return r.json()


def _card(c: dict) -> dict:
    return {"id": c.get("id"), "name": c.get("name"), "description": c.get("desc"), "due": c.get("due"),
            "list_id": c.get("idList"), "labels": [lab.get("name") for lab in c.get("labels") or []], "url": c.get("url")}


@itool(PID, "trello_list_boards")
async def trello_list_boards(ctx: ToolContext) -> dict:
    """List the user's open Trello boards."""
    rows = await tapi(ctx, "GET", "/members/me/boards", params={"filter": "open", "fields": "name,url,desc"})
    return {"boards": [{"id": b["id"], "name": b.get("name"), "url": b.get("url")} for b in rows]}


@itool(PID, "trello_list_lists")
async def trello_list_lists(ctx: ToolContext, board_id: str) -> dict:
    """List the lists (columns like "To Do", "Doing") on a Trello board."""
    rows = await tapi(ctx, "GET", f"/boards/{board_id}/lists", params={"filter": "open"})
    return {"board_id": board_id, "lists": [{"id": lst["id"], "name": lst.get("name")} for lst in rows]}


@itool(PID, "trello_list_cards")
async def trello_list_cards(ctx: ToolContext, list_id: str | None = None, board_id: str | None = None) -> dict:
    """List cards in a Trello list, or every open card on a board."""
    if list_id:
        rows = await tapi(ctx, "GET", f"/lists/{list_id}/cards")
    elif board_id:
        rows = await tapi(ctx, "GET", f"/boards/{board_id}/cards", params={"filter": "open"})
    else:
        raise IntegrationError("Give a list_id or a board_id.")
    return {"cards": [_card(c) for c in rows]}


@itool(PID, "trello_create_card", risk=Risk.write)
async def trello_create_card(ctx: ToolContext, list_id: str, name: str, description: str | None = None,
                             due: str | None = None, position: str = "bottom") -> dict:
    """Create a card in a Trello list. `due` is an ISO date/time; `position` is "top" or "bottom"."""
    params: dict[str, Any] = {"idList": list_id, "name": name, "pos": position if position in {"top", "bottom"} else "bottom"}
    if description:
        params["desc"] = description
    if due:
        params["due"] = due
    return {"created": True, "card": _card(await tapi(ctx, "POST", "/cards", params=params))}


@itool(PID, "trello_move_card", risk=Risk.write)
async def trello_move_card(ctx: ToolContext, card_id: str, list_id: str, position: str = "top") -> dict:
    """Move a Trello card to another list (e.g. from "Doing" to "Done")."""
    card = await tapi(ctx, "PUT", f"/cards/{card_id}",
                      params={"idList": list_id, "pos": position if position in {"top", "bottom"} else "top"})
    return {"moved": True, "card": _card(card)}


class TrelloPlugin(IntegrationPlugin):
    id = PID
    display_name = "Trello"
    description = "Manage your Trello boards: see boards, lists and cards, add new cards and move cards between lists."
    category = "productivity"
    icon = "trello"
    auth_type = "api_key"
    selection_hint = "Use for Trello boards, lists and cards (view, create or move cards)."
    setup_fields = [
        SetupField("api_key", "API key", secret=False, required=True, help="From trello.com/power-ups/admin → your Power-Up → API key."),
        SetupField("token", "Token", secret=True, required=True, help="Generated from the link next to your API key."),
    ]
    docs_url = "https://developer.atlassian.com/cloud/trello/guides/rest-api/api-introduction/"
    instructions_md = (
        "1. Sign in to Trello and open https://trello.com/power-ups/admin.\n"
        "2. Click **New**, name it `Sentient`, pick any workspace, fill in your email, and click **Create**.\n"
        "3. Open the **API key** tab and click **Generate a new API key**. Copy the **API key**.\n"
        "4. On the same page, click the **Token** link next to the key (or open "
        "`https://trello.com/1/authorize?expiration=never&scope=read,write&response_type=token&key=YOUR_KEY`).\n"
        "5. Click **Allow** and copy the token shown.\n"
        "6. Paste the API key and token here and click **Connect**.\n"
    )
    tools = [trello_list_boards, trello_list_lists, trello_list_cards, trello_create_card, trello_move_card]

    async def validate(self, fields: dict[str, str], mgr: IntegrationManager) -> tuple[dict, str | None]:
        key, token = str(fields.get("api_key", "")).strip(), str(fields.get("token", "")).strip()
        if not key or not token:
            raise IntegrationError("Please paste both the Trello API key and the token.")
        c = {"api_key": key, "token": token}
        async with http_client() as http:
            r = await http.get(f"{API}/members/me", params={"key": key, "token": token, "fields": "username,fullName"})
        if r.status_code in (400, 401):
            raise IntegrationError("Trello didn't accept that key and token. Generate the token again from your API key page.")
        r.raise_for_status()
        return c, r.json().get("username")


PLUGIN = TrelloPlugin()
