"""Google Sheets: create, read, append, update."""

from __future__ import annotations

from urllib.parse import quote

from sentient.integrations.base import IntegrationError, itool
from sentient.integrations.google import gapi
from sentient.integrations.plugins._google_base import GooglePlugin
from sentient.tools.base import Risk, ToolContext

API = "https://sheets.googleapis.com/v4/spreadsheets"
PID = "gsheets"
Cell = str | int | float | bool | None


def _url(sid: str) -> str:
    return f"https://docs.google.com/spreadsheets/d/{sid}/edit"


def _range(r: str) -> str:
    return quote(r, safe="!:$'")


async def _first_sheet(ctx: ToolContext, sid: str) -> str:
    meta = await gapi(ctx, PID, "GET", f"{API}/{sid}", params={"fields": "sheets(properties(title))"})
    sheets = meta.get("sheets") or []
    if not sheets:
        raise IntegrationError("That spreadsheet has no sheets.")
    return sheets[0]["properties"]["title"]


def _quote_sheet(title: str) -> str:
    return f"'{title}'" if not title.replace("_", "").isalnum() else title


@itool(PID, "gsheets_create_spreadsheet", risk=Risk.write)
async def gsheets_create_spreadsheet(ctx: ToolContext, title: str, headers: list[str] | None = None,
                                     rows: list[list[Cell]] | None = None, sheet_title: str = "Sheet1") -> dict:
    """Create a Google Sheet, optionally filled with a header row and data rows. Returns its id and link."""
    res = await gapi(ctx, PID, "POST", API, json={"properties": {"title": title},
                                                  "sheets": [{"properties": {"title": sheet_title}}]})
    sid = res["spreadsheetId"]
    values = ([list(headers)] if headers else []) + [list(r) for r in rows or []]
    if values:
        rng = f"{_quote_sheet(sheet_title)}!A1"
        await gapi(ctx, PID, "PUT", f"{API}/{sid}/values/{_range(rng)}", params={"valueInputOption": "USER_ENTERED"},
                   json={"range": rng, "majorDimension": "ROWS", "values": values})
    return {"spreadsheet_id": sid, "title": title, "rows_written": len(values), "url": _url(sid)}


@itool(PID, "gsheets_get_info")
async def gsheets_get_info(ctx: ToolContext, spreadsheet_id: str) -> dict:
    """Get a spreadsheet's title and its sheets (tab names and sizes)."""
    meta = await gapi(ctx, PID, "GET", f"{API}/{spreadsheet_id}",
                      params={"fields": "properties(title),sheets(properties(title,sheetId,gridProperties))"})
    return {"spreadsheet_id": spreadsheet_id, "title": (meta.get("properties") or {}).get("title"), "url": _url(spreadsheet_id),
            "sheets": [{"title": s["properties"]["title"], "sheet_id": s["properties"].get("sheetId"),
                        "rows": (s["properties"].get("gridProperties") or {}).get("rowCount"),
                        "columns": (s["properties"].get("gridProperties") or {}).get("columnCount")}
                       for s in meta.get("sheets") or []]}


@itool(PID, "gsheets_read_range")
async def gsheets_read_range(ctx: ToolContext, spreadsheet_id: str, range: str | None = None) -> dict:
    """Read cells from a sheet in A1 notation, e.g. "Sheet1!A1:D50" or just "Budget" for a whole tab.
    Without a range, reads the first tab (up to 500 rows)."""
    rng = range or f"{_quote_sheet(await _first_sheet(ctx, spreadsheet_id))}!A1:Z500"
    res = await gapi(ctx, PID, "GET", f"{API}/{spreadsheet_id}/values/{_range(rng)}")
    values = res.get("values") or []
    return {"spreadsheet_id": spreadsheet_id, "range": res.get("range", rng), "row_count": len(values), "values": values}


@itool(PID, "gsheets_append_rows", risk=Risk.write)
async def gsheets_append_rows(ctx: ToolContext, spreadsheet_id: str, rows: list[list[Cell]],
                              sheet: str | None = None) -> dict:
    """Add rows after the last filled row of a tab (default: the first tab). Formulas like =SUM(A1:A5) work."""
    if not rows:
        raise IntegrationError("There are no rows to add.")
    rng = _quote_sheet(sheet or await _first_sheet(ctx, spreadsheet_id))
    res = await gapi(ctx, PID, "POST", f"{API}/{spreadsheet_id}/values/{_range(rng)}:append",
                     params={"valueInputOption": "USER_ENTERED", "insertDataOption": "INSERT_ROWS"},
                     json={"majorDimension": "ROWS", "values": [list(r) for r in rows]})
    upd = res.get("updates") or {}
    return {"spreadsheet_id": spreadsheet_id, "updated_range": upd.get("updatedRange"),
            "rows_added": upd.get("updatedRows", len(rows)), "url": _url(spreadsheet_id)}


@itool(PID, "gsheets_update_range", risk=Risk.write)
async def gsheets_update_range(ctx: ToolContext, spreadsheet_id: str, range: str, values: list[list[Cell]]) -> dict:
    """Overwrite cells starting at a range in A1 notation, e.g. range "Sheet1!B2" with values [["Paid", 120]]."""
    res = await gapi(ctx, PID, "PUT", f"{API}/{spreadsheet_id}/values/{_range(range)}",
                     params={"valueInputOption": "USER_ENTERED"},
                     json={"range": range, "majorDimension": "ROWS", "values": [list(r) for r in values]})
    return {"spreadsheet_id": spreadsheet_id, "updated_range": res.get("updatedRange"),
            "updated_cells": res.get("updatedCells"), "url": _url(spreadsheet_id)}


class GSheetsPlugin(GooglePlugin):
    id = PID
    display_name = "Google Sheets"
    description = (
        "Build and update spreadsheets. Sentient can create a sheet from a table of data, read ranges, add rows "
        "(like logging expenses) and update cells, including formulas."
    )
    category = "productivity"
    icon = "google-sheets"
    api_name = "Google Sheets API"
    api_slug = "sheets.googleapis.com"
    selection_hint = "Use to create, read or edit spreadsheets in Google Sheets."
    tools = [gsheets_create_spreadsheet, gsheets_get_info, gsheets_read_range, gsheets_append_rows, gsheets_update_range]


PLUGIN = GSheetsPlugin()
