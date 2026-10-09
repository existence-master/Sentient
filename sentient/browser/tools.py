"""Browser tools (plugin ``browser``). Contract: docs/API.md section 12.

Tools are thin wrappers over :class:`BrowserService`. Failures come back to the model as a
friendly ``{"error": ...}``. Clicks, typing with submit and key presses carry a ``risk_fn`` so a
click on "Place order" is approved as ``send`` rather than ``write``.
"""

from __future__ import annotations

import functools
from typing import Any

from sentient.browser.service import BrowserError, _friendly_playwright_error, service_from
from sentient.tools.base import Risk, Tool, ToolContext, ToolPlugin, tool

SAFETY = (
    " Never type passwords, card numbers or one-time codes and never complete a purchase, payment or post without "
    "the user's approval: when a site needs signing in, ask the user to click 'Open browser' and sign in themselves."
)


def _risk(kind: str, arguments: dict | None, ctx: Any) -> Risk | None:
    try:
        return service_from(ctx).risk_for(kind, arguments or {})
    except Exception:
        return Risk.send if kind == "click" else None


def _describe(kind: str, arguments: dict | None, ctx: Any) -> dict:
    try:
        return service_from(ctx).describe_for(kind, arguments or {})
    except Exception:
        return {}


def _opened(arguments: dict | None, ctx: Any) -> str:
    return str((arguments or {}).get("url") or "")


def _link(arguments: dict | None, ctx: Any) -> str:
    try:
        return service_from(ctx).link_for(arguments or {})
    except Exception:
        return ""


# the web address a call loads, so an address that could carry data to a new site asks first (ADR 0018)
_ADDRESS = {"browser_open": _opened, "browser_click": _link}


def btool(name: str, *, risk: Risk, risk_kind: str | None = None):
    def wrap(fn):
        @functools.wraps(fn)
        async def safe(ctx: ToolContext, **kwargs: Any) -> Any:
            try:
                return await fn(ctx, **kwargs)
            except BrowserError as exc:
                return {"error": str(exc)}
            except Exception as exc:
                if type(exc).__module__.startswith("playwright"):
                    return {"error": _friendly_playwright_error(exc)}
                raise

        # every browser result shows a page someone else wrote; typing puts text into that page (ADR 0018)
        t: Tool = tool(
            name, risk=risk, untrusted_output=True, exfiltrates=name == "browser_type", url_fn=_ADDRESS.get(name)
        )(safe)
        if risk_kind:
            t.risk_fn = functools.partial(_risk, risk_kind)  # type: ignore[attr-defined]
            t.describe_fn = functools.partial(_describe, risk_kind)  # type: ignore[attr-defined]
        return t

    return wrap


@btool("browser_open", risk=Risk.read)
async def browser_open(ctx: ToolContext, url: str, profile: str = "") -> dict:
    """Open a web address in Sentient's own browser and return a snapshot of the page: its buttons, links and
    fields with refs like [e12], plus the page text. Use for websites without an integration or when you need to
    click or fill things in; to only read an article, web_fetch is faster. `profile` picks a named browser profile
    (each has its own sign-ins) for this and the following browser calls; leave it empty to keep the current one."""
    return await service_from(ctx).open(ctx, url, profile)


@btool("browser_snapshot", risk=Risk.read)
async def browser_snapshot(ctx: ToolContext) -> dict:
    """Describe the current page: interactive elements with refs like [e12] button "Sign in" and the visible text.
    Call it after anything changes the page; refs from an older snapshot stop working."""
    return await service_from(ctx).snapshot(ctx)


@btool("browser_click", risk=Risk.write, risk_kind="click")
async def browser_click(ctx: ToolContext, ref: str) -> dict:
    """Click the element with this ref (from browser_snapshot), e.g. "e12". Buttons that buy, pay, send, post,
    delete or confirm need the user's approval."""
    return await service_from(ctx).click(ctx, ref)


@btool("browser_type", risk=Risk.write, risk_kind="type")
async def browser_type(ctx: ToolContext, ref: str, text: str, submit: bool = False) -> dict:
    """Replace the text in a field (ref from browser_snapshot) with `text`. submit=true presses Enter afterwards,
    for example to run a search."""
    return await service_from(ctx).type(ctx, ref, text, submit)


@btool("browser_select", risk=Risk.write)
async def browser_select(ctx: ToolContext, ref: str, option: str) -> dict:
    """Choose an option in a dropdown list (a combobox ref from browser_snapshot) by its visible label."""
    return await service_from(ctx).select(ctx, ref, option)


@btool("browser_press", risk=Risk.write, risk_kind="press")
async def browser_press(ctx: ToolContext, key: str) -> dict:
    """Press a key in the page: Enter, Tab, Escape, ArrowDown, PageDown, or a combination like Control+a."""
    return await service_from(ctx).press(ctx, key)


@btool("browser_scroll", risk=Risk.read)
async def browser_scroll(ctx: ToolContext, direction: str = "down") -> dict:
    """Scroll the page: down, up, top, bottom, left or right. Then call browser_snapshot."""
    return await service_from(ctx).scroll(ctx, direction)


@btool("browser_back", risk=Risk.read)
async def browser_back(ctx: ToolContext) -> dict:
    """Go back to the previous page in the current tab."""
    return await service_from(ctx).back(ctx)


@btool("browser_tabs", risk=Risk.read)
async def browser_tabs(ctx: ToolContext, profile: str = "") -> dict:
    """List the open browser tabs with their index, address and title. `profile` works as in browser_open."""
    return await service_from(ctx).tabs(ctx, profile)


@btool("browser_switch_tab", risk=Risk.read)
async def browser_switch_tab(ctx: ToolContext, index: int) -> dict:
    """Make the tab with this index (from browser_tabs) the one the other browser tools act on."""
    return await service_from(ctx).switch_tab(ctx, index)


@btool("browser_extract", risk=Risk.read)
async def browser_extract(ctx: ToolContext, question: str = "") -> dict:
    """Return the readable main text of the current page without menus and scripts. Pass `question` to keep the
    most relevant parts of a long page."""
    return await service_from(ctx).extract(ctx, question)


@btool("browser_screenshot", risk=Risk.read)
async def browser_screenshot(ctx: ToolContext) -> dict:
    """Save a picture of the current page to the files folder (outputs/browser/) and return its file name."""
    return await service_from(ctx).screenshot(ctx)


@btool("browser_close", risk=Risk.read)
async def browser_close(ctx: ToolContext) -> dict:
    """Close the browser when you are done with websites. It reopens automatically when needed."""
    return await service_from(ctx).close_tool(ctx)


# the safety rules ride on the tools the model uses to act
for _t in (browser_open, browser_click, browser_type):
    _t.description += SAFETY


class BrowserPlugin(ToolPlugin):
    id = "browser"
    display_name = "Web Browser"
    description = (
        "Uses a real web browser (Edge or Chrome) on your computer for websites without an integration: searching, "
        "filling in forms, reading pages. You sign in yourself, and it asks before buying, sending or deleting anything."
    )
    category = "utilities"
    icon = "browser"
    auth = "none"
    selection_hint = (
        "Use to operate a website step by step (open, click, type, choose, read) when no integration or web_fetch "
        "can do the job, for example booking forms, shopping carts or sites that need the user's sign-in."
    )
    tools = [
        browser_open, browser_snapshot, browser_click, browser_type, browser_select, browser_press,
        browser_scroll, browser_back, browser_tabs, browser_switch_tab, browser_extract, browser_screenshot,
        browser_close,
    ]
