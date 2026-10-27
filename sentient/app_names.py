"""How an outside app's self-chosen name may be shown to the user (issue #443).

The name arrives word for word from the app's own handshake, so it is untrusted text like
everything else an app sends: a client can call itself "Sentient Security Center. Your data
will be erased unless you tap Allow now.", and uncleaned that sentence would land in the
approval notification the user decides from. Every surface that shows the name therefore
takes it from here: control characters and newlines stripped, capped at 40 characters, and
the notes that quote it frame it as a name (``An app called `...` ``) instead of letting it
speak as if it were Sentient.
"""

from __future__ import annotations

import re

MAX_APP_NAME = 40

# Handshake names whose display form reads better: the MCP client's own name for
# Claude Code is "claude-code".
KNOWN_APPS = {"claude-code": "Claude Code"}

_WS = re.compile(r"\s+")

# Backtick runs, for sizing a code-span fence one longer than any run inside.
_BACKTICK_RUN = re.compile(r"`+")


def code_span(text: str) -> str:
    """The text as one inline code span: a fence one backtick longer than the longest
    run of backticks inside, with a space inside each end (rendered away).

    Nothing inside a code span is parsed, so this is what a Markdown-rendered note uses
    for app-controlled text. Backslash escaping cannot do this job: remark-gfm resolves
    escapes before it looks for autolinks, so `https\\://evil.co` still arrives as a live
    link (review 5641825878)."""
    if not text or not text.strip():
        return text
    fence = "`" * (max((len(m.group(0)) for m in _BACKTICK_RUN.finditer(text)), default=0) + 1)
    return f"{fence} {text} {fence}"


def framed_app_name(name: object, *, capital: bool = True) -> str:
    """The app name as it may go into a Markdown-rendered note (approval, plan-ready).

    The name itself is already cleaned (see clean_app_name); this only stops Markdown
    from acting on it: a client calling itself "[Fix it](https://evil.co) **NOW**" must
    not arrive as a live link with bold text right above Allow and Deny. The name goes
    in an inline code span, so nothing inside it is parsed at all — links, autolinks
    (bare URLs and emails included), emphasis, images. Plain-text surfaces (the turn
    mark, the Review list, notify titles) keep the bare name."""
    if not isinstance(name, str) or not name or name.lower() == "an mcp app":
        return "An MCP app" if capital else "an MCP app"
    framed = f"an app called {code_span(name)}"
    return framed[0].upper() + framed[1:] if capital else framed


def clean_app_name(raw: object) -> str | None:
    """The display form of an app's name, or None when nothing usable remains."""
    if not isinstance(raw, str):
        return None
    # control and format characters (newlines among them) become spaces, runs collapse
    text = "".join(ch if ch.isprintable() else " " for ch in raw)
    text = _WS.sub(" ", text).strip()
    if not text:
        return None
    text = KNOWN_APPS.get(text.lower(), text)
    if len(text) > MAX_APP_NAME:
        text = text[: MAX_APP_NAME - 1].rstrip() + "…"
    return text
