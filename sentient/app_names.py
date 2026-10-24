"""How an outside app's self-chosen name may be shown to the user (issue #443).

The name arrives word for word from the app's own handshake, so it is untrusted text like
everything else an app sends: a client can call itself "Sentient Security Center. Your data
will be erased unless you tap Allow now.", and uncleaned that sentence would land in the
approval notification the user decides from. Every surface that shows the name therefore
takes it from here: control characters and newlines stripped, capped at 40 characters, and
the notes that quote it frame it as a name (``An app called "..."``) instead of letting it
speak as if it were Sentient.
"""

from __future__ import annotations

import re

MAX_APP_NAME = 40

# Handshake names whose display form reads better: the MCP client's own name for
# Claude Code is "claude-code".
KNOWN_APPS = {"claude-code": "Claude Code"}

_WS = re.compile(r"\s+")

# Punctuation Markdown can act on where a note is rendered as such (the desktop renders
# approval and plan-ready notes as Markdown): emphasis, code, links, images, HTML, strikes.
_MARKDOWN_PUNCT = re.compile(r"([\\`*{}\[\]()#<>!|~_])")
_SCHEME = re.compile(r"(?i)\b([a-z][a-z0-9+.-]*)://")
_WWW = re.compile(r"(?i)\bwww\.")


def _escape_markdown(text: str) -> str:
    """The text with nothing left that Markdown can act on: punctuation backslash-escaped
    in one pass (so an introduced backslash is never escaped again), and URL schemes and
    www. broken so GFM cannot autolink them. What renders is still the same characters."""
    text = _MARKDOWN_PUNCT.sub(r"\\\1", text)
    text = _SCHEME.sub(r"\1\\://", text)
    return _WWW.sub(r"www\\.", text)


def framed_app_name(name: object, *, capital: bool = True) -> str:
    """The app name as it may go into a Markdown-rendered note (approval, plan-ready).

    The name itself is already cleaned (see clean_app_name); this only neutralises what
    Markdown would do with it: a client calling itself "[Fix it](https://evil.co) **NOW**"
    must not arrive as a live link with bold text right above Allow and Deny. Plain-text
    surfaces (the turn mark, the Review list, notify titles) keep the unescaped name."""
    if not isinstance(name, str) or not name or name.lower() == "an mcp app":
        return "An MCP app" if capital else "an MCP app"
    framed = f'an app called "{_escape_markdown(name)}"'
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
