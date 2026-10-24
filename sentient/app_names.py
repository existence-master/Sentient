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
