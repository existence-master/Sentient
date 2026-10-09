"""Hide one-time codes, magic sign-in links and password reset links in email before a model reads it.

A prompt injection that can read a sign-in code or a reset link can take over an account, so these never reach a
prompt. Everything here is deterministic (regexes plus subject and sender hints) and leaves ordinary numbers (order
numbers, dates, prices, phone numbers) and ordinary links alone. The user still sees the original in their mail app.

- :func:`hide_secrets` masks one piece of text.
- :func:`hide_email_secrets` masks the ``subject``, ``snippet`` and ``body`` of a normalized email item.
- :func:`enabled` reads the ``integrations.hide_one_time_codes`` switch from an integration manager or app.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qsl, urlsplit

CODE_PLACEHOLDER = "[one-time code hidden]"
LINK_PLACEHOLDER = "[sign-in link hidden]"
RESET_PLACEHOLDER = "[password reset link hidden]"

# ---------------------------------------------------------------------------- codes
# Words right before "code" that make it something else (an error, a postcode, a discount...).
_NOT_A_CODE = {
    "error", "status", "exit", "return", "response", "zip", "postal", "post", "area", "country", "promo", "promotion",
    "promotional", "discount", "coupon", "voucher", "gift", "referral", "invite", "tracking", "product", "item",
    "source", "dress", "qr", "bar", "reference", "booking", "tax", "hs", "sort", "swift", "ifsc", "airport", "pin",
    "colour", "color", "sic", "naics", "billing", "branch", "bank", "class", "course", "event", "order",
    "reservation", "flight", "ticket", "seat", "room", "door", "lock", "dial", "phone", "postcode", "ups",
}
_STRONG_CUE = (
    r"one[- ]?time[- ](?:code|password|passcode|pin|key)|otp|verification[- ](?:code|number|pin)|verify[- ]code|"
    r"security[- ](?:code|key|pin)|(?:sign|log)[- ]?in[- ](?:code|pin)|login[- ](?:code|pin)|"
    r"authenticat(?:ion|or)[- ]code|auth[- ]code|(?:2fa|mfa|two[- ](?:factor|step))(?:[- ](?:code|verification))?|"
    r"passcode|pass[- ]code|access[- ]code|activation[- ]code|recovery[- ]code|backup[- ]code|reset[- ]code|"
    r"pin(?![- ]?codes?\b)"
)
_WEAK_CUE = r"code"
_CUE_RE = re.compile(rf"\b(?:(?P<strong>{_STRONG_CUE})|(?P<weak>{_WEAK_CUE}))\b", re.IGNORECASE)
# what may sit between the cue and the code: ":", "is", or a few short words ending in "is" or ":"
_GAP_RE = re.compile(
    r"[^\S\n]*(?:"
    r"[:=#\-\u2013]"
    r"|(?:[A-Za-z'\u2019]{1,15}[^\S\n]+){0,4}?(?:is|was|are)\b[^\S\n]*:?"
    r"|(?:[A-Za-z'\u2019]{1,15}[^\S\n]+){0,3}[A-Za-z'\u2019]{1,15}:"
    r")?[^\S\n]*(?:\n[^\S\n]*){0,3}",
    re.IGNORECASE,
)
_NUM = r"\d{3}[ -]\d{3}|\d{4,8}"
_ALNUM = r"(?=[A-Z0-9-]*\d)(?=[A-Z0-9-]*[A-Z])[A-Z0-9]{1,6}-?[A-Z0-9]{2,8}"
_SPACED = r"(?:\d[ ]){3,7}\d"
# a code token: not glued to other digits, words, prices, decimals, dates or phone numbers
_CURRENCY = "\u20ac\u00a3\u20b9\u00a5"  # euro, pound, rupee, yen
_BEFORE = r"(?<![\w.,/$" + _CURRENCY + r"#+@-])"
_AFTER = r"(?![\w/%@]|[.,:/-]\d)"
_NUM_TOKEN_RE = re.compile(rf"{_BEFORE}(?:{_NUM}){_AFTER}")
_ANY_TOKEN_RE = re.compile(rf"{_BEFORE}(?:{_NUM}|{_ALNUM}){_AFTER}")
# after a weak cue ("code") the code must end the sentence or be followed by a typical next word
_WEAK_TAIL_RE = re.compile(
    r"[^\S\n]*(?:$|[\n.,;!)\]]|\b(?:to|and|for|is|expires|it|valid|which|this|in|on|when|if)\b)", re.IGNORECASE
)
# "482913 is your verification code", "G-482913 is your Google code"
_CODE_FIRST_RE = re.compile(
    rf"{_BEFORE}(?P<code>(?:G-)?(?:{_NUM}|{_ALNUM})){_AFTER}"
    r"(?P<rest>[^\S\n]+is[^\S\n]+(?:your|the)\b[^.\n]{0,40}?\b(?:code|otp|pin|passcode|password)\b)",
)
_LINE_CODE_RE = re.compile(rf"(?P<code>(?:G-)?(?:{_NUM}|{_ALNUM}|{_SPACED}))[.!]?")

# ---------------------------------------------------------------------------- whole-email hints
_AUTH_SUBJECT_RE = re.compile(
    r"\b(?:verification|verify|your code|one[- ]?time|otp|passcode|security code|sign[- ]?in|log[- ]?in|login|"
    r"2fa|two[- ](?:factor|step)|authenticat\w*|confirm your (?:email|account|identity|address)|password|"
    r"magic link|access code|activation|activate your|reset)\b",
    re.IGNORECASE,
)
_AUTH_SENDER_RE = re.compile(
    r"(?:^|[.@_+-])(?:verify|verification|security|auth|accounts?|account-security|login|signin|otp|2fa|identity|"
    r"passwordreset|password)(?:$|[.@_+-])",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------- links
_URL_RE = re.compile(r"https?://[^\s<>\"'\])}]+", re.IGNORECASE)
_URL_AUTH_RE = re.compile(
    r"reset|password|passwd|recover|magic|sign[-_]?in|log[-_]?in|login|auth|verif|confirm|activat|otp|one[-_]?time|"
    r"passwordless|unlock|oobcode|token|session|invite|accept|validate|2fa|mfa",
    re.IGNORECASE,
)
_URL_RESET_RE = re.compile(r"reset|password|passwd|recover", re.IGNORECASE)
_SECRET_PARAMS = {
    "token", "code", "otp", "key", "auth", "sig", "signature", "hash", "ticket", "nonce", "oobcode", "apikey",
    "access_token", "id_token", "auth_token", "login_token", "magic", "secret", "t", "k", "h", "s", "l",
}
_TOKENISH_RE = re.compile(r"[A-Za-z0-9_\-.~%=+]{16,}")
_LINK_CONTEXT_RE = re.compile(
    r"\b(?:sign[- ]?in|log[- ]?in|login|magic link|reset (?:your )?password|password reset|forgot (?:your )?password|"
    r"set (?:a |your )?(?:new )?password|choose a new password|change (?:your )?password|recover (?:your )?account|"
    r"verify (?:your )?(?:email|account|address|identity)|confirm (?:your )?(?:email|account|address|identity|sign[- ]?in)|"
    r"activate (?:your )?account|one[- ]?time link|secure link|this link (?:will )?expires?|link expires|"
    r"access your account)\b",
    re.IGNORECASE,
)
_RESET_CONTEXT_RE = re.compile(r"\b(?:reset|password|recover)\b", re.IGNORECASE)


def _tokenish(value: str) -> bool:
    """A long opaque value: at least 16 characters, with letters and digits mixed (not a plain word or slug)."""
    return bool(_TOKENISH_RE.fullmatch(value)) and any(c.isdigit() for c in value) and any(c.isalpha() for c in value)


def _carries_secret(url: str) -> bool:
    """True when the link carries something that looks like a one-off key (a token-like value or parameter)."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    for name, value in parse_qsl(parts.query, keep_blank_values=True):
        if name.lower() in _SECRET_PARAMS and len(value) >= 6:
            return True
        if _tokenish(value):
            return True
    if _tokenish(parts.fragment.split("=")[-1]):
        return True
    return any(_tokenish(seg) for seg in parts.path.split("/"))


def _hide_links(text: str, auth_email: bool) -> str:
    def replace(m: re.Match) -> str:
        url = m.group(0).rstrip(".,;:!?")
        tail = m.group(0)[len(url):]
        if not _carries_secret(url):
            return m.group(0)
        before = text[max(0, m.start() - 160):m.start()]
        try:
            where = urlsplit(url)
            target = f"{where.path}?{where.query}"
        except ValueError:
            target = url
        context = bool(_LINK_CONTEXT_RE.search(before))
        if not (_URL_AUTH_RE.search(target) or context or auth_email):
            return m.group(0)
        reset = _URL_RESET_RE.search(target) or (context and _RESET_CONTEXT_RE.search(before))
        return (RESET_PLACEHOLDER if reset else LINK_PLACEHOLDER) + tail

    return _URL_RE.sub(replace, text)


def _qualified(text: str, cue_start: int) -> bool:
    """False when the word before a bare "code" makes it an error code, postcode, discount code..."""
    words = re.findall(r"[A-Za-z]+", text[max(0, cue_start - 24):cue_start])
    return not (words and words[-1].lower() in _NOT_A_CODE)


def _hide_inline_codes(text: str, auth_email: bool) -> str:
    spans: list[tuple[int, int]] = []
    for cue in _CUE_RE.finditer(text):
        strong = cue.group("strong") is not None
        if not strong and not _qualified(text, cue.start()):
            continue
        gap = _GAP_RE.match(text, cue.end())
        token_re = _ANY_TOKEN_RE if (strong or auth_email) else _NUM_TOKEN_RE
        tok = token_re.match(text, gap.end()) if gap else None
        if tok is None:
            continue
        if not strong and not _WEAK_TAIL_RE.match(text, tok.end()):
            continue
        spans.append(tok.span())
    for m in _CODE_FIRST_RE.finditer(text):
        spans.append(m.span("code"))
    return _apply(text, spans, CODE_PLACEHOLDER)


def _hide_line_codes(text: str, auth_email: bool) -> str:
    """A line that is only a code, right after a line that talks about a code (or anywhere in a sign-in email)."""
    lines = text.split("\n")
    recent: list[str] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        m = _LINE_CODE_RE.fullmatch(stripped)
        if m and not _looks_like_plain_number(m.group("code")):
            near = " ".join(recent[-3:])
            if auth_email or _CUE_RE.search(near):
                lines[i] = line.replace(m.group("code"), CODE_PLACEHOLDER, 1)
        recent.append(stripped)
    return "\n".join(lines)


def _looks_like_plain_number(token: str) -> bool:
    """A year on its own line (1900-2099) is far likelier than a code that happens to look like one."""
    return bool(re.fullmatch(r"(?:19|20)\d\d", token))


def _apply(text: str, spans: list[tuple[int, int]], placeholder: str) -> str:
    if not spans:
        return text
    out, last = [], 0
    for start, end in sorted(set(spans)):
        if start < last:
            continue
        out.append(text[last:start])
        out.append(placeholder)
        last = end
    out.append(text[last:])
    return "".join(out)


def is_auth_email(subject: str = "", sender: str = "") -> bool:
    """Subject or sender says this is a sign-in, verification or password email."""
    return bool(_AUTH_SUBJECT_RE.search(subject or "") or _AUTH_SENDER_RE.search((sender or "").split("@")[0])
                or _AUTH_SENDER_RE.search(_sender_domain(sender)))


def _sender_domain(sender: str) -> str:
    domain = (sender or "").rpartition("@")[2]
    return domain.rsplit(".", 2)[0] if domain.count(".") >= 2 else ""  # subdomains only: accounts.google.com


def hide_secrets(text: str, *, subject: str = "", sender: str = "") -> str:
    """``text`` with one-time codes, magic sign-in links and password reset links replaced by placeholders."""
    if not text:
        return text or ""
    auth = is_auth_email(subject, sender)
    text = _hide_links(text, auth)
    text = _hide_inline_codes(text, auth)
    return _hide_line_codes(text, auth)


def hide_email_secrets(item: dict) -> dict:
    """Mask ``subject``, ``snippet`` and ``body`` of a normalized email item in place and return it."""
    subject = str(item.get("subject") or "")
    sender = str(item.get("sender_email") or "")
    for key in ("subject", "snippet", "body"):
        value = item.get(key)
        if isinstance(value, str) and value:
            item[key] = hide_secrets(value, subject=subject, sender=sender)
    return item


def enabled(owner: Any) -> bool:
    """The ``integrations.hide_one_time_codes`` switch of an integration manager or app (on unless turned off)."""
    app = getattr(owner, "app", owner)
    cfg = getattr(getattr(app, "config", None), "integrations", None)
    return bool(getattr(cfg, "hide_one_time_codes", True))
