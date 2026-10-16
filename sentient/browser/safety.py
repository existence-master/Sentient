"""Pure safety heuristics for the browser tools (no Playwright here, easy to test).

- :func:`click_risk` decides whether clicking an element is an ordinary ``write`` or an
  outward, hard-to-undo ``send`` (buy, pay, send, post, delete, confirm ...).
- :func:`sensitive_field` recognises password, card, one-time-code and similar fields the
  assistant must never type into; the user signs in or pays themselves.
- :func:`url_problem` applies the allow/block domain lists.
- :func:`devtools_endpoint` accepts a DevTools address to attach to only when it is on this computer (loopback).

Element descriptors are the dicts produced by the snapshot script (``snapshot.py``):
``{role, name, tag, type, value, href, autocomplete, id, name_attr, label, placeholder,
aria_label, title, in_form, form_payment, form_submit_text, is_submit, near, near_cut}``. ``near`` is the text around the
element (its card or form); ``near_cut`` says it stopped early at another offer. Missing keys are fine.
"""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlsplit

from sentient.tools.base import Risk

# ----------------------------------------------------------------------------- risky clicks
_RISKY_WORDS = [
    r"buy", r"buy now", r"purchase", r"pay", r"pay now", r"place (?:your )?order", r"order now",
    r"complete (?:the )?(?:order|purchase|payment)", r"confirm (?:and )?pay", r"check ?out", r"proceed to (?:pay|payment|checkout)",
    r"send", r"send now", r"post", r"publish", r"tweet", r"share now", r"submit (?:payment|order)",
    r"delete", r"remove", r"erase", r"discard", r"trash", r"confirm", r"subscribe", r"unsubscribe",
    r"transfer", r"donate", r"book now", r"reserve", r"make payment", r"add money", r"withdraw",
    r"cancel (?:my )?(?:order|subscription|account|booking)", r"close (?:my )?account", r"deactivate",
]
RISKY_CLICK_RE = re.compile(r"\b(?:" + "|".join(_RISKY_WORDS) + r")\b", re.IGNORECASE)

# approval wording: first matching rule names the kind of action for the approval card
_LABEL_RULES = [
    (re.compile(r"\b(?:buy|purchase|pay|payment|order|check ?out|donate|book now|reserve|transfer|"
                r"add money|withdraw)\b", re.IGNORECASE), "Purchase"),
    (re.compile(r"\b(?:delete|remove|erase|discard|trash|deactivate|close (?:my )?account)\b", re.IGNORECASE),
     "Deletes"),
    (re.compile(r"\b(?:post|publish|tweet|share)\b", re.IGNORECASE), "Posts publicly"),
    (re.compile(r"\b(?:send|submit|confirm|reply)\b", re.IGNORECASE), "Sends"),
]
_DEFAULT_LABELS = {"click": "Clicks", "type": "Types into the page", "press": "Presses a key"}
SIGNUP_LABEL = "Signs up"

# subscribe, follow, notify me: "Signs up" only with a free signal and no money signal on or around it (#378)
_SIGNUP_RE = re.compile(
    r"\b(?:subscribe|sign ?up|follow|notify me|get (?:the )?updates|"
    r"join (?:the |our )?(?:newsletter|mailing list|wait ?list))\b",
    re.IGNORECASE,
)
# anything that says it may cost money: a price or currency, a billing period, a plan, a trial or an upgrade
_PAID_RE = re.compile(
    r"[$¢£¥€₹₩₽₺₫₱฿₦₪₴₨]"
    r"|\b(?:usd|eur|gbp|inr|jpy|cad|aud|rs|rupees?|dollars?|euros?|pounds?)\b"
    r"|/\s*(?:mo|mon|month|yr|year|wk|week|day)s?\b"
    r"|\bper\s+(?:month|year|week|day|user|seat)\b|\b(?:a|each|every)\s+(?:month|year|week)\b"
    # an amount next to a billing word ("499 monthly", "billed annually at 4,999"); "monthly newsletter" stays free
    r"|\d[\d,.]*\D{0,20}?\b(?:monthly|yearly|annually|annual|quarterly|per\s+annum)\b"
    r"|\b(?:monthly|yearly|annually|annual|quarterly|per\s+annum)\b\D{0,20}?\d"
    r"|\b(?:plans?|trial|upgrade|premium|pricing|prices?|billed|billing|"
    r"paid|pay|payment|buy|purchase|checkout|membership|pro)\b",
    re.IGNORECASE,
)
# what a free sign-up says about itself; "Signs up" needs one of these, so a bare "Subscribe" stays a purchase
_FREE_RE = re.compile(
    r"\b(?:free|news ?letters?|updates?|wait ?list|notify me|notifications?|follow(?:ing)?|digest|alerts?|"
    r"mailing list|no spam|inbox|emails?|blog|podcast|channel)\b",
    re.IGNORECASE,
)


def approval_target(el: dict | None) -> str:
    """The element's readable name for an approval card, e.g. 'Place order'."""
    if not el:
        return ""
    for key in ("name", "aria_label", "value", "title", "label", "placeholder"):
        text = re.sub(r"\s+", " ", str(el.get(key) or "")).strip()
        if text:
            return text[:80]
    return str(el.get("role") or el.get("tag") or "")[:80]


def _partial_offer(text: str, near: str, near_cut: bool) -> bool:
    """True when the context stopped early and holds little besides the element itself: no whole card was read,
    so a price may sit just out of reach."""
    if not near_cut:
        return False
    rest = (near or "").replace(text, " ") if text else (near or "")
    return len(re.sub(r"\W+", "", rest)) < 20


def paid_signup(text: str, near: str = "", near_cut: bool = False) -> bool:
    """True when ``text`` is a subscribe, sign up or follow that the approval card calls a 'Purchase'."""
    text = text or ""
    return bool(_SIGNUP_RE.search(text)) and not _LABEL_RULES[0][0].search(text) and (
        risk_label_for(text, "click", near, near_cut) == _LABEL_RULES[0][1])


def risk_label_for(text: str, kind: str, near: str = "", near_cut: bool = False) -> str:
    """'Purchase', 'Signs up', 'Deletes', 'Posts publicly', 'Sends', else a neutral label for the action kind.

    ``near`` is the text around the element (its card or form). A subscribe, follow or "notify me" reads
    'Signs up' only when the element or ``near`` says what free thing it is (a newsletter, updates, a waitlist,
    alerts, "free" ...) and neither shows a price, currency, billing period, plan, trial or upgrade. Anything else
    may cost money, so it is 'Purchase', which always asks: a bare "Subscribe" is a purchase. A "Sign up" with
    neither a free nor a money signal keeps the plain label it had before (#378). ``near_cut`` says the context stopped
    early (at another offer's button); then only a whole card of text can show it is free. A 'Purchase' label on a
    sign-up comes with ``send`` risk (:func:`paid_signup`, :func:`click_risk`), so its hard stop can't be skipped."""
    text = text or ""
    purchase_rule, *other_rules = _LABEL_RULES
    if purchase_rule[0].search(text):
        return purchase_rule[1]
    if _SIGNUP_RE.search(text):
        context = f"{text} {near or ''}"
        paid = _PAID_RE.search(context)
        if _FREE_RE.search(context) and not paid and not _partial_offer(text, near, near_cut):
            return SIGNUP_LABEL
        if paid or RISKY_CLICK_RE.search(text):
            return purchase_rule[1]
    for rx, label in other_rules:
        if rx.search(text):
            return label
    return _DEFAULT_LABELS.get(kind, "Changes something")


# typing + Enter into these sends a message somewhere
_MESSAGE_FIELD_RE = re.compile(
    r"\b(?:message|comment|reply|tweet|post|what'?s happening|write something|chat|compose)\b", re.IGNORECASE
)


def _norm(text: object) -> str:
    """Split camelCase and snake/kebab names into words so ``cardNumber`` reads ``card number``."""
    s = str(text or "")
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    s = re.sub(r"[_\-.\[\]:/]+", " ", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def element_label(el: dict) -> str:
    """Everything a person could read on or about the element, joined."""
    parts = [el.get(k) for k in ("name", "aria_label", "title", "label")]
    if str(el.get("type") or "").lower() in {"submit", "button", "image", "reset"}:
        parts.append(el.get("value"))
    return " ".join(_norm(p) for p in parts if p)


def click_risk(el: dict | None) -> Risk:
    """``send`` when clicking looks like it buys, pays, sends, posts, deletes or confirms; else ``write``."""
    if not el:
        return Risk.write
    if RISKY_CLICK_RE.search(element_label(el)):
        return Risk.send
    if el.get("is_submit") and el.get("form_payment"):
        return Risk.send
    # a sign-up the approval card calls a purchase is at least send, so "Purchases always ask" holds (#378)
    if paid_signup(approval_target(el), str(el.get("near") or ""), bool(el.get("near_cut"))):
        return Risk.send
    return Risk.write


def type_risk(el: dict | None, submit: bool) -> Risk:
    """Typing is ``write``; pressing Enter afterwards is ``send`` when that submits a payment form,
    a form whose button sends/posts/buys, or a message box."""
    if not el or not submit:
        return Risk.write
    if el.get("form_payment"):
        return Risk.send
    if el.get("form_submit_text") and RISKY_CLICK_RE.search(_norm(el.get("form_submit_text"))):
        return Risk.send
    label = " ".join(_norm(el.get(k)) for k in ("name", "aria_label", "label", "placeholder", "name_attr", "id"))
    if _MESSAGE_FIELD_RE.search(label):
        return Risk.send
    context = " ".join([approval_target(el), str(el.get("form_submit_text") or "")])
    if paid_signup(context, str(el.get("near") or ""), bool(el.get("near_cut"))):
        return Risk.send
    return Risk.write


def press_risk(key: str, focused: dict | None) -> Risk:
    """Enter on a focused field behaves like ``browser_type(..., submit=True)``."""
    if str(key or "").strip().lower() in {"enter", "return", "numpadenter"}:
        return type_risk(focused, True) if focused else Risk.write
    return Risk.write


# ----------------------------------------------------------------------------- sensitive fields
_SENSITIVE_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("password", re.compile(r"\b(?:password|passwd|pass word|pwd|passcode|pass phrase|passphrase)\b")),
    ("PIN", re.compile(r"\b(?:pin(?!\s*code)|mpin|upi pin|atm pin)\b")),
    ("card", re.compile(
        r"\b(?:card ?(?:number|no|num)|cc ?(?:num|number|no)|credit ?card|debit ?card|cvv2?|cvc2?|csc|"
        r"security code|card verification|card expir\w*|name on card|cardholder)\b"
    )),
    ("one-time code", re.compile(
        r"\b(?:otp|one ?time|verification code|verify code|2fa|two ?factor|mfa|auth(?:entication)? code|"
        r"sms code|security key|totp)\b"
    )),
    ("bank or ID number", re.compile(
        r"\b(?:ssn|social security|iban|routing number|account number|acct number|aadhaar|aadhar|pan number|"
        r"passport number|sort code|swift|ifsc)\b"
    )),
    ("secret key", re.compile(r"\b(?:api ?key|secret ?key|access ?token|private ?key|seed phrase|recovery phrase)\b")),
]


def sensitive_field(el: dict | None) -> str | None:
    """Return what kind of secret the field asks for ("password", "card" ...) or None."""
    if not el:
        return None
    ftype = str(el.get("type") or "").lower()
    if ftype == "password":
        return "password"
    tokens = str(el.get("autocomplete") or "").lower().split()
    for tok in tokens:
        if tok in {"current-password", "new-password"}:
            return "password"
        if tok == "one-time-code":
            return "one-time code"
        if tok.startswith("cc-") and tok not in {"cc-name"}:
            return "card"
    text = " ".join(
        _norm(el.get(k)) for k in ("name", "aria_label", "label", "placeholder", "name_attr", "id", "title")
    )
    for kind, pattern in _SENSITIVE_PATTERNS:
        if pattern.search(text):
            return kind
    return None


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def looks_like_card_number(text: str) -> bool:
    """A 13-19 digit number (spaces/dashes allowed) that passes the Luhn check."""
    for m in re.finditer(r"(?:\d[ -]?){13,19}", text or ""):
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            return True
    return False


def needs_user(kind: str) -> dict:
    """The tool result for a field only the user may fill: the refusal, plus ``needs_user`` so a task run stops and
    tells the user why it is stuck (tasks/stuck.py)."""
    thing = "card details" if kind.startswith("card") else kind
    return {"error": sensitive_refusal(kind), "needs_user": f"the page asks for your {thing}, which only you should enter"}


def sensitive_refusal(kind: str) -> str:
    return (
        f"This looks like a {kind} field. For the user's safety you never type {kind}s or other secrets. "
        "Ask the user to fill it in themselves: they can click 'Open browser' in Sentient to see this page "
        "in a visible window, sign in or pay there, and then tell you to continue."
    )


# ----------------------------------------------------------------------------- domains
BLOCKED_SCHEMES = {"file", "javascript", "chrome", "edge", "about", "data", "view-source", "chrome-extension", "blob"}


def host_matches(host: str, pattern: str) -> bool:
    host = (host or "").lower().strip(".")
    pat = (pattern or "").lower().strip()
    if "://" in pat:
        pat = urlsplit(pat).hostname or ""
    pat = pat.removeprefix("*.").strip(".").split("/")[0]
    return bool(pat) and (host == pat or host.endswith("." + pat))


def normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return url
    if url == "about:blank":
        return url
    if "://" not in url and not url.lower().startswith(("about:", "javascript:", "data:", "file:", "blob:")):
        url = "https://" + url
    return url


def url_problem(url: str, allow: list[str], block: list[str]) -> str | None:
    """A plain-language reason the URL must not be opened, or None when it is fine."""
    if url == "about:blank":
        return None
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    if scheme in BLOCKED_SCHEMES or scheme not in {"http", "https"}:
        return "Only normal web addresses (http or https) can be opened in the browser."
    host = parts.hostname or ""
    if not host:
        return "That web address is missing the site name."
    if any(host_matches(host, b) for b in block or []):
        return f"{host} is on the blocked sites list in Settings, so the browser won't open it."
    if allow and not any(host_matches(host, a) for a in allow):
        return f"{host} isn't on the allowed sites list in Settings, so the browser won't open it."
    return None


# ----------------------------------------------------------------------------- attaching to a running browser
NOT_LOCAL = (
    "Sentient only attaches to a browser on this computer. Use an address like http://127.0.0.1:9333 "
    "(the port you started the browser with)."
)


def is_loopback_host(host: str) -> bool:
    """True for this computer only: localhost, 127.0.0.0/8 and ::1."""
    h = (host or "").strip().strip("[]").lower()
    if h == "localhost":
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def devtools_endpoint(raw: str) -> tuple[str, str | None]:
    """Normalize a DevTools address (``9333``, ``127.0.0.1:9333``, ``http://localhost:9333``) and say why it can't
    be used, or ``(address, None)``. Only loopback is allowed: a DevTools port gives full control of the browser."""
    text = str(raw or "").strip()
    if not text:
        return "", "Add the DevTools address, for example http://127.0.0.1:9333."
    if text.isdigit():
        text = f"http://127.0.0.1:{text}"
    elif "://" not in text:
        text = "http://" + text
    try:
        parts = urlsplit(text)
        port = parts.port
    except ValueError:
        return "", "That DevTools address isn't valid. Use one like http://127.0.0.1:9333."
    scheme = parts.scheme.lower()
    if scheme not in {"http", "https", "ws", "wss"}:
        return "", "That DevTools address isn't valid. Use one like http://127.0.0.1:9333."
    if not is_loopback_host(parts.hostname or ""):
        return "", NOT_LOCAL
    if not port:
        return "", "Add the port to the DevTools address, for example http://127.0.0.1:9333."
    host = parts.hostname or ""
    host = f"[{host}]" if ":" in host else host
    if scheme in {"ws", "wss"}:
        return f"{scheme}://{host}:{port}{parts.path}", None
    return f"{scheme}://{host}:{port}", None
