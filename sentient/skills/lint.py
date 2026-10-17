"""Skill linter (#164): advisory checks on a SKILL.md, in plain language for the Skills screen.

Each finding is ``{"rule", "level", "message"}`` with ``level`` ``error`` (the skill can't load, or it asks for
something unsafe) or ``warning`` (it works but could be better). Rules:

- ``frontmatter``: the ``---`` header is missing or isn't readable YAML
- ``required``: name, description or steps are missing; the name doesn't match the folder
- ``size``: the description or the body is too long to read cheaply
- ``reference``: a linked file is missing, a needed tool isn't installed, or the skill sprawls over many files
- ``danger``: instructions to skip approvals, ignore other instructions, wipe files, run downloaded code, send
  secrets away, send your data to a fixed address, read secrets or the keychain, or hide what it does from you
- ``hidden``: text you can't see when reading the skill (zero-width or other invisible characters, text styled to
  be invisible such as white on white, HTML comments). The ``danger`` rules also read that hidden text.
- ``outbound``: it always sends something to one fixed email address or web address (a warning: check who it is)
- ``log_shaped``: it reads like a diary of past runs instead of a lesson ("lessons, not logs")
- ``sections``: there is no Procedure section
- ``validate``: a check in the ``validate`` header can't be used (``sentient/skills/checks.py``, #386); the ``danger``
  rules read a check's code too

The linter never blocks a user's own save. ``skill_save`` stages a skill with a ``danger`` finding for review.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

import yaml

from sentient.skills.checks import parse as parse_checks

_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,63}$")

MAX_DESCRIPTION = 300
MAX_BODY = 12_000
MAX_REFERENCES = 8
LOG_LINES = 3

_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_PATH_REF = re.compile(r"(?<![\w/.-])((?:references|scripts|assets|templates|examples)/[\w./-]+\.\w+)")
_DATED_LINE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:\*\*)?(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|(?:mon|tues|wednes|thurs|fri|satur|sun)day\b)",
    re.IGNORECASE,
)
_DIARY = re.compile(
    r"\b(?:last time|this time|yesterday|today|earlier today|in (?:the|my) last run|on the \d+(?:st|nd|rd|th)?)\b"
    r"[^.\n]{0,60}\b(?:I|we|it)\b",
    re.IGNORECASE,
)

# (pattern, message, negatable): a negatable rule is skipped when the sentence says not to do it ("never send passwords")
_DANGER: list[tuple[re.Pattern[str], str, bool]] = [
    (
        re.compile(
            r"\b(?:ignore|disregard|forget)\b[^.\n]{0,30}\b(?:previous|prior|earlier|other|all|system)\b[^.\n]{0,20}"
            r"\b(?:instructions|rules|prompt)",
            re.IGNORECASE,
        ),
        "It tells Sentient to ignore its other instructions.",
        True,
    ),
    (
        re.compile(
            r"\b(?:without|never|don'?t|do not)\s+(?:ask(?:ing)?|check(?:ing)?|confirm(?:ing)?|wait(?:ing)? for)\b[^.\n]{0,25}"
            r"\b(?:user|me|permission|approval|confirmation|anyone)\b",
            re.IGNORECASE,
        ),
        "It tells Sentient to act without asking you first.",
        False,
    ),
    (
        re.compile(
            r"\b(?:skip|bypass|turn off|disable)\b[^.\n]{0,15}\b(?:approvals?|confirmations?|permission prompts?)\b"
            r"|\bauto-?approve\b",
            re.IGNORECASE,
        ),
        "It tells Sentient to act without asking you first.",
        True,
    ),
    (
        re.compile(
            r"\brm\s+-[a-z]*r[a-z]*\s+(?:/|~|\*|\$HOME|--no-preserve-root)|\bdel\s+/[sq]\b|\bformat\s+[a-z]:"
            r"|\bRemove-Item\b[^\n]{0,60}-Recurse[^\n]{0,60}(?:\b[A-Z]:\\?(?=[\s'\"]|$)|\$HOME|\$env:USERPROFILE|~(?=[\s'\"]|$))",
            re.IGNORECASE | re.MULTILINE,
        ),
        "It runs a command that can wipe whole folders or drives.",
        True,
    ),
    (
        re.compile(
            r"\b(?:curl|wget|iwr|Invoke-WebRequest)\b[^\n|]{0,200}\|\s*(?:sudo\s+)?(?:ba|z)?sh\b|\biex\s*\(|\bInvoke-Expression\b",
            re.IGNORECASE,
        ),
        "It downloads a script and runs it straight away.",
        True,
    ),
    (
        re.compile(
            r"\b(?:send|post|upload|email|share|paste)\b[^.\n]{0,40}"
            r"\b(?:passwords?|api[ _-]?keys?|tokens?|secrets?|credentials|private keys?)\b",
            re.IGNORECASE,
        ),
        "It sends passwords, keys or other secrets somewhere.",
        True,
    ),
    (
        re.compile(
            r"\b(?:disable|turn off|uninstall)\b[^.\n]{0,20}\b(?:antivirus|firewall|defender|security software)\b",
            re.IGNORECASE,
        ),
        "It turns off your computer's security protection.",
        True,
    ),    (
        re.compile(
            r"\b(?:download|fetch|pull|grab)\b[^.\n]{0,80}\b(?:and|then)\s+(?:run|execute|launch|install|source)\b"
            r"|\bpip3?\s+install\s+(?:git\+)?https?://|\bbase64\s+(?:-d|--decode)\b[^\n|]{0,80}\|\s*(?:ba|z)?sh\b"
            r"|\b(?:powershell|pwsh)(?:\.exe)?\b[^\n]{0,60}\s-(?:e|enc|encodedcommand)\s|\b(?:ba|z)?sh\s+<\(\s*(?:curl|wget)\b"
            r"|\b(?:ba|z)?sh\s+-c\s+[\"']?\$\(\s*(?:curl|wget)\b",
            re.IGNORECASE,
        ),
        "It downloads a script and runs it straight away.",
        True,
    ),
    (
        re.compile(
            r"\b(?:set|switch|change|put)\b[^.\n]{0,20}\bapprovals?\b[^.\n]{0,20}\boff\b"
            r"|\balways\s+(?:click|choose|pick|answer|say)\s+[\"'`]?(?:allow|yes|approve)",
            re.IGNORECASE,
        ),
        "It tells Sentient to act without asking you first.",
        True,
    ),
    (
        re.compile(
            r"\b(?:read|cat|type|open|copy|dump|export|print|grab|collect|harvest|extract|look up|access|upload|send)\b"
            r"[^.\n]{0,60}(?:\bkeychain\b|\bkeyring\b|credential manager|\.ssh\b|\bid_rsa\b|\bid_ed25519\b|"
            r"(?<![\w])\.env\b|\.aws[/\\]credentials|\.netrc\b|\bauth\.json\b|\bcookies\b|saved passwords|"
            r"password manager|login data|wallet\.dat|seed phrase|recovery phrase|private keys?)"
            r"|\bsecurity\s+(?:find-(?:generic|internet)-password|dump-keychain)\b|\bcmdkey\s+/list\b"
            r"|\bsecret-tool\s+lookup\b|\bkeyring\s+get\b|\bGet-StoredCredential\b|\bvaultcmd\b",
            re.IGNORECASE,
        ),
        "It reads passwords, keys or other secrets stored on your computer.",
        True,
    ),
    (
        re.compile(
            r"\b(?:do not|don'?t|never)\s+(?:tell|inform|mention|show|let|alert|notify|reveal|disclose)\b[^.\n]{0,25}"
            r"\b(?:user|them|him|her|anyone|this step|these steps|this instruction)\b"
            r"|\bwithout\s+(?:telling|informing|letting|notifying|alerting)\b[^.\n]{0,15}\b(?:user|them|anyone)\b"
            r"|\b(?:secretly|quietly|silently|covertly|discreetly)\s+(?:also\s+)?(?:send|e-?mail|forward|upload|copy|post|"
            r"share|delete|run|bcc|cc|sync|add)\b",
            re.IGNORECASE,
        ),
        "It tells Sentient to hide what it does from you.",
        False,
    ),
    (
        re.compile(
            r"\byou are now (?:a|an|in|no longer|free|unrestricted|dan)\b|\bnew (?:system )?instructions\s*:"
            r"|</?\s*(?:important|system|instructions?|admin)\s*>|\[(?:system|admin|developer)\]"
            r"|\b(?:enter|enable|switch to)\s+(?:developer|god|jailbreak|unrestricted)\s+mode\b"
            r"|\boverride\s+(?:your|the|all)\s+(?:rules|instructions|safety|guidelines)\b"
            r"|\b(?:reveal|print|show|repeat)\s+(?:your|the)\s+system\s+prompt\b",
            re.IGNORECASE,
        ),
        "It tries to give Sentient new orders or change its rules.",
        True,
    ),
]
_NEGATION = re.compile(r"\b(?:never|don'?t|do not|must not|mustn'?t|should not|shouldn'?t|avoid|no)\b[^.\n]*$", re.IGNORECASE)

# sending to one fixed address (#146): a send verb and an email or web address in the same sentence. With the user's
# data named too it is ``danger`` ("email the contacts to x@evil.example"), otherwise an ``outbound`` warning.
_SEND_VERB = re.compile(
    r"\b(?:send|e-?mail|mail|forward|post|upload|share|sync|copy|bcc|cc|transmit|exfiltrate|report|deliver|push|beacon)"
    r"(?:s|ed|ing)?\b",
    re.IGNORECASE,
)
_ADDRESS = re.compile(r"[\w.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+|\bhttps?://[^\s)\"'<>]+", re.IGNORECASE)
_USER_DATA = re.compile(
    r"\b(?:contact list|address book|contacts?|inbox|e-?mails|messages|files|documents|notes|memor(?:y|ies)|history|"
    r"conversations?|chats?|transcripts?|passwords?|api[ _-]?keys?|tokens?|secrets?|credentials|calendar|clipboard|"
    r"screenshots?|photos|location|everything|all (?:of )?(?:your|the user'?s|my) \w+|personal (?:data|details|info)|"
    r"phone numbers|browsing data|cookies)\b",
    re.IGNORECASE,
)

# text a person can't see when reading the skill (#146)
_INVISIBLE = re.compile(
    "[­᠎​‌‎‏‪-‮⁠-⁤⁦-⁩﻿\U000e0000-\U000e007f]"
    "|(?<=[\x00-\x7f])‍|‍(?=[\x00-\x7f])"  # a joiner between emoji is fine; next to plain letters it isn't
)
_HIDDEN_STYLE = re.compile(
    r"<[a-z][^>]*\bstyle\s*=\s*[\"'][^\"']*(?:display\s*:\s*none|visibility\s*:\s*hidden|font-size\s*:\s*0(?:px|pt|em)?\b"
    r"|opacity\s*:\s*0(?:\.0+)?\s*(?:;|[\"'])|color\s*:\s*(?:#fff(?:fff)?\b|white\b|transparent\b|"
    r"rgba?\(\s*255\s*,\s*255\s*,\s*255\b))|<[a-z][^>]*\shidden(?:\s|>|=)",
    re.IGNORECASE,
)
_HTML_COMMENT = re.compile(r"<!--(.*?)-->", re.DOTALL)
_PADDING = re.compile(r"(?:[ \t]*\n){12,}[ \t]*\S| {80,}\S")  # instructions pushed far below or right of what shows


def reveal(text: str) -> str:
    """The text with hidden characters removed and look-alike letters folded, so the danger rules read what a
    model would read (``e<zero-width space>mail`` -> ``email``; fullwidth letters -> ASCII)."""
    return unicodedata.normalize("NFKC", _INVISIBLE.sub("", text or ""))


def hidden_findings(text: str) -> list[dict]:
    """``hidden`` findings: invisible characters, text styled to be invisible, comments and padded-away text."""
    text = text or ""
    out: list[dict] = []
    count = len(_INVISIBLE.findall(text))
    if count:
        out.append(_finding("hidden", "error", f"It has {count} invisible character{'s' if count != 1 else ''} that you "
                            "can't see when reading it. They can hide words from you."))
    if _HIDDEN_STYLE.search(text):
        out.append(_finding("hidden", "error", "It has text styled to be invisible (for example white on white), "
                            "so you wouldn't see it in a preview."))
    comments = [c for c in _HTML_COMMENT.findall(text) if len(re.sub(r"\W", "", c)) >= 12]
    if comments:
        risky = any(_ADDRESS.search(c) or any(_dangerous(p, reveal(c), n) for p, _m, n in _DANGER) for c in comments)
        out.append(_finding("hidden", "error" if risky else "warning",
                            "It has a hidden comment with instructions in it." if risky
                            else "It has a hidden comment (<!-- -->) that doesn't show when the skill is displayed."))
    if _PADDING.search(text):
        out.append(_finding("hidden", "warning", "It has text pushed far out of view by blank lines or spaces."))
    return out


def outbound_findings(text: str) -> list[dict]:
    """``danger`` when a sentence sends the user's data to a fixed address, else an ``outbound`` warning."""
    out: list[dict] = []
    seen: set[str] = set()
    for sentence in re.split(r"(?<=[.!?])\s+|\n", text or ""):
        if not _SEND_VERB.search(sentence):
            continue
        for addr in _ADDRESS.findall(sentence):
            addr = addr.rstrip(".,;:")
            key = addr.lower()
            if key in seen or key.endswith(("@example.com", "@example.org", "@example.net")):
                continue
            seen.add(key)
            data = _USER_DATA.search(sentence)
            if data and not _NEGATION.search(sentence[: data.start()]):
                out.append(_finding("danger", "error", f"It sends your {data.group(0).lower()} to a fixed address ({addr})."))
            else:
                out.append(_finding("outbound", "warning", f"It always sends to {addr}. Make sure you know who that is."))
    return out


def _dangerous(pattern: re.Pattern[str], text: str, negatable: bool) -> bool:
    for match in pattern.finditer(text):
        before = text[max(0, match.start() - 40):match.start()]
        if negatable and _NEGATION.search(before):
            continue
        return True
    return False


def _finding(rule: str, level: str, message: str) -> dict:
    return {"rule": rule, "level": level, "message": message}


def _references(body: str) -> set[str]:
    refs: set[str] = set()
    for target in _LINK.findall(body):
        t = target.split("#", 1)[0].strip()
        if not t or re.match(r"^[a-z][a-z0-9+.-]*:", t, re.IGNORECASE) or t.startswith(("/", "\\")):
            continue  # web links, mailto:, anchors and absolute paths aren't skill files
        refs.add(t.removeprefix("./"))
    refs.update(m.rstrip(".,;:)") for m in _PATH_REF.findall(body))
    return refs


SECURITY_RULES = frozenset({"danger", "hidden", "outbound"})


def security_findings(text: str, *, hidden_in: str | None = None) -> list[dict]:
    """The safety rules alone (``hidden``, ``danger``, ``outbound``) on any text: a SKILL.md's steps, a script or a
    reference file in a skill's folder, an MCP tool's description. ``hidden_in`` is where to look for hidden text
    (the whole file, header included); it defaults to ``text``."""
    out = hidden_findings(text if hidden_in is None else hidden_in)
    readable = reveal(text)
    seen: set[str] = set()
    for pattern, message, negatable in _DANGER:
        if message not in seen and _dangerous(pattern, readable, negatable):
            seen.add(message)
            out.append(_finding("danger", "error", message))
    out.extend(outbound_findings(readable))
    return out


def lint_text(
    text: str,
    *,
    folder: Path | None = None,
    known_tools: set[str] | None = None,
    check_name: bool = True,
) -> list[dict]:
    """Lint a SKILL.md. ``folder`` (the skill's folder) enables the missing-file and name checks;
    ``known_tools`` (installed tool plugin ids) enables the needed-tool check; ``check_name=False`` skips the
    name checks when the name is about to be rewritten (an import)."""
    out: list[dict] = []
    text = (text or "").replace("\r\n", "\n").lstrip("\ufeff")
    m = _FRONTMATTER.match(text)
    if not m:
        out.append(_finding("frontmatter", "error", "The skill has no header. Start the file with a --- block that holds its name and description."))
        meta: dict[str, Any] = {}
        body = text.strip()
    else:
        body = m.group(2).strip()
        try:
            loaded = yaml.safe_load(m.group(1))
        except yaml.YAMLError:
            loaded = None
            out.append(_finding("frontmatter", "error", "The header between the --- lines can't be read. Check for a stray colon or quote."))
        else:
            if loaded is not None and not isinstance(loaded, dict):
                out.append(_finding("frontmatter", "error", "The header should be a list of fields like name: and description:."))
        meta = loaded if isinstance(loaded, dict) else {}

    header_ok = not out  # the field checks only make sense once the header reads
    name = str(meta.get("name") or "").strip()
    folder_name = folder.name if folder is not None else ""
    if header_ok and check_name:
        effective = (name or folder_name).lower()
        if not effective:
            out.append(_finding("required", "error", "The skill has no name."))
        elif not _NAME_RE.match(effective) or effective in {"pending", "archived"}:
            out.append(_finding("required", "error", "The name must be 2 to 64 lowercase letters, digits and dashes."))
        elif name and folder_name and name.lower() != folder_name.lower():
            out.append(_finding("required", "warning", f"The name '{name}' doesn't match its folder '{folder_name}'."))
    description = " ".join(str(meta.get("description") or "").split())
    if header_ok and not description:
        out.append(_finding("required", "error", "The skill has no description, so Sentient can't tell when to use it."))
    elif len(description) > MAX_DESCRIPTION:
        out.append(_finding("size", "warning", f"The description is {len(description)} characters. Keep it under {MAX_DESCRIPTION}: it is added to every chat."))
    if not body:
        out.append(_finding("required", "error", "The skill has no steps."))
    elif len(body) > MAX_BODY:
        out.append(_finding("size", "warning", f"The steps are {len(body):,} characters long. Keep the core steps and move details into a reference file."))
    if body and not re.search(r"^#{1,6}\s*procedure\b", body, re.IGNORECASE | re.MULTILINE):
        out.append(_finding("sections", "warning", "There is no Procedure section with numbered steps."))

    refs = _references(body)
    if folder is not None:
        missing = sorted(r for r in refs if not (folder / r).exists())
        for r in missing[:5]:
            out.append(_finding("reference", "warning", f"It links to {r}, which isn't in the skill's folder."))
        if len(missing) > 5:
            out.append(_finding("reference", "warning", f"{len(missing) - 5} more linked files are missing."))
    if len(refs) > MAX_REFERENCES:
        out.append(_finding("reference", "warning", f"It points to {len(refs)} other files. Skills work best when the main steps fit in one place."))
    requires = meta.get("requires_tools") or []
    if known_tools is not None and isinstance(requires, list):
        for tool in [str(t) for t in requires if str(t) not in known_tools][:5]:
            out.append(_finding("reference", "warning", f"It needs the {tool} tools, which aren't installed or turned on."))

    declared, problems = parse_checks(meta.get("validate")) if header_ok else ([], [])
    for problem in problems:
        out.append(_finding("validate", "warning", problem))
    code = "\n".join(str(c["run"]) for c in declared if "run" in c)
    out.extend(security_findings(body + "\n" + description + ("\n" + code if code else ""), hidden_in=text))

    dated = sum(1 for line in body.splitlines() if _DATED_LINE.match(line))
    diary = len(_DIARY.findall(body))
    if dated >= LOG_LINES or diary >= LOG_LINES:
        out.append(_finding("log_shaped", "warning", "It reads like a log of past runs. Write the lesson instead: what to do next time, not what happened."))
    return out


def lint_file(path: Path, *, known_tools: set[str] | None = None, check_name: bool = True) -> list[dict]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return [_finding("frontmatter", "error", f"The file can't be read: {exc.strerror or exc}.")]
    return lint_text(text, folder=path.parent, known_tools=known_tools, check_name=check_name)


def has_danger(findings: list[dict]) -> bool:
    """A finding that should make a person look before the skill is used: ``danger``, or hidden text that is an error."""
    return any(f.get("rule") == "danger" or (f.get("rule") == "hidden" and f.get("level") == "error") for f in findings)
