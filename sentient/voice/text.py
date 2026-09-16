"""Turning a streamed markdown reply into speakable sentences.

``SentenceSplitter`` is fed ``text_delta`` chunks as they arrive and returns
complete sentences as early as it can prove they are complete, so the first
sentence can be synthesized while the model is still writing the second.
Fenced code blocks are skipped entirely; ``clean_for_speech`` removes the
remaining markdown, URLs and symbols a TTS engine would read out literally.
"""

from __future__ import annotations

import re
import unicodedata

# lowercased tokens (without the trailing dot) that end in "." but do not end a sentence
ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e", "eg", "ie",
    "a.m", "p.m", "u.s", "u.k", "u.s.a", "approx", "inc", "ltd", "co", "corp", "dept", "est",
    "fig", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    "mt", "ft", "lb", "lbs", "oz", "vol", "ch", "pp", "cf", "al", "gen", "gov", "sen", "rep",
    "capt", "col", "lt", "sgt", "rev", "hon", "ave", "blvd", "rd", "mins", "hrs",
}  # fmt: skip
# only abbreviations when a number follows ("No. 5", "pg. 12")
NUMERIC_ABBREVIATIONS = {"no", "nos", "pg", "art", "sec", "op"}

_BOUNDARY = re.compile(r"[.!?…]+[\"'”’)\]*_`]*(?=\s|$)|\n")  # noqa: RUF001  (closing quotes/markdown)
_TOKEN_BEFORE = re.compile(r"(\S+)$")
MAX_SENTENCE_CHARS = 240


_CLAUSE = re.compile(r"[,;:](?=\s)|\s[-–—]{1,2}(?=\s)")  # noqa: RUF001


class SentenceSplitter:
    """``first_clause_chars`` > 0 lets the *first* chunk of a reply end at a clause
    boundary (comma, semicolon, colon, dash) once it is at least that long, so speech
    can start before the first full sentence has been written."""

    def __init__(self, max_chars: int = MAX_SENTENCE_CHARS, first_clause_chars: int = 0):
        self.buf = ""
        self.in_code = False
        self.max_chars = max_chars
        self.first_clause_chars = max(0, int(first_clause_chars))
        self.emitted = 0

    def feed(self, delta: str) -> list[str]:
        self.buf += delta
        return self._drain(final=False)

    def flush(self) -> list[str]:
        out = self._drain(final=True)
        self.buf = ""
        self.in_code = False
        self.emitted = 0
        return out

    def _first_clause_cut(self, region: str) -> int | None:
        if not self.first_clause_chars or self.emitted or len(region) < self.first_clause_chars:
            return None
        for m in _CLAUSE.finditer(region):
            if m.end() >= self.first_clause_chars and m.end() < len(region):
                return m.end()
        return None

    # ------------------------------------------------------------------ internals
    def _drain(self, *, final: bool) -> list[str]:
        out: list[str] = []
        while self.buf:
            if self.in_code:
                end = self.buf.find("```")
                if end < 0:
                    self.buf = "" if final else self.buf[-2:]  # keep a partial closing fence
                    break
                self.buf = self.buf[end + 3 :]
                self.in_code = False
                continue
            fence = self.buf.find("```")
            region = self.buf if fence < 0 else self.buf[:fence]
            if fence < 0 and not final:
                # a trailing "`" or "``" may be the start of a fence: never cut into it
                tail = len(region) - len(region.rstrip("`"))
                if tail:
                    region = region[:-tail]
            cut = self._boundary(region, end_is_boundary=fence >= 0 or final)
            if cut is not None:
                self._emit(region[:cut], out)
                self.buf = self.buf[cut:]
                continue
            if fence >= 0:
                self._emit(region, out)
                self.buf = self.buf[fence + 3 :]
                self.in_code = True
                continue
            if final:
                self._emit(self.buf, out)
                self.buf = ""
            elif (clause := self._first_clause_cut(region)) is not None:
                self._emit(region[:clause], out)
                self.buf = self.buf[clause:]
                continue
            elif len(region) > self.max_chars:
                cut = self._soft_cut(region)
                self._emit(region[:cut], out)
                self.buf = self.buf[cut:]
                continue
            break
        return out

    def _emit(self, chunk: str, out: list[str]) -> None:
        spoken = clean_for_speech(chunk)
        if is_speakable(spoken):
            out.append(spoken)
            self.emitted += 1

    def _boundary(self, text: str, *, end_is_boundary: bool) -> int | None:
        for m in _BOUNDARY.finditer(text):
            end = m.end()
            if m.group() == "\n":
                if text[: m.start()].strip():
                    return end
                continue
            if end >= len(text) and not end_is_boundary:
                return None  # cannot see what follows yet ("3." may become "3.5")
            if m.group() == "." and self._is_abbreviation(text, m.start(), end):
                continue
            return end
        if end_is_boundary and text.strip():
            return len(text)
        return None

    @staticmethod
    def _is_abbreviation(text: str, dot: int, end: int) -> bool:
        tm = _TOKEN_BEFORE.search(text[:dot])
        if not tm:
            return False
        token = tm.group(1).lstrip("(\"'“*_")
        low = token.lower()
        after = text[end:].lstrip()
        if low in ABBREVIATIONS:
            return True
        if low in NUMERIC_ABBREVIATIONS and after[:1].isdigit():
            return True
        if re.fullmatch(r"[A-Za-z]", token):  # initials: "J. K. Rowling"
            return True
        if re.fullmatch(r"(?:[A-Za-z]\.)+[A-Za-z]", token):  # "U.S", "e.g"
            return True
        # numbered list item at the start of a line: "1. Buy milk"
        line_start = text.rfind("\n", 0, tm.start()) + 1
        return token.isdigit() and not text[line_start : tm.start()].strip()

    def _soft_cut(self, text: str) -> int:
        window = text[: self.max_chars]
        for sep in ("; ", ": ", ", ", " - ", " "):
            i = window.rfind(sep)
            if i > self.max_chars // 3:
                return i + len(sep)
        return self.max_chars


# ----------------------------------------------------------------------------- markdown -> speech
_FENCE = re.compile(r"```.*?(?:```|$)", re.S)
_THINK = re.compile(r"<think>.*?(?:</think>|$)", re.S | re.I)
_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_REF_LINK = re.compile(r"\[\^?\d+\]")
_URL = re.compile(r"(?:https?://|www\.)[^\s)>\]]*[^\s)>\].,;:!?'\"]", re.I)
_EMAIL_ANGLE = re.compile(r"<([^<>@\s]+@[^<>\s]+)>")
_HTML = re.compile(r"</?[a-zA-Z][^>]*>")
_INLINE_CODE = re.compile(r"`([^`\n]*)`")
_HEADER = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]*", re.M)
_QUOTE = re.compile(r"^[ \t]*>+[ \t]?", re.M)
_HR = re.compile(r"^[ \t]*([-*_])(?:[ \t]*\1){2,}[ \t]*$", re.M)
_BULLET = re.compile(r"^[ \t]*(?:[-*+•]|\d+[.)])[ \t]+", re.M)
_TABLE_SEP = re.compile(r"^[ \t]*\|?[ \t]*:?-{2,}:?[ \t]*(?:\|[ \t]*:?-{2,}:?[ \t]*)*\|?[ \t]*$", re.M)
_BOLD = re.compile(r"(\*\*|__)(.+?)\1", re.S)
_ITALIC_STAR = re.compile(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])")
_ITALIC_UNDER = re.compile(r"(?<![\w_])_(?!\s)(.+?)(?<!\s)_(?![\w_])")
_STRIKE = re.compile(r"~~(.+?)~~")
_CHECKBOX = re.compile(r"\[[ xX]\][ \t]*")
_LEFTOVER = re.compile(r"[*#`~|<>^]+")
_SPACES = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")
_REPEATED_COMMAS = re.compile(r"(?:,\s*){2,}")


def clean_for_speech(text: str) -> str:
    if not text:
        return ""
    t = _THINK.sub(" ", text)
    t = _FENCE.sub(" ", t)
    t = _IMAGE.sub(r"\1", t)
    t = _LINK.sub(r"\1", t)
    t = _REF_LINK.sub("", t)
    t = _EMAIL_ANGLE.sub(r"\1", t)
    t = _URL.sub(" ", t)
    t = _HTML.sub(" ", t)
    t = _INLINE_CODE.sub(r"\1", t)
    t = _TABLE_SEP.sub(" ", t)
    t = _HR.sub(" ", t)
    t = _HEADER.sub("", t)
    t = _QUOTE.sub("", t)
    t = _BULLET.sub("", t)
    t = _CHECKBOX.sub("", t)
    t = _BOLD.sub(r"\2", t)
    t = _STRIKE.sub(r"\1", t)
    t = _ITALIC_STAR.sub(r"\1", t)
    t = _ITALIC_UNDER.sub(r"\1", t)
    t = t.replace("|", ", ").replace("&", " and ")
    t = t.replace("—", ", ").replace(" -- ", ", ")
    t = _LEFTOVER.sub(" ", t)
    t = "".join(ch for ch in t if unicodedata.category(ch) not in {"So", "Cs", "Co"})
    t = _SPACES.sub(" ", t).strip()
    t = _SPACE_BEFORE_PUNCT.sub(r"\1", t)
    t = _REPEATED_COMMAS.sub(", ", t)
    return t.strip(" ,")


def is_speakable(text: str) -> bool:
    return any(ch.isalnum() for ch in text)


def split_sentences(text: str) -> list[str]:
    s = SentenceSplitter()
    return [*s.feed(text), *s.flush()]
