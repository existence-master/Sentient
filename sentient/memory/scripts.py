"""Facts written in Devanagari (Hindi, Marathi...): who they are about, and whether two names are the same person
across scripts (#339).

:func:`facts.fact_subjects` reads Latin text: it finds a name by its capital letter. Devanagari has no capitals, so a
fact like "प्रिया पुणे में रहती है" ("Priya lives in Pune") had no subject at all, and an unclear subject counts as the
same as any other. That let a small model file it as an update of "Riya lives in Pune". This module gives such facts
a subject (the first word that isn't a time word or pronoun, as in an SOV sentence) and compares names across scripts
by a rough romanization, so "माया" matches "Maya" and "प्रिया" doesn't match "Riya".

It is deliberately small and conservative: when it can't tell, two facts count as being about different people,
which keeps both facts instead of merging them.
"""

from __future__ import annotations

import re
import unicodedata

_DEVANAGARI = re.compile(f"[{chr(0x900)}-{chr(0x97F)}]")
# letters and vowel signs of Devanagari (not the danda "।" or Devanagari digits), or Latin letters
_WORD = re.compile(f"[{chr(0x900)}-{chr(0x963)}{chr(0x971)}-{chr(0x97F)}]+|[A-Za-z][A-Za-z'{chr(0x2019)}]*")

# time words, particles and helper verbs that start or fill a Hindi sentence but are never its subject
HINDI_SKIP = frozenset({
    "पिछले", "पिछला", "पिछली", "अगले", "अगला", "अगली", "इस", "उस", "हर", "साल", "महीने", "महीना", "हफ्ते", "हफ़्ते",
    "सप्ताह", "दिन", "आज", "कल", "परसों", "अब", "अभी", "सुबह", "शाम", "रात", "कभी", "हमेशा", "फिर", "तब", "जब",
    "पहले", "बाद", "में", "से", "को", "का", "की", "के", "ने", "और", "लेकिन", "पर", "भी", "तो", "ही", "आजकल", "हाल",
    "शायद", "बस", "एक", "है", "हैं", "था", "थी", "थे", "हूँ", "हूं", "हो"
})
# "he", "she", "this": the subject is someone named in another sentence, so it is unclear here
HINDI_PRONOUNS = frozenset({
    "वह", "वो", "यह", "ये", "वे", "उसने", "उन्होंने", "उसका", "उसकी", "उसके", "इसका", "इसकी", "इसके", "उनका",
    "उनकी", "उनके"
})
# "I", "my": the fact is about the user
HINDI_FIRST_PERSON = frozenset({"मैं", "मैंने", "मुझे", "मुझको", "हम", "हमने", "हमें"})
HINDI_MY = frozenset({"मेरा", "मेरी", "मेरे", "हमारा", "हमारी", "हमारे"})
HINDI_POSSESSIVE = frozenset({"का", "की", "के"})
# relation words, in English so "माया की बहन" and "Maya's sister" have the same relation
HINDI_RELATIONS = {
    "बहन": "sister", "दीदी": "sister", "भाई": "brother", "भैया": "brother", "माँ": "mother", "मां": "mother",
    "मम्मी": "mother", "माता": "mother", "पिता": "father", "पापा": "father", "पिताजी": "father", "पत्नी": "wife",
    "पति": "husband", "बेटा": "son", "बेटी": "daughter", "दोस्त": "friend", "मित्र": "friend", "सहेली": "friend",
    "मैनेजर": "manager", "बॉस": "boss", "चाचा": "uncle", "मामा": "uncle", "चाची": "aunt", "मौसी": "aunt",
    "बुआ": "aunt", "दादी": "grandmother", "नानी": "grandmother", "दादा": "grandfather", "नाना": "grandfather",
    "कंपनी": "company", "कुत्ता": "dog", "बिल्ली": "cat",
}

# ------------------------------------------------------------------------------------------------ romanization
_CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n", "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n", "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r", "ल": "l", "ळ": "l", "व": "v",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
}
_NUKTA_FORMS = {"ज": "z", "फ": "f", "ड": "r", "ढ": "rh", "क": "k", "ख": "kh", "ग": "g"}
_VOWELS = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ii", "उ": "u", "ऊ": "uu", "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o",
    "औ": "au", "ऑ": "o", "ॲ": "a",
}
_SIGNS = {
    "ा": "aa", "ि": "i", "ी": "ii", "ु": "u", "ू": "uu", "ृ": "ri", "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
    "ॉ": "o", "ॅ": "e",
}
_VIRAMA, _NUKTA = "्", "़"
_NASAL = {"ं": "n", "ँ": "n", chr(0x903): "h"}  # visarga


def has_devanagari(text: str) -> bool:
    return bool(_DEVANAGARI.search(text or ""))


def leads_with_devanagari(text: str) -> bool:
    """Whether the first letter of ``text`` is Devanagari (the fact is written in Hindi, Marathi...)."""
    for ch in text or "":
        if unicodedata.category(ch)[0] in "LM":
            return bool(_DEVANAGARI.match(ch))
    return False


def romanize(text: str) -> str:
    """A rough Latin spelling of Devanagari text ("प्रिया" -> "priyaa"); other characters are kept."""
    text = unicodedata.normalize("NFD", text or "")  # क़ -> क + nukta
    out: list[str] = []
    pending = False  # a consonant waits for its vowel (the inherent "a" unless a sign or virama follows)
    for i, ch in enumerate(text):
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if ch == _NUKTA:
            continue
        if ch in _CONSONANTS:
            if pending:
                out.append("a")
            out.append(_NUKTA_FORMS.get(ch, _CONSONANTS[ch]) if nxt == _NUKTA else _CONSONANTS[ch])
            pending = True
            continue
        if ch in _SIGNS:
            out.append(_SIGNS[ch])
        elif ch == _VIRAMA:
            pass
        else:
            if pending:
                out.append("a")
            out.append(_VOWELS.get(ch) or _NASAL.get(ch) or ch)
        pending = False
    if pending:
        out.append("a")
    return "".join(out)


def name_key(name: str) -> str:
    """A spelling-tolerant key for comparing one name across scripts: "Maya" and "माया" both give "my", "Priya"
    and "प्रिया" give "pry", "Riya" gives "ry". Aspiration, long vowels, doubled letters and every "a" after the first
    letter are dropped, since romanized Hindi spells them in many ways."""
    s = re.sub(r"[^a-z]", "", romanize(name).lower())
    s = s.replace("ph", "f").replace("ee", "i").replace("oo", "u")
    s = re.sub(r"([^aeiouh])h", r"\1", s)  # kh, sh, th, bh...: the h is optional in romanized Hindi
    s = s.translate(str.maketrans({"w": "v", "z": "j", "q": "k", "c": "k", "x": "k"}))
    s = re.sub(r"(.)\1+", r"\1", s)
    return s[:1] + s[1:].replace("a", "")


def names_match(a: str, b: str) -> bool:
    """The same name: exactly within one script, by :func:`name_key` across scripts ("माया" and "maya")."""
    if a == b:
        return True
    if has_devanagari(a) == has_devanagari(b):
        return unicodedata.normalize("NFC", a) == unicodedata.normalize("NFC", b)
    ka, kb = name_key(a), name_key(b)
    return bool(ka) and ka == kb


# ------------------------------------------------------------------------------------------------ subjects
def devanagari_subjects(text: str, user_name: str = "") -> set[str]:
    """Who a Devanagari fact is about, in the shape of :func:`facts.fact_subjects`: {"प्रिया"} for "पिछले साल प्रिया
    पुणे गई", {"maya's sister"} for "मेरी बहन ..." when the user is Maya, {"माया's sister"} for "माया की बहन ...".
    The first word that isn't a time word or pronoun, within the first four. Empty when unclear."""
    user = (user_name or "").strip().lower() or "user"
    words = _WORD.findall(unicodedata.normalize("NFC", text or ""))
    for i, w in enumerate(words[:4]):
        if w in HINDI_FIRST_PERSON:
            return {user}
        nxt = words[i + 1] if i + 1 < len(words) else ""
        if w in HINDI_MY:
            return {f"{user}'s {HINDI_RELATIONS[nxt]}"} if nxt in HINDI_RELATIONS else {user}
        if w in HINDI_PRONOUNS:
            return set()
        if w in HINDI_SKIP:
            continue
        name = w if has_devanagari(w) else w.lower().replace(chr(0x2019), "'").removesuffix("'s")
        rel = words[i + 2] if i + 2 < len(words) else ""
        if nxt in HINDI_POSSESSIVE and rel in HINDI_RELATIONS:
            return {f"{name}'s {HINDI_RELATIONS[rel]}"}
        return {name}
    return set()


def script_words(text: str) -> set[str]:
    """Words of ``text`` in a script other than Latin (Devanagari, Cyrillic...), which the English word helpers miss.
    Small particles and time words of Hindi are left out."""
    words: set[str] = set()
    current: list[str] = []
    for ch in unicodedata.normalize("NFC", text or "") + " ":
        if unicodedata.category(ch)[0] in "LM" and (current or unicodedata.category(ch)[0] == "L"):
            current.append(ch)
            continue
        if current:
            word = "".join(current).lower()
            if not word.isascii():
                words.add(word)
            current = []
    return words - HINDI_SKIP - HINDI_PRONOUNS
