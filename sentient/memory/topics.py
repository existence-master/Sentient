"""The v2 memory topic taxonomy (ported verbatim from ``mcp_hub/memory/constants.py``).

Every stored fact carries one or more of these eight topic names. Models (and
older v3 builds) sometimes answer with free-form labels, so ``normalize_topics``
maps anything else onto the canonical set.
"""

from __future__ import annotations

TOPICS: list[dict[str, str]] = [
    {"name": "Personal Identity", "description": "Core traits, personality, beliefs, values, ethics, and preferences"},
    {"name": "Interests & Lifestyle", "description": "Hobbies, recreational activities, habits, routines, daily behavior"},
    {
        "name": "Work & Learning",
        "description": "Career, jobs, professional achievements, academic background, skills, certifications",
    },
    {"name": "Health & Wellbeing", "description": "Mental and physical health, self-care practices"},
    {
        "name": "Relationships & Social Life",
        "description": "Family, friends, romantic connections, social interactions, social media",
    },
    {"name": "Financial", "description": "Income, expenses, investments, financial goals"},
    {"name": "Goals & Challenges", "description": "Aspirations, objectives, obstacles, and difficulties faced"},
    {"name": "Miscellaneous", "description": "Anything that doesn't clearly fit into the above"},
]
TOPIC_NAMES: list[str] = [t["name"] for t in TOPICS]
DEFAULT_TOPIC = "Miscellaneous"

# free-form labels (older v3 prompt, model drift) -> canonical v2 topic
_ALIASES: dict[str, str] = {
    "identity": "Personal Identity",
    "personal": "Personal Identity",
    "preferences": "Personal Identity",
    "preference": "Personal Identity",
    "values": "Personal Identity",
    "beliefs": "Personal Identity",
    "personality": "Personal Identity",
    "location": "Personal Identity",
    "hobbies": "Interests & Lifestyle",
    "hobby": "Interests & Lifestyle",
    "interests": "Interests & Lifestyle",
    "lifestyle": "Interests & Lifestyle",
    "routines": "Interests & Lifestyle",
    "routine": "Interests & Lifestyle",
    "habits": "Interests & Lifestyle",
    "travel": "Interests & Lifestyle",
    "food": "Interests & Lifestyle",
    "tech": "Work & Learning",
    "work": "Work & Learning",
    "career": "Work & Learning",
    "job": "Work & Learning",
    "education": "Work & Learning",
    "learning": "Work & Learning",
    "skills": "Work & Learning",
    "projects": "Work & Learning",
    "health": "Health & Wellbeing",
    "wellbeing": "Health & Wellbeing",
    "fitness": "Health & Wellbeing",
    "family": "Relationships & Social Life",
    "friends": "Relationships & Social Life",
    "relationships": "Relationships & Social Life",
    "social": "Relationships & Social Life",
    "finance": "Financial",
    "finances": "Financial",
    "money": "Financial",
    "goals": "Goals & Challenges",
    "plans": "Goals & Challenges",
    "challenges": "Goals & Challenges",
    "other": "Miscellaneous",
    "misc": "Miscellaneous",
}


def normalize_topics(raw: object, limit: int = 3) -> list[str]:
    """Map a model's topic answer onto the canonical names. Never returns an empty list."""
    items: list[str]
    if isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, list | tuple):
        items = [str(x) for x in raw if x]
    else:
        items = []
    out: list[str] = []
    lower_names = {n.lower(): n for n in TOPIC_NAMES}
    for item in items:
        key = item.strip().lower()
        name = lower_names.get(key) or lower_names.get(key.replace(" and ", " & "))
        if name is None:
            name = _ALIASES.get(key)
        if name is None:
            # partial match: "Work" -> "Work & Learning", "relationships and social" -> ...
            for lname, canonical in lower_names.items():
                first = lname.split(" ")[0].strip("&")
                if key and (key in lname or first == key.split(" ")[0]):
                    name = canonical
                    break
        name = name or DEFAULT_TOPIC
        if name not in out:
            out.append(name)
    if len(out) > 1 and DEFAULT_TOPIC in out:
        out.remove(DEFAULT_TOPIC)
    return out[:limit] or [DEFAULT_TOPIC]
