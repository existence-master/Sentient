"""Prompts for memory. Ported from v2 ``mcp_hub/memory/prompts.py`` + ``formats.py``
(fact analysis, CUD decision, fact extraction) and the v2 ``summarize_old_conversations``
narrative prompt, tightened for small local models.

v3 changes: extraction returns ``{"facts": [...]}`` (JSON object mode), never writes
"the user"/"USERNAME" when the name is known, and ignores requests, dates and tool
output; CUD additionally allows SKIP when the fact is already covered.
"""

from __future__ import annotations

import json

from sentient.memory.topics import TOPIC_NAMES, TOPICS

TOPIC_LIST_STR = ", ".join(TOPIC_NAMES)
TOPIC_LINES = "\n".join(f"- {t['name']}: {t['description']}" for t in TOPICS)

# small local models over-use Miscellaneous; concrete routing examples fix most of it
TOPIC_GUIDE = """Topic guide (pick every topic that fits; "Miscellaneous" ONLY when none of the other seven fit):
- Name, where they live or are from, languages, nationality, beliefs, values, likes and dislikes -> Personal Identity
- Hobbies, sports, travel, food, habits, daily routines, pets -> Interests & Lifestyle
- Jobs, companies they work at or founded, roles, projects, education, degrees, skills, certifications -> Work & Learning
- Illness, injuries, pain, allergies, diet for health, exercise for fitness, sleep, mental health -> Health & Wellbeing
- Family members, friends, partners, colleagues and what happens in their lives -> Relationships & Social Life
- Income, spending, savings, investments, fundraising for a company -> Financial
- Plans, ambitions, targets, struggles -> Goals & Challenges

Duration guide: facts tied to a date or a near-future event ("next month", "on Friday", "this week") are "short-term" with a duration that covers the event (e.g. "6 weeks"); identity, relationships, skills, history and lasting preferences are "long-term".
"""

# ---------------------------------------------------------------- formats (v2 formats.py)
FACT_ANALYSIS_FORMAT = {
    "type": "object",
    "properties": {
        "topics": {
            "type": "array",
            "items": {"type": "string", "enum": TOPIC_NAMES},
            "description": "A list of one or more relevant topics for the given text. If none fit, use ['Miscellaneous'].",
        },
        "memory_type": {
            "type": "string",
            "enum": ["long-term", "short-term"],
            "description": "The type of memory. Use 'short-term' for transient info like reminders or temporary context.",
        },
        "duration": {
            "type": ["string", "null"],
            "description": "If memory_type is 'short-term', provide a human-readable duration (e.g., '1 hour', '3 days'). Otherwise, this should be null.",
        },
    },
    "required": ["topics", "memory_type", "duration"],
}

CUD_DECISION_FORMAT = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["ADD", "UPDATE", "DELETE", "SKIP"]},
        "fact_id": {
            "type": ["integer", "null"],
            "description": "The ID of the fact to be updated or deleted. This should be null if the action is ADD.",
        },
        "content": {
            "type": ["string", "null"],
            "description": "The new, full content of the fact if the action is ADD or UPDATE. Should be null for DELETE.",
        },
        "analysis": {
            "type": ["object", "null"],
            "properties": FACT_ANALYSIS_FORMAT["properties"],
            "description": "A full analysis of the new content. Required for ADD or UPDATE actions, null for DELETE.",
        },
    },
    "required": ["action", "fact_id", "content", "analysis"],
}

# ---------------------------------------------------------------- fact analysis (v2)
FACT_ANALYSIS_SYSTEM = f"""You are an information analysis system. Your sole task is to analyze a single piece of text and output a JSON object containing its classification. Adhere strictly to the provided JSON schema.

Topics:
{TOPIC_LINES}

{TOPIC_GUIDE}
Instructions:
1. Read the input text carefully.
2. Topic Classification: Select one or more relevant topics, using the exact topic names above. If none fit, use "Miscellaneous".
3. Memory Duration: Decide if the information is 'long-term' (core facts, preferences, relationships, work) or 'short-term' (transient info, plans with a date, reminders, temporary states).
4. Duration Estimation: If 'short-term', estimate a reasonable expiration duration (e.g., '2 hours', '1 day', '2 weeks'). If 'long-term', set duration to null.
5. Your response MUST be a single, valid JSON object that strictly adheres to the following schema. Do not include any other text or explanations.

JSON Schema:
{json.dumps(FACT_ANALYSIS_FORMAT, indent=2)}
"""
FACT_ANALYSIS_USER = 'Analyze the following text: "{text}"'

# ---------------------------------------------------------------- CUD decision (v2 + SKIP)
CUD_DECISION_SYSTEM = f"""You are a memory management reasoning engine. Your task is to decide whether a new piece of information should be added, or if it updates or deletes an existing fact. You must also perform a full analysis for any new or updated content. Adhere strictly to the provided JSON schema.

Actions:
- ADD: The new information is entirely new. `content` is the new fact (lightly cleaned, same meaning), and `analysis` must be completed. `fact_id` is null.
- UPDATE: The new information modifies or supersedes one existing fact (same subject, newer or different value). `content` is the new, full, updated fact, and `analysis` must be completed for it. `fact_id` is the ID of the original fact.
- DELETE: The new information says an existing fact is no longer true and nothing replaces it. `fact_id` is the ID of the fact to remove. `content` and `analysis` are null.
- SKIP: The new information is already fully covered by an existing fact and adds NO new detail (no new place, date, time, name, number or change of state). `fact_id` is that fact's ID. If it adds detail, use UPDATE to write one merged fact that keeps both.

Instructions:
1. Analyze the new information to understand its intent.
2. Compare it with the existing facts listed (they come with their IDs and similarity scores). Only use IDs from that list. A fact marked "same_subject_attribute" is about the same person and the same attribute (where they live, their job, relationship, diet...): if the new information changes that attribute (e.g. "moved to Bengaluru" vs "lives in Pune"), UPDATE that fact instead of adding a second one.
3. Decide the action. When the existing list is empty, the action is ADD.
4. For ADD or UPDATE you MUST perform a complete analysis on the new `content`:
   - topics: one or more of: {TOPIC_LIST_STR}
   - memory_type: "long-term" for stable facts; "short-term" for time-bound facts (plans, deadlines, temporary states)
   - duration: for short-term facts, e.g. "3 days", "2 weeks"; null for long-term

{TOPIC_GUIDE}5. Your response MUST be a single, valid JSON object that strictly adheres to the following schema. Do not include any other text or explanations.

JSON Schema:
{json.dumps(CUD_DECISION_FORMAT, indent=2)}

Example:
{{"action": "ADD", "fact_id": null, "content": "Maya's sister Riya lives in Pune.", "analysis": {{"topics": ["Relationships & Social Life"], "memory_type": "long-term", "duration": null}}}}
"""
CUD_DECISION_USER = (
    "New information: '{information}'\n\nHere are the most similar facts already in memory:\n{similar_facts}\n\n"
    "Decide the correct action and provide all required fields."
)

# ---------------------------------------------------------------- fact extraction (v2, tightened)
_NAME_RULE_KNOWN = (
    'Write in third person using the name "{username}". "I" and "my" become "{username}" and "{username}\'s". '
    'Never write the words "the user", "user", or "USERNAME".'
)
_NAME_RULE_UNKNOWN = 'Write in third person, referring to the person as "The user".'

FACT_EXTRACTION_SYSTEM = """You are an expert system for information decomposition. Your primary goal is to study incoming text and convert it into a list of "atomic" facts about {username}. An atomic fact is a single, indivisible piece of information that is meaningful and personally relevant to {username}. Extract ONLY durable facts that are directly about {username}. If there are none, return an empty list.

Key Instructions:
1. Deconstruct compound sentences into ATOMIC FACTS: split sentences containing 'and', 'but', or 'while' into separate, self-contained facts. Each fact must be a complete thought.
2. {name_rule}
3. Output ONLY a JSON object of the form {{"facts": ["fact 1", "fact 2"]}}. No commentary.

YOU MUST IGNORE (never output these):
- What {username} is asking or requesting right now ("asked what day it is", "wants a summary", "needs help with an email").
- The current date, time, weather, or anything an assistant or a tool reported.
- Boilerplate and formatting: signatures, headers, footers, navigation links, confidentiality notices, unsubscribe links.
- UI text and metadata: button labels, image alt text, system messages, structural titles like "Subject:" or "Fwd:".
- Vague or procedural statements: "see below", "let me know your thoughts", "the task was completed".
- Trivial, temporary details with no lasting value ("the meeting is at 2 PM today"). A recurring pattern IS valuable ("{username}'s weekly marketing meeting is on Tuesdays at 2 PM").

YOU MUST ONLY EXTRACT:
- Personal details ("{username}'s sister Riya lives in Pune.")
- Preferences ("{username} drinks only black coffee.")
- Professional context ("{username} is the project lead for Project Phoenix.")
- Relationships ("{username}'s manager is Jane Doe.")
- Commitments and plans ("{username} promised to send the report by Friday.")
- Goals, habits, routines, health, skills, education, tools they use, places they go.

Example 1:
Text: "Hi team, just a reminder that I'm the lead on the new mobile app project. My manager, Jane, and I decided that the deadline is next Friday. Also, my favorite snack is almonds."
Output: {{"facts": ["{username} is the lead on the new mobile app project.", "{username}'s manager is Jane.", "{username}'s mobile app project deadline is next Friday.", "{username}'s favorite snack is almonds."]}}

Example 2:
Text: "Notification from Asana: Task 'Update Website Copy' was completed by you. Due Date: Yesterday. Click here to view the task. Avatar of Alex."
Output: {{"facts": []}}

Example 3:
Text: "what's the weather like today? can you check my calendar"
Output: {{"facts": []}}
"""

FACT_EXTRACTION_USER = "Text from {username}:\n{text}"


def extraction_system(username: str) -> str:
    known = bool(username.strip())
    name = username.strip() or "the user"
    rule = (_NAME_RULE_KNOWN if known else _NAME_RULE_UNKNOWN).format(username=name)
    return FACT_EXTRACTION_SYSTEM.format(username=name, name_rule=rule)


# ---------------------------------------------------------------- episodic summaries (v2 narrative prompt)
SUMMARIZE_CHUNK_SYSTEM = """You are the AI assistant in the provided conversation log. Your task is to write a summary of the conversation from your own perspective, as if you are recalling the memory of the interaction.

Core Instructions:
1. Adopt a First-Person Narrative: Use "I", "me", and "my" to refer to your own actions and thoughts. Refer to the other party as {user_ref}.
2. Describe the Flow: Recount the conversation as a sequence of events. For example: "{user_ref_cap} told me about their project...", "I then asked for clarification on...", "We then discussed...".
3. File uploads: If a message mentions an attached file, describe the action factually ("{user_ref_cap} uploaded a file named 'report.pdf' and asked for a summary."). Never say you could not process it.
4. Goal: A dense, narrative paragraph that captures the key information, decisions, names, dates and open questions from my point of view. Focus on information useful for future context.
5. Format: No preamble or sign-off. Respond only with the summary paragraph.
"""


FLUSH_USER = """These conversation turns are about to be compressed. Extract only lasting facts about {username}: things {username} said about themselves, their people, preferences and plans, decisions {username} made and commitments either side agreed to.
Ignore greetings, questions, requests, and anything the assistant or a tool reported that {username} did not confirm.

Conversation:
{transcript}"""


# ---------------------------------------------------------------- user model (dialectic refresh)
USER_MODEL_DIMENSIONS = [
    "preferences", "communication", "goals", "routines", "relationships", "values", "work_style", "dislikes", "context",
]

USER_MODEL_SYSTEM = """You keep a model of who {name} is: patterns in preferences, communication, goals, routines, relationships, values, work style, dislikes and life context. Use ONLY the evidence given.

Reply with JSON only, in this shape:
{{"operations": [
  {{"op": "add", "dimension": "preferences", "statement": "{name} prefers short, direct answers.", "confidence": 0.5, "evidence": ["m2", "f14"]}},
  {{"op": "support", "id": "i3", "evidence": ["m5"]}},
  {{"op": "contradict", "id": "i2", "evidence": ["m7"], "question": "Do you still run every morning?"}},
  {{"op": "retire", "id": "i4", "reason": "the project ended"}}
],
"summary": "short markdown portrait of {name}"}}

Rules:
- dimension is one of: {dimensions}.
- An insight is a pattern or trait in one third-person sentence, not a single event, request or task. Do not copy a fact word for word.
- If an existing insight already covers it, use support instead of add.
- contradict only when evidence goes against an insight. retire only when it clearly no longer applies.
- Use only ids shown below. Evidence ids come from the message (m), fact (f) and summary (s) lists.
- confidence for a new insight: 0.4 for one signal, up to 0.7 for repeated signals.
- summary: at most 150 words, about {name}, in third person.
- Nothing new: {{"operations": [], "summary": "<the current portrait>"}}."""

USER_MODEL_USER = """Current portrait:
{summary}

Current insights (id | dimension | confidence | status | statement):
{insights}

Recent messages from {name}:
{messages}

New or changed facts:
{facts}

Conversation summaries:
{summaries}"""

USER_MODEL_ANSWER_SYSTEM = """You asked {name} a question to check something you believed about them. Decide what the answer means.
Reply with JSON only: {{"verdict": "confirm" | "retire" | "rewrite", "statement": "<the corrected belief in one third-person sentence, for rewrite>", "fact": "<one third-person fact about {name} learned from the answer>"}}
- confirm: the belief is right. retire: it is wrong and nothing replaces it. rewrite: it is partly right or changed."""

USER_MODEL_ANSWER_USER = 'Belief: "{statement}"\nQuestion: "{question}"\nAnswer from {name}: "{answer}"'

USER_MODEL_ASK_SYSTEM = """You predict what {name} would want, using only what is known about them below. Answer in two or three sentences. Say which insight or fact you relied on. If nothing known applies, say you are not sure and suggest asking {name}."""

# ---------------------------------------------------------------- dreaming
DREAM_MERGE_SYSTEM = """These memories about {name} say the same thing. Write ONE fact in third person that keeps every name, place, date and number from all of them and adds nothing new.
Reply with JSON only: {{"keep_id": <id of the memory to keep>, "fact": "<merged fact>"}}
If they are not about the same thing, reply {{"keep_id": null, "fact": null}}."""

DREAM_CONTRADICTION_SYSTEM = """Two memories about {name} may conflict. A was saved before B. Decide whether both can be true today, and if not, which one is true now.
Reply with JSON only: {{"conflict": true | false, "current": "A" | "B", "change": "<what changed, under 12 words>"}}
Pick B unless A's own wording says it is the more recent situation."""

DREAM_CONTRADICTION_USER = 'A (saved {a_at}): "{a}"\nB (saved {b_at}): "{b}"'


def summarize_system(user_name: str) -> str:
    ref = user_name.strip() or "the user"
    return SUMMARIZE_CHUNK_SYSTEM.format(user_ref=ref, user_ref_cap=ref[:1].upper() + ref[1:])
