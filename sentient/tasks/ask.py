"""``ask_user``: a running task pauses with one question and carries on with the answer.

The tool only records the question in the slot the executor puts in ``ctx.extra``; the
executor then stops the loop after that round (no model call while waiting), stores the
transcript and the question on the run (status ``waiting_for_user``), and the answer
later replaces the tool's placeholder result before the run resumes. Offered only inside
task runs: the plugin is ``scoped``, so chat, subagents and the planner never see it.
"""

from __future__ import annotations

import json
from typing import Any

from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool

ASK_TOOL = "ask_user"
PLUGIN_ID = "task_questions"
STATE_KEY = "ask_user"  # ctx.extra slot the executor provides for one run
MAX_OPTIONS = 6
MAX_OPTION_CHARS = 80
MAX_QUESTION_CHARS = 1000
MAX_ANSWER_CHARS = 4000
MAX_QUESTIONS_PER_RUN = 5

WAITING_NOTE = "Your question was sent to the user. This task is paused until they answer."
CANCELLED_NOTE = "The user cancelled this run instead of answering. Stop here."


def clean_options(raw: Any) -> list[str]:
    """Up to ``MAX_OPTIONS`` short, distinct, non-empty choices."""
    out: list[str] = []
    for item in raw if isinstance(raw, list) else []:
        text = " ".join(str(item or "").split())[:MAX_OPTION_CHARS].strip()
        if text and text.lower() not in {o.lower() for o in out}:
            out.append(text)
    return out[:MAX_OPTIONS]


def answer_content(answer: str) -> str:
    """The tool message the model reads once the user answered."""
    return json.dumps(
        {"answer": answer, "note": "This is the user's answer to your question. Continue the task with it."},
        ensure_ascii=False,
    )


def cancelled_content() -> str:
    return json.dumps({"error": CANCELLED_NOTE}, ensure_ascii=False)


def count_questions(messages: list[dict] | None) -> int:
    """How many times this run already called ask_user."""
    n = 0
    for m in messages or []:
        calls = m.get("tool_calls") if isinstance(m, dict) else None
        for tc in calls if isinstance(calls, list) else []:
            if isinstance(tc, dict) and (tc.get("function") or {}).get("name", tc.get("name")) == ASK_TOOL:
                n += 1
    return n


def last_call_id(messages: list[dict]) -> str:
    """The id of the most recent ask_user tool result (fallback when the tool could not see its call id)."""
    for m in reversed(messages):
        if m.get("role") == "tool" and m.get("name") == ASK_TOOL and m.get("tool_call_id"):
            return str(m["tool_call_id"])
    return ""


def fill_result(messages: list[dict], call_id: str | None, content: str) -> list[dict]:
    """Replace the placeholder result of the ask_user call ``call_id`` with ``content``.

    Without a matching tool message (an old or damaged checkpoint) the answer is added as a
    user message, so the model still sees it.
    """
    out: list[dict] = []
    found = False
    for m in messages or []:
        if call_id and m.get("role") == "tool" and m.get("tool_call_id") == call_id:
            m = {**m, "content": content}
            found = True
        out.append(m)
    if not found:
        try:
            data = json.loads(content)
            text = data.get("answer") or data.get("error") or content
        except (TypeError, ValueError, AttributeError):
            text = content
        out.append({"role": "user", "content": f"Reply to your question: {text}"})
    return out


@tool(ASK_TOOL, risk=Risk.write, internal=True)
async def ask_user(ctx: ToolContext, question: str, options: list[str] | None = None) -> dict:
    """Ask the user one question and pause this task until they answer. Use it only when you truly
    cannot continue without the user's choice (for example, which of two flights to book). Never use it
    to confirm risky actions; the app handles that. Give `options` (2 to 6 short choices) when there are
    clear choices. The user's answer comes back as the result of this call."""
    state = ctx.extra.get(STATE_KEY)
    if not isinstance(state, dict):
        return {"error": "ask_user only works inside a running task."}
    if state.get("question"):
        return {"error": "You already asked a question. Wait for that answer before asking another."}
    if int(state.get("asked") or 0) >= MAX_QUESTIONS_PER_RUN:
        return {"error": "You have asked enough questions in this run. Make a sensible choice and continue."}
    text = str(question or "").strip()[:MAX_QUESTION_CHARS]
    if not text:
        return {"error": "The question is empty. Call ask_user again with the question to ask."}
    state.update(question=text, options=clean_options(options), tool_call_id=ctx.call_id or "")
    return {"status": "waiting_for_user", "note": WAITING_NOTE}


class TaskQuestionsPlugin(ToolPlugin):
    id = PLUGIN_ID
    display_name = "Questions for you"
    description = "Lets a running task pause and ask you a question."
    category = "core"
    icon = "IconMessageQuestion"
    scoped = True
    tools = [ask_user]
