"""The agent loop.

Two layers:

- ``Agent.run_loop`` is the reusable tool-calling engine: given messages, a
  role/model and a tool subset, it streams typed events, executes tools
  (through approvals when asked), records token usage, and leaves the final
  text in a ``LoopResult``. Chat turns, task runs, swarm workers, subagents,
  proactive reasoning and voice all use it. Tools run in their own asyncio task
  so their ``ctx.progress`` output streams as ``tool_progress`` while they work;
  look-up tools requested together run concurrently; results keep their order.
- ``Agent.run_turn`` is a chat turn: persists the user message (with
  attachments), builds the system prompt (persona, profile, recalled memory,
  user model, skills, clock, running conversation summary), runs the loop,
  persists the transcript, then kicks off background work (fact extraction,
  auto title, context compression). Messages the user sends while a reply runs
  (``Agent.steer``) are fed to the model at the next round.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import inspect
import json
import logging
import mimetypes
import re
import time
import weakref
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from sentient import paths
from sentient.agent.approvals import ApprovalBroker
from sentient.agent.prompt import build_system_prompt
from sentient.agent.toolselect import ToolSelector
from sentient.config.schema import SentientConfig
from sentient.files.extract import extract_text, is_image
from sentient.llm.events import (
    AgentEvent,
    ApprovalRequest,
    Done,
    Error,
    TextDelta,
    ThinkingDelta,
    ToolCallEvent,
    ToolProgress,
    ToolResultEvent,
    Usage,
    UserInterjection,
    tool_progress_event,
)
from sentient.llm.provider import LLMProvider, ProviderError, ToolCall
from sentient.memory.facts import FactMemory
from sentient.memory.workspace import Workspace
from sentient.services import cancel_tasks
from sentient.skills.loader import SkillLibrary
from sentient.store.db import Store, new_id
from sentient.tools.base import Risk, Tool, ToolContext, bind_call, describe_call, effective_risk
from sentient.tools.registry import ToolRegistry
from sentient.tools.rules import never_message, unattended_ask_message

log = logging.getLogger(__name__)

MAX_ATTACHMENT_CHARS = 60_000
EMPTY_ANSWER_NUDGE = (
    "You have not replied yet. If the request still needs a tool call, make it now; "
    "otherwise answer the user now, based on the tool results above."
)
USER_MODEL_TIMEOUT_S = 2.0
SPOKEN_CHANNELS = {"voice", "glasses", "phone"}  # these turns use the voice role
# Real qwen3:8b told the user "the Place order button was clicked" after a plain decline, so say it bluntly.
DECLINED = (
    "NOT DONE. The user declined this action, so it did not happen. Tell the user plainly that it was not done "
    "and never say or imply that it succeeded."
)
PersistFn = Callable[..., Awaitable[Any]]
# policy(tool, effective_risk, arguments) -> refusal message or None (sync or async)
PolicyFn = Callable[[Tool, Risk, dict], "str | Awaitable[str | None] | None"]


# ---------------------------------------------------------------------- transcript helpers
def _normalize_tool_calls(calls: Any) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for i, tc in enumerate(calls if isinstance(calls, list) else []):
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        name = fn.get("name") or tc.get("name")
        if not name:
            continue
        args = fn.get("arguments", tc.get("arguments", {}))
        if not isinstance(args, str):
            args = json.dumps(args if args is not None else {}, ensure_ascii=False, default=str)
        call_id = str(tc.get("id") or f"call_{i}")
        if call_id in seen:
            continue
        seen.add(call_id)
        out.append({"id": call_id, "type": "function", "function": {"name": name, "arguments": args}})
    return out


def history_to_openai(rows: list[dict]) -> list[dict]:
    """Stored transcript rows -> OpenAI-format messages every provider accepts.

    Repairs what history windows and interrupted turns leave behind: tool messages whose call was cut off,
    assistant tool calls without results (dropped, keeping any text), empty assistant messages,
    and tool calls stored with non-string arguments.
    """
    out: list[dict] = []
    for r in rows:
        role = r.get("role")
        if role == "assistant":
            calls = _normalize_tool_calls(r.get("tool_calls"))
            content = r.get("content") or ""
            if not calls and not content.strip():
                continue
            msg: dict[str, Any] = {"role": "assistant", "content": content}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
        elif role == "tool":
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": r.get("tool_call_id") or "",
                    "name": r.get("name") or "",
                    "content": r.get("content") or "",
                }
            )
        elif role in {"user", "system"}:
            content = r.get("content") or ""
            if r.get("attachments"):
                content += "\n\n(Attached earlier: " + ", ".join(r["attachments"]) + ")"
            out.append({"role": role, "content": content})

    # every assistant tool call needs its reply right after it; tool messages need their call
    fixed: list[dict] = []
    i = 0
    while i < len(out):
        m = out[i]
        if m["role"] == "tool":  # orphan: its call fell out of the window or was never recorded
            i += 1
            continue
        if m["role"] == "assistant" and m.get("tool_calls"):
            ids = {tc["id"] for tc in m["tool_calls"]}
            answered: dict[str, dict] = {}
            j = i + 1
            while j < len(out) and out[j]["role"] == "tool":
                tid = out[j].get("tool_call_id")
                if tid in ids and tid not in answered:
                    answered[tid] = out[j]
                j += 1
            calls = [tc for tc in m["tool_calls"] if tc["id"] in answered]
            if calls:
                fixed.append({**m, "tool_calls": calls})
                fixed.extend(answered[tc["id"]] for tc in calls)
            elif (m.get("content") or "").strip():
                fixed.append({"role": "assistant", "content": m["content"]})
            i = j
            continue
        fixed.append(m)
        i += 1
    return fixed


def is_tool_format_error(exc: Exception) -> bool:
    """True when the provider rejected the request because the model cannot do tool calling
    (e.g. an old Ollama model template emitting malformed tool calls)."""
    m = str(exc).lower()
    return "does not support tools" in m or "invalid character" in m or ("tool" in m and "pars" in m)


def _json_safe(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return json.dumps(str(value))


def _error_text(res: Any) -> str | None:
    if isinstance(res, dict) and res.get("error"):
        return str(res["error"])[:500]
    return None


@dataclass
class LoopResult:
    text: str = ""
    thinking: str = ""
    tool_calls: int = 0
    tools_used: list[str] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    hit_step_limit: bool = False
    tool_errors: list[dict] = field(default_factory=list)   # [{name, error}] (declined approvals excluded)
    skills_viewed: list[str] = field(default_factory=list)  # names passed to skill_view
    interjections: list[str] = field(default_factory=list)  # steer messages the model received
    paused: bool = False  # ``stop`` ended the loop after a round of tool results (a task run asked the user)
    # set when an "ask" rule stopped an unattended run (also copied to ``error``); plain words for the user
    stopped_by_rule: str | None = None


class SteerQueue:
    """Messages the user sent while a reply was running, applied at the next model round."""

    def __init__(self) -> None:
        self._items: list[str] = []
        self.accepting = True

    def put(self, text: str) -> bool:
        if not self.accepting:
            return False
        self._items.append(text)
        return True

    def take(self) -> list[str]:
        items, self._items = self._items, []
        return items

    def restore(self, items: list[str]) -> None:
        self._items = [*items, *self._items]

    def close(self) -> list[str]:
        self.accepting = False
        return self.take()


@dataclass
class _CallPlan:
    tc: ToolCall
    tool: Tool | None
    risk: Risk = Risk.read
    preset: tuple[Any, bool, int] | None = None  # result decided without running the tool
    needs_approval: bool = False
    concurrent: bool = False  # may run at the same time as neighbouring look-ups
    outcome: tuple[Any, bool, int] | None = None
    rule_stop: bool = False  # an "ask" rule refused this call in a run nobody can answer


class Agent:
    def __init__(
        self,
        *,
        config: SentientConfig,
        store: Store,
        llm: LLMProvider,
        registry: ToolRegistry,
        memory: FactMemory | None,
        workspace: Workspace,
        skills: SkillLibrary,
        approvals: ApprovalBroker,
        app: Any = None,
    ):
        self.config = config
        self.store = store
        self.llm = llm
        self.registry = registry
        self.memory = memory
        self.workspace = workspace
        self.skills = skills
        self.approvals = approvals
        self.app = app  # SentientApp; gives tools access to feature services via ctx.extra["app"]
        self.tool_selector = ToolSelector(self)
        self._background: set[asyncio.Task] = set()
        self._steers: dict[str, SteerQueue] = {}
        self._turn_tasks: weakref.WeakSet[asyncio.Task] = weakref.WeakSet()  # tasks running a chat reply

    # ------------------------------------------------------------------ steering
    def is_running(self, session_id: str) -> bool:
        """True while a chat reply for this session is running and can take steer messages."""
        q = self._steers.get(session_id)
        return bool(q and q.accepting)

    def steer(self, session_id: str, text: str) -> bool:
        """Queue ``text`` for the reply running in ``session_id``. Returns False when no reply is running
        (the caller should start a normal turn instead)."""
        text = (text or "").strip()
        q = self._steers.get(session_id)
        if not text or q is None:
            return False
        return q.put(text)

    # ------------------------------------------------------------------ context
    def tool_context(self, session_id: str | None, channel: str) -> ToolContext:
        return ToolContext(
            store=self.store,
            config=self.config,
            llm=self.llm,
            memory=self.memory,
            session_id=session_id,
            channel=channel,
            extra=self.app.tool_extra() if self.app is not None else {
                "skills": self.skills, "workspace": self.workspace, "registry": self.registry
            },
        )

    async def _user_model_context(self, user_text: str) -> str:
        um = getattr(self.app, "user_model", None) if self.app is not None else None
        fn = getattr(um, "context_for", None)
        if not callable(fn):
            return ""
        try:
            out = await asyncio.wait_for(fn(user_text), timeout=USER_MODEL_TIMEOUT_S)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # includes the timeout: the user model must never slow a turn down much
            log.debug("user model context skipped: %s", exc)
            return ""
        return out.strip() if isinstance(out, str) else ""

    async def _recall(self, user_text: str) -> list[dict]:
        if self.memory is None or self.config.memory.facts_top_k <= 0 or not user_text.strip():
            return []
        try:
            return await self.memory.recall(user_text)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("recall failed: %s", exc)
            return []

    async def system_prompt(self, user_text: str, channel: str, session: dict | None = None) -> str:
        facts, user_context = await asyncio.gather(self._recall(user_text), self._user_model_context(user_text))
        prompt = build_system_prompt(
            snapshot=self.workspace.snapshot(),
            facts=facts,
            skills_index=self.skills.prompt_index(),
            assistant_name=self.config.assistant.name,
            user_name=self.config.assistant.user_name,
            timezone=self.config.assistant.timezone,
            channel=channel,
            location=self.config.assistant.location,
            user_context=user_context,
            tool_names=[t.name for t in self.registry.tools()],
        )
        if session and session.get("context_summary"):
            prompt += "\n\n## Earlier in this conversation\n" + session["context_summary"]
        return prompt

    # ------------------------------------------------------------------ reusable engine
    async def run_loop(
        self,
        messages: list[dict],
        ctx: ToolContext,
        *,
        result: LoopResult,
        role: str = "primary",
        model: str | None = None,
        tool_names: list[str] | None = None,
        max_rounds: int | None = None,
        use_approvals: bool = True,
        persist: PersistFn | None = None,
        source: str = "chat",
        ev: dict | None = None,
        steer: SteerQueue | None = None,
        policy: PolicyFn | None = None,
        stop: Callable[[], bool] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Stream a tool-calling conversation. Mutates ``messages`` and fills ``result``.

        ``persist(role, content, **fields)`` is called for intermediate assistant
        tool-call messages, tool results and steer messages (not for the final answer).
        ``steer`` feeds user messages in at round boundaries. ``policy`` may refuse a call
        before it runs (subagents use it). ``stop()`` is checked after each round of tool
        results; when it returns True the loop ends without another model call and sets
        ``result.paused`` (task runs use it to wait for the user's answer). Yields everything except ``Done``.
        """
        ev = ev or {}
        tools = self.registry.openai_schemas(tool_names) or None
        rounds = max_rounds or self.config.models.max_tool_rounds
        text_acc = ""
        tools_disabled = False
        empty_nudged = False
        round_no = 0
        while round_no < rounds:
            round_no += 1
            if steer is not None:
                for text in steer.take():
                    async for e in self._interject(text, messages, persist, result, ev):
                        yield e
            text_acc = ""
            think_acc = ""
            tool_calls: list[ToolCall] = []
            try:
                async for chunk in self.llm.stream(role, messages, tools, model=model):
                    if chunk.thinking:
                        think_acc += chunk.thinking
                        yield ThinkingDelta(text=chunk.thinking, **ev)
                    if chunk.text:
                        text_acc += chunk.text
                        yield TextDelta(text=chunk.text, **ev)
                    if chunk.done:
                        tool_calls = chunk.tool_calls
                        if chunk.usage:
                            result.prompt_tokens += chunk.usage.get("prompt_tokens", 0)
                            result.completion_tokens += chunk.usage.get("completion_tokens", 0)
                            yield Usage(model=chunk.model, **chunk.usage, **ev)
                            with contextlib.suppress(Exception):
                                await self.store.record_usage(
                                    chunk.model, chunk.usage.get("prompt_tokens", 0),
                                    chunk.usage.get("completion_tokens", 0), role=role, source=source,
                                )
            except ProviderError as exc:
                if tools and not tools_disabled and not text_acc and not think_acc and is_tool_format_error(exc):
                    # degrade instead of failing the whole turn: answer without tools
                    log.warning("model rejected tool calling (%s); retrying without tools", exc)
                    tools = None
                    tools_disabled = True
                    yield Error(
                        message="This model can't use tools, so I'm answering without them. "
                        "Choose a tool-capable model in Settings → Models.",
                        recoverable=True,
                        **ev,
                    )
                    continue
                result.error = str(exc)
                yield Error(message=str(exc), recoverable=False, **ev)
                return
            result.thinking += think_acc

            if not tool_calls:
                pending = steer.take() if steer is not None else []
                if pending and round_no < rounds:
                    # the user added something while this answer streamed: keep the answer, then continue
                    if text_acc.strip():
                        messages.append({"role": "assistant", "content": text_acc})
                        if persist:
                            await persist("assistant", text_acc, thinking=think_acc or None)
                    for text in pending:
                        async for e in self._interject(text, messages, persist, result, ev):
                            yield e
                    continue
                if pending and steer is not None:
                    steer.restore(pending)  # no rounds left: the caller starts a new turn with them
                if not text_acc.strip() and not empty_nudged and round_no < rounds:
                    # small local models sometimes end a round with only hidden thinking: ask once for a reply
                    empty_nudged = True
                    messages.append({"role": "user", "content": EMPTY_ANSWER_NUDGE})
                    continue
                if empty_nudged and messages and messages[-1].get("content") == EMPTY_ANSWER_NUDGE:
                    messages.pop()  # the nudge is scaffolding, not part of the conversation
                result.text = text_acc
                result.messages = messages
                return

            assistant_msg = {
                "role": "assistant",
                "content": text_acc or "",
                "tool_calls": [tc.to_openai() for tc in tool_calls],
            }
            messages.append(assistant_msg)
            if persist:
                await persist("assistant", text_acc or None, tool_calls=assistant_msg["tool_calls"], thinking=think_acc or None)

            plans = [await self._plan_call(tc, ctx, tool_names, use_approvals, policy) for tc in tool_calls]
            result.tool_calls += len(plans)
            for group in self._groups(plans):
                for p in group:
                    yield ToolCallEvent(call_id=p.tc.id, name=p.tc.name, arguments=p.tc.arguments, **ev)
                runnable: list[_CallPlan] = []
                for p in group:
                    if p.preset is not None:
                        p.outcome = p.preset
                    elif p.needs_approval:
                        assert p.tool is not None
                        approval_id = new_id()
                        self.approvals.create(approval_id)
                        wording = await describe_call(p.tool, p.tc.arguments, ctx, p.risk)
                        yield ApprovalRequest(
                            approval_id=approval_id, call_id=p.tc.id, name=p.tc.name, arguments=p.tc.arguments,
                            risk=p.risk.name, reason=f"{p.tool.plugin}: {p.tool.description[:160]}",
                            risk_label=wording["risk_label"], target=wording["target"], **ev,
                        )
                        decision = await self.approvals.wait(approval_id, ctx.session_id, p.tc.name, p.risk)
                        if decision in {"allow", "allow_session"}:
                            runnable.append(p)
                        else:
                            p.outcome = ({"error": DECLINED, "declined": True}, True, 0)
                    else:
                        runnable.append(p)
                if runnable:
                    async for progress in self._execute(runnable, ctx, ev):
                        yield progress
                for p in group:
                    async for e in self._record(p, messages, persist, result, ev):
                        yield e
            ruled = next((p for p in plans if p.rule_stop), None)
            if ruled is not None:  # an "ask" rule and nobody to ask: end the run and say why
                assert ruled.preset is not None
                result.stopped_by_rule = result.error = _error_text(ruled.preset[0])
                result.text = text_acc
                result.messages = messages
                return
            if stop is not None and stop():
                result.paused = True
                result.text = ""
                result.messages = messages
                return

        result.hit_step_limit = True
        result.text = text_acc or "I reached the step limit before finishing. Tell me how to continue."
        result.messages = messages

    async def _interject(
        self, text: str, messages: list[dict], persist: PersistFn | None, result: LoopResult, ev: dict
    ) -> AsyncIterator[AgentEvent]:
        messages.append({"role": "user", "content": text})
        result.interjections.append(text)
        if persist:
            await persist("user", text, interjection=True)
        yield UserInterjection(text=text, **ev)

    async def _plan_call(
        self,
        tc: ToolCall,
        ctx: ToolContext,
        tool_names: list[str] | None,
        use_approvals: bool,
        policy: PolicyFn | None,
    ) -> _CallPlan:
        tool = self.registry.get(tc.name)
        rule = self.approvals.rule(tool) if tool is not None else None
        if tool is not None and rule == "never":
            # lasting rule (ADR 0016): the tool is not offered, and a call made anyway is refused without running
            refusal = never_message(self.approvals.label(tool, self.registry))
            return _CallPlan(tc=tc, tool=None, preset=({"error": refusal}, True, 0))
        if tool is None or not (tool_names is None or tc.name in tool_names):
            return _CallPlan(tc=tc, tool=None, preset=({"error": f"unknown tool {tc.name}"}, True, 0))
        plan = _CallPlan(tc=tc, tool=tool, risk=tool.risk)
        if "_raw" in tc.arguments and len(tc.arguments) == 1:
            plan.preset = self._raw_arguments_error(tool, tc)
            return plan
        plan.risk = await effective_risk(tool, tc.arguments, ctx)
        if policy is not None:
            refusal = policy(tool, plan.risk, tc.arguments)
            if inspect.isawaitable(refusal):
                refusal = await refusal
            if refusal:
                plan.preset = ({"error": str(refusal)}, True, 0)
                return plan
        if use_approvals:
            plan.needs_approval = await self.approvals.decide(tool, ctx.session_id, plan.risk, tc.arguments, ctx)
        elif rule == "ask":  # task runs and other unattended loops cannot stop to ask: the run stops here
            plan.preset = ({"error": unattended_ask_message(self.approvals.label(tool, self.registry))}, True, 0)
            plan.rule_stop = True
        return plan

    def _groups(self, plans: list[_CallPlan]) -> list[list[_CallPlan]]:
        """Consecutive look-ups that need no approval run together; everything else runs alone, in order."""
        parallel = self.config.chat.parallel_read_tools
        groups: list[list[_CallPlan]] = []
        for p in plans:
            p.concurrent = parallel and p.tool is not None and p.preset is None and not p.needs_approval and p.risk == Risk.read
            if p.concurrent and groups and groups[-1][-1].concurrent:
                groups[-1].append(p)
            else:
                groups.append([p])
        return groups

    async def _execute(self, plans: list[_CallPlan], ctx: ToolContext, ev: dict) -> AsyncIterator[ToolProgress]:
        """Run tools as tasks and stream their ``ctx.progress`` output until all finish."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[ToolProgress] = asyncio.Queue()

        def sink(call_id: str, name: str, payload: dict) -> None:
            event = tool_progress_event(call_id, name, payload, **ev)
            try:
                running = asyncio.get_running_loop()
            except RuntimeError:
                running = None
            if running is loop:
                queue.put_nowait(event)
            else:  # a worker thread (asyncio.to_thread) reporting output
                loop.call_soon_threadsafe(queue.put_nowait, event)

        async def run_one(p: _CallPlan) -> None:
            bind_call(sink, p.tc.id, p.tc.name)
            p.outcome = await self._run_tool(p.tc, ctx)

        tasks = [asyncio.create_task(run_one(p), name=f"tool:{p.tc.name}") for p in plans]
        pending: set[asyncio.Future] = set(tasks)
        getter: asyncio.Future | None = None
        try:
            while pending:
                if getter is None:
                    getter = asyncio.ensure_future(queue.get())
                done, _ = await asyncio.wait({*pending, getter}, return_when=asyncio.FIRST_COMPLETED)
                if getter in done:
                    yield getter.result()
                    getter = None
                pending -= done
            if getter is not None:
                getter.cancel()
                getter = None
            await asyncio.sleep(0)  # let thread-safe callbacks scheduled at the very end land
            while not queue.empty():
                yield queue.get_nowait()
        finally:
            if getter is not None:
                getter.cancel()
            for t in tasks:
                if not t.done():
                    t.cancel()
        for p, t in zip(plans, tasks, strict=True):
            if p.outcome is None:  # the task died without recording (should not happen)
                exc = t.exception() if t.done() and not t.cancelled() else None
                p.outcome = ({"error": f"tool did not finish: {exc}"}, True, 0)

    async def _record(
        self, p: _CallPlan, messages: list[dict], persist: PersistFn | None, result: LoopResult, ev: dict
    ) -> AsyncIterator[AgentEvent]:
        tc = p.tc
        res, is_error, ms = p.outcome or ({"error": "tool did not run"}, True, 0)
        if p.tool is not None and p.tool.plugin not in result.tools_used:
            result.tools_used.append(p.tool.plugin)
        err = _error_text(res)
        if (is_error or err) and err != DECLINED:
            result.tool_errors.append({"name": tc.name, "error": err or str(res)[:500]})
        elif (
            tc.name == "skill_view"
            and isinstance(tc.arguments.get("name"), str)
            and tc.arguments["name"] not in result.skills_viewed
        ):
            result.skills_viewed.append(tc.arguments["name"])
        # record the result before yielding so a checkpoint taken on this event is complete
        content = await self._tool_content(res, tc.id)
        messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": content})
        if persist:
            await persist("tool", content, tool_call_id=tc.id, name=tc.name)
        yield ToolResultEvent(call_id=tc.id, name=tc.name, result=res, is_error=is_error, duration_ms=ms, **ev)

    async def _tool_content(self, res: Any, call_id: str) -> str:
        """The tool message the model reads. Long results are cut; the full text goes to files/outputs/."""
        content = _json_safe(res)
        limit = self.config.chat.tool_result_max_chars
        if not limit or len(content) <= limit:
            return content
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", call_id or "")[:80] or new_id()
        rel = f"outputs/tool-{safe}.txt"
        full = res if isinstance(res, str) else json.dumps(res, ensure_ascii=False, indent=2, default=str)

        def write() -> None:
            path = paths.files_dir() / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(full, encoding="utf-8")

        try:
            await asyncio.to_thread(write)
            note = (
                f"\n\n[Result cut to {limit} of {len(content)} characters. The full result is saved as files/{rel}; "
                f'read it with file_read(name="{rel}") if you need the rest.]'
            )
        except OSError as exc:
            log.warning("could not save long tool result: %s", exc)
            note = f"\n\n[Result cut to {limit} of {len(content)} characters.]"
        return content[:limit] + note

    # ------------------------------------------------------------------ chat turn
    async def _user_content(self, text: str, attachments: list[str]) -> str | list[dict]:
        if not attachments:
            return text
        parts_text = [text] if text else []
        images: list[dict] = []
        root = paths.files_dir().resolve()
        for name in attachments:
            p = (root / name).resolve()
            if root not in p.parents or not p.is_file():
                parts_text.append(f"\n[Attachment {name} could not be found]")
                continue
            if is_image(p):
                mime = mimetypes.guess_type(p.name)[0] or "image/png"
                data = base64.b64encode(p.read_bytes()).decode()
                images.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}})
                parts_text.append(f"\n[Image attached: {name}]")
                continue
            try:
                body = await asyncio.to_thread(extract_text, p, MAX_ATTACHMENT_CHARS)
                parts_text.append(f"\n--- Attached file: {name} ---\n{body}\n--- end of {name} ---")
            except Exception as exc:
                parts_text.append(f"\n[Attachment {name} ({p.suffix}) could not be read: {exc}. It is saved as files/{name}.]")
        joined = "\n".join(parts_text).strip()
        if images:
            return [{"type": "text", "text": joined}, *images]
        return joined

    async def run_turn(
        self,
        session_id: str,
        user_text: str,
        *,
        channel: str = "desktop",
        attachments: list[str] | None = None,
        model: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        turn_id = new_id()
        ev = {"session_id": session_id, "turn_id": turn_id}
        attachments = attachments or []
        # register for steering; a second concurrent turn on the same session (rare) does not take it over
        steer: SteerQueue | None = None
        if not self.is_running(session_id):
            steer = SteerQueue()
            self._steers[session_id] = steer

        turn_task = asyncio.current_task()  # Stop everything cancels this task (halt)
        if turn_task is not None:
            self._turn_tasks.add(turn_task)

        def release() -> list[str]:
            if turn_task is not None:
                self._turn_tasks.discard(turn_task)
            if steer is None:
                return []
            left = steer.close()
            if self._steers.get(session_id) is steer:
                del self._steers[session_id]
            return left

        result = LoopResult()
        partial = ""
        try:
            await self.store.add_message(session_id, "user", user_text, attachments=attachments)
            await self.store.touch_session(session_id, title=(user_text or (attachments[0] if attachments else ""))[:60])

            session = await self.store.get_session(session_id)
            system = await self.system_prompt(user_text, channel, session)
            history = await self.store.recent_messages(session_id, self.config.chat.history_window)
            convo = history_to_openai(history)
            # replace the just-stored user message with the rich version (attachments inlined)
            if convo and convo[-1]["role"] == "user":
                convo[-1] = {"role": "user", "content": await self._user_content(user_text, attachments)}
            # spoken channels use the voice role: its own model and reasoning effort (default: thinking off)
            role = "voice" if channel in SPOKEN_CHANNELS else "primary"
            if attachments and any(is_image(paths.files_dir() / a) for a in attachments) and self.config.models.roles.vision:
                role = "vision"
            messages: list[dict] = [{"role": "system", "content": system}, *convo]
            ctx = self.tool_context(session_id, channel)
            recent_plugins: set[str] = set()
            for row in history[-12:]:
                for tc in row.get("tool_calls") or []:
                    t = self.registry.get((tc.get("function") or {}).get("name", ""))
                    if t is not None:
                        recent_plugins.add(t.plugin)
            try:
                tool_names = await self.tool_selector.select(
                    user_text, model=model or self.llm.model_for(role), recent_plugins=recent_plugins
                )
            except Exception as exc:  # selection must never break a turn
                log.warning("tool selection failed, offering all tools: %s", exc)
                tool_names = None

            async def persist(role_: str, content: str | None, **fields: Any) -> None:
                await self.store.add_message(session_id, role_, content, **fields)

            async for event in self.run_loop(
                messages, ctx, result=result, role=role, model=model, tool_names=tool_names,
                persist=persist, source="chat", ev=ev, steer=steer,
            ):
                if isinstance(event, TextDelta):
                    partial += event.text
                elif isinstance(event, ToolResultEvent | UserInterjection):
                    partial = ""  # text before a tool call or a steer was already persisted
                yield event
        except (asyncio.CancelledError, GeneratorExit):
            release()
            # the user pressed Stop (or the window went away): keep what was shown
            with contextlib.suppress(Exception):
                await asyncio.shield(
                    self.store.add_message(session_id, "assistant", (partial.rstrip() + "\n\n_(stopped)_").strip())
                )
            raise
        except BaseException:
            release()
            raise
        leftover = release()

        if not (result.error and not result.text):
            message_id = await self.store.add_message(
                session_id, "assistant", result.text, thinking=result.thinking or None
            )
            yield Done(content=result.text, message_id=message_id, **ev)

            said = "\n\n".join([user_text, *result.interjections]).strip()
            if self.memory is not None and self.config.memory.extract_after_turn:
                self._spawn(self._extract(said))
            if self.config.chat.auto_title and session and (session.get("title") or "") == (user_text or "")[:60]:
                self._spawn(self._auto_title(session_id, user_text, result.text))
            self._spawn(self._maybe_compress(session_id))
            if self.app is not None:
                self.app.bus.publish(
                    "chat.turn_completed",
                    {
                        "session_id": session_id,
                        "turn_id": turn_id,
                        "tool_calls": result.tool_calls,
                        "tool_errors": result.tool_errors,
                        "skills_viewed": result.skills_viewed,
                        "user_text": said,
                        "reply": result.text,
                    },
                )

        if leftover:
            # steer messages that arrived after the final round start a new turn
            async for event in self.run_turn(session_id, "\n\n".join(leftover), channel=channel, model=model):
                yield event

    # ------------------------------------------------------------------ tools
    @staticmethod
    def _raw_arguments_error(tool: Tool, tc: ToolCall) -> tuple[Any, bool, int]:
        schema = tool.openai_schema()["function"]["parameters"]
        return (
            {
                "error": "Your arguments were not valid JSON. Call the tool again with a JSON object "
                "matching this schema.",
                "received": str(tc.arguments["_raw"])[:500],
                "schema": schema,
            },
            True,
            0,
        )

    async def _run_tool(self, tc: ToolCall, ctx: ToolContext) -> tuple[Any, bool, int]:
        tool = self.registry.get(tc.name)
        if tool is None:
            return {"error": f"unknown tool {tc.name}"}, True, 0
        if self.approvals.is_never(tool):  # a "never" rule set while this call waited for approval still wins
            return {"error": never_message(self.approvals.label(tool, self.registry))}, True, 0
        started = time.perf_counter()
        if "_raw" in tc.arguments and len(tc.arguments) == 1:
            return self._raw_arguments_error(tool, tc)
        try:
            result = await tool.call(ctx, tc.arguments)
            return result, False, int((time.perf_counter() - started) * 1000)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(x) for x in e.get('loc', ())) or 'arguments'}: {e.get('msg')}" for e in exc.errors()[:5]
            )
            return (
                {
                    "error": f"Invalid arguments for {tc.name}: {problems}. Call it again with arguments matching the schema.",
                    "schema": tool.openai_schema()["function"]["parameters"],
                },
                True,
                int((time.perf_counter() - started) * 1000),
            )
        except Exception as exc:
            log.exception("tool %s failed", tc.name)
            return {"error": f"{type(exc).__name__}: {exc}"}, True, int((time.perf_counter() - started) * 1000)

    # ------------------------------------------------------------------ background
    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    def _publish_memory(self, results: Any) -> None:
        if self.app is None or not isinstance(results, list):
            return
        for r in results:
            if isinstance(r, dict) and r.get("action") in {"ADD", "UPDATE", "DELETE"}:
                self.app.bus.publish("memory.updated", {"action": r["action"], "id": r.get("id"), "content": r.get("content")})

    async def _extract(self, user_text: str) -> None:
        """Only the user's own words are mined for facts; the assistant's reply is
        mostly restated context (dates, tool output) and produced junk memories."""
        assert self.memory is not None
        if len(user_text.split()) < 4:
            return
        try:
            results = await self.memory.extract_and_store(user_text, self.config.assistant.user_name)
            self._publish_memory(results)
        except Exception as exc:
            log.warning("background extraction failed: %s", exc)

    async def _auto_title(self, session_id: str, user_text: str, reply: str) -> None:
        try:
            title = await self.llm.complete_text(
                "fast",
                [
                    {"role": "system", "content": "Write a 2-6 word title for this chat. Title case. No quotes, no punctuation at the end. Reply with the title only."},
                    {"role": "user", "content": f"User: {user_text[:500]}\nAssistant: {reply[:500]}"},
                ],
            )
            title = title.strip().strip('"').strip()[:80]
            if title:
                await self.store.rename_session(session_id, title)
                if self.app is not None:
                    self.app.bus.publish("session.updated", {"session_id": session_id, "title": title})
        except Exception as exc:
            log.debug("auto title failed: %s", exc)

    async def _flush_memory(self, transcript: str) -> None:
        """Let the memory package keep durable facts from turns about to be folded into a summary."""
        flush = getattr(self.memory, "flush_conversation", None) if self.memory is not None else None
        if not callable(flush):
            return
        try:
            results = await flush(transcript, self.config.assistant.user_name)
            self._publish_memory(results)
        except Exception as exc:
            log.warning("memory flush before compression failed: %s", exc)

    async def _maybe_compress(self, session_id: str) -> None:
        """Hermes-style context compression: fold turns that fell out of the history
        window into a running summary injected into the system prompt."""
        try:
            total = await self.store.count_messages(session_id)
            window = self.config.chat.history_window
            if total <= max(self.config.chat.compress_after_messages, window + 10):
                return
            session = await self.store.get_session(session_id)
            upto = (session or {}).get("context_upto") or ""
            rows = await self.store.fetchall(
                "SELECT role, content, created_at FROM messages WHERE session_id = ? AND created_at > ?"
                " AND role IN ('user','assistant') AND content IS NOT NULL ORDER BY created_at",
                (session_id, upto),
            )
            older = rows[: max(0, len(rows) - window)]
            if len(older) < 10:
                return
            transcript = "\n".join(f"{r['role']}: {(r['content'] or '')[:800]}" for r in older)
            await self._flush_memory(transcript)
            prev = (session or {}).get("context_summary") or ""
            summary = await self.llm.complete_text(
                "fast",
                [
                    {"role": "system", "content": "Update the running summary of a conversation. Keep decisions, facts, names, dates, open questions and anything the assistant promised. Max 250 words. Plain text."},
                    {"role": "user", "content": f"Current summary:\n{prev or '(none)'}\n\nNew turns:\n{transcript}"},
                ],
            )
            await self.store.execute(
                "UPDATE sessions SET context_summary = ?, context_upto = ? WHERE id = ?",
                (summary.strip(), older[-1]["created_at"], session_id),
            )
        except Exception as exc:
            log.debug("context compression skipped: %s", exc)

    def drop_queued(self) -> dict[str, list[str]]:
        """Stop everything: take the messages waiting for running replies (steers) so they are never sent.
        Returns ``{session_id: [text]}``. A message sent after this starts a new turn as usual."""
        dropped: dict[str, list[str]] = {}
        for session_id, queue in list(self._steers.items()):
            if items := queue.close():
                dropped[session_id] = items
        return dropped

    async def halt(self) -> int:
        """Stop everything: cancel every running chat reply (the caller's own task excepted) and background
        job (memory notes, titles). Returns how many replies were cancelled."""
        replies = await cancel_tasks(self._turn_tasks)
        await cancel_tasks(self._background)
        return replies

    async def drain(self, timeout: float | None = None) -> None:
        """Wait for background work (fact extraction, titles, compression).

        With ``timeout``, anything still running after that many seconds is cancelled, so shutting
        Sentient down can never hang on a stuck model call.
        """
        loop = asyncio.get_running_loop()
        deadline = None if timeout is None else loop.time() + timeout
        while self._background:
            pending = list(self._background)
            remaining = None if deadline is None else deadline - loop.time()
            if remaining is not None and remaining <= 0:
                log.warning("cancelling %d background job(s) still running at shutdown", len(pending))
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                self._background.difference_update(pending)
                return
            done, _ = await asyncio.wait(pending, timeout=remaining)
            self._background.difference_update(done)
