"""One local model job at a time, chats first, background work yields (#149)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import litellm
import pytest
from fastapi.testclient import TestClient

from sentient.app import SentientApp
from sentient.config.schema import SentientConfig
from sentient.gateway.app import create_app
from sentient.llm import jobs as jobs_mod
from sentient.llm.jobs import ModelJobs, as_kind, current_kind, detached, is_local
from sentient.llm.provider import LiteLLMProvider
from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool
from tests.conftest import FakeProvider, tool_call

LOCAL = "ollama_chat/qwen3:8b"
WAIT = 3.0  # seconds before a test calls it a deadlock


def make_jobs(*, quiet: int = 0, battery: bool = False, on_battery: bool = False, **kw) -> ModelJobs:
    models = SentientConfig().models
    models.background_quiet_s = quiet
    models.background_on_battery = on_battery
    state = {"battery": battery}
    jobs = ModelJobs(lambda: models, on_battery=lambda: state["battery"], **kw)
    jobs.test_battery = state  # type: ignore[attr-defined]
    jobs.test_models = models  # type: ignore[attr-defined]
    return jobs


class Meter:
    """Counts how many local calls run at once."""

    def __init__(self) -> None:
        self.now = 0
        self.most = 0
        self.order: list[str] = []

    async def use(self, jobs: ModelJobs, name: str, seconds: float = 0.01, model: str = LOCAL) -> None:
        async with jobs.slot(model):
            self.now += 1
            self.most = max(self.most, self.now)
            self.order.append(name)
            await asyncio.sleep(seconds)
            self.now -= 1


async def started(*coros) -> list[asyncio.Task]:
    """Start tasks in this order, each queued before the next."""
    tasks = []
    for coro in coros:
        tasks.append(asyncio.create_task(coro))
        await asyncio.sleep(0)
    return tasks


def as_task(kind: str, coro) -> asyncio.Task:
    return asyncio.create_task(coro, context=detached(kind))


# ---------------------------------------------------------------------------- the queue
async def test_local_calls_never_overlap_and_cloud_calls_never_wait():
    jobs, meter = make_jobs(), Meter()
    await asyncio.gather(*(meter.use(jobs, f"local{i}") for i in range(6)))
    assert meter.most == 1 and len(meter.order) == 6

    cloud = Meter()
    await asyncio.gather(*(cloud.use(jobs, f"cloud{i}", 0.05, "anthropic/claude-sonnet-5") for i in range(4)))
    assert cloud.most == 4  # no queue for cloud models
    assert is_local("ollama/llama3.2:3b") and not is_local("openai/gpt-5") and not is_local(None)


async def test_turned_off_the_queue_lets_calls_overlap():
    jobs, meter = make_jobs(), Meter()
    jobs.test_models.local_queue = False  # type: ignore[attr-defined]
    await asyncio.gather(*(meter.use(jobs, f"local{i}", 0.05) for i in range(3)))
    assert meter.most == 3


async def test_chat_jumps_the_queue_then_people_then_tasks_then_upkeep():
    jobs, meter = make_jobs(), Meter()
    holder = as_task("memory", meter.use(jobs, "running", 0.05))
    await asyncio.sleep(0.01)
    waiting = [
        as_task("titles", meter.use(jobs, "titles")),
        as_task("task", meter.use(jobs, "task")),
        as_task("memory", meter.use(jobs, "memory")),
        as_task("interactive", meter.use(jobs, "interactive")),
        as_task("chat", meter.use(jobs, "chat")),
        as_task("task", meter.use(jobs, "task2")),
    ]
    await asyncio.gather(holder, *waiting)
    # the running call finishes (nothing is cut off); the chat goes next; same rank keeps arrival order
    assert meter.order == ["running", "chat", "interactive", "task", "task2", "titles", "memory"]
    assert meter.most == 1


async def test_unknown_callers_count_as_someone_waiting():
    assert current_kind() == "interactive"
    with as_kind("task"):
        assert current_kind() == "task"
    assert current_kind() == "interactive"


# ---------------------------------------------------------------------------- background work yields
async def test_background_waits_while_you_chat_and_until_it_has_been_quiet():
    jobs, meter = make_jobs(quiet=1), Meter()
    loop = asyncio.get_running_loop()
    with jobs.chat_turn():
        task = as_task("task", meter.use(jobs, "task"))
        await asyncio.sleep(0.1)
        assert meter.order == [] and jobs.status()["deferred_reason"] == "chat"
        await meter.use(jobs, "chat")  # the chat itself runs at once
        interactive = asyncio.create_task(meter.use(jobs, "button"))  # a person waiting is never held back
        await asyncio.wait_for(interactive, WAIT)
    ended = loop.time()
    await asyncio.wait_for(task, WAIT)
    assert loop.time() - ended >= 0.9  # the quiet time after the reply
    assert meter.order == ["chat", "button", "task"]


async def test_background_waits_on_battery_but_chats_do_not(monkeypatch):
    monkeypatch.setattr(jobs_mod, "RECHECK_S", 0.05)
    jobs, meter = make_jobs(battery=True), Meter()
    task = as_task("suggestions", meter.use(jobs, "suggestions"))
    await asyncio.sleep(0.1)
    assert meter.order == [] and jobs.status() == {
        "busy": False, "job": None, "model": None, "since": None, "waiting": 0, "deferred": 1,
        "deferred_reason": "battery",
    }
    await asyncio.wait_for(as_task("chat", meter.use(jobs, "chat")), WAIT)
    jobs.test_battery["battery"] = False  # type: ignore[attr-defined]  plugged in: it goes on the next check
    await asyncio.wait_for(task, WAIT)
    assert meter.order == ["chat", "suggestions"]


async def test_background_on_battery_can_be_allowed():
    jobs, meter = make_jobs(battery=True, on_battery=True), Meter()
    await asyncio.wait_for(as_task("memory", meter.use(jobs, "memory")), WAIT)


# ---------------------------------------------------------------------------- no deadlocks
async def test_a_call_inside_a_held_slot_does_not_wait_on_itself():
    jobs, meter = make_jobs(), Meter()

    async def outer() -> None:
        async with jobs.slot(LOCAL):
            await meter.use(jobs, "nested")  # e.g. between a stream's chunks

    await asyncio.wait_for(outer(), WAIT)
    assert meter.order == ["nested"]


async def test_a_stale_slot_from_a_finished_call_does_not_skip_the_queue():
    jobs = make_jobs()
    leaked: list = []

    async def call() -> None:
        async with jobs.slot(LOCAL):
            leaked.append(asyncio.create_task(asyncio.sleep(0)))  # a task started while holding copies the slot

    await call()
    meter = Meter()
    other = asyncio.create_task(meter.use(jobs, "other", 0.1))
    await asyncio.sleep(0.01)
    # a later call from that copied context waits like anyone else: the slot it remembers was released
    await asyncio.wait_for(meter.use(jobs, "late"), WAIT)
    await other
    assert meter.most == 1


async def test_work_a_chat_waits_on_is_not_held_back_by_that_chat():
    jobs, meter = make_jobs(quiet=60), Meter()
    with jobs.chat_turn():
        # a tool in the reply runs memory work inline, or starts a helper task and waits for it
        with as_kind("memory"):
            await asyncio.wait_for(meter.use(jobs, "inline memory"), WAIT)
        helper = asyncio.create_task(meter.use(jobs, "helper"))
        await asyncio.wait_for(helper, WAIT)
        # work started from the reply that nobody waits on waits for a quiet moment
        later = as_task("titles", meter.use(jobs, "title"))
        await asyncio.sleep(0.1)
        assert "title" not in meter.order
    later.cancel()
    assert meter.order == ["inline memory", "helper"]


async def test_safety_valve_when_a_reply_stops_using_the_model(monkeypatch):
    monkeypatch.setattr(jobs_mod, "STALL_S", 0.2)
    jobs, meter = make_jobs(quiet=60), Meter()
    with jobs.chat_turn():
        # the reply waits on something (a lock) held by background work that needs the model
        background = as_task("memory", meter.use(jobs, "memory"))
        await asyncio.wait_for(background, WAIT)
    assert meter.order == ["memory"]


# ---------------------------------------------------------------------------- cancellation
async def test_cancelled_waiters_and_holders_free_the_slot():
    jobs, meter = make_jobs(), Meter()
    holder = asyncio.create_task(meter.use(jobs, "holder", 10))
    await asyncio.sleep(0.01)
    waiter = asyncio.create_task(meter.use(jobs, "waiter", 10))
    await asyncio.sleep(0.01)
    waiter.cancel()
    holder.cancel()  # Stop everything
    await asyncio.gather(holder, waiter, return_exceptions=True)
    assert jobs.status()["busy"] is False
    await asyncio.wait_for(meter.use(jobs, "after"), WAIT)
    assert meter.order == ["holder", "after"]


async def test_a_waiter_cancelled_just_as_it_got_the_slot_hands_it_on():
    jobs, meter = make_jobs(), Meter()
    gate = asyncio.Event()

    async def hold() -> None:
        async with jobs.slot(LOCAL):
            await gate.wait()

    holder = asyncio.create_task(hold())
    await asyncio.sleep(0.01)
    waiter = asyncio.create_task(meter.use(jobs, "waiter"))
    await asyncio.sleep(0.01)
    gate.set()
    await holder  # the slot passes to the waiter's future ...
    waiter.cancel()  # ... which is cancelled before it ran
    await asyncio.gather(waiter, return_exceptions=True)
    await asyncio.wait_for(meter.use(jobs, "next"), WAIT)
    assert jobs.status()["busy"] is False


# ---------------------------------------------------------------------------- the busy signal
async def test_busy_signal_says_what_runs_and_what_waits():
    seen: list[dict] = []
    jobs = make_jobs(quiet=60, on_change=seen.append)
    gate = asyncio.Event()

    async def hold() -> None:
        async with jobs.slot(LOCAL):
            await gate.wait()

    holder = as_task("task", hold())
    await asyncio.sleep(0.3)
    assert seen[-1]["busy"] is True and seen[-1]["job"] == "task" and seen[-1]["model"] == LOCAL
    chat = as_task("chat", Meter().use(jobs, "chat"))
    await asyncio.sleep(0.3)
    assert seen[-1]["waiting"] == 1
    gate.set()
    await asyncio.gather(holder, chat)
    await asyncio.sleep(0.3)
    assert seen[-1]["busy"] is False and seen[-1]["waiting"] == 0
    assert all(s.keys() == {"busy", "job", "model", "since", "waiting", "deferred", "deferred_reason"} for s in seen)


# ---------------------------------------------------------------------------- the real provider
@pytest.fixture
def scripted(monkeypatch):
    """litellm returns slow streams; ``calls`` records when each call started and ended."""
    calls: list[tuple[str, str]] = []
    pauses: dict[str, asyncio.Event] = {}

    def chunk(text: str):
        return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text, reasoning_content=None))])

    async def acompletion(**kwargs):
        name = kwargs["messages"][-1]["content"]
        calls.append(("start", name))
        if kwargs.get("stream"):
            async def gen():
                try:
                    for word in ("one ", "two ", "three"):
                        if name in pauses:
                            await pauses[name].wait()
                        yield chunk(word)
                finally:
                    calls.append(("end", name))

            return gen()
        await asyncio.sleep(0.01)
        calls.append(("end", name))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))])

    monkeypatch.setattr(litellm, "acompletion", acompletion)
    monkeypatch.setattr(litellm, "stream_chunk_builder", lambda chunks, messages=None: None)
    monkeypatch.setattr("sentient.llm.provider.secrets.get_secret", lambda *a, **k: None)
    monkeypatch.setattr(LiteLLMProvider, "_model_max_context", lambda self, model, base: _none())
    return SimpleNamespace(calls=calls, pauses=pauses)


async def _none():
    return None


def local_provider() -> LiteLLMProvider:
    cfg = SentientConfig()
    cfg.models.roles.primary = LOCAL
    cfg.models.roles.fast = LOCAL
    prov = LiteLLMProvider(cfg)
    prov.jobs._on_battery = lambda: False
    return prov


def msgs(name: str) -> list[dict]:
    return [{"role": "user", "content": name}]


async def test_a_stream_holds_the_model_until_it_ends(scripted):
    prov = local_provider()
    scripted.pauses["stream"] = asyncio.Event()

    async def read() -> None:
        async for _ in prov.stream("primary", msgs("stream")):
            pass

    reader = asyncio.create_task(read())
    await asyncio.sleep(0.05)
    other = asyncio.create_task(prov.complete_text("fast", msgs("text")))
    await asyncio.sleep(0.05)
    assert scripted.calls == [("start", "stream")]  # the text call waits for the stream
    scripted.pauses["stream"].set()
    await asyncio.wait_for(asyncio.gather(reader, other), WAIT)
    assert scripted.calls == [("start", "stream"), ("end", "stream"), ("start", "text"), ("end", "text")]


async def test_the_model_is_free_before_the_last_chunk_and_between_chunks(scripted):
    prov = local_provider()
    seen = []
    async for chunk in prov.stream("primary", msgs("stream")):
        if chunk.text == "one ":
            seen.append(await asyncio.wait_for(prov.complete_text("fast", msgs("mid")), WAIT))  # nested: no wait
        if chunk.done:
            # the caller runs tools that call the model while this generator is still open; even a call that does
            # not share this context (another chat, a helper) gets the model now
            other = asyncio.create_task(prov.complete_json("fast", msgs("tool")), context=detached("chat"))
            seen.append(await asyncio.wait_for(other, WAIT))
    assert seen == ['{"ok": true}', {"ok": True}]


async def test_stopping_a_stream_frees_the_model(scripted):
    prov = local_provider()
    scripted.pauses["stream"] = asyncio.Event()

    async def read() -> None:
        async for _ in prov.stream("primary", msgs("stream")):
            pass

    reader = asyncio.create_task(read())
    await asyncio.sleep(0.05)
    assert prov.jobs.status()["busy"] is True
    reader.cancel()  # Stop everything cancels the reply
    await asyncio.gather(reader, return_exceptions=True)
    assert prov.jobs.status()["busy"] is False
    assert await asyncio.wait_for(prov.complete_text("fast", msgs("after")), WAIT) == '{"ok": true}'
    # the stopped request was closed (Ollama stops writing) before the next one started
    assert scripted.calls == [("start", "stream"), ("end", "stream"), ("start", "after"), ("end", "after")]


# ---------------------------------------------------------------------------- the app
class ScheduledFake(FakeProvider):
    """The scripted model behind the real scheduler, used the way LiteLLMProvider uses it."""

    def __init__(self, config, **kw):
        super().__init__(**kw)
        self.jobs = ModelJobs(lambda: config.models, on_battery=lambda: False)
        self.meter = Meter()
        self.kinds: list[str] = []

    async def stream(self, role, messages, tools=None, *, model=None):
        self.kinds.append(current_kind())
        async with self.jobs.slot(LOCAL):
            self.meter.now += 1
            self.meter.most = max(self.meter.most, self.meter.now)
            try:
                chunks = [c async for c in super().stream(role, messages, tools, model=model)]
                for c in chunks[:-1]:
                    yield c
                    await asyncio.sleep(0)
            finally:
                self.meter.now -= 1
        yield chunks[-1]

    async def complete_text(self, role, messages, *, model=None):
        self.kinds.append(current_kind())
        async with self.jobs.slot(LOCAL):
            return await super().complete_text(role, messages, model=model)


async def start_app(config, isolated_home, llm, name: str) -> SentientApp:
    return await SentientApp(config, llm=llm, db_path=isolated_home / f"{name}.db", enable_background=False).start()


async def test_a_tool_and_a_helper_in_a_chat_reply_call_the_model_without_deadlock(config, isolated_home):
    config.subagents.enabled = True
    llm = ScheduledFake(config, replies=[
        [tool_call("ask_the_model")],                                # chat round 1
        [tool_call("delegate_task", goal="Say hello")],            # chat round 2
        "Hello from the helper.",                                    # the helper (a subagent)
        "All done.",                                                 # chat final
    ])

    @tool("ask_the_model", risk=Risk.read)
    async def ask_the_model(ctx: ToolContext) -> str:
        """Ask the model something."""
        return await ctx.llm.complete_text("fast", [{"role": "user", "content": "hi"}])

    app = await start_app(config, isolated_home, llm, "nested")
    try:
        class P(ToolPlugin):
            id = "asker"
            display_name = "Asker"
            tools = [ask_the_model]

        app.registry.register(P())
        sid = await app.store.create_session(channel="desktop")

        async def turn() -> list:
            return [e async for e in app.agent.run_turn(sid, "go")]

        events = await asyncio.wait_for(turn(), 10)
        assert events[-1].type == "done" and events[-1].content == "All done."
        assert llm.meter.most == 1
        assert set(llm.kinds) == {"chat"}  # the tool and the helper count as the chat
        assert app.model_jobs._turns == 0 and app.model_jobs.status()["busy"] is False
    finally:
        await app.stop()


async def test_work_after_a_reply_and_task_runs_count_as_background(config, isolated_home):
    app = await start_app(config, isolated_home, FakeProvider(), "kinds")
    try:
        seen: list[str] = []

        async def record() -> None:
            seen.append(current_kind())

        with app.model_jobs.chat_turn():
            app.agent._spawn(record(), "titles")
            app.tasks._spawn(record(), "probe")
            await asyncio.sleep(0.05)
        assert seen == ["titles", "task"]
    finally:
        await app.stop()


async def test_services_start_their_loops_as_their_kind(config, isolated_home, monkeypatch):
    app = SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "svc.db", enable_background=False)
    seen: dict[str, str] = {}
    for svc in (app.tasks, app.proactivity, app.dreaming, app.channels):
        async def fake_start(name=svc.name) -> None:
            seen[name] = current_kind()

        monkeypatch.setattr(svc, "start", fake_start)
    await app.start()
    try:
        assert seen == {"tasks": "task", "proactivity": "suggestions", "dreaming": "memory", "channels": "interactive"}
        assert current_kind() == "interactive"  # nothing leaks out of start()
    finally:
        await app.stop()


async def test_busy_event_is_published(config, isolated_home):
    app = await start_app(config, isolated_home, FakeProvider(), "event")
    try:
        async with app.bus.subscribe() as q:
            app.model_jobs.on_change({"busy": True, "job": "task"})  # what the scheduler publishes
            event = await asyncio.wait_for(q.get(), 1)
        assert event["type"] == "model.busy" and event["data"]["job"] == "task"
    finally:
        await app.stop()


def test_busy_route(config, isolated_home, monkeypatch):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "test-token")
    core = SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "route.db", enable_background=False)
    with TestClient(create_app(core)) as c:
        c.headers.update({"Authorization": "Bearer test-token"})
        assert c.get("/api/models/busy").json() == {
            "busy": False, "job": None, "model": None, "since": None, "waiting": 0, "deferred": 0,
            "deferred_reason": None,
        }


# ---------------------------------------------------------------------------- battery
def test_linux_power_status(tmp_path):
    from sentient.llm import power

    def supply(name: str, **files: str) -> None:
        (tmp_path / name).mkdir()
        for key, value in files.items():
            (tmp_path / name / key).write_text(value + "\n")

    assert power._linux(tmp_path / "missing") is None
    supply("BAT0", type="Battery", status="Discharging")
    assert power._linux(tmp_path) is True
    supply("AC", type="Mains", online="1")
    assert power._linux(tmp_path) is False  # plugged in wins


def test_battery_reading_is_remembered_and_never_raises(monkeypatch):
    from sentient.llm import power

    calls = []

    def probe():
        calls.append(1)
        raise OSError("no power info")

    monkeypatch.setattr(power, "_probe", probe)
    monkeypatch.setattr(power, "_cache", None)
    assert power.on_battery() is False and power.on_battery() is False
    assert len(calls) == 1
