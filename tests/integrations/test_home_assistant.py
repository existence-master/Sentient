"""Home Assistant (#175): connecting, finding devices, risk levels, quick light commands and live updates.

REST calls are mocked with respx. The websocket (rooms, devices and live updates) is a small server on 127.0.0.1 that
speaks Home Assistant's websocket protocol, so nothing leaves this computer.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from websockets.asyncio.server import serve

from sentient.agent import fastpath, outcomes
from sentient.agent.fastpath import en
from sentient.app import SentientApp
from sentient.gateway.app import create_app
from sentient.integrations.base import IntegrationError
from sentient.integrations.plugins import home_assistant as ha
from sentient.llm.events import ApprovalRequest, Done, TextDelta, ToolCallEvent, ToolResultEvent
from sentient.tools.base import Risk
from tests.conftest import FakeProvider

TOKEN = "fake-maya-token"


def _state(eid: str, state: str, name: str, **attrs) -> dict:
    return {"entity_id": eid, "state": state, "attributes": {"friendly_name": name, **attrs},
            "last_changed": "2026-10-15T08:00:00+00:00"}


STATES = [
    _state("light.living_room_ceiling", "on", "Living room ceiling", brightness=200),
    _state("light.living_room_floor_lamp", "on", "Living room floor lamp"),
    _state("light.bedroom_lamp", "off", "Bedroom lamp"),
    _state("light.bedroom_strip", "off", "Bedroom strip"),
    _state("climate.hallway_thermostat", "heat", "Hallway thermostat", temperature=20, current_temperature=19.5),
    _state("lock.front_door", "locked", "Front door"),
    _state("cover.garage_door", "closed", "Garage door", device_class="garage"),
    _state("cover.bedroom_blinds", "open", "Bedroom blinds", device_class="blind"),
    _state("switch.garage_opener", "off", "Garage opener"),
    _state("switch.coffee_maker", "off", "Coffee maker"),
    _state("sensor.power", "120", "Power use"),
    _state("scene.movie_night", "scening", "Movie night"),
]
REGISTRY = {
    "areas": [{"area_id": "living_room", "name": "Living room"}, {"area_id": "bedroom", "name": "Bedroom"},
              {"area_id": "hallway", "name": "Hallway"}],
    "devices": [{"id": "d1", "name": "Hue ceiling", "area_id": "living_room"},
                {"id": "d2", "name": "Nest", "area_id": "hallway", "name_by_user": "Thermostat"}],
    "entities": [
        {"entity_id": "light.living_room_ceiling", "device_id": "d1"},
        {"entity_id": "light.living_room_floor_lamp", "area_id": "living_room"},
        {"entity_id": "light.bedroom_lamp", "area_id": "bedroom"},
        {"entity_id": "light.bedroom_strip", "area_id": "bedroom"},
        {"entity_id": "climate.hallway_thermostat", "device_id": "d2"},
        {"entity_id": "lock.front_door", "area_id": "hallway"},
        {"entity_id": "switch.hidden_thing", "hidden_by": "user"},
    ],
}


class FakeHA:
    """respx routes for the REST API, recording service calls."""

    def __init__(self, router: respx.Router, url: str = "http://homeassistant.local:8123"):
        self.url = url
        self.calls: list[tuple[str, dict]] = []
        self.states = [dict(s) for s in STATES]
        self.ignore: set[str] = set()  # devices that take the call and do nothing (#386)
        self.lag: dict[str, int] = {}  # devices that report the change only after this many more reads
        self.pending: dict[str, str] = {}
        self.reads: list[str] = []

        def authed(request: httpx.Request) -> bool:
            return request.headers.get("authorization") == f"Bearer {TOKEN}"

        def config(request):
            if not authed(request):
                return httpx.Response(401, text="401: Unauthorized")
            return httpx.Response(200, json={"location_name": "Maya's home", "version": "2026.10.0"})

        def states(request):
            return httpx.Response(200, json=self.states) if authed(request) else httpx.Response(401)

        def one(request, entity_id):
            self.reads.append(entity_id)
            hit = next((s for s in self.states if s["entity_id"] == entity_id), None)
            if hit and entity_id in self.pending:
                self.lag[entity_id] = self.lag.get(entity_id, 0) - 1
                if self.lag[entity_id] < 0:
                    hit["state"] = self.pending.pop(entity_id)
            return httpx.Response(200, json=hit) if hit else httpx.Response(404, json={"message": "Entity not found."})

        def service(request, domain, service):
            body = json.loads(request.content or b"{}")
            self.calls.append((f"{domain}.{service}", body))
            changed = []
            for s in self.states:
                if s["entity_id"] in body.get("entity_id", []) and s["entity_id"] not in self.ignore:
                    after = {"turn_off": "off", "turn_on": "on", "unlock": "unlocked", "lock": "locked"}.get(
                        service, s["state"])
                    if self.lag.get(s["entity_id"]):
                        self.pending[s["entity_id"]] = after
                        continue
                    s["state"] = after
                    changed.append(s)
            return httpx.Response(200, json=changed)

        router.get(f"{url}/api/config").mock(side_effect=config)
        router.get(f"{url}/api/states").mock(side_effect=states)
        router.get(url__regex=rf"{url}/api/states/(?P<entity_id>[^/]+)$").mock(side_effect=one)
        router.post(url__regex=rf"{url}/api/services/(?P<domain>\w+)/(?P<service>\w+)$").mock(side_effect=service)


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    ha.forget_cache()

    async def registry(c):  # the websocket part has its own test below
        return REGISTRY

    monkeypatch.setattr(ha, "_registry", registry)
    monkeypatch.setattr(ha, "SETTLE_S", 0.0)  # the read-back's waits, kept short here
    monkeypatch.setattr(ha, "RETRY_S", 0.01)
    yield
    ha.forget_cache()


async def _connected(app: SentientApp) -> None:
    await app.integrations.connect("home_assistant", {"url": "homeassistant.local:8123", "token": TOKEN})


async def _ha_app(config, isolated_home, llm=None, mode: str = "ask") -> SentientApp:
    config.chat.tool_selection = "all"
    config.tools.approvals.mode = mode
    return await SentientApp(config, llm=llm or FakeProvider(), db_path=isolated_home / "ha.db",
                             enable_background=False).start()


async def _turn(s: SentientApp, sid: str, text: str, decision: str = "allow") -> tuple[list[ApprovalRequest], list]:
    asked, events = [], []
    async for ev in s.agent.run_turn(sid, text, channel="desktop"):
        events.append(ev)
        if isinstance(ev, ApprovalRequest):
            asked.append(ev)
            s.approvals.resolve(ev.approval_id, decision)
    return asked, events


def _reply(events: list) -> str:
    return "".join(e.text for e in events if isinstance(e, TextDelta))


# ---------------------------------------------------------------------------- address, token, connect
def test_addresses_are_cleaned_and_plain_http_stays_at_home():
    assert ha.check_url("homeassistant.local:8123/") == "http://homeassistant.local:8123"
    assert ha.check_url("http://192.168.1.20:8123/api") == "http://192.168.1.20:8123"
    assert ha.check_url("https://maya.duckdns.org") == "https://maya.duckdns.org"
    assert ha.check_url("http://127.0.0.1:18123") == "http://127.0.0.1:18123"
    assert ha.ws_url("https://maya.duckdns.org") == "wss://maya.duckdns.org/api/websocket"
    with pytest.raises(IntegrationError, match="https"):
        ha.check_url("http://maya.duckdns.org:8123")
    with pytest.raises(IntegrationError, match="https"):
        ha.check_url("http://[2606:4700::1111]:8123")  # a public IPv6 address has no dots but is not at home
    assert ha.check_url("http://[fe80::1]:8123") == "http://[fe80::1]:8123"
    with pytest.raises(IntegrationError):
        ha.check_url("")


async def test_connect_saves_url_and_token_in_the_keychain(app, keychain):
    with respx.mock(assert_all_called=False) as router:
        FakeHA(router)
        out = await app.integrations.connect("home_assistant", {"url": "homeassistant.local:8123", "token": TOKEN})
    assert out["connected"] and out["account_label"] == "Maya's home" and out["category"] == "home"
    assert out["entity_picker"] is True
    assert {f["key"]: f["secret"] for f in out["setup"]["fields"]} == {"url": False, "token": True}
    assert json.loads(keychain["integration:home_assistant"]) == {"url": "http://homeassistant.local:8123", "token": TOKEN}
    assert out["triggers"] == [{"event": "state_changed", "label": "Device change"}]
    risks = {t["name"]: t["risk"] for t in out["tools"]}
    assert risks["home_assistant_find_entities"] == "read" and risks["home_assistant_turn_off"] == "write"
    assert risks["home_assistant_unlock"] == "send" and risks["home_assistant_set_alarm"] == "send"


async def test_a_wrong_token_says_what_to_do(app):
    with respx.mock(assert_all_called=False) as router:
        FakeHA(router)
        with pytest.raises(IntegrationError, match="long-lived access token"):
            await app.integrations.connect("home_assistant", {"url": "homeassistant.local:8123", "token": "nope"})
    assert not (await app.integrations.integration("home_assistant"))["connected"]


# ---------------------------------------------------------------------------- tools
async def test_find_entities_narrows_down_and_never_dumps_everything(app, ctx):
    with respx.mock(assert_all_called=False) as router:
        FakeHA(router)
        await _connected(app)
        tools = {t.name: t for t in ha.PLUGIN.tools}
        res = await tools["home_assistant_find_entities"].call(ctx, {"area": "living room", "kind": "lights"})
        assert [e["entity_id"] for e in res["entities"]] == ["light.living_room_ceiling", "light.living_room_floor_lamp"]
        res = await tools["home_assistant_find_entities"].call(ctx, {"search": "thermostat"})
        assert res["entities"] == [{"entity_id": "climate.hallway_thermostat", "name": "Hallway thermostat",
                                    "state": "heat", "area": "Hallway"}]
        res = await tools["home_assistant_find_entities"].call(ctx, {"limit": 3})
        assert len(res["entities"]) == 3 and res["more"] == len(STATES) - 3 and "Narrow" in res["hint"]
        areas = await tools["home_assistant_list_areas"].call(ctx, {})
        assert {"name": "Bedroom", "kinds": {"light": 2}} in areas["areas"]
        state = await tools["home_assistant_get_state"].call(ctx, {"entity_id": "light.living_room_ceiling"})
        assert state["state"] == "on" and state["details"]["brightness_pct"] == 78
        # the picker limits what Sentient may see and use
        await app.integrations.set_entity_choice("home_assistant", "picked", ["light.bedroom_lamp"])
        res = await tools["home_assistant_find_entities"].call(ctx, {})
        assert [e["entity_id"] for e in res["entities"]] == ["light.bedroom_lamp"]
        res = await tools["home_assistant_turn_off"].call(ctx, {"entity_id": "light.living_room_ceiling"})
        assert "isn't allowed" in res["error"]


async def test_changes_call_the_right_services(app, ctx):
    with respx.mock(assert_all_called=False) as router:
        fake = FakeHA(router)
        await _connected(app)
        tools = {t.name: t for t in ha.PLUGIN.tools}
        res = await tools["home_assistant_turn_off"].call(
            ctx, {"entity_id": "light.living_room_ceiling, light.living_room_floor_lamp"})
        assert res["done"] and fake.calls[-1] == (
            "light.turn_off", {"entity_id": ["light.living_room_ceiling", "light.living_room_floor_lamp"]})
        await tools["home_assistant_turn_on"].call(ctx, {"entity_id": "light.bedroom_lamp", "brightness_pct": 140})
        assert fake.calls[-1] == ("light.turn_on", {"entity_id": ["light.bedroom_lamp"], "brightness_pct": 100})
        await tools["home_assistant_set_temperature"].call(ctx, {"entity_id": "climate.hallway_thermostat",
                                                                 "temperature": 22})
        assert fake.calls[-1] == ("climate.set_temperature", {"entity_id": ["climate.hallway_thermostat"],
                                                              "temperature": 22.0})
        await tools["home_assistant_activate_scene"].call(ctx, {"entity_id": "scene.movie_night"})
        assert fake.calls[-1][0] == "scene.turn_on"
        # a lock is never switched with turn on or off: it has its own tools, which always ask
        res = await tools["home_assistant_turn_on"].call(ctx, {"entity_id": "lock.front_door"})
        assert "home_assistant_unlock" in res["error"]
        await tools["home_assistant_unlock"].call(ctx, {"entity_id": "lock.front_door"})
        assert fake.calls[-1] == ("lock.unlock", {"entity_id": ["lock.front_door"]})
        res = await tools["home_assistant_set_alarm"].call(ctx, {"entity_id": "lock.front_door", "mode": "disarm"})
        assert "isn't an alarm" in res["error"]


async def test_security_things_ask_every_time():
    tools = {t.name: t for t in ha.PLUGIN.tools}
    # nothing cached yet: anything that could be a door asks, a light never does
    assert tools["home_assistant_open_cover"].risk_fn({"entity_id": "cover.main_entrance"}, None) == Risk.send
    assert tools["home_assistant_turn_off"].risk_fn({"entity_id": "light.porch"}, None) == Risk.write
    ha._CACHE["http://x"] = {"states": (1e18, STATES)}
    tools = {t.name: t for t in ha.PLUGIN.tools}
    for name in ("home_assistant_lock", "home_assistant_unlock", "home_assistant_set_alarm"):
        assert tools[name].risk == Risk.send and tools[name].allow_for_chat is False
    cover = tools["home_assistant_open_cover"]
    assert cover.risk_fn({"entity_id": "cover.garage_door"}, None) == Risk.send
    assert cover.risk_fn({"entity_id": "cover.bedroom_blinds"}, None) == Risk.write
    off = tools["home_assistant_turn_off"]
    assert off.risk_fn({"entity_id": "switch.garage_opener"}, None) == Risk.send  # sounds like a garage door
    assert off.risk_fn({"entity_id": "switch.coffee_maker, light.bedroom_lamp"}, None) == Risk.write
    assert await tools["home_assistant_unlock"].describe_fn({"entity_id": "lock.front_door"}, None) == {
        "risk_label": "Unlocks a door", "target": "Front door"}


# ---------------------------------------------------------------------------- quick commands (#356)
HOME_PHRASES = [
    ("turn off the living room lights", "off", "living room lights", "light", True, False, None),
    ("Turn on the bedroom lamp, please", "on", "bedroom lamp", "light", False, False, None),
    ("switch off all the lights", "off", "lights", "light", True, True, None),
    ("living room lights off", "off", "living room lights", "light", True, False, None),
    ("turn the lights in the living room off", "off", "living room lights", "light", True, False, None),
    ("dim the living room lights to 30%", "on", "living room lights", "light", True, False, 30),
    ("set the bedroom lamp to 50 percent", "on", "bedroom lamp", "light", False, False, 50),
    ("hey Sentient turn on the coffee maker", "on", "coffee maker", "", False, False, None),
]
NOT_HOME = ["is the porch light on", "unlock the front door", "turn off the alarm", "lock the front door",
            "turn off notifications", "set the thermostat to 22", "turn off everything", "turn off the timer",
            "dim the lights", "leave the lights on"]


@pytest.mark.parametrize(("text", "power", "which", "device", "plural", "everything", "level"), HOME_PHRASES)
def test_home_phrasings(text, power, which, device, plural, everything, level):
    cmd = en.parse(text, None, ["Sentient"])  # type: ignore[arg-type]
    assert cmd is not None and cmd.action == "home"
    assert (cmd.power, cmd.which, cmd.device, cmd.plural, cmd.all, cmd.level) == (
        power, which, device, plural, everything, level)


@pytest.mark.parametrize("text", NOT_HOME)
def test_not_a_home_command(text):
    cmd = en.parse(text, None, ["Sentient"])  # type: ignore[arg-type]
    assert cmd is None or cmd.action != "home"


def test_matching_is_exact_or_it_goes_to_the_model():
    index = ha.build_index(STATES, REGISTRY)
    ids = lambda found: [e["entity_id"] for e in found or []]  # noqa: E731
    assert ids(ha.match(index, "living room lights", device="light", plural=True)) == [
        "light.living_room_ceiling", "light.living_room_floor_lamp"]
    assert ids(ha.match(index, "bedroom lamp", device="light", plural=False)) == ["light.bedroom_lamp"]
    assert ha.match(index, "bedroom light", device="light", plural=False) is None  # two lights: which one?
    assert len(ha.match(index, "bedroom lights", device="light", plural=True) or []) == 2
    assert len(ha.match(index, "lights", device="light", plural=True, everywhere=True) or []) == 4
    assert ha.match(index, "lights", device="light", plural=True) is None  # which room?
    assert ids(ha.match(index, "coffee maker", device="", plural=False)) == ["switch.coffee_maker"]
    assert ha.match(index, "garage opener", device="", plural=False) is None  # sounds like a garage door
    assert ha.match(index, "front door", device="", plural=False) is None  # a lock
    assert ha.match(index, "kitchen lights", device="light", plural=True) is None
    # "all the bedroom lights" means both, even when one of them is named exactly that
    named = [*index, {**index[2], "entity_id": "light.bedroom_lights", "name": "Bedroom lights"}]
    assert len(ha.match(named, "bedroom lights", device="light", plural=True) or []) == 1
    assert len(ha.match(named, "bedroom lights", device="light", plural=True, everywhere=True) or []) == 3
    assert en.home_done(en.parse("turn off the living room lights", None), 2) == "Turned off the living room lights."
    assert en.home_done(en.parse("switch off all the lights", None), 4) == "Turned off all the lights."
    assert en.home_done(en.parse("dim my bedroom lamp to 30%", None), 1) == "Set the bedroom lamp to 30%."


async def test_living_room_lights_go_off_with_no_model_call(config, isolated_home):
    llm = FakeProvider()
    s = await _ha_app(config, isolated_home, llm)
    try:
        with respx.mock(assert_all_called=False) as router:
            fake = FakeHA(router)
            await _connected(s)
            sid = await s.store.create_session(channel="desktop")
            asked, events = await _turn(s, sid, "turn off the living room lights")
        assert llm.calls == [] and llm.text_calls == [] and asked == []
        assert [type(e).__name__ for e in events] == ["ToolCallEvent", "ToolResultEvent", "TextDelta", "Done"]
        call = next(e for e in events if isinstance(e, ToolCallEvent))
        assert call.name == "home_assistant_turn_off"
        assert fake.calls == [("light.turn_off", {"entity_id": ["light.living_room_ceiling",
                                                                 "light.living_room_floor_lamp"]})]
        assert _reply(events) == "Turned off the living room lights."
        assert next(e for e in events if isinstance(e, ToolResultEvent)).is_error is False
        assert isinstance(events[-1], Done) and events[-1].content == "Turned off the living room lights."
    finally:
        await s.stop()


async def test_unclear_or_unsafe_commands_go_to_the_model(config, isolated_home):
    llm = FakeProvider(replies=["Which bedroom light?", "model 2", "model 3", "model 4", "model 5"])
    s = await _ha_app(config, isolated_home, llm)
    try:
        with respx.mock(assert_all_called=False) as router:
            fake = FakeHA(router)
            sid = await s.store.create_session(channel="desktop")
            # not connected yet: the model answers
            assert await fastpath.plan(s, "turn off the living room lights") is None
            await _connected(s)
            _, events = await _turn(s, sid, "turn off the bedroom light")  # two of them
            assert _reply(events) == "Which bedroom light?"
            _, events = await _turn(s, sid, "turn on the garage opener")  # sounds like a garage door
            assert _reply(events) == "model 2"
            s.config.tools.approvals.mode = "always"  # the user OKs every call
            _, events = await _turn(s, sid, "turn off the living room lights")
            assert _reply(events) == "model 3"
            s.config.tools.approvals.mode = "ask"
            s.config.tools.approvals.rules = {"home_assistant": "ask"}
            _, events = await _turn(s, sid, "turn off the living room lights")
            assert _reply(events) == "model 4"
            s.config.tools.approvals.rules = {}
            await s.integrations.set_access("home_assistant", "read")  # Read only (#141)
            _, events = await _turn(s, sid, "turn off the living room lights")
            assert _reply(events) == "model 5"
            assert fake.calls == []
    finally:
        await s.stop()


async def test_unlocking_asks_first_with_only_allow_once(config, isolated_home):
    llm = FakeProvider(replies=[
        [{"id": "c1", "name": "home_assistant_unlock", "arguments": {"entity_id": "lock.front_door"}}],
        "I unlocked the front door.",
    ])
    s = await _ha_app(config, isolated_home, llm)
    try:
        with respx.mock(assert_all_called=False) as router:
            fake = FakeHA(router)
            await _connected(s)
            sid = await s.store.create_session(channel="desktop")
            asked, _ = await _turn(s, sid, "unlock the front door", decision="deny")
            assert len(asked) == 1 and asked[0].risk == "send" and asked[0].choices == ["allow", "deny"]
            assert asked[0].risk_label == "Unlocks a door" and asked[0].target == "Front door"
            assert fake.calls == []  # denied: nothing happened
    finally:
        await s.stop()


# ---------------------------------------------------------------------------- entity picker routes
def test_entity_picker_routes(config, isolated_home, keychain, monkeypatch):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "t")
    s = SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "routes.db", enable_background=False)
    with respx.mock(assert_all_called=False) as router:
        router.route(host="testserver").pass_through()
        FakeHA(router)
        with TestClient(create_app(s)) as client:
            h = {"Authorization": "Bearer t"}
            assert client.get("/api/integrations/home_assistant/entities", headers=h).status_code == 400  # not yet
            r = client.post("/api/integrations/home_assistant/connect", headers=h,
                            json={"fields": {"url": "homeassistant.local:8123", "token": TOKEN}})
            assert r.status_code == 200, r.text
            body = client.get("/api/integrations/home_assistant/entities", headers=h).json()
            assert body["mode"] == "all" and len(body["entities"]) == len(STATES)
            assert all(e["allowed"] for e in body["entities"])
            r = client.put("/api/integrations/home_assistant/entities", headers=h,
                           json={"mode": "picked", "picked": ["light.bedroom_lamp"]})
            assert r.json() == {"mode": "picked", "picked": ["light.bedroom_lamp"]}
            body = client.get("/api/integrations/home_assistant/entities", headers=h).json()
            assert [e["entity_id"] for e in body["entities"] if e["allowed"]] == ["light.bedroom_lamp"]
            assert client.get("/api/integrations/gmail/entities", headers=h).status_code == 400
            assert client.put("/api/integrations/home_assistant/entities", headers=h,
                              json={"mode": "some"}).status_code == 422


# ---------------------------------------------------------------------------- websocket: rooms and live updates
class FakeSocket:
    """Home Assistant's websocket protocol: auth, registry lists and state_changed events."""

    def __init__(self):
        self.events: asyncio.Queue = asyncio.Queue()
        self.subscribed = asyncio.Event()

    async def handler(self, ws):
        await ws.send(json.dumps({"type": "auth_required"}))
        auth = json.loads(await ws.recv())
        if auth.get("access_token") != TOKEN:
            await ws.send(json.dumps({"type": "auth_invalid", "message": "Invalid access token"}))
            return
        await ws.send(json.dumps({"type": "auth_ok"}))
        pusher = None
        try:
            async for raw in ws:
                msg = json.loads(raw)
                kind = msg["type"]
                lists = {"config/area_registry/list": "areas", "config/device_registry/list": "devices",
                         "config/entity_registry/list": "entities"}
                if kind in lists:
                    await ws.send(json.dumps({"id": msg["id"], "type": "result", "success": True,
                                              "result": REGISTRY[lists[kind]]}))
                elif kind == "subscribe_events":
                    await ws.send(json.dumps({"id": msg["id"], "type": "result", "success": True, "result": None}))
                    sub_id = msg["id"]

                    async def push(sub_id=sub_id):
                        while True:
                            event = await self.events.get()
                            await ws.send(json.dumps({"id": sub_id, "type": "event", "event": event}))

                    pusher = asyncio.create_task(push())
                    self.subscribed.set()
        finally:
            if pusher:
                pusher.cancel()


def _door(old: str, new: str, at: str) -> dict:
    return {"event_type": "state_changed", "data": {
        "entity_id": "lock.front_door",
        "old_state": {"state": old, "attributes": {"friendly_name": "Front door"}},
        "new_state": {"state": new, "attributes": {"friendly_name": "Front door"}, "last_changed": at}}}


async def test_rooms_come_from_the_websocket(monkeypatch):
    sock = FakeSocket()
    async with serve(sock.handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        c = {"url": f"http://127.0.0.1:{port}", "token": TOKEN}
        monkeypatch.undo()  # the real _registry, through the fake socket
        reg = await ha._registry(c)
        assert [a["name"] for a in reg["areas"]] == ["Living room", "Bedroom", "Hallway"]
        index = {e["entity_id"]: e for e in ha.build_index(STATES, reg)}
        assert index["light.living_room_ceiling"]["area"] == "Living room"  # through its device
        assert index["climate.hallway_thermostat"]["device"] == "Thermostat"
        with pytest.raises(IntegrationError, match="token"):
            async with ha.Socket({**c, "token": "wrong"}):
                pass


async def test_live_updates_start_tasks_when_the_door_unlocks(app):
    sock = FakeSocket()
    now = app.tasks.now_iso()
    task_id = await app.tasks.repo.insert_task({
        "name": "Tell me when the front door unlocks", "description": "Send me a note.", "status": "active",
        "schedule": {"type": "triggered", "source": "home_assistant", "event": "state_changed",
                     "filter": {"entity_id": "lock.front_door", "to": "unlocked"}},
        "plan": [{"tool": "time", "description": "Check the time"}], "created_at": now, "updated_at": now,
    })
    async with serve(sock.handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        url = f"http://127.0.0.1:{port}"
        with respx.mock(assert_all_called=False) as router:
            FakeHA(router, url=url)
            await app.integrations.connect("home_assistant", {"url": url, "token": TOKEN})
            seen: list[dict] = []
            async with app.bus.subscribe() as queue:
                watcher = asyncio.create_task(ha.PLUGIN.watch(app.integrations))
                try:
                    await asyncio.wait_for(sock.subscribed.wait(), 5)
                    await sock.events.put(_door("locked", "locked", "2026-10-15T09:00:00+00:00"))  # attribute-only
                    await sock.events.put(_door("locked", "unavailable", "2026-10-15T09:00:01+00:00"))  # restarting
                    await sock.events.put(_door("locked", "unlocked", "2026-10-15T09:00:02+00:00"))
                    while not seen:
                        ev = await asyncio.wait_for(queue.get(), 5)
                        if ev.get("type") == "source.items":
                            seen.append(ev["data"])
                    for _ in range(100):  # the tasks package picks source.items up by itself
                        if (await app.tasks.get(task_id))["runs"]:
                            break
                        await asyncio.sleep(0.05)
                finally:
                    watcher.cancel()
                    await app.tasks.drain()
    assert len(seen) == 1 and seen[0]["source"] == "home_assistant" and seen[0]["event"] == "state_changed"
    item = seen[0]["items"][0]
    assert (item["entity_id"], item["from"], item["to"], item["name"], item["area"]) == (
        "lock.front_door", "locked", "unlocked", "Front door", "Hallway")
    assert len((await app.tasks.get(task_id))["runs"]) == 1
    status = {f["source"]: f for f in await app.integrations.feed_status()}["home_assistant"]
    assert status["kind"] == "home_assistant_live"


def test_trigger_filters_ignore_case_for_home_assistant():
    from sentient.tasks.triggers import event_matches_filter

    item = {"entity_id": "lock.front_door", "name": "Front door", "from": "locked", "to": "unlocked"}
    # what a small model wrote for "when the front door unlocks"
    wanted = {"name": "front door", "entity_id": "lock.front_door", "from": "locked", "to": "unlocked"}
    assert event_matches_filter(item, wanted, "home_assistant")
    assert event_matches_filter(item, {"name": {"$in": ["FRONT DOOR"]}}, "home_assistant")
    assert not event_matches_filter(item, {"to": "locked"}, "home_assistant")
    assert not event_matches_filter(item, wanted, "webhook")  # other sources stay exact
    assert event_matches_filter({**item, "name": "Straße"}, {"name": {"$regex": "^Stra.e$"}}, "home_assistant")
    assert event_matches_filter({**item, "name": "Straße"}, {"name": {"$contains": "Straße"}}, "home_assistant")


# ---------------------------------------------------------------------------- opening the home always asks (ADR 0028)
def _call(name: str, **args) -> list[dict]:
    return [{"id": f"c_{name}", "name": name, "arguments": args}]


OPENING = [
    ("home_assistant_unlock", {"entity_id": "lock.front_door"}, True),
    ("home_assistant_set_alarm", {"entity_id": "alarm_control_panel.home", "mode": "disarm"}, True),
    ("home_assistant_open_cover", {"entity_id": "cover.garage_door"}, True),
    ("home_assistant_turn_on", {"entity_id": "switch.garage_opener"}, True),  # sounds like a garage door
    ("home_assistant_lock", {"entity_id": "lock.front_door"}, False),  # locking follows the normal rules
    ("home_assistant_set_alarm", {"entity_id": "alarm_control_panel.home", "mode": "arm_away"}, False),
    ("home_assistant_open_cover", {"entity_id": "cover.bedroom_blinds"}, False),
]


@pytest.mark.parametrize(("name", "args", "asks"), OPENING)
async def test_opening_the_home_asks_even_with_approvals_off(config, isolated_home, name, args, asks):
    llm = FakeProvider(replies=[_call(name, **args), "Done."])
    s = await _ha_app(config, isolated_home, llm, mode="off")
    states = [*STATES, _state("alarm_control_panel.home", "armed_away", "House alarm")]
    try:
        with respx.mock(assert_all_called=False) as router:
            fake = FakeHA(router)
            fake.states = [dict(x) for x in states]
            await _connected(s)
            # an Allow rule never lifts it either, like a purchase
            s.config.tools.approvals.rules = {name: "allow"}
            sid = await s.store.create_session(channel="desktop")
            asked, _ = await _turn(s, sid, "do it", decision="deny")
        if asks:
            assert len(asked) == 1 and asked[0].choices == ["allow", "deny"] and fake.calls == []
        else:
            assert asked == [] and len(fake.calls) == 1
    finally:
        await s.stop()


async def test_a_task_run_holds_an_unlock_and_scripts_refuse_it(app, ctx):
    from sentient.agent.loop import LoopResult
    from sentient.sandbox.policy import BridgePolicy, Refused

    with respx.mock(assert_all_called=False) as router:
        fake = FakeHA(router)
        await _connected(app)
        app.config.tools.approvals.mode = "off"
        app.agent.llm.replies = [_call("home_assistant_unlock", entity_id="lock.front_door"), "Waiting."]
        result = LoopResult()
        messages = [{"role": "user", "content": "unlock the front door"}]
        async for _ in app.agent.run_loop(messages, ctx, result=result, use_approvals=False, source="subagent"):
            pass
        tool_msg = next(m for m in messages if m.get("role") == "tool")
        assert "opens the user's home" in tool_msg["content"] and fake.calls == []
        unlock = next(t for t in ha.PLUGIN.tools if t.name == "home_assistant_unlock")
        policy = BridgePolicy.build(approvals_mode="off", allowed_tools=None, read_only=False, max_tool_calls=10)
        with pytest.raises(Refused, match="open the user's home"):
            await policy.check(unlock, unlock.name, {"entity_id": "lock.front_door"}, ctx, 0)


# ---------------------------------------------------------------------------- reading the change back (#386)
def _tools() -> dict:
    return {t.name: t for t in ha.PLUGIN.tools}


async def _checked(app, ctx, name: str, args: dict) -> dict | None:
    tool = _tools()[name]
    res = await tool.call(ctx, args)
    return await app.agent.check_outcome(tool, args, res, False, ctx)


async def test_a_device_that_obeys_is_read_back_and_one_that_ignores_the_call_is_not_done(app, ctx):
    app.config.integrations.home_assistant_check_s = 0.2
    with respx.mock(assert_all_called=False) as router:
        fake = FakeHA(router)
        await _connected(app)
        ok = await _checked(app, ctx, "home_assistant_turn_on", {"entity_id": "light.bedroom_lamp"})
        assert ok["ok"] is True and ok["how"] == "read_back" and "light.bedroom_lamp" in fake.reads
        fake.ignore.add("light.bedroom_strip")
        tool = _tools()["home_assistant_turn_on"]
        res = await tool.call(ctx, {"entity_id": "light.bedroom_strip"})
        assert res["done"] is True  # Home Assistant took the call...
        bad = await app.agent.check_outcome(tool, {"entity_id": "light.bedroom_strip"}, res, False, ctx)
        assert bad["ok"] is False and bad["how"] == "read_back"  # ...but the strip stayed off
        assert bad["detail"] == "Bedroom strip still reports off"
        assert fake.reads.count("light.bedroom_strip") >= 2  # it read again before saying so
        assert bad["targets"] == ["light.bedroom_strip"] and bad["missed"] == ["light.bedroom_strip"]
        [final] = outcomes.finalize([bad], "I turned on the bedroom strip.")
        assert final["message"] == "Sentient said it ran the home action, but Bedroom strip still reports off."
        # another light that came on doesn't settle it; the strip coming on at the next try does
        assert outcomes.failures(outcomes.finalize([bad, ok]))
        fake.ignore.clear()
        retry = await _checked(app, ctx, "home_assistant_turn_on", {"entity_id": "light.bedroom_strip"})
        assert retry["ok"] is True and not outcomes.failures(outcomes.finalize([bad, ok, retry]))


async def test_a_slow_device_is_read_again_until_it_reports_the_change(app, ctx):
    app.config.integrations.home_assistant_check_s = 1.0
    with respx.mock(assert_all_called=False) as router:
        fake = FakeHA(router)
        await _connected(app)
        await _tools()["home_assistant_unlock"].call(ctx, {"entity_id": "lock.front_door"})
        fake.lag["lock.front_door"] = 2  # still unlocked on the first two reads
        check = await _checked(app, ctx, "home_assistant_lock", {"entity_id": "lock.front_door"})
        assert check["ok"] is True and fake.reads.count("lock.front_door") == 3


def test_what_counts_as_done_for_each_kind():
    def verdict(action, args, eid, state, **attrs):
        want = ha._expected(action, args, eid)
        return None if want is None else ha._compare(want, _state(eid, state, "Thing", **attrs))[0]

    assert verdict("turn_on", {}, "light.a", "on") == "ok"
    assert verdict("turn_off", {}, "switch.a", "on") == "wrong"
    assert verdict("turn_on", {}, "media_player.tv", "playing") == "ok"  # "on" can be many states
    assert verdict("turn_on", {}, "media_player.tv", "off") == "wrong"
    assert verdict("turn_on", {}, "scene.movie_night", "x") is None  # nothing clear to expect
    assert ha._expected("run_script", {}, "script.morning") is None
    assert verdict("lock", {}, "lock.a", "locking") == "moving"
    assert verdict("lock", {}, "lock.a", "jammed") == "wrong"
    assert verdict("unlock", {}, "lock.a", "unavailable") == "wrong"
    assert verdict("close_cover", {}, "cover.garage", "closing") == "moving"
    assert verdict("open_cover", {}, "cover.blinds", "closed") == "wrong"
    assert verdict("set_alarm", {"mode": "arm_away"}, "alarm_control_panel.a", "armed_away") == "ok"
    assert verdict("set_alarm", {"mode": "disarm"}, "alarm_control_panel.a", "armed_home") == "wrong"
    assert verdict("set_temperature", {"temperature": 22}, "climate.a", "heat", temperature=22.0) == "ok"
    assert verdict("set_temperature", {"temperature": 22}, "climate.a", "heat", temperature=20) == "wrong"
    assert verdict("set_temperature", {"temperature": 22, "hvac_mode": "cool"}, "climate.a", "heat",
                   temperature=22) == "wrong"
    want = ha._expected("set_temperature", {"temperature": 22}, "climate.a")
    assert ha._compare(want, _state("climate.a", "heat", "Hallway thermostat", temperature=20))[1] == (
        "Hallway thermostat is still set to 20°, not 22°")
    assert ha._compare(ha._expected("turn_on", {}, "light.a"), _state("light.a", "unavailable", "Porch"))[1] == (
        "Porch isn't responding (Home Assistant says it is unavailable)")


async def test_a_door_still_moving_is_not_a_failure_and_the_check_can_be_off(app, ctx):
    app.config.integrations.home_assistant_check_s = 0.05
    with respx.mock(assert_all_called=False) as router:
        fake = FakeHA(router)
        await _connected(app)
        fake.ignore.add("cover.garage_door")
        next(s for s in fake.states if s["entity_id"] == "cover.garage_door")["state"] = "opening"
        check = await _checked(app, ctx, "home_assistant_open_cover", {"entity_id": "cover.garage_door"})
        assert check["ok"] is True and check["how"] == "read_back"
        assert check["detail"] == "Garage door is still opening"
        scene = await _checked(app, ctx, "home_assistant_activate_scene", {"entity_id": "scene.movie_night"})
        assert scene["ok"] is True and scene["how"] == "result"  # nothing clear to read back
        app.config.integrations.home_assistant_check_s = 0
        fake.ignore.add("light.bedroom_strip")
        reads = len(fake.reads)
        off = await _checked(app, ctx, "home_assistant_turn_on", {"entity_id": "light.bedroom_strip"})
        assert off["ok"] is True and off["how"] == "result" and len(fake.reads) == reads


async def test_the_model_and_the_reply_hear_that_the_light_stayed_off(config, isolated_home):
    llm = FakeProvider(replies=[_call("home_assistant_turn_on", entity_id="light.bedroom_strip"),
                                "The bedroom strip did not turn on."])
    s = await _ha_app(config, isolated_home, llm, mode="off")
    s.config.integrations.home_assistant_check_s = 0.1
    try:
        with respx.mock(assert_all_called=False) as router:
            fake = FakeHA(router)
            fake.ignore.add("light.bedroom_strip")
            await _connected(s)
            sid = await s.store.create_session(channel="desktop")
            _, events = await _turn(s, sid, "make the strip in the bedroom glow")
        result = next(e for e in events if isinstance(e, ToolResultEvent))
        assert result.check["ok"] is False and result.check["detail"] == "Bedroom strip still reports off"
        # the model read the read-back, not only the result that said done
        assert "it did not work: Bedroom strip still reports off" in json.dumps(llm.calls[-1], default=str)
        done = events[-1]
        assert isinstance(done, Done) and done.checks[0]["message"].endswith("Bedroom strip still reports off.")
    finally:
        await s.stop()


async def test_a_quick_command_never_says_turned_on_when_the_lamp_stayed_off(config, isolated_home):
    llm = FakeProvider()
    s = await _ha_app(config, isolated_home, llm)
    s.config.integrations.home_assistant_check_s = 0.1
    try:
        with respx.mock(assert_all_called=False) as router:
            fake = FakeHA(router)
            fake.ignore.add("light.bedroom_lamp")
            await _connected(s)
            sid = await s.store.create_session(channel="desktop")
            _, events = await _turn(s, sid, "turn on the bedroom lamp")
            fake.ignore.clear()
            _, again = await _turn(s, sid, "turn on the bedroom lamp")
        assert llm.calls == [] and fake.calls[0] == ("light.turn_on", {"entity_id": ["light.bedroom_lamp"]})
        assert _reply(events) == "I tried, but Bedroom lamp still reports off."
        done = events[-1]
        assert done.checks[0]["ok"] is False and "Bedroom lamp still reports off" in done.checks[0]["message"]
        result = next(e for e in events if isinstance(e, ToolResultEvent))
        assert result.check["ok"] is False
        assert _reply(again) == "Turned on the bedroom lamp." and again[-1].checks[0]["how"] == "read_back"
    finally:
        await s.stop()
