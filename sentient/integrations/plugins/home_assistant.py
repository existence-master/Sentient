"""Home Assistant (#175): lights, switches, thermostats, locks and more, with a URL and a long-lived access token.

The token lives in the keychain (``integration:home_assistant``, ``{url, token}``). Look-ups and changes use Home
Assistant's REST API (``/api/states``, ``/api/services/<domain>/<service>``); rooms and devices come from its
websocket API (``config/area_registry/list``, ``config/device_registry/list``, ``config/entity_registry/list``), and
so do live updates (``subscribe_events`` for ``state_changed``), which start tasks ("when the front door opens...").

The model never sees every entity: it finds the few it needs with ``home_assistant_find_entities`` (by words, room or
kind), and the entity picker (``entity_choice``) can limit what Sentient may see and use at all.

Risk: look-ups read; lights, switches, thermostats, scenes and blinds are a change (``write``); locks, alarms, garage
doors, gates and anything that sounds like one are ``send`` and ask every time, with only "Allow once" offered.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import json
import logging
import re
import ssl
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urlsplit

import httpx

from sentient.integrations.base import (
    IntegrationError,
    IntegrationPlugin,
    SetupField,
    creds,
    itool,
    manager_from,
)
from sentient.integrations.common import http_client
from sentient.tools.base import Risk, ToolContext
from sentient.tools.rules import (
    DISARMS_LABEL,
    OPENS_GATE_LABEL,
    SECURITY_SWITCH_LABEL,
    UNLOCKS_LABEL,
)

if TYPE_CHECKING:  # pragma: no cover
    from sentient.integrations.service import IntegrationManager

log = logging.getLogger(__name__)

PID = "home_assistant"
STATES_TTL_S = 10.0
REGISTRY_TTL_S = 300.0
WS_TIMEOUT_S = 15.0
BACKOFF_BASE_S = 30.0
FIND_LIMIT = 25

# what "turn on" and "turn off" may switch; everything else has its own tool or none
SWITCHABLE = {
    "light", "switch", "fan", "input_boolean", "media_player", "climate", "humidifier", "water_heater", "siren",
    "automation", "remote", "scene",
}
SECURITY_DOMAINS = {"lock", "alarm_control_panel"}
SECURITY_COVER_CLASSES = {"garage", "gate", "door"}
# a switch, script or button with one of these words may open a door or turn off an alarm: it asks like a lock does
SECURITY_WORDS = re.compile(r"(lock|door|garage|gate|alarm|security|safe\b|vault)", re.I)
WORDY_DOMAINS = {"switch", "script", "button", "input_button", "input_boolean", "automation", "cover"}
# live updates skip these: they change all the time (power use, the sun) and would start tasks for nothing
NOISY_DOMAINS = {
    "sensor", "number", "input_number", "sun", "weather", "update", "zone", "image", "event", "camera", "time", "date",
    "datetime", "tts", "stt", "conversation", "persistent_notification", "button", "input_button", "input_text",
}
UNSETTLED = {"unavailable", "unknown", "", None}
ALARM_MODES = {"arm_home", "arm_away", "arm_night", "arm_vacation", "disarm"}
# attributes worth telling the model about; the rest are noise (icons, supported features, color lists)
USEFUL_ATTRS = (
    "unit_of_measurement", "device_class", "brightness", "color_mode", "color_temp_kelvin", "rgb_color",
    "current_temperature", "temperature", "target_temp_high", "target_temp_low", "hvac_mode", "hvac_action",
    "hvac_modes", "preset_mode", "fan_mode", "current_position", "percentage", "volume_level", "media_title",
    "source", "is_volume_muted", "battery_level", "humidity", "current_humidity",
)


# ---------------------------------------------------------------------------- address and token
def normalize_url(raw: str) -> str:
    """``homeassistant.local:8123`` -> ``http://homeassistant.local:8123``; no path, query or trailing slash."""
    s = str(raw or "").strip().rstrip("/")
    if not s:
        raise IntegrationError("Please paste the address of your Home Assistant, like http://homeassistant.local:8123.")
    if "://" not in s:
        s = "http://" + s
    parts = urlsplit(s)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise IntegrationError("That doesn't look like a Home Assistant address. It starts with http:// or https://.")
    path = parts.path.rstrip("/")
    if path.endswith("/api"):
        path = path[: -len("/api")]
    return f"{parts.scheme}://{parts.netloc}{path}"


def _local_host(host: str) -> bool:
    h = host.strip("[]").lower()
    try:  # an IP address first: a public IPv6 address has no dots but is not at home
        ip = ipaddress.ip_address(h)
    except ValueError:
        return h == "localhost" or "." not in h or h.endswith((
            ".local", ".lan", ".home", ".internal", ".localdomain", ".home.arpa", ".localhost", ".test"))
    return ip.is_private or ip.is_loopback or ip.is_link_local


def check_url(url: str) -> str:
    """The cleaned address; plain http only on this computer or the home network, so the token never crosses the
    internet in the clear."""
    url = normalize_url(url)
    parts = urlsplit(url)
    if parts.scheme == "http" and not _local_host(parts.hostname or ""):
        raise IntegrationError(
            "For a Home Assistant you reach over the internet, use its https:// address so your token stays private."
        )
    return url


def ws_url(url: str) -> str:
    return ("wss://" + url[len("https://"):] if url.startswith("https://") else "ws://" + url[len("http://"):]) + (
        "/api/websocket")


# ---------------------------------------------------------------------------- REST and websocket
_SSL: ssl.SSLContext | None = None


def _ssl_context() -> ssl.SSLContext:
    """One certificate store for every call: building it takes most of a second on some computers, and a light
    command should feel instant."""
    global _SSL
    if _SSL is None:
        _SSL = httpx.create_ssl_context()
    return _SSL


async def rest(c: dict, method: str, path: str, *, body: Any = None) -> Any:
    headers = {"Authorization": f"Bearer {c['token']}", "Content-Type": "application/json"}
    async with http_client(timeout=20.0, headers=headers, verify=_ssl_context()) as http:
        r = await http.request(method, f"{c['url']}{path}", content=None if body is None else json.dumps(body))
    if r.status_code in (401, 403):
        raise IntegrationError("Home Assistant didn't accept the saved token. Make a new long-lived access token and "
                               "reconnect Home Assistant.")
    if r.status_code == 404 and path.startswith("/api/states/"):
        raise IntegrationError("Home Assistant has no device with that id. Use home_assistant_find_entities first.")
    if r.status_code == 400 and path.startswith("/api/services/"):
        raise IntegrationError(f"Home Assistant couldn't do that: {r.text[:200].strip() or 'it refused the request'}.")
    r.raise_for_status()
    return r.json() if r.content else None


class Socket:
    """One authenticated connection to Home Assistant's websocket API."""

    def __init__(self, c: dict):
        self.c = c
        self.ws: Any = None
        self._next = 0

    async def __aenter__(self) -> Socket:
        from websockets.asyncio.client import connect

        self.ws = await connect(ws_url(self.c["url"]), open_timeout=WS_TIMEOUT_S, max_size=2**25)
        try:
            await self._auth()
        except BaseException:
            await self.ws.close()
            raise
        return self

    async def _auth(self) -> None:
        first = json.loads(await asyncio.wait_for(self.ws.recv(), WS_TIMEOUT_S))
        if first.get("type") == "auth_required":
            await self.ws.send(json.dumps({"type": "auth", "access_token": self.c["token"]}))
            first = json.loads(await asyncio.wait_for(self.ws.recv(), WS_TIMEOUT_S))
        if first.get("type") != "auth_ok":
            raise IntegrationError("Home Assistant didn't accept the saved token. Make a new long-lived access "
                                   "token and reconnect Home Assistant.")

    async def __aexit__(self, *exc: Any) -> None:
        with contextlib.suppress(Exception):
            await self.ws.close()

    async def send(self, payload: dict) -> int:
        self._next += 1
        await self.ws.send(json.dumps({**payload, "id": self._next}))
        return self._next

    async def call(self, payload: dict) -> Any:
        msg_id = await self.send(payload)
        while True:
            msg = json.loads(await asyncio.wait_for(self.ws.recv(), WS_TIMEOUT_S))
            if msg.get("id") == msg_id and msg.get("type") == "result":
                if not msg.get("success"):
                    err = (msg.get("error") or {}).get("message") or "it refused the request"
                    raise IntegrationError(f"Home Assistant couldn't do that: {err}.")
                return msg.get("result")

    async def events(self) -> AsyncIterator[dict]:
        async for raw in self.ws:
            msg = json.loads(raw)
            if msg.get("type") == "event":
                yield msg.get("event") or {}


# ---------------------------------------------------------------------------- the home, cached
# url -> {"states": (monotonic, list), "registry": (monotonic, dict)}
_CACHE: dict[str, dict[str, tuple[float, Any]]] = {}


def forget_cache() -> None:
    _CACHE.clear()


async def _registry(c: dict) -> dict:
    """Rooms, devices and entity entries, or empty when the websocket can't be used (names still work)."""
    slot = _CACHE.setdefault(c["url"], {})
    hit = slot.get("registry")
    if hit and time.monotonic() - hit[0] < REGISTRY_TTL_S:
        return hit[1]
    try:
        async with Socket(c) as ws:
            reg = {
                "areas": await ws.call({"type": "config/area_registry/list"}) or [],
                "devices": await ws.call({"type": "config/device_registry/list"}) or [],
                "entities": await ws.call({"type": "config/entity_registry/list"}) or [],
            }
    except IntegrationError:
        raise
    except Exception as exc:  # an older or locked-down Home Assistant: no rooms, names still work
        log.info("Home Assistant rooms could not be read (%s); using names only", type(exc).__name__)
        reg = {"areas": [], "devices": [], "entities": []}
    slot["registry"] = (time.monotonic(), reg)
    return reg


async def _states(c: dict, *, fresh: bool = False) -> list[dict]:
    slot = _CACHE.setdefault(c["url"], {})
    hit = slot.get("states")
    if hit and not fresh and time.monotonic() - hit[0] < STATES_TTL_S:
        return hit[1]
    rows = await rest(c, "GET", "/api/states")
    rows = [r for r in rows or [] if isinstance(r, dict) and r.get("entity_id")]
    slot["states"] = (time.monotonic(), rows)
    return rows


def build_index(states: list[dict], reg: dict) -> list[dict]:
    """One small record per entity: ``{entity_id, name, domain, area, device, state, device_class}``. Entities
    Home Assistant hides or turned off are left out."""
    areas = {a.get("area_id"): a.get("name") for a in reg.get("areas") or [] if isinstance(a, dict)}
    devices = {d.get("id"): d for d in reg.get("devices") or [] if isinstance(d, dict)}
    entries = {e.get("entity_id"): e for e in reg.get("entities") or [] if isinstance(e, dict)}
    out = []
    for s in states:
        eid = str(s["entity_id"])
        entry = entries.get(eid) or {}
        if entry.get("hidden_by") or entry.get("disabled_by"):
            continue
        attrs = s.get("attributes") or {}
        device = devices.get(entry.get("device_id")) or {}
        area_id = entry.get("area_id") or device.get("area_id")
        out.append({
            "entity_id": eid,
            "name": str(attrs.get("friendly_name") or entry.get("name") or entry.get("original_name") or eid),
            "domain": eid.split(".", 1)[0],
            "area": areas.get(area_id) or "",
            "device": str(device.get("name_by_user") or device.get("name") or ""),
            "state": s.get("state"),
            "device_class": attrs.get("device_class") or entry.get("device_class") or entry.get("original_device_class"),
        })
    return out


def entity_choice(mgr: IntegrationManager | None) -> dict:
    """``{mode: "all"|"picked", picked: [entity ids]}`` from the entity picker."""
    if mgr is None or not hasattr(mgr, "entity_choice"):
        return {"mode": "all", "picked": []}
    return mgr.entity_choice(PID)


def allowed(entity_id: str, choice: dict) -> bool:
    return choice.get("mode") != "picked" or entity_id in set(choice.get("picked") or [])


async def home(mgr: IntegrationManager, *, fresh: bool = False, everything: bool = False) -> list[dict]:
    """The entities Sentient may use (all of them with ``everything``, for the picker)."""
    c = await mgr.get_credentials(PID)
    if not c:
        raise IntegrationError("Home Assistant isn't connected yet. Connect it from Integrations.")
    states, reg = await asyncio.gather(_states(c, fresh=fresh), _registry(c))
    index = build_index(states, reg)
    if everything:
        return index
    choice = entity_choice(mgr)
    return [e for e in index if allowed(e["entity_id"], choice)]


def cached_entity(entity_id: str) -> dict | None:
    """What the cache knows about one entity, without any network call (for risk and approval wording)."""
    for slot in _CACHE.values():
        hit = slot.get("states")
        for s in (hit[1] if hit else []):
            if s.get("entity_id") == entity_id:
                attrs = s.get("attributes") or {}
                return {"entity_id": entity_id, "name": attrs.get("friendly_name") or entity_id,
                        "domain": entity_id.split(".", 1)[0], "device_class": attrs.get("device_class")}
    return None


# ---------------------------------------------------------------------------- what is security
def is_security(entity: dict) -> bool:
    """Locks, alarms, garage doors, gates and doors, and switches, scripts or buttons that sound like one."""
    domain = entity.get("domain") or str(entity.get("entity_id", "")).split(".", 1)[0]
    if domain in SECURITY_DOMAINS:
        return True
    if domain == "cover" and str(entity.get("device_class") or "").lower() in SECURITY_COVER_CLASSES:
        return True
    if domain in WORDY_DOMAINS:
        words = f"{entity.get('entity_id', '')} {entity.get('name', '')}".replace("_", " ")
        return bool(SECURITY_WORDS.search(words))
    return False


def split_ids(entity_id: str | list | None) -> list[str]:
    raw = entity_id if isinstance(entity_id, list) else re.split(r"[,\s]+", str(entity_id or ""))
    return list(dict.fromkeys(str(x).strip().lower() for x in raw if str(x).strip()))


# kinds that are never a door, lock or alarm, whatever they are called
NEVER_SECURITY = {"light", "fan", "climate", "media_player", "scene", "humidifier", "water_heater", "remote"}


def _maybe_security(entity_id: str) -> bool:
    """True when the entity is security, or the cache doesn't know it and its kind could be (never a light)."""
    known = cached_entity(entity_id)
    if known is None:
        return entity_id.split(".", 1)[0] not in NEVER_SECURITY
    return is_security(known)


def _security_risk(args: dict, ctx: Any) -> Risk:
    """``send`` when any entity of the call is security (asks every time), else ``write``. Uses only the entity ids
    and the cache, never the network, so the agent loop and quick commands can call it anywhere. An entity the cache
    doesn't know yet counts as security unless its kind never is (a light), so a call never looks safer than it is."""
    return Risk.send if any(_maybe_security(e) for e in split_ids(args.get("entity_id"))) else Risk.write


def _names(args: dict) -> str:
    names = [str((cached_entity(e) or {}).get("name") or e) for e in split_ids(args.get("entity_id"))]
    return ", ".join(names)[:120]


def _describe(label: str, security: str | None = None):
    """Approval wording: ``label``, or ``security`` when the call touches a door, lock or alarm. The labels in
    ``rules.OPENS_HOME_LABELS`` (unlock, turn off the alarm, open a garage door or gate) make the call a hard stop
    that asks whatever the approvals mode says (ADR 0028), so an unknown entity gets the security label."""

    async def describe(args: dict, ctx: Any) -> dict:
        ids = split_ids(args.get("entity_id"))
        if any(cached_entity(e) is None for e in ids):
            with contextlib.suppress(Exception):  # the names make the question plainer, and say what is security
                await home(manager_from(ctx))
        said = security if security and any(_maybe_security(e) for e in ids) else label
        return {"risk_label": said, "target": _names(args)}

    return describe


async def _describe_alarm(args: dict, ctx: Any) -> dict:
    out = await _describe("Arms your alarm")(args, ctx)
    if str(args.get("mode") or "").strip().lower().replace(" ", "_") not in ALARM_MODES - {"disarm"}:
        out["risk_label"] = DISARMS_LABEL  # disarming, or a mode it can't read: asks like turning the alarm off
    return out


# ---------------------------------------------------------------------------- shared tool helpers
async def _creds(ctx: ToolContext) -> dict:
    return await creds(ctx, PID)


async def _resolve(ctx: ToolContext, entity_id: str | list | None, domains: set[str] | None = None,
                   wrong: str = "") -> list[dict]:
    """The entities of a call, checked against the picker and the kinds this tool handles."""
    ids = split_ids(entity_id)
    if not ids:
        raise IntegrationError("Give an entity id, like light.living_room. Use home_assistant_find_entities to find it.")
    mgr = manager_from(ctx)
    by_id = {e["entity_id"]: e for e in await home(mgr, everything=True)}
    choice = entity_choice(mgr)
    out = []
    for eid in ids:
        e = by_id.get(eid)
        if e is None:
            raise IntegrationError(f"Home Assistant has no device called {eid}. Use home_assistant_find_entities.")
        if not allowed(eid, choice):
            raise IntegrationError(f"Sentient isn't allowed to use {e['name']}. You can change which devices it may "
                                   "use in Integrations > Home Assistant.")
        if domains is not None and e["domain"] not in domains:
            raise IntegrationError(wrong.format(name=e["name"]) if wrong else
                                   f"{e['name']} can't do that.")
        out.append(e)
    return out


async def _service(ctx: ToolContext, domain: str, service: str, entities: list[dict], data: dict | None = None) -> dict:
    c = await _creds(ctx)
    body = {"entity_id": [e["entity_id"] for e in entities], **(data or {})}
    changed = await rest(c, "POST", f"/api/services/{domain}/{service}", body=body)
    _CACHE.get(c["url"], {}).pop("states", None)  # the next look-up sees the change
    names = {e["entity_id"]: e["name"] for e in entities}
    now = [{"name": names.get(s.get("entity_id"), s.get("entity_id")), "state": s.get("state")}
           for s in changed or [] if isinstance(s, dict) and s.get("entity_id") in names]
    return {"done": True, "service": f"{domain}.{service}", "devices": [e["name"] for e in entities],
            **({"now": now} if now else {})}


def _short(e: dict) -> dict:
    out = {"entity_id": e["entity_id"], "name": e["name"], "state": e["state"]}
    if e.get("area"):
        out["area"] = e["area"]
    return out


def _words(text: str) -> list[str]:
    return [w for w in re.split(r"[^a-z0-9]+", str(text or "").lower()) if w]


# ---------------------------------------------------------------------------- tools: look-ups
@itool(PID, "home_assistant_list_areas")
async def ha_list_areas(ctx: ToolContext) -> dict:
    """List the rooms (areas) in Home Assistant and how many devices of each kind are in them."""
    rooms: dict[str, dict[str, int]] = {}
    for e in await home(manager_from(ctx)):
        kinds = rooms.setdefault(e["area"] or "No room", {})
        kinds[e["domain"]] = kinds.get(e["domain"], 0) + 1
    return {"areas": [{"name": name, "kinds": kinds} for name, kinds in sorted(rooms.items())]}


@itool(PID, "home_assistant_list_devices")
async def ha_list_devices(ctx: ToolContext, area: str | None = None) -> dict:
    """List devices in Home Assistant (a lamp, a thermostat, a lock), optionally only those in one room (`area`)."""
    want = " ".join(_words(area or ""))
    devices: dict[str, dict] = {}
    for e in await home(manager_from(ctx)):
        if want and " ".join(_words(e["area"])) != want:
            continue
        key = e["device"] or e["name"]
        d = devices.setdefault(key, {"name": key, "area": e["area"], "entities": []})
        d["entities"].append(e["entity_id"])
    rows = list(devices.values())
    return {"devices": rows[:60], **({"more": len(rows) - 60} if len(rows) > 60 else {})}


@itool(PID, "home_assistant_find_entities")
async def ha_find_entities(ctx: ToolContext, search: str | None = None, area: str | None = None,
                           kind: str | None = None, limit: int = FIND_LIMIT) -> dict:
    """Find smart home devices in Home Assistant and their entity ids: lights, switches, thermostats (climate),
    locks, covers (blinds, garage doors), sensors, scenes and scripts. Narrow it down with `search` (words in the
    name, like "living room lamp"), `area` (a room, like "Bedroom") or `kind` (light, switch, climate, lock, cover,
    scene, script, sensor, binary_sensor, fan, media_player). Use the entity ids with the other Home Assistant tools."""
    entities = await home(manager_from(ctx))
    words = _words(search or "")
    room = " ".join(_words(area or ""))
    kinds = {k.strip().lower() for k in re.split(r"[,\s]+", kind or "") if k.strip()}
    kinds |= {k[:-1] for k in kinds if k.endswith("s")}  # "lights"
    found = []
    for e in entities:
        if kinds and e["domain"] not in kinds:
            continue
        if room and " ".join(_words(e["area"])) != room:
            continue
        hay = set(_words(f"{e['name']} {e['entity_id']} {e['area']} {e['device']}"))
        hay |= {w[:-1] for w in hay if w.endswith("s")}
        if words and not all(w in hay or w.rstrip("s") in hay for w in words):
            continue
        found.append(_short(e))
    limit = max(1, min(int(limit or FIND_LIMIT), 100))
    out: dict[str, Any] = {"entities": found[:limit]}
    if len(found) > limit:
        out["more"] = len(found) - limit
        out["hint"] = "Narrow it down with search, area or kind."
    if not found:
        out["hint"] = ("Nothing matched. Try fewer words, or home_assistant_list_areas to see the rooms."
                       if entities else "Sentient may not use any devices yet. Pick them in Integrations > Home Assistant.")
    return out


@itool(PID, "home_assistant_get_state")
async def ha_get_state(ctx: ToolContext, entity_id: str) -> dict:
    """Get the current state of one Home Assistant device: on or off, brightness, temperature, locked or unlocked,
    open or closed."""
    e = (await _resolve(ctx, entity_id))[0]
    s = await rest(await _creds(ctx), "GET", f"/api/states/{quote(e['entity_id'])}")
    attrs = (s or {}).get("attributes") or {}
    details = {k: attrs[k] for k in USEFUL_ATTRS if k in attrs}
    if isinstance(details.get("brightness"), int | float):
        details["brightness_pct"] = round(float(details.pop("brightness")) / 255 * 100)
    return {"entity_id": e["entity_id"], "name": e["name"], "state": (s or {}).get("state"), "area": e["area"],
            "last_changed": (s or {}).get("last_changed"), **({"details": details} if details else {})}


# ---------------------------------------------------------------------------- tools: changes
_NOT_SWITCHABLE = ("{name} can't be turned on or off here. Locks use home_assistant_lock or home_assistant_unlock, "
                   "blinds and garage doors home_assistant_open_cover or home_assistant_close_cover, alarms "
                   "home_assistant_set_alarm and scripts home_assistant_run_script.")


def _by_domain(entities: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for e in entities:
        groups.setdefault(e["domain"], []).append(e)
    return groups


async def _switch(ctx: ToolContext, entity_id: str, on: bool, data: dict | None = None) -> dict:
    entities = await _resolve(ctx, entity_id, SWITCHABLE, _NOT_SWITCHABLE)
    results = []
    for domain, group in _by_domain(entities).items():
        if domain == "scene" and not on:
            raise IntegrationError("A scene can't be turned off. Turn off its lights instead.")
        extra = data if domain == "light" else None
        results.append(await _service(ctx, domain, "turn_on" if on else "turn_off", group, extra))
    if len(results) == 1:
        return results[0]
    return {"done": True, "devices": [n for r in results for n in r["devices"]],
            "now": [x for r in results for x in r.get("now", [])]}


@itool(PID, "home_assistant_turn_on", outcome="home", risk=Risk.write, risk_fn=_security_risk,
       describe_fn=_describe("Turns something on", SECURITY_SWITCH_LABEL))
async def ha_turn_on(ctx: ToolContext, entity_id: str, brightness_pct: int | None = None,
                     color: str | None = None) -> dict:
    """Turn on lights, switches, fans and other Home Assistant devices. `entity_id` is one id or several separated
    by commas. For lights, `brightness_pct` (1 to 100) sets the brightness and `color` a color name like "warm
    white" or "blue"."""
    data: dict[str, Any] = {}
    if brightness_pct is not None:
        data["brightness_pct"] = max(1, min(int(brightness_pct), 100))
    if color:
        data["color_name"] = str(color).strip().lower().replace("warm white", "navajowhite")
    return await _switch(ctx, entity_id, True, data or None)



@itool(PID, "home_assistant_turn_off", outcome="home", risk=Risk.write, risk_fn=_security_risk,
       describe_fn=_describe("Turns something off", SECURITY_SWITCH_LABEL))
async def ha_turn_off(ctx: ToolContext, entity_id: str) -> dict:
    """Turn off lights, switches, fans and other Home Assistant devices. `entity_id` is one id or several separated
    by commas."""
    return await _switch(ctx, entity_id, False)



@itool(PID, "home_assistant_set_temperature", outcome="home", risk=Risk.write, describe_fn=_describe("Sets a temperature"))
async def ha_set_temperature(ctx: ToolContext, entity_id: str, temperature: float, hvac_mode: str | None = None) -> dict:
    """Set a thermostat (climate entity) to a temperature, in the units Home Assistant uses. `hvac_mode` is optional:
    heat, cool, heat_cool, auto or off."""
    entities = await _resolve(ctx, entity_id, {"climate", "water_heater"}, "{name} isn't a thermostat.")
    if not -50 <= float(temperature) <= 120:
        raise IntegrationError("That temperature looks wrong. Give it in degrees, like 21.")
    data: dict[str, Any] = {"temperature": float(temperature)}
    if hvac_mode:
        data["hvac_mode"] = str(hvac_mode).strip().lower()
    domain = entities[0]["domain"]
    return await _service(ctx, domain, "set_temperature", [e for e in entities if e["domain"] == domain], data)


@itool(PID, "home_assistant_activate_scene", outcome="home", risk=Risk.write, describe_fn=_describe("Starts a scene"))
async def ha_activate_scene(ctx: ToolContext, entity_id: str) -> dict:
    """Activate a Home Assistant scene, like "Movie night" (scene.movie_night)."""
    return await _service(ctx, "scene", "turn_on", await _resolve(ctx, entity_id, {"scene"}, "{name} isn't a scene."))


@itool(PID, "home_assistant_run_script", outcome="home", risk=Risk.write, risk_fn=_security_risk,
       describe_fn=_describe("Runs a script", SECURITY_SWITCH_LABEL))
async def ha_run_script(ctx: ToolContext, entity_id: str) -> dict:
    """Run a Home Assistant script (script.<name>). One that sounds like it opens, unlocks or disarms something
    asks every time."""
    return await _service(ctx, "script", "turn_on", await _resolve(ctx, entity_id, {"script"}, "{name} isn't a script."))



@itool(PID, "home_assistant_open_cover", outcome="home", risk=Risk.write, risk_fn=_security_risk,
       describe_fn=_describe("Opens something", OPENS_GATE_LABEL))
async def ha_open_cover(ctx: ToolContext, entity_id: str) -> dict:
    """Open blinds, shades, curtains, a garage door or a gate (cover entities). Garage doors, gates and doors ask
    every time."""
    return await _service(ctx, "cover", "open_cover", await _resolve(ctx, entity_id, {"cover"}, "{name} isn't a cover."))



@itool(PID, "home_assistant_close_cover", outcome="home", risk=Risk.write, risk_fn=_security_risk,
       describe_fn=_describe("Closes something"))
async def ha_close_cover(ctx: ToolContext, entity_id: str) -> dict:
    """Close blinds, shades, curtains, a garage door or a gate (cover entities). Garage doors, gates and doors ask
    every time."""
    return await _service(ctx, "cover", "close_cover", await _resolve(ctx, entity_id, {"cover"}, "{name} isn't a cover."))



@itool(PID, "home_assistant_lock", outcome="home", risk=Risk.send, allow_for_chat=False,
       describe_fn=_describe("Locks a door"))
async def ha_lock(ctx: ToolContext, entity_id: str) -> dict:
    """Lock a door lock in Home Assistant."""
    return await _service(ctx, "lock", "lock", await _resolve(ctx, entity_id, {"lock"}, "{name} isn't a lock."))



@itool(PID, "home_assistant_unlock", outcome="home", risk=Risk.send, allow_for_chat=False,
       describe_fn=_describe(UNLOCKS_LABEL))
async def ha_unlock(ctx: ToolContext, entity_id: str) -> dict:
    """Unlock a door lock in Home Assistant. Always asks first."""
    return await _service(ctx, "lock", "unlock", await _resolve(ctx, entity_id, {"lock"}, "{name} isn't a lock."))



@itool(PID, "home_assistant_set_alarm", outcome="home", risk=Risk.send, allow_for_chat=False,
       describe_fn=_describe_alarm)
async def ha_set_alarm(ctx: ToolContext, entity_id: str, mode: str) -> dict:
    """Arm or disarm a Home Assistant alarm panel. `mode` is arm_home, arm_away, arm_night, arm_vacation or disarm.
    Disarming always asks first. An alarm that needs a code has to be disarmed in Home Assistant itself."""
    m = str(mode or "").strip().lower().replace(" ", "_")
    if m not in ALARM_MODES:
        raise IntegrationError("Choose arm_home, arm_away, arm_night, arm_vacation or disarm.")
    entities = await _resolve(ctx, entity_id, {"alarm_control_panel"}, "{name} isn't an alarm panel.")
    return await _service(ctx, "alarm_control_panel", f"alarm_{m}", entities)



# ---------------------------------------------------------------------------- quick commands (#356)
def match(entities: list[dict], words: str, *, device: str, plural: bool, everywhere: bool = False) -> list[dict] | None:
    """The entities a quick command means, or None when it is not exactly clear (the model then answers).

    ``words`` is what the user called it without the article ("living room lights", "bedroom lamp"); ``device`` the
    kind it named ("light", "switch" or "" for none); ``plural`` says they asked for several ("lights");
    ``everywhere`` is "all the lights". A match is, in order: the one entity with exactly that name (a light, switch
    or fan), every one of that kind in the room of that name, or the ones of that kind whose names hold every word.
    Without a kind named only an exact name counts. Security entities never match, and if one would have, the whole
    command goes to the model."""
    kinds = {"light": {"light"}, "switch": {"switch", "input_boolean"}}.get(device, set())
    said = _words(words)
    base = [w for w in said if w not in DEVICE_WORDS]

    def singular(ws: list[str]) -> str:
        return " ".join(w[:-1] if w.endswith("s") and len(w) > 3 else w for w in ws)

    exact = [e for e in entities if " ".join(_words(e["name"])) in {" ".join(said), singular(said)}]
    if exact and not everywhere:  # "all the kitchen lights" means every one, not only the one named that
        e = exact[0]
        return [e] if len(exact) == 1 and e["domain"] in QUICK_DOMAINS and not is_security(e) else None
    if not kinds:
        return None
    if everywhere and not base:
        found = [e for e in entities if e["domain"] in kinds]
    elif not base:
        return None
    else:
        room = " ".join(base)
        found = [e for e in entities if e["area"] and " ".join(_words(e["area"])) == room and e["domain"] in kinds]
        if not found:
            found = [e for e in entities if e["domain"] in kinds and all(
                w in set(_words(e["name"])) or singular([w]) in set(_words(e["name"])) for w in base)]
    if not found or any(is_security(e) for e in found):
        return None
    if len(found) > 1 and not plural:
        return None  # "the bedroom light" and there are two: which one?
    return found


QUICK_DOMAINS = {"light", "switch", "input_boolean", "fan"}
DEVICE_WORDS = {"light", "lights", "lamp", "lamps", "switch", "switches", "plug", "plugs", "bulb", "bulbs"}


async def quick_entities(mgr: IntegrationManager) -> list[dict] | None:
    """The entities quick commands may use, from the cache when it is fresh; None when Home Assistant can't be
    reached (the model then answers)."""
    try:
        return await home(mgr)
    except Exception as exc:
        log.info("quick command: Home Assistant not reachable (%s)", type(exc).__name__)
        return None


# ---------------------------------------------------------------------------- live updates
def state_item(event: dict, by_id: dict[str, dict], choice: dict) -> dict | None:
    """A ``source.items`` item for one ``state_changed`` event, or None when it should not start anything: an
    attribute-only change, a device going unavailable or coming back, a noisy kind (sensors, the sun) or a device
    Sentient may not use."""
    data = event.get("data") or {}
    eid = str(data.get("entity_id") or "")
    old, new = data.get("old_state") or {}, data.get("new_state") or {}
    before, after = old.get("state"), new.get("state")
    domain = eid.split(".", 1)[0]
    if not eid or before == after or before in UNSETTLED or after in UNSETTLED or domain in NOISY_DOMAINS:
        return None
    if not allowed(eid, choice):
        return None
    known = by_id.get(eid) or {}
    attrs = new.get("attributes") or {}
    changed_at = new.get("last_changed") or event.get("time_fired") or ""
    return {
        "id": f"{eid}@{changed_at}", "entity_id": eid, "name": attrs.get("friendly_name") or known.get("name") or eid,
        "domain": domain, "area": known.get("area") or "", "from": before, "to": after, "changed_at": changed_at,
        "device_class": attrs.get("device_class"),
    }


# ---------------------------------------------------------------------------- the plugin
INSTRUCTIONS = (
    "1. Open Home Assistant in your browser and note its address, for example `http://homeassistant.local:8123`.\n"
    "2. Click your name at the bottom left to open your profile, then open the **Security** tab.\n"
    "3. Under **Long-lived access tokens**, click **Create token**, name it `Sentient` and click **OK**.\n"
    "4. Copy the token right away. Home Assistant shows it only once.\n"
    "5. Paste the address and the token here and click **Connect**.\n\n"
    "Sentient can turn lights and switches on and off, set thermostats and run scenes on its own when you ask. "
    "Locks, alarms, garage doors and gates always ask you first. After connecting, you can choose which devices "
    "Sentient may use."
)


class HomeAssistantPlugin(IntegrationPlugin):
    id = PID
    display_name = "Home Assistant"
    description = ("Control your smart home: lights, switches, thermostats, scenes and scripts, with locks, alarms "
                   "and garage doors always asking first. Tasks can start when something changes, like a door opening.")
    category = "home"
    icon = "home-assistant"
    auth_type = "api_key"
    selection_hint = ("Use for the smart home in Home Assistant: lights, lamps, switches, plugs, fans, thermostats, "
                      "temperature, locks, doors, garage, blinds, alarms, sensors, scenes and scripts.")
    setup_fields = [
        SetupField("url", "Home Assistant address", secret=False, required=True,
                   placeholder="http://homeassistant.local:8123",
                   help="The address you open Home Assistant at."),
        SetupField("token", "Long-lived access token", secret=True, required=True,
                   help="From your Home Assistant profile, Security tab, Long-lived access tokens."),
    ]
    docs_url = "https://www.home-assistant.io/docs/authentication/#your-account-profile"
    instructions_md = INSTRUCTIONS
    feed_kind = "home_assistant_live"
    triggers = [{"event": "state_changed", "label": "Device change"}]
    entity_picker = True
    tools = [ha_find_entities, ha_list_areas, ha_list_devices, ha_get_state, ha_turn_on, ha_turn_off,
             ha_set_temperature, ha_activate_scene, ha_run_script, ha_open_cover, ha_close_cover, ha_lock, ha_unlock,
             ha_set_alarm]

    async def validate(self, fields: dict[str, str], mgr: IntegrationManager) -> tuple[dict, str | None]:
        url = check_url(str(fields.get("url", "")))
        token = str(fields.get("token", "")).strip()
        if not token:
            raise IntegrationError("Please paste a long-lived access token from your Home Assistant profile.")
        c = {"url": url, "token": token}
        try:
            info = await rest(c, "GET", "/api/config")
        except IntegrationError:
            raise IntegrationError("Home Assistant didn't accept that token. Create a new long-lived access token "
                                   "in your profile, on the Security tab, and paste it here.") from None
        except Exception as exc:
            raise IntegrationError(f"Couldn't reach Home Assistant at {url}. Check the address, and that this "
                                   f"computer is on the same network ({type(exc).__name__}).") from None
        _CACHE.pop(url, None)
        return c, str((info or {}).get("location_name") or "") or urlsplit(url).hostname

    async def quick_match(self, mgr: IntegrationManager, words: str, *, device: str, plural: bool,
                          everywhere: bool = False) -> list[dict] | None:
        """For quick commands: the devices the words name exactly, or None (see ``match``)."""
        entities = await quick_entities(mgr)
        if not entities:
            return None
        return match(entities, words, device=device, plural=plural, everywhere=everywhere)

    async def entities(self, mgr: IntegrationManager) -> list[dict]:
        """Every entity, for the picker (``GET /api/integrations/home_assistant/entities``)."""
        return [{k: e[k] for k in ("entity_id", "name", "domain", "area", "state")}
                for e in await home(mgr, fresh=True, everything=True)]

    async def watch(self, mgr: IntegrationManager) -> None:
        """Live updates: ``state_changed`` events become ``source.items`` (event ``state_changed``), reconnecting with
        backoff. Off with ``integrations.home_assistant_live``."""
        while True:
            if not getattr(mgr.app.config.integrations, "home_assistant_live", True):
                return
            c = await mgr.get_credentials(PID)
            if not c:
                return
            try:
                await self._listen(mgr, c)
                failure: Exception = ConnectionError("Home Assistant closed the connection")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                failure = exc
                if isinstance(exc, IntegrationError) and "token" in str(exc):
                    await mgr.set_error(PID, str(exc))
            res = await mgr.feeds.record_failure(PID, failure, base=BACKOFF_BASE_S)
            await asyncio.sleep(res["retry_in_s"])

    async def _listen(self, mgr: IntegrationManager, c: dict) -> None:
        """One live connection, until Home Assistant closes it."""
        by_id = {e["entity_id"]: e for e in await home(mgr, everything=True)}
        unknown: set[str] = set()
        async with Socket(c) as ws:
            await ws.call({"type": "subscribe_events", "event_type": "state_changed"})
            await mgr.feeds.record_success(PID, cursor="live", note="Listening for changes")
            log.info("Home Assistant live updates on (%d devices)", len(by_id))
            async for event in ws.events():
                eid = str((event.get("data") or {}).get("entity_id") or "")
                if eid and eid not in by_id and eid not in unknown:  # a new device: look once more
                    by_id = {e["entity_id"]: e for e in await home(mgr, everything=True)}
                    if eid not in by_id:
                        unknown.add(eid)
                item = state_item(event, by_id, entity_choice(mgr))
                if item is not None:
                    await mgr.emit_items(PID, "feed", [item], event="state_changed")


PLUGIN = HomeAssistantPlugin()
