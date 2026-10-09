"""Agent tools for the user's devices (plugin ``devices``, hidden while no usable device is online)."""

from __future__ import annotations

import base64
import secrets
from datetime import UTC, datetime
from typing import Any

from sentient import paths
from sentient.nodes.service import DeviceError
from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool

IMAGE_EXT = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
NO_VISION_HINTS = ("vision", "image", "multimodal", "multi-modal", "not support", "unsupported content", "image_url")
# a missing library or an unreachable server is not the model lacking vision
SETUP_HINTS = ("pip install", "pillow", "no module named", "connection", "timed out", "refused")


def _app(ctx: ToolContext) -> Any:
    app = ctx.extra.get("app")
    if app is None:
        raise DeviceError("Devices are not available here.")
    return app


def _camera_risk(arguments: dict, ctx: ToolContext) -> Risk | None:
    """Photos and screenshots are private: ask first unless the user turned that off."""
    nodes = getattr(getattr(ctx, "config", None), "nodes", None)
    return Risk.send if nodes is None or nodes.camera_requires_approval else None


def _camera_describe(label: str):
    def describe(arguments: dict, ctx: ToolContext) -> dict:
        device = str((arguments or {}).get("device") or "").strip()
        return {"risk_label": label, "target": device or "your device"}

    return describe


def _save_image(data: dict, prefix: str) -> tuple[str, str, str]:
    """Save ``{mime, base64}`` under files/outputs/devices. Returns (relative file, mime, base64)."""
    b64 = data.get("base64")
    if not isinstance(b64, str) or not b64:
        raise DeviceError("The device did not send an image.")
    mime = str(data.get("mime") or "image/jpeg").lower()
    try:
        raw = base64.b64decode(b64, validate=False)
    except ValueError as exc:
        raise DeviceError("The device sent an unreadable image.") from exc
    out_dir = paths.files_dir() / "outputs" / "devices"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{prefix}-{datetime.now(UTC):%Y%m%d-%H%M%S}-{secrets.token_hex(2)}{IMAGE_EXT.get(mime, '.jpg')}"
    (out_dir / name).write_bytes(raw)
    return f"outputs/devices/{name}", mime, b64


def _pcm_from_wav(wav: bytes) -> tuple[bytes, int, int] | None:
    """Split a PCM16 WAV into (samples, sample_rate, channels) so tiny devices get raw frames.

    Returns None for anything that is not 16-bit PCM, and the caller falls back to sending the WAV.
    """
    if len(wav) < 44 or wav[:4] != b"RIFF" or wav[8:12] != b"WAVE":
        return None
    pos, rate, channels, bits, pcm = 12, 16000, 1, 16, None
    while pos + 8 <= len(wav):
        chunk, size, body = wav[pos : pos + 4], int.from_bytes(wav[pos + 4 : pos + 8], "little"), pos + 8
        if chunk == b"fmt " and size >= 16:
            if int.from_bytes(wav[body : body + 2], "little") != 1:  # 1 = uncompressed PCM
                return None
            channels = int.from_bytes(wav[body + 2 : body + 4], "little")
            rate = int.from_bytes(wav[body + 4 : body + 8], "little")
            bits = int.from_bytes(wav[body + 14 : body + 16], "little")
        elif chunk == b"data":
            pcm = wav[body : body + size]
            break
        pos = body + size + (size & 1)  # chunks are word aligned
    if pcm is None or bits != 16 or not rate:
        return None
    return pcm, rate, channels


async def _describe(app: Any, mime: str, b64: str, question: str, source: str) -> tuple[str | None, str | None]:
    prompt = (
        f"This image was just captured from the user's {source}. {question.strip()}\n"
        "Answer in a few clear sentences. Only describe what you can actually see."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ],
        }
    ]
    try:
        text = await app.llm.complete_text("vision", messages)
    except Exception as exc:
        model = ""
        try:
            model = app.llm.model_for("vision")
        except Exception:
            model = "the current model"
        low = str(exc).lower()
        if any(h in low for h in NO_VISION_HINTS) and not any(h in low for h in SETUP_HINTS):
            return None, (
                f"The image was saved, but {model} cannot look at images. Choose a model that supports images "
                "for the Vision role in Settings > Models."
            )
        return None, f"The image was saved, but describing it failed: {exc}"
    return (text or "").strip() or None, None


@tool(risk=Risk.read)
async def device_list(ctx: ToolContext) -> dict:
    """List the user's devices (phone, smart glasses, watch, this computer): which are online and what each can do."""
    nodes = await _app(ctx).nodes.list_nodes()
    return {
        "devices": [
            {
                "name": n["name"], "kind": n["kind"], "online": n["online"], "battery": n["battery"],
                "can": n["capabilities"] if n["online"] else [],
            }
            for n in nodes
        ]
    }  # fmt: skip


@tool(risk=Risk.read, untrusted_output=True)  # a photo can show text someone else wrote (ADR 0018)
async def device_take_photo(ctx: ToolContext, device: str = "", question: str = "") -> dict:
    """Take a photo with a device camera. Smart glasses see what the user is looking at.
    question: what you want to know about the photo, e.g. "What am I looking at?". Leave empty to just save it.
    device: device name or kind ("glasses", "phone"); empty picks the best one."""
    app = _app(ctx)
    conn, cap = app.nodes.resolve(device, ["camera.photo"])
    data = await app.nodes.call(conn, cap, {"facing": "back"}, timeout_ms=max(app.config.nodes.invoke_timeout_s, 20) * 1000)
    file, mime, b64 = _save_image(data, "photo")
    out: dict[str, Any] = {"file": file, "device": conn.name, "description": None}
    if question.strip():
        description, note = await _describe(app, mime, b64, question, f"{conn.kind} camera ({conn.name})")
        out["description"] = description
        if note:
            out["note"] = note
    return out


@tool(risk=Risk.read, untrusted_output=True)
async def device_capture_screen(ctx: ToolContext, device: str = "", question: str = "") -> dict:
    """Take a screenshot of a device screen (usually this computer).
    question: what you want to know about the screen; empty just saves the screenshot.
    device: device name or kind; empty picks this computer when possible."""
    app = _app(ctx)
    conn, cap = app.nodes.resolve(device, ["screen.capture"])
    data = await app.nodes.call(conn, cap, {})
    file, mime, b64 = _save_image(data, "screen")
    out: dict[str, Any] = {"file": file, "device": conn.name, "description": None}
    if question.strip():
        description, note = await _describe(app, mime, b64, question, f"screen ({conn.name})")
        out["description"] = description
        if note:
            out["note"] = note
    return out


@tool(risk=Risk.read)
async def device_get_location(ctx: ToolContext, device: str = "") -> dict:
    """Get the current location of the user's phone or another device. device: name or kind; empty picks the best one."""
    app = _app(ctx)
    conn, cap = app.nodes.resolve(device, ["location.get"])
    data = await app.nodes.call(conn, cap, {})
    lat, lon = data.get("lat"), data.get("lon")
    out = {"device": conn.name, "lat": lat, "lon": lon, "accuracy_m": data.get("accuracy_m")}
    if data.get("label"):
        out["label"] = data["label"]
    if lat is not None and lon is not None:
        out["map"] = f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map=17/{lat}/{lon}"
    return out


@tool(risk=Risk.write, internal=True)
async def device_notify(ctx: ToolContext, text: str, device: str = "") -> dict:
    """Show a short notification on the user's device (phone, glasses, watch). device: name or kind; empty picks the best one."""
    app = _app(ctx)
    conn, cap = app.nodes.resolve(device, ["notify.show", "display.text", "display.card"])
    params = {"title": app.config.assistant.name, "text": text} if cap != "display.text" else {"text": text}
    await app.nodes.call(conn, cap, params)
    return {"ok": True, "device": conn.name}


@tool(risk=Risk.write, internal=True)
async def device_display(ctx: ToolContext, text: str, device: str = "") -> dict:
    """Show text on a device screen, like the display in smart glasses. Keep it short. device: name or kind; empty picks the best one."""
    app = _app(ctx)
    conn, cap = app.nodes.resolve(device, ["display.card", "display.text", "notify.show"])
    params = {"text": text} if cap == "display.text" else {"title": app.config.assistant.name, "text": text}
    await app.nodes.call(conn, cap, params)
    return {"ok": True, "device": conn.name}


@tool(risk=Risk.write, internal=True)
async def device_speak(ctx: ToolContext, text: str, device: str = "") -> dict:
    """Say something out loud through a device speaker (glasses, phone). device: name or kind; empty picks the best one."""
    app = _app(ctx)
    conn, cap = app.nodes.resolve(device, ["speak", "audio.pcm", "audio.play"])
    if cap == "speak":
        await app.nodes.call(conn, cap, {"text": text})
        return {"ok": True, "device": conn.name}
    voice = getattr(app, "voice", None)
    if voice is None:
        raise DeviceError(f"{conn.name} can only play audio and voice is not available.")
    try:
        wav = await voice.speak(text)
    except Exception as exc:
        raise DeviceError(f"Could not turn the text into speech: {exc}") from exc
    split = _pcm_from_wav(wav) if cap == "audio.pcm" else None
    if split is not None:
        # microcontrollers get raw PCM16 in one binary frame: no base64, no big JSON string in RAM
        pcm, rate, channels = split
        params = {"format": "pcm16", "sample_rate": rate, "channels": channels, "text": text}
        await app.nodes.call(conn, "audio.pcm", params, timeout_ms=60_000, payload=pcm)
    else:
        target = "audio.play" if cap == "audio.pcm" else cap
        if target not in conn.capabilities:
            raise DeviceError(f"{conn.name} cannot play this audio format.")
        await app.nodes.call(conn, target, {"mime": "audio/wav", "base64": base64.b64encode(wav).decode(), "text": text})
    return {"ok": True, "device": conn.name}


# Approvals use the effective risk (docs/API.md section 10).
# Photos and screenshots ask first (nodes.camera_requires_approval); approvals honour risk_fn and describe_fn.
device_take_photo.risk_fn = _camera_risk  # type: ignore[attr-defined]
device_capture_screen.risk_fn = _camera_risk  # type: ignore[attr-defined]
device_take_photo.describe_fn = _camera_describe("Takes a photo")  # type: ignore[attr-defined]
device_capture_screen.describe_fn = _camera_describe("Looks at the screen")  # type: ignore[attr-defined]


class DevicesPlugin(ToolPlugin):
    id = "devices"
    display_name = "Devices"
    description = "Your phone, smart glasses and other paired devices: camera, screen, location, notifications, speech."
    category = "utilities"
    icon = "IconDeviceMobile"
    selection_hint = (
        "the user's phone or smart glasses: take a photo of what they see, where they are, show or say something on the device"
    )
    tools = [
        device_list, device_take_photo, device_capture_screen, device_get_location,
        device_notify, device_display, device_speak,
    ]  # fmt: skip
