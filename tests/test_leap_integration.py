"""Cross-package behaviour wired by the lead after the V3 leap builders finished."""

from __future__ import annotations

from sentient.app import SentientApp
from sentient.browser import safety
from sentient.tools.base import ToolContext
from tests.conftest import FakeProvider


async def _app(config) -> SentientApp:
    return await SentientApp(config, llm=FakeProvider(), enable_background=False).start()


async def test_deleting_a_webhook_switches_off_its_tasks_and_says_so(config):
    app = await _app(config)
    try:
        hook = await app.integrations.hooks.create("Door sensor")
        now = app.tasks.now_iso()
        listening = await app.tasks.repo.insert_task({
            "name": "Door alert", "status": "active", "enabled": True,
            "schedule": {"type": "triggered", "source": "webhook", "event": hook["id"], "filter": {}},
            "plan": [{"tool": "time", "description": "Note the time"}], "created_at": now, "updated_at": now,
        })
        other = await app.tasks.repo.insert_task({
            "name": "Other hook", "status": "active", "enabled": True,
            "schedule": {"type": "triggered", "source": "webhook", "event": "someotherhook", "filter": {}},
            "plan": [{"tool": "time", "description": "Note the time"}], "created_at": now, "updated_at": now,
        })
        assert await app.integrations.hooks.delete(hook["id"])
        assert (await app.tasks.get(listening))["enabled"] is False
        assert (await app.tasks.get(other))["enabled"] is True
        notes = [n for n in await app.notifications.list() if (n.get("payload") or {}).get("event") == "webhook_removed"]
        assert notes and "Door alert" in notes[0]["message"]
    finally:
        await app.stop()


async def test_lan_listener_serves_webhooks(config):
    import httpx

    from sentient.nodes.lan import create_lan_app

    app = await _app(config)
    try:
        hook = await app.integrations.hooks.create("Garden sensor")
        gate = create_lan_app(app, "lan-secret")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gate), base_url="https://lan.test") as client:
            ok = await client.post(f"/hooks/{hook['id']}", json={"moisture": 12},
                                   headers={"X-Sentient-Secret": hook["secret"]})
            bad = await client.post(f"/hooks/{hook['id']}", json={}, headers={"X-Sentient-Secret": "nope"})
            manage = await client.get("/api/hooks", headers={"Authorization": "Bearer lan-secret"})
        assert ok.status_code == 200 and ok.json()["ok"] is True
        assert bad.status_code == 401
        assert manage.status_code == 404  # hook management never reaches the network
    finally:
        await app.stop()


def test_device_capture_approvals_have_plain_wording(config):
    from sentient.nodes.tools import device_capture_screen, device_take_photo

    ctx = ToolContext(store=None, config=config, llm=None)
    assert device_take_photo.describe_fn({"device": "Glasses"}, ctx) == {"risk_label": "Takes a photo", "target": "Glasses"}
    assert device_capture_screen.describe_fn({}, ctx)["target"] == "your device"


def test_browser_approval_wording():
    order = {"role": "button", "name": "Place order", "is_submit": True}
    assert safety.approval_target(order) == "Place order"
    assert safety.risk_label_for("Place order", "click") == "Purchase"
    assert safety.risk_label_for("Delete my account", "click") == "Deletes"
    assert safety.risk_label_for("Post", "click") == "Posts publicly"
    assert safety.risk_label_for("Send message", "type") == "Sends"
    assert safety.risk_label_for("Next page", "click") == "Clicks"
    assert safety.approval_target(None) == ""


def test_browser_ref_from_whole_snapshot_line():
    """Real qwen3:8b passed '[e4] button "Place order"' as the ref; the risk check must still find e4."""
    from sentient.browser.service import _clean_ref

    assert _clean_ref("e4") == "e4"
    assert _clean_ref("[e4]") == "e4"
    assert _clean_ref('[e4] button "Place order"') == "e4"
    assert _clean_ref("E12") == "e12"
    assert _clean_ref("e2e test") is None
    assert _clean_ref("") is None


async def test_shutdown_cancels_stuck_background_work(config):
    """Seen on a Linux CI runner: a background job that never finished made app.stop() hang forever."""
    import asyncio

    import sentient.app as app_module

    app = await _app(config)
    stuck = asyncio.Event()
    app.agent._spawn(stuck.wait())  # never set
    original = app_module.SHUTDOWN_GRACE_S
    app_module.SHUTDOWN_GRACE_S = 0.2
    try:
        await asyncio.wait_for(app.stop(), 10)
    finally:
        app_module.SHUTDOWN_GRACE_S = original
    assert not app.agent._background
