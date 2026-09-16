from fastapi.testclient import TestClient

from sentient.app import SentientApp
from sentient.evolution.log import log_event
from sentient.gateway.app import create_app
from tests.conftest import FakeProvider

BODY = "\n".join(
    [
        "# Weekly review",
        "",
        "## When to use",
        "Fridays",
        "",
        "## Procedure",
        "1. Gather",
        "",
        "## Pitfalls",
        "None",
        "",
        "## Verification",
        "Sent",
    ]
)


def test_edit_pending_proposal_and_reason(config, isolated_home, monkeypatch):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "skills-token")
    core = SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "sk.db", enable_background=False)
    with TestClient(create_app(core)) as c:
        c.headers["Authorization"] = "Bearer skills-token"
        created = c.post("/api/skills", json={"name": "weekly-review", "description": "v1", "body": BODY})
        assert created.status_code == 200
        active_before = c.get("/api/skills/weekly-review").json()

        async def propose() -> None:
            core.skills.write(
                "weekly-review", "v2 proposal", BODY + "\n5. Health", author="assistant", pending=True
            )
            await log_event(
                core.store,
                "skill_patched",
                {"name": "weekly-review", "pending": True, "reason": "You asked for health stats twice", "task_id": "t1"},
            )

        c.portal.call(propose)
        listing = c.get("/api/skills").json()
        pending = next(p for p in listing["pending"] if p["name"] == "weekly-review")
        assert pending["reason"] == "You asked for health stats twice"
        assert pending["origin"] == {"task_id": "t1"}
        assert pending["proposed_at"]

        edited = c.put("/api/skills/weekly-review?target=pending", json={"description": "v2 edited"}).json()
        assert edited["description"] == "v2 edited"
        active_after = c.get("/api/skills/weekly-review").json()
        assert active_after["description"] == active_before["description"]
        assert active_after["version"] == active_before["version"]

        assert c.put("/api/skills/weekly-review", json={"target": "sideways"}).status_code == 400
        assert c.put("/api/skills/nothing-here?target=pending", json={"body": "x"}).status_code == 404
