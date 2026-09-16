from __future__ import annotations

import pytest

from sentient.app import SentientApp
from tests.conftest import FakeProvider


class FakeService:
    name = "fake"

    async def start(self):
        return None

    async def stop(self):
        return None


class FakeTasks(FakeService):
    def __init__(self):
        self.created: list[dict] = []
        self.events: list[tuple] = []
        self.tasks: list[dict] = []

    async def create_task(self, prompt, *, source="user", original_context=None, **kw):
        task = {"task_id": f"task{len(self.created) + 1}", "name": prompt.splitlines()[0], "status": "planning",
                "original_context": original_context, "prompt": prompt}
        self.created.append(task)
        return task

    async def handle_event(self, source, event_type, event_data, event_id=None):
        self.events.append((source, event_type, event_id))
        return []

    async def list(self):
        return self.tasks


class FakeIntegrations(FakeService):
    def __init__(self, items=None, email="sarthak@example.com"):
        self.items = {"gmail": list(items or []), "gcalendar": []}
        self.email = email
        self.polls: list[tuple] = []

    async def is_connected(self, plugin_id):
        return plugin_id in {"gmail", "gcalendar"}

    async def poll_source(self, source, since=None):
        self.polls.append((source, since))
        out, self.items[source] = self.items.get(source, []), []
        return out

    async def integration(self, plugin_id):
        return {"id": plugin_id, "account_label": self.email}


class FakeUserModel(FakeService):
    def __init__(self):
        self.context = ""
        self.queries: list[str] = []

    async def context_for(self, text):
        self.queries.append(text)
        return self.context


MEETING_EMAIL = {
    "id": "msg-1", "thread_id": "th-1", "from": "Jane Doe <jane@acme.com>", "sender_email": "jane@acme.com",
    "to": "sarthak@example.com", "subject": "Meeting about Project Phoenix",
    "snippet": "Can we meet Tuesday at 2pm to go over the launch plan?",
    "body": "Hi Sarthak, can we meet Tuesday at 2pm to go over the Project Phoenix launch plan? Thanks, Jane",
    "date": "2026-09-15T08:00:00+00:00", "labels": ["INBOX", "IMPORTANT"],
    "url": "https://mail.google.com/mail/u/0/#inbox/th-1",
}

ACTIONABLE = {
    "actionable": True, "confidence_score": 0.82,
    "reasoning": "Jane asks for a meeting; calendar is free.",
    "suggestion_description": "Draft a reply to Jane confirming Tuesday at 2pm",
    "suggestion_type_description": "A suggestion to draft an email confirming a proposed meeting.",
    "suggestion_action_details": {"action_type": "draft_email", "recipient": "jane@acme.com"},
}


@pytest.fixture
def fake_tasks():
    return FakeTasks()


@pytest.fixture
async def app(config, isolated_home, fake_tasks, monkeypatch):
    llm = FakeProvider()
    a = SentientApp(config, llm=llm, db_path=isolated_home / "pro.db", enable_background=False)
    await a.start()
    monkeypatch.setattr(a, "tasks", fake_tasks)
    monkeypatch.setattr(a, "integrations", FakeIntegrations([MEETING_EMAIL]))
    monkeypatch.setattr(a, "user_model", FakeUserModel())
    a.config.proactivity.context_agent_rounds = 0
    a.config.assistant.timezone = "UTC"
    a.fake = llm
    yield a
    await a.stop()


def script_pipeline(llm, reasoner: dict, type_name: str = "draft_meeting_confirmation_email") -> None:
    """Queue the LLM replies one process_event call consumes (no memory hits in these tests)."""
    llm.json_replies.append({"event_specific_context": "Project Phoenix", "calendar_availability": "Tuesday calendar"})
    llm.json_replies.append(reasoner)
    llm.text_replies.append(type_name)
