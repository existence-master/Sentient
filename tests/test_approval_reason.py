"""The reason line of an approval card is never the tool's instructions for the model (#398)."""

from __future__ import annotations

from sentient.app import SentientApp
from sentient.llm.events import ApprovalRequest
from sentient.tools.base import Risk, ToolContext, ToolPlugin, approval_reason, tool
from tests.conftest import FakeProvider, tool_call

MODEL_TEXT = "Click the element with this ref (from snapshot), e.g. \"e12\". Never type passwords. Buttons that buy"


def test_only_a_label_that_says_more_than_the_risk_is_a_reason():
    assert approval_reason({"risk_label": "Sends", "target": "Send message"}, Risk.send) == ""
    assert approval_reason({"risk_label": "Changes something"}, Risk.write) == ""
    assert approval_reason({"risk_label": "Purchase", "target": "Place order"}, Risk.send) == "Purchase"
    assert approval_reason({"risk_label": "Unlocks a door"}, Risk.send) == "Unlocks a door"
    assert approval_reason({}, Risk.send) == ""


async def test_an_approval_request_carries_no_model_instructions(config, isolated_home):
    def describe(arguments, ctx):
        return {"risk_label": "Purchase" if "order" in arguments.get("label", "").lower() else "Sends",
                "target": arguments.get("label")}

    # the description is what the model reads, as a plugin tool's docstring would be
    @tool("page_press", risk=Risk.send, description=MODEL_TEXT, describe_fn=describe)
    async def page_press(ctx: ToolContext, label: str) -> dict:
        return {"pressed": label}

    class Page(ToolPlugin):
        id = "page"
        display_name = "Page"
        tools = [page_press]

    config.tools.approvals.mode = "ask"
    llm = FakeProvider(replies=[
        [tool_call("page_press", label="Send message")], "not sent",
        [tool_call("page_press", label="Place order")], "not bought",
    ])
    s = await SentientApp(config, llm=llm, db_path=isolated_home / "reason.db").start()
    try:
        s.registry.register(Page())
        sid = await s.store.create_session(channel="desktop")
        asked: list[ApprovalRequest] = []
        for text in ("send it", "order it"):
            async for ev in s.agent.run_turn(sid, text, channel="desktop"):
                if isinstance(ev, ApprovalRequest):
                    asked.append(ev)
                    s.approvals.resolve(ev.approval_id, "deny")
        assert [(a.reason, a.risk_label, a.target) for a in asked] == [
            ("", "Sends", "Send message"), ("Purchase", "Purchase", "Place order"),
        ]
        assert all("Click the element" not in a.model_dump_json() for a in asked)
    finally:
        await s.stop()
