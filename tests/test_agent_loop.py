
from sentient.agent import outcomes
from sentient.agent.loop import action_call_history, browser_click_history
from sentient.app import SentientApp
from sentient.llm.events import ApprovalRequest, Done, TextDelta, ToolResultEvent
from sentient.llm.provider import StreamChunk, ToolCall
from tests.conftest import FakeProvider, tool_call


async def _collect(agent, sid, text, on_event=None):
    events = []
    async for ev in agent.run_turn(sid, text, channel="cli"):
        events.append(ev)
        if on_event:
            await on_event(ev)
    return events


async def test_plain_reply_persists_and_streams(config, isolated_home):
    llm = FakeProvider(replies=["Hello Maya!"])
    s = await SentientApp(config, llm=llm, db_path=isolated_home / "a.db").start()
    try:
        sid = await s.store.create_session(channel="cli")
        events = await _collect(s.agent, sid, "hi")
        text = "".join(e.text for e in events if isinstance(e, TextDelta))
        assert text == "Hello Maya!"
        assert isinstance(events[-1], Done) and events[-1].content == "Hello Maya!"
        msgs = await s.store.recent_messages(sid, 10)
        assert [m["role"] for m in msgs] == ["user", "assistant"]
        system = llm.calls[0]["messages"][0]["content"]
        assert "Maya" in system and "## Working rules" in system
        assert llm.calls[0]["tools"], "tools must be offered to the model"
    finally:
        await s.stop()


async def test_tool_round_trip(config, isolated_home):
    llm = FakeProvider(replies=[[tool_call("current_datetime")], "It is now."])
    s = await SentientApp(config, llm=llm, db_path=isolated_home / "b.db").start()
    try:
        sid = await s.store.create_session(channel="cli")
        events = await _collect(s.agent, sid, "what time is it")
        kinds = [type(e).__name__ for e in events]
        assert "ToolCallEvent" in kinds and "ToolResultEvent" in kinds
        res = next(e for e in events if isinstance(e, ToolResultEvent))
        assert not res.is_error and "iso" in res.result
        msgs = await s.store.recent_messages(sid, 10)
        assert [m["role"] for m in msgs] == ["user", "assistant", "tool", "assistant"]
        # the second model call must see the tool result
        second = llm.calls[1]["messages"]
        assert second[-1]["role"] == "tool" and second[-2]["tool_calls"]
    finally:
        await s.stop()


async def test_approval_gate_deny_and_allow(config, isolated_home):
    from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool

    sent: list[str] = []

    @tool("send_postcard", risk=Risk.write)
    async def send_postcard(ctx: ToolContext, text: str) -> dict:
        """Post a postcard to someone outside Sentient."""
        sent.append(text)
        return {"posted": text}

    class Postcards(ToolPlugin):
        id = "postcards"
        display_name = "Postcards"
        tools = [send_postcard]

    config.tools.approvals.mode = "ask"
    llm = FakeProvider(
        replies=[
            [tool_call("send_postcard", text="x")], "declined noted",
            [tool_call("send_postcard", text="y")], "posted",
        ]
    )
    s = await SentientApp(config, llm=llm, db_path=isolated_home / "c.db").start()
    try:
        s.registry.register(Postcards())
        sid = await s.store.create_session(channel="cli")

        async def deny(ev):
            """Decline the browser-click approval shown by this test."""
            if isinstance(ev, ApprovalRequest):
                assert ev.risk == "write"
                s.approvals.resolve(ev.approval_id, "deny")

        events = await _collect(s.agent, sid, "send a postcard", deny)
        res = next(e for e in events if isinstance(e, ToolResultEvent))
        assert res.is_error and "declined" in str(res.result)
        assert sent == []

        async def allow(ev):
            """Allow the postcard approval shown by this test."""
            if isinstance(ev, ApprovalRequest):
                s.approvals.resolve(ev.approval_id, "allow_session")

        events = await _collect(s.agent, sid, "send another", allow)
        res = next(e for e in events if isinstance(e, ToolResultEvent))
        assert not res.is_error and sent == ["y"]
        # session allowance now skips the prompt for the same tool
        assert not s.approvals.needs_approval(s.registry.get("send_postcard"), sid)
    finally:
        await s.stop()


async def test_reply_corrects_a_browser_click_it_never_made(config, isolated_home):
    """A natural-language click claim needs evidence from a browser click call."""
    llm = FakeProvider(replies=["I clicked the subscription button. "])
    s = await SentientApp(config, llm=llm, db_path=isolated_home / "never-clicked.db").start()
    try:
        sid = await s.store.create_session(channel="cli")
        events = await _collect(s.agent, sid, "click the subscription button")
        text = "".join(e.text for e in events if isinstance(e, TextDelta))
        assert text == "I clicked the subscription button. \n\nCorrection: I didn't click it, so nothing changed on the page."
        assert (await s.store.recent_messages(sid, 1))[-1]["content"] == text
    finally:
        await s.stop()


async def test_reply_copying_an_earlier_correction_gets_no_second_one(config, isolated_home):
    """A reply that already carries the correction, copied from the chat, is not corrected again (#488)."""
    copied = "I clicked the subscription button.\n\nCorrection: I didn't click it, so nothing changed on the page."
    llm = FakeProvider(replies=[copied])
    s = await SentientApp(config, llm=llm, db_path=isolated_home / "copied-correction.db").start()
    try:
        sid = await s.store.create_session(channel="cli")
        events = await _collect(s.agent, sid, "click the subscription button")
        text = "".join(e.text for e in events if isinstance(e, TextDelta))
        assert text == copied
        assert text.count("Correction:") == 1
    finally:
        await s.stop()


async def test_reply_quoting_the_note_without_the_correction_line_is_still_corrected(config, isolated_home):
    """Only a copied correction line counts as one: the same words elsewhere in the reply do not (#488)."""
    reply = "I didn't click it, so nothing changed on the page. Now it is done: I clicked the subscription button."
    llm = FakeProvider(replies=[reply])
    s = await SentientApp(config, llm=llm, db_path=isolated_home / "quoted-note.db").start()
    try:
        sid = await s.store.create_session(channel="cli")
        events = await _collect(s.agent, sid, "click the subscription button")
        text = "".join(e.text for e in events if isinstance(e, TextDelta))
        assert text == reply + "\n\nCorrection: I didn't click it, so nothing changed on the page."
    finally:
        await s.stop()


async def test_reply_corrects_a_browser_click_the_user_declined(config, isolated_home):
    """A denied browser-click approval must remain visible in the final reply."""
    config.tools.approvals.mode = "ask"
    llm = FakeProvider(replies=[[tool_call("browser_click", ref="e1")], "It was clicked, but the action was declined."])
    s = await SentientApp(config, llm=llm, db_path=isolated_home / "declined-click.db").start()
    try:
        sid = await s.store.create_session(channel="cli")

        async def deny(ev):
            """Decline the browser-click approval shown by this test."""
            if isinstance(ev, ApprovalRequest):
                s.approvals.resolve(ev.approval_id, "deny")

        events = await _collect(s.agent, sid, "click the subscription button", deny)
        text = "".join(e.text for e in events if isinstance(e, TextDelta))
        assert text.endswith("Correction: I didn't click it: you declined, so nothing changed on the page.")
        result = next(e for e in events if isinstance(e, ToolResultEvent))
        assert result.result["declined"] is True
    finally:
        await s.stop()


async def test_step_limit_corrects_a_declined_browser_click_claim(config, isolated_home):
    """A streamed claim is corrected even when its tool call exhausts the final round."""
    class ClaimAndClick(FakeProvider):
        async def stream(self, role, messages, tools=None, *, model=None):
            """Stream a textual click claim and its tool call in one final round."""
            self.calls.append({"role": role, "messages": messages, "tools": tools, "model": model})
            yield StreamChunk(text="The subscription button was clicked.", model="fake")
            yield StreamChunk(done=True, tool_calls=[ToolCall(**tool_call("browser_click", ref="e1"))], model="fake")

    config.models.max_tool_rounds = 1
    config.tools.approvals.mode = "ask"
    s = await SentientApp(config, llm=ClaimAndClick(), db_path=isolated_home / "limit-click.db").start()
    try:
        sid = await s.store.create_session(channel="cli")

        async def deny(ev):
            if isinstance(ev, ApprovalRequest):
                s.approvals.resolve(ev.approval_id, "deny")

        events = await _collect(s.agent, sid, "click the subscription button", deny)
        text = "".join(e.text for e in events if isinstance(e, TextDelta))
        assert text.endswith("Correction: I didn't click it: you declined, so nothing changed on the page.")
        assert isinstance(events[-1], Done) and events[-1].content == text
    finally:
        await s.stop()


def test_browser_click_correction_keeps_a_click_that_succeeded_before_a_declined_retry():
    """A true claim stays alone when any click of the claimed control succeeded, also if a later try was declined
    (#488)."""
    subscribe_then_declined_subscribe = [
        {"name": "browser_click", "ok": True, "target": 'button "Subscribe"'},
        {"name": "browser_click", "declined": True, "target": 'button "Subscribe"'},
    ]
    assert outcomes.unmade_browser_click_note("I clicked the Subscribe button.", subscribe_then_declined_subscribe) == ""
    assert outcomes.unmade_browser_click_note(
        "I clicked the Subscribe button.", [], subscribe_then_declined_subscribe
    ) == ""
    # with no successful click of that control, the latest outcome still words the correction
    assert outcomes.unmade_browser_click_note(
        "I clicked the Subscribe button.", [{"name": "browser_click", "declined": True, "target": 'button "Subscribe"'}]
    ) == "I didn't click it: you declined, so nothing changed on the page."


def test_browser_click_correction_uses_the_latest_call():
    """The latest known click outcome controls its correction wording."""
    assert outcomes.unmade_browser_click_note(
        "The button was clicked.", [{"name": "browser_click", "ok": True, "declined": False}]
    ) == ""
    assert outcomes.unmade_browser_click_note(
        "The button was clicked.", [
            {"name": "browser_click", "ok": True, "declined": False},
            {"name": "browser_click", "ok": False, "declined": True},
        ]
    ) == ""
    assert outcomes.unmade_browser_click_note(
        "The button was clicked.", [{"name": "browser_click", "ok": False, "uncertain": True}]
    ) == "I couldn't confirm whether it was clicked."
    assert outcomes.unmade_browser_click_note("The button hasn't been clicked yet.", []) == ""
    assert outcomes.unmade_browser_click_note("My friend clicked a link in a phishing email.", []) == ""
    assert outcomes.unmade_browser_click_note("The subscription button was clicked.", []) == ""
    assert outcomes.unmade_browser_click_note(
        "I did not click A, but I clicked B.", []
    ) == "I didn't click it, so nothing changed on the page."
    assert outcomes.unmade_browser_click_note(
        "You clicked A, and I clicked B.", []
    ) == "I didn't click it, so nothing changed on the page."
    prior_success = [{"name": "browser_click", "ok": True}]
    assert outcomes.unmade_browser_click_note(
        "I clicked the upgrade button.", [], prior_success
    ) == ""
    assert outcomes.unmade_browser_click_note("I clicked it.", [], prior_success) == ""
    prior_subscribe = [{"name": "browser_click", "ok": True, "target": 'button "Subscribe"'}]
    assert outcomes.unmade_browser_click_note("I clicked the Subscribe button.", [], prior_subscribe) == ""
    assert outcomes.unmade_browser_click_note(
        "I clicked the Delete account button.", [], prior_subscribe
    ) == ""
    assert outcomes.unmade_browser_click_note(
        "The Subscribe button was clicked, but the action failed.", [], [{"name": "browser_click", "declined": True}]
    ) == "I didn't click it: you declined, so nothing changed on the page."
    assert outcomes.unmade_browser_click_note("If I clicked it, you would be charged $19.", []) == ""
    assert outcomes.unmade_browser_click_note("I've clicked the button.", []) == "I didn't click it, so nothing changed on the page."
    subscribe_then_declined_delete = [
        {"name": "browser_click", "ok": True, "target": 'button "Subscribe"'},
        {"name": "browser_click", "declined": True, "target": 'button "Delete account"'},
    ]
    assert outcomes.unmade_browser_click_note(
        "I clicked the Subscribe button.", subscribe_then_declined_delete
    ) == ""
    assert outcomes.unmade_browser_click_note(
        "I clicked the Delete account button.", subscribe_then_declined_delete
    ) == "I didn't click it: you declined, so nothing changed on the page."
    assert outcomes.unmade_browser_click_note(
        "I clicked the Subscribe button.", [], subscribe_then_declined_delete
    ) == ""
    assert outcomes.unmade_browser_click_note(
        "I clicked the Delete account button.", [], subscribe_then_declined_delete
    ) == "I didn't click it: you declined, so nothing changed on the page."
    assert outcomes.unmade_browser_click_note(
        "I clicked the account settings button.", [{"name": "browser_click", "ok": True, "target": 'button "Delete account"'}]
    ) == ""
    assert outcomes.unmade_browser_click_note(
        "If it were clicked, you would be charged $19.", [{"name": "browser_click", "declined": True}]
    ) == ""
    successful_subscribe_then_declined_delete = [
        {"name": "browser_click", "ok": True, "target": 'button "Subscribe"'},
        {"name": "browser_click", "declined": True, "target": 'button "Delete account"'},
    ]
    for reply in (
        'I clicked the "Subscribe" button (e1) for the newsletter and attempted to click the "Delete account" button (e2).',
        "The subscription button was clicked successfully.",
        "Yes, I clicked the Subscribe button for the free weekly newsletter.",
        'Today I clicked just one thing: the "Subscribe" button on the Garden Notes page.',
    ):
        assert outcomes.unmade_browser_click_note(reply, successful_subscribe_then_declined_delete) == ""
    assert outcomes.unmade_browser_click_note(
        "Nothing was clicked.", successful_subscribe_then_declined_delete
    ) == ""
    assert outcomes.unmade_browser_click_note(
        "Yes, I clicked the Subscribe button for the free weekly newsletter.",
        [{"name": "browser_click", "declined": True, "target": 'button "Delete account"'}],
        [{"name": "browser_click", "ok": True, "target": 'button "Subscribe"'}],
    ) == ""
    assert outcomes.unmade_browser_click_note(
        "I clicked the Delete account button.",
        [
            {"name": "browser_click", "ok": True, "target": 'button "Subscribe"'},
            {"name": "browser_click", "declined": True},
        ],
    ) == ""
    assert outcomes.unmade_browser_click_note(
        "I clicked the Subscribe button and I clicked the Delete account button.",
        successful_subscribe_then_declined_delete,
    ) == "I didn't click it: you declined, so nothing changed on the page."
    assert outcomes.unmade_browser_click_note(
        "I successfully clicked the Subscribe button.", successful_subscribe_then_declined_delete
    ) == ""
    assert outcomes.unmade_browser_click_note(
        "Today I clicked Subscribe, which succeeded. I also clicked Delete account, but that one was declined.",
        successful_subscribe_then_declined_delete,
    ) == ""
    assert outcomes.unmade_browser_click_note(
        "I also clicked the Delete account button.", successful_subscribe_then_declined_delete
    ) == "I didn't click it: you declined, so nothing changed on the page."


def test_curly_apostrophe_negations_are_not_click_claims():
    """A curly apostrophe in wasn't, didn't, hasn't, or haven't is still a negation (#498)."""
    failed = [{"name": "browser_click", "ok": False, "target": 'button "Subscribe"'}]
    for reply in (
        "No, the Subscribe button wasn't clicked.",
        "I didn't get it clicked.",
        "The Subscribe button hasn't been clicked.",
        "I haven't clicked the Subscribe button.",
    ):
        curly = reply.replace("'", "\u2019")
        assert outcomes.unmade_browser_click_note(reply, failed) == "", reply
        assert outcomes.unmade_browser_click_note(curly, failed) == "", curly
    assert outcomes.unmade_browser_click_note(
        "I clicked the Subscribe button.", failed
    ) == "I couldn't click it, so nothing changed on the page."


def test_browser_click_history_retains_a_truncated_successful_result():
    """A context-truncated result must not turn a recorded click into a false failure."""
    history = browser_click_history([
        {"role": "assistant", "tool_calls": [tool_call("browser_click", ref="e1")]},
        {"role": "tool", "tool_call_id": "call_browser_click", "content":
         '{"ok": true, "clicked": "[e1] button \\"Subscribe\\"", "url": "https://example.test/'},
    ])
    assert history == [{"name": "browser_click", "target": '[e1] button "Subscribe"', "declined": False,
                        "ok": True, "uncertain": False}]
    declined = browser_click_history([
        {"role": "assistant", "tool_calls": [tool_call("browser_click", ref="e2")]},
        {"role": "tool", "tool_call_id": "call_browser_click", "content": '{"declined": true}',
         "approval": {"target": "Delete account"}},
    ])
    assert declined == [{"name": "browser_click", "target": "Delete account", "declined": True,
                         "ok": False, "uncertain": False}]


async def test_reply_uses_the_chat_history_for_a_declined_browser_click(config, isolated_home):
    """A later unsupported click claim retains the preceding declined-click context."""
    config.tools.approvals.mode = "ask"
    llm = FakeProvider(replies=[[tool_call("browser_click", ref="e1")], "I didn't click it: you declined.", "I clicked it."])
    s = await SentientApp(config, llm=llm, db_path=isolated_home / "prior-declined-click.db").start()
    try:
        sid = await s.store.create_session(channel="cli")

        async def deny(ev):
            """Decline the browser-click approval shown by this test."""
            if isinstance(ev, ApprovalRequest):
                s.approvals.resolve(ev.approval_id, "deny")

        first = await _collect(s.agent, sid, "click the subscription button", deny)
        assert "Correction:" not in "".join(e.text for e in first if isinstance(e, TextDelta))
        second = await _collect(s.agent, sid, "did it work?")
        text = "".join(e.text for e in second if isinstance(e, TextDelta))
        assert text.endswith("Correction: I didn't click it: you declined, so nothing changed on the page.")
    finally:
        await s.stop()


async def test_reply_corrects_an_email_it_never_sent(config, isolated_home):
    """A send claim with no send call is reported as not done (#492)."""
    llm = FakeProvider(replies=["I sent the email to Alice. "])
    s = await SentientApp(config, llm=llm, db_path=isolated_home / "never-sent.db").start()
    try:
        sid = await s.store.create_session(channel="cli")
        events = await _collect(s.agent, sid, "send an email to Alice")
        text = "".join(e.text for e in events if isinstance(e, TextDelta))
        assert text == "I sent the email to Alice. \n\nCorrection: I didn't send the email, so nothing changed."
        assert (await s.store.recent_messages(sid, 1))[-1]["content"] == text
    finally:
        await s.stop()


async def test_reply_corrects_a_send_the_user_declined(config, isolated_home):
    """A denied send approval must remain visible in the final reply (#492)."""
    from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool

    @tool("fake_send_email", risk=Risk.send, outcome="email")
    async def fake_send_email(ctx: ToolContext, to: str) -> dict:
        """Send an email (test double)."""
        return {"sent": [to]}

    class FakeMailer(ToolPlugin):
        id = "fake_mailer"
        display_name = "Fake mailer"
        tools = [fake_send_email]

    config.tools.approvals.mode = "ask"
    llm = FakeProvider(replies=[[tool_call("fake_send_email", to="a@x.com")],
                                "The email was sent, but the action was declined."])
    s = await SentientApp(config, llm=llm, db_path=isolated_home / "declined-send.db").start()
    try:
        s.registry.register(FakeMailer())
        sid = await s.store.create_session(channel="cli")

        async def deny(ev):
            """Decline the send approval shown by this test."""
            if isinstance(ev, ApprovalRequest):
                s.approvals.resolve(ev.approval_id, "deny")

        events = await _collect(s.agent, sid, "send an email", deny)
        text = "".join(e.text for e in events if isinstance(e, TextDelta))
        assert text.endswith("Correction: I didn't send the email: you declined, so nothing changed.")
        result = next(e for e in events if isinstance(e, ToolResultEvent))
        assert result.result["declined"] is True
    finally:
        await s.stop()


def _action_call(kind, declined=False, ok=True, uncertain=False, targets=None):
    return {"kind": kind, "declined": declined, "ok": ok, "uncertain": uncertain,
            "targets": targets or []}


def test_action_correction_never_corrects_a_true_reply():
    """A claim backed by a successful call is left alone, for every kind (#492)."""
    assert outcomes.unmade_action_note(
        "I sent the email.", [_action_call("email", targets=["a@x.com"])]
    ) == ""
    assert outcomes.unmade_action_note(
        "I posted the message.", [_action_call("message", targets=["general"])]
    ) == ""
    assert outcomes.unmade_action_note(
        "I saved the file.", [_action_call("file", targets=["notes.md"])]
    ) == ""
    assert outcomes.unmade_action_note(
        "I booked the room.", [_action_call("event", targets=["book club"])]
    ) == ""
    assert outcomes.unmade_action_note(
        "I turned on the lights.", [_action_call("home", targets=["light.living_room"])]
    ) == ""
    # a successful earlier turn protects a summary too
    assert outcomes.unmade_action_note(
        "I sent the email.", [], [_action_call("email", targets=["a@x.com"])]
    ) == ""


def test_action_correction_reports_declined_failed_and_never_made():
    """A declined, failed or never-made send/save is reported as not done (#492)."""
    assert outcomes.unmade_action_note(
        "I sent the email.", [_action_call("email", declined=True, ok=False, targets=["a@x.com"])]
    ) == "I didn't send the email: you declined, so nothing changed."
    assert outcomes.unmade_action_note(
        "I sent it.", [_action_call("email", declined=True, ok=False, targets=["a@x.com"])]
    ) == "I didn't send the email: you declined, so nothing changed."
    assert outcomes.unmade_action_note(
        "I saved the file.", [_action_call("file", ok=False, targets=["notes.md"])]
    ) == "I couldn't write the file, so nothing changed."
    assert outcomes.unmade_action_note(
        "I turned on the lights.", [_action_call("home", ok=False, uncertain=True)]
    ) == "I couldn't confirm whether I ran the home action."
    assert outcomes.unmade_action_note("I sent the email to Alice.", []) == \
        "I didn't send the email, so nothing changed."
    # a never-made claim that doesn't name the action is left alone: the verb alone may describe an
    # untracked action (#492 review)
    assert outcomes.unmade_action_note("I booked it.", []) == ""
    # ...but one that names it is still corrected
    assert outcomes.unmade_action_note("I booked the meeting.", []) == \
        "I didn't create the event, so nothing changed."
    assert outcomes.unmade_action_note("I saved plan.md.", []) == \
        "I didn't write the file, so nothing changed."
    # or when this turn actually tried that action, even for a verb-only claim
    assert outcomes.unmade_action_note(
        "I emailed Bob.", [_action_call("email", declined=True, ok=False, targets=["a@x.com"])]
    ) == "I didn't send the email, so nothing changed."


def test_action_correction_leaves_unsure_text_alone():
    """Other people's actions, negations, hypotheticals and ordinary chat verbs are never corrected (#492)."""
    assert outcomes.unmade_action_note("You sent the email.", []) == ""
    assert outcomes.unmade_action_note("My assistant sent the file.", []) == ""
    assert outcomes.unmade_action_note("I didn't send the email.", []) == ""
    assert outcomes.unmade_action_note("The email was never sent.", []) == ""
    assert outcomes.unmade_action_note("If I sent the email, you'd get a copy.", []) == ""
    assert outcomes.unmade_action_note("I ran into an issue while sending.", []) == ""
    assert outcomes.unmade_action_note("I wrote a summary of the report.", []) == ""
    assert outcomes.unmade_action_note("I added some context above.", []) == ""
    # a shared verb without a naming noun stays ambiguous across kinds
    assert outcomes.unmade_action_note("I sent it.", []) == ""
    assert outcomes.unmade_action_note(
        "I sent it.", [_action_call("email", targets=["a@x.com"])], [_action_call("message")]
    ) == ""


def test_action_correction_leaves_untracked_action_verbs_alone():
    """True replies about actions this check doesn't track are never corrected (#492 review).

    qwen3:8b over-corrections from the review: "saved" after memory_remember, "scheduled"/"set"/"ran" for
    tasks and reminders became file/event/home corrections. A bare verb without a naming noun stays alone.
    """
    assert outcomes.unmade_action_note("I saved that your balcony faces west.", []) == ""
    assert outcomes.unmade_action_note("I've scheduled a task to check the prices.", []) == ""
    assert outcomes.unmade_action_note("I set up a daily task for backups.", []) == ""
    assert outcomes.unmade_action_note("I set a reminder for tomorrow morning.", []) == ""
    assert outcomes.unmade_action_note("I ran a quick check and nothing was sent.", []) == ""


def test_action_correction_matches_the_full_target():
    """The full address or file name is kept when matching a named claim (#492 review).

    The target used to be cut at ".", so "arjun@example.com" and "plan.md" never matched their recorded call.
    """
    assert outcomes.unmade_action_note(
        "I sent the email to arjun@example.com.",
        [_action_call("email", declined=True, ok=False, targets=["arjun@example.com"])],
    ) == "I didn't send the email: you declined, so nothing changed."
    assert outcomes.unmade_action_note(
        "I saved plan.md.",
        [_action_call("file", declined=True, ok=False, targets=["plan.md"])],
    ) == "I didn't write the file: you declined, so nothing changed."
    # and a true full-target claim is left alone
    assert outcomes.unmade_action_note(
        "I sent the email to arjun@example.com.",
        [_action_call("email", targets=["arjun@example.com"])],
    ) == ""


def test_action_correction_says_couldnt_confirm_for_uncertain_calls():
    """An uncertain call is never reported as "didn't happen": it says it couldn't confirm (#492 review)."""
    assert outcomes.unmade_action_note(
        "I sent the email to Bob.",
        [_action_call("email", ok=False, uncertain=True, targets=["arjun@example.com"])],
    ) == "I couldn't confirm whether I sent the email."


def test_action_correction_accepts_curly_apostrophe_negations():
    """A negation with the curly apostrophe (U+2019) counts: it is not corrected (#492 review)."""
    declined = [_action_call("email", declined=True, ok=False, targets=["a@x.com"])]
    assert outcomes.unmade_action_note("The email wasn\u2019t sent.", declined) == ""
    assert outcomes.unmade_action_note("The email wasn't sent.", declined) == ""
    assert outcomes.unmade_action_note("I didn\u2019t send the email.", declined) == ""


def test_action_correction_treats_incomplete_history_as_unsure():
    """Calls outside the loaded history window count as "unsure", not "never made" (#492 review)."""
    assert outcomes.unmade_action_note(
        "I sent the email to Alice.", [], history_complete=False
    ) == "I couldn't confirm whether I sent the email."
    assert outcomes.unmade_action_note(
        "I sent the email to Alice.", [],
    ) == "I didn't send the email, so nothing changed."


async def test_resumed_approved_send_is_not_corrected(config, isolated_home):
    """A resumed task's approved send is the resumed segment's history: no correction (#492 review).

    FakeProvider repro of the review's case: the approved call runs in ``_run_approved_call`` (no history),
    then the resumed segment summarizes it as "I sent the email to Arjun." — which must stand.
    """
    import json

    from sentient.tasks import ask
    from sentient.tasks.executor import _run_approved_call
    from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool

    @tool("fake_resume_send", risk=Risk.send, outcome="email")
    async def fake_resume_send(ctx: ToolContext, to: str) -> dict:
        """Send an email (test double)."""
        return {"sent": [to]}

    class FakeResumeMailer(ToolPlugin):
        id = "fake_resume_mailer"
        display_name = "Fake resume mailer"
        tools = [fake_resume_send]

    class StubRepo:
        async def update_run(self, run_id, fields):
            return None

        async def get_run(self, run_id):
            return {}

    class StubSvc:
        def __init__(self, app):
            self.app = app
            self.repo = StubRepo()

        async def progress(self, task_id, run_id, event):
            return None

    s = await SentientApp(config, llm=FakeProvider(replies=[]),
                          db_path=isolated_home / "resumed-send.db").start()
    try:
        s.registry.register(FakeResumeMailer())
        call_id = "call_approved_send"
        checkpoint = [
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": call_id, "type": "function",
                "function": {"name": "fake_resume_send",
                             "arguments": json.dumps({"to": "arjun@example.com"})},
            }]},
            {"role": "tool", "tool_call_id": call_id,
             "content": json.dumps({"error": ask.INTERRUPTED_NOTE}), ask.APPROVED_KEY: True},
        ]
        svc = StubSvc(s)
        ctx = s.agent.tool_context(None, "task")
        messages, action = await _run_approved_call(svc, "task-1", "run-1", ctx, checkpoint)
        assert action == {"kind": "email", "declined": False, "ok": True, "uncertain": False,
                          "targets": ["arjun@example.com"]}
        # the resumed segment sees the approved call as history: the summary stands, uncorrected
        assert outcomes.unmade_action_note("I sent the email to Arjun.", [], [action]) == ""
        assert ask.take_approved(messages) is None  # the approval mark was consumed: it runs once
    finally:
        await s.stop()


def test_action_call_history_reads_a_prior_turn():
    """Stored tool calls map back to their action kind, so a later summary is not corrected (#492)."""
    from types import SimpleNamespace

    class FakeRegistry:
        def get(self, name):
            tools = {
                "gmail_send": SimpleNamespace(outcome="email", name="gmail_send"),
                "file_write": SimpleNamespace(outcome="file", name="file_write"),
            }
            return tools.get(name)

    history = action_call_history(
        [
            {"role": "assistant", "tool_calls": [tool_call("gmail_send", to="a@x.com")]},
            {"role": "tool", "tool_call_id": "call_gmail_send",
             "content": '{"ok": true, "sent": ["a@x.com"]}'},
            {"role": "assistant", "tool_calls": [tool_call("file_write", name="n.md")]},
            {"role": "tool", "tool_call_id": "call_file_write", "content": '{"declined": true}'},
        ],
        FakeRegistry(),
    )
    assert history == [
        {"kind": "email", "targets": ["a@x.com"], "declined": False, "ok": True, "uncertain": False},
        {"kind": "file", "targets": ["n.md"], "declined": True, "ok": False, "uncertain": False},
    ]
    assert outcomes.unmade_action_note("I sent the email.", [], history) == ""


def test_action_call_history_reads_leading_json_and_outcome_notes():
    """A stored result can carry text after its JSON: an appended outcome note (#386) or a cut marker.

    The leading JSON is decoded; a note means the read-back said it did not work (never a later success),
    while a cut marker stays a success so a true reply is never corrected (#492 review).
    """
    from types import SimpleNamespace

    class FakeRegistry:
        def get(self, name):
            tools = {
                "gmail_send": SimpleNamespace(outcome="email", name="gmail_send"),
                "ha_turn_on": SimpleNamespace(outcome="home", name="ha_turn_on"),
            }
            return tools.get(name)

    note = outcomes.model_note({"ok": False, "how": "read_back", "detail": "the light is still off"})
    assert note.startswith("[Sentient checked afterwards")
    history = action_call_history(
        [
            {"role": "assistant", "tool_calls": [tool_call("ha_turn_on", entity_id="light.kitchen")]},
            {"role": "tool", "tool_call_id": "call_ha_turn_on",
             "content": f'{{"ok": true}}\n{note}'},
            {"role": "assistant", "tool_calls": [tool_call("gmail_send", to="a@x.com")]},
            {"role": "tool", "tool_call_id": "call_gmail_send",
             "content": '{"ok": true, "sent": ["a@x.com"]}\n\n[Result cut: this is only the first 10 of 80 characters.]'},
        ],
        FakeRegistry(),
    )
    assert [call["kind"] for call in history] == ["home", "email"]
    assert history[0]["ok"] is False  # the read-back note says the light is still off
    assert history[1]["ok"] is True  # a cut marker is not an outcome
    # a false claim about the noted call is corrected, not hidden by the fake success
    assert outcomes.unmade_action_note("I turned on the lights.", [], history) == \
        "I couldn't run the home action, so nothing changed."


async def test_resumed_segment_rebuilds_action_history_from_the_whole_checkpoint(config, isolated_home):
    """The resumed segment rebuilds its action history from the whole checkpoint, not just the approved call.

    A true summary of an earlier action call in the checkpoint is never "corrected" (#492 review).
    """
    import json

    from sentient.agent.loop import action_call_history
    from sentient.tasks import ask
    from sentient.tasks.executor import _run_approved_call
    from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool

    @tool("fake_resume_send", risk=Risk.send, outcome="email")
    async def fake_resume_send(ctx: ToolContext, to: str) -> dict:
        """Send an email (test double)."""
        return {"sent": [to]}

    class FakeResumeMailer(ToolPlugin):
        id = "fake_resume_mailer"
        display_name = "Fake resume mailer"
        tools = [fake_resume_send]

    class StubRepo:
        async def update_run(self, run_id, fields):
            return None

        async def get_run(self, run_id):
            return {}

    class StubSvc:
        def __init__(self, app):
            self.app = app
            self.repo = StubRepo()

        async def progress(self, task_id, run_id, event):
            return None

    s = await SentientApp(config, llm=FakeProvider(replies=[]),
                          db_path=isolated_home / "resumed-checkpoint.db").start()
    try:
        s.registry.register(FakeResumeMailer())
        approved_id = "call_approved_send"
        checkpoint = [
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "call_earlier_send", "type": "function",
                "function": {"name": "fake_resume_send",
                             "arguments": json.dumps({"to": "earlier@example.com"})},
            }]},
            {"role": "tool", "tool_call_id": "call_earlier_send",
             "content": json.dumps({"sent": ["earlier@example.com"]})},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": approved_id, "type": "function",
                "function": {"name": "fake_resume_send",
                             "arguments": json.dumps({"to": "arjun@example.com"})},
            }]},
            {"role": "tool", "tool_call_id": approved_id,
             "content": json.dumps({"error": ask.INTERRUPTED_NOTE}), ask.APPROVED_KEY: True},
        ]
        svc = StubSvc(s)
        ctx = s.agent.tool_context(None, "task")
        messages, _ = await _run_approved_call(svc, "task-1", "run-1", ctx, checkpoint)
        # what execute_single does on resume: rebuild from the whole updated checkpoint
        history = action_call_history(messages, s.registry)
        assert [call["targets"] for call in history] == [["earlier@example.com"], ["arjun@example.com"]]
        assert all(call["ok"] for call in history)
        assert outcomes.unmade_action_note("I sent the email to earlier@example.com.", [], history) == ""
        assert outcomes.unmade_action_note("I sent the email to Arjun.", [], history) == ""
    finally:
        await s.stop()


async def test_internal_tools_do_not_prompt_in_ask_mode(config, isolated_home):
    config.tools.approvals.mode = "ask"
    llm = FakeProvider(replies=[[tool_call("file_write", name="note.txt", content="hi")], "saved"])
    s = await SentientApp(config, llm=llm, db_path=isolated_home / "internal.db").start()
    try:
        sid = await s.store.create_session(channel="cli")
        events = await _collect(s.agent, sid, "save a note")
        assert not any(isinstance(e, ApprovalRequest) for e in events)
        assert (isolated_home / "files" / "note.txt").read_text() == "hi"
        for name in ("memory_remember", "file_write", "skill_save"):
            tool = s.registry.get(name)
            assert tool is not None and tool.internal, name
            assert not s.approvals.needs_approval(tool, sid)
        # irreversible internal actions still ask, and "always" asks for everything
        assert s.approvals.needs_approval(s.registry.get("memory_forget"), sid)
        s.approvals.config.mode = "always"
        assert s.approvals.needs_approval(s.registry.get("file_write"), sid)
    finally:
        await s.stop()

async def test_memory_remember_via_tool_then_recalled_in_prompt(config, isolated_home):
    # this test checks that recalled facts reach the prompt, not the similarity threshold
    # (the fake bag-of-words embedder scores paraphrases low)
    config.memory.min_similarity = 0.0
    llm = FakeProvider(
        replies=[[tool_call("memory_remember", fact="Maya's sister lives in Pune")], "Got it.", "Pune."],
        json_replies=[{"action": "ADD", "fact_id": None, "content": "Maya's sister lives in Pune",
                       "memory_type": "long-term", "duration": None, "topics": ["family"]}],
    )
    s = await SentientApp(config, llm=llm, db_path=isolated_home / "d.db").start()
    try:
        sid = await s.store.create_session(channel="cli")
        await _collect(s.agent, sid, "my sister lives in Pune")
        assert await s.memory.count() == 1
        await _collect(s.agent, sid, "where does my sister live")
        system = llm.calls[-1]["messages"][0]["content"]
        assert "Maya's sister lives in Pune" in system, "recalled facts must be injected into the prompt"
    finally:
        await s.stop()


async def test_background_extraction_runs(config, isolated_home):
    config.memory.extract_after_turn = True
    llm = FakeProvider(
        replies=["Nice, noted."],
        json_replies=[{"facts": ["Maya drinks black coffee"]},
                      {"action": "ADD", "fact_id": None, "content": "Maya drinks black coffee",
                       "memory_type": "long-term", "duration": None, "topics": ["preferences"]}],
    )
    s = await SentientApp(config, llm=llm, db_path=isolated_home / "e.db").start()
    try:
        sid = await s.store.create_session(channel="cli")
        await _collect(s.agent, sid, "I only ever drink black coffee, no sugar")
        await s.agent.drain()
        assert await s.memory.count() == 1
    finally:
        await s.stop()


async def test_voice_channel_uses_voice_role(config, isolated_home):
    llm = FakeProvider(replies=["Sure thing."])
    s = await SentientApp(config, llm=llm, db_path=isolated_home / "v.db").start()
    try:
        sid = await s.store.create_session(channel="voice")
        async for _ in s.agent.run_turn(sid, "what's up", channel="voice"):
            pass
        assert llm.calls[-1]["role"] == "voice"
        assert "You are speaking out loud" in llm.calls[-1]["messages"][0]["content"]
    finally:
        await s.stop()


async def test_model_without_tool_support_degrades_gracefully(config, isolated_home):
    from sentient.llm.events import Error
    from sentient.llm.provider import ProviderError

    class NoToolsProvider(FakeProvider):
        async def stream(self, role, messages, tools=None, *, model=None):
            if tools:
                self.calls.append({"role": role, "messages": messages, "tools": tools, "model": model})
                raise ProviderError("Ollama_chatException - invalid character '<' looking for beginning of value")
            async for chunk in super().stream(role, messages, None, model=model):
                yield chunk

    llm = NoToolsProvider(replies=["It is sunny."])
    s = await SentientApp(config, llm=llm, db_path=isolated_home / "nt.db").start()
    try:
        sid = await s.store.create_session(channel="cli")
        events = await _collect(s.agent, sid, "weather?")
        errors = [e for e in events if isinstance(e, Error)]
        assert len(errors) == 1 and errors[0].recoverable and "can't use tools" in errors[0].message
        assert isinstance(events[-1], Done) and events[-1].content == "It is sunny."
        assert llm.calls[0]["tools"] and llm.calls[-1]["tools"] is None
    finally:
        await s.stop()


async def test_stopped_reply_is_saved(config, isolated_home):
    llm = FakeProvider(replies=["Once upon a time there was a very long story"])
    s = await SentientApp(config, llm=llm, db_path=isolated_home / "stop.db").start()
    try:
        sid = await s.store.create_session(channel="cli")
        gen = s.agent.run_turn(sid, "tell me a story", channel="cli")
        async for ev in gen:
            if isinstance(ev, TextDelta):
                break
        await gen.aclose()
        msgs = await s.store.recent_messages(sid, 10)
        assert msgs[-1]["role"] == "assistant"
        assert msgs[-1]["content"].startswith("Once") and "(stopped)" in msgs[-1]["content"]
    finally:
        await s.stop()


async def test_malformed_tool_arguments_get_clear_error(config, isolated_home):
    bad = [{"id": "c1", "name": "memory_recall", "arguments": {"_raw": "{query: sister"}}]
    llm = FakeProvider(replies=[bad, "ok"])
    s = await SentientApp(config, llm=llm, db_path=isolated_home / "raw.db").start()
    try:
        sid = await s.store.create_session(channel="cli")
        events = await _collect(s.agent, sid, "who is my sister")
        res = next(e for e in events if isinstance(e, ToolResultEvent))
        assert res.is_error and "not valid JSON" in res.result["error"]
        assert res.result["schema"]["properties"]["query"]["type"] == "string"
    finally:
        await s.stop()
