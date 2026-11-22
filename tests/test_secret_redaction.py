"""Known secret values are hidden from logs and tool errors (#178)."""

import logging

import pytest

from sentient import secrets

KEY = "sk-live-abcdef123456"


@pytest.fixture(autouse=True)
def _clean():
    saved = set(secrets._known)
    secrets._known.clear()
    yield
    secrets._known.clear()
    secrets._known.update(saved)


def test_redact_replaces_known_values_only():
    secrets.remember(KEY)
    secrets.remember("short")  # too short to hide safely
    assert secrets.redact(f"auth failed for {KEY} short") == "auth failed for <secret> short"


def test_get_secret_from_env_is_remembered(monkeypatch):
    monkeypatch.setenv("MY_TEST_KEY", "tok-1234567890")
    assert secrets.get_secret("nope-not-in-keychain", env_var="MY_TEST_KEY") == "tok-1234567890"
    assert secrets.redact("x tok-1234567890 y") == "x <secret> y"


def test_log_filter_hides_values_in_message_args_and_traceback():
    secrets.remember(KEY)
    logger = logging.getLogger("sentient.test_redaction")
    handler = logging.StreamHandler()
    lines: list[str] = []
    handler.emit = lambda r: lines.append(handler.format(r))  # type: ignore[method-assign]
    handler.addFilter(secrets.RedactSecrets())
    logger.addHandler(handler)
    try:
        logger.error("call with %s", KEY)
        try:
            raise RuntimeError(f"bad key {KEY}")
        except RuntimeError:
            logger.exception("tool failed")
    finally:
        logger.removeHandler(handler)
    assert len(lines) == 2
    assert all(KEY not in line for line in lines)
    assert "<secret>" in lines[0] and "<secret>" in lines[1]


async def test_tool_error_hides_secret(config, isolated_home):
    from sentient.app import SentientApp
    from sentient.llm.events import ToolResultEvent
    from tests.conftest import FakeProvider, tool_call

    secrets.remember(KEY)
    llm = FakeProvider(replies=[[tool_call("current_datetime")], "done"])
    s = await SentientApp(config, llm=llm, db_path=isolated_home / "r.db").start()
    try:

        async def broken(ctx, **kwargs):
            raise RuntimeError(f"401 for key {KEY}")

        s.registry.get("current_datetime").fn = broken
        sid = await s.store.create_session(channel="cli")
        events = [e async for e in s.agent.run_turn(sid, "what time is it", channel="cli")]
        res = next(e for e in events if isinstance(e, ToolResultEvent))
        assert res.is_error
        assert KEY not in res.result["error"]
        assert "<secret>" in res.result["error"]
    finally:
        await s.stop()


def test_json_credential_fields_are_remembered():
    secrets.remember_fields({"a": {"t": "rt-0123456789"}, "n": 5, "l": ["lst-abcdefghij"]})
    assert secrets.redact("x rt-0123456789 lst-abcdefghij") == "x <secret> <secret>"


def test_json_credential_fields_in_tuples_are_remembered():
    secrets.remember_fields({"t": ("tup-0123456789", ["in-abcdefghij"])})
    assert secrets.redact("x tup-0123456789 in-abcdefghij") == "x <secret> <secret>"


def test_create_app_installs_redaction_on_a_bare_root_logger():
    # a `serve --reload` worker starts with no root handler and never ran the CLI's logging setup
    from sentient.gateway.app import create_app

    root = logging.getLogger()
    saved = root.handlers[:]
    root.handlers.clear()
    try:
        create_app()
        assert root.handlers
        assert all(any(isinstance(f, secrets.RedactSecrets) for f in h.filters) for h in root.handlers)
    finally:
        root.handlers[:] = saved
