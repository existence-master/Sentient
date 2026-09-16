from __future__ import annotations

import json

import httpx
import pytest
import respx

from sentient.integrations.base import IntegrationError


def _drain(q) -> list[dict]:
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


async def test_list_shape_and_builtins(app):
    items = {i["id"]: i for i in await app.integrations.list_integrations()}
    for key in ("gmail", "gcalendar", "gdrive", "gdocs", "gsheets", "gslides", "gpeople", "github", "slack", "notion",
                "discord", "trello", "whatsapp", "internet_search", "web", "weather", "maps", "news", "charts"):
        assert key in items, key
    gmail = items["gmail"]
    assert set(gmail) >= {"id", "display_name", "description", "category", "icon", "auth_type", "connected",
                          "account_label", "status", "error", "setup", "privacy_filters", "triggers", "tools"}
    assert gmail["auth_type"] == "oauth" and gmail["connected"] is False and gmail["status"] == "disconnected"
    assert gmail["privacy_filters"] == {"supported": True, "fields": ["keywords", "emails", "labels"]}
    assert gmail["triggers"] == [{"event": "new_email", "label": "New email"}]
    assert {f["key"] for f in gmail["setup"]["fields"]} == {"client_id", "client_secret"}
    assert "console.cloud.google.com" in gmail["setup"]["instructions_md"]
    risks = {t["name"]: t["risk"] for t in gmail["tools"]}
    assert risks["gmail_search"] == "read" and risks["gmail_send"] == "send" and risks["gmail_trash"] == "send"
    assert risks["gmail_create_draft"] == "write" and risks["gmail_archive"] == "write"
    assert items["weather"]["connected"] is True and items["weather"]["status"] == "connected"
    assert await app.integrations.connected_plugins() >= {"internet_search", "web", "weather", "maps", "news", "charts"}
    assert "gmail" not in await app.integrations.connected_plugins()


async def test_every_tool_has_risk_schema_and_unique_name(app):
    names = set()
    for p in app.integrations.plugins():
        for t in p.tools:
            assert t.name not in names
            names.add(t.name)
            schema = t.openai_schema()["function"]
            assert schema["description"], t.name
            assert len(t.name) <= 64
    assert not names & {"memory_recall", "memory_remember", "memory_forget", "memory_search_history",
                        "current_datetime", "skill_view", "skill_save", "file_list", "file_read", "file_write"}


def offered(app) -> set[str]:
    """Tool names the model is offered right now."""
    return {t["function"]["name"] for t in app.registry.openai_schemas()}


async def test_disconnected_tools_hidden_and_friendly_error(app, ctx):
    # hide_disconnected_tools default: disconnected tools are not offered to the model, but still resolve
    assert "gmail_search" not in offered(app) and app.registry.is_hidden("gmail")
    assert "web_search" in offered(app)
    tool = next(t for t in app.integrations.plugin("gmail").tools if t.name == "gmail_search")
    assert await tool.call(ctx, {"query": "invoice"}) == {"error": "Gmail isn't connected yet. Connect it from Integrations."}


async def test_github_pat_connect_disconnect(app, ctx, keychain):
    disabled: list[str] = []

    async def disable_tasks_for_plugin(plugin_id: str) -> None:
        disabled.append(plugin_id)

    app.tasks.disable_tasks_for_plugin = disable_tasks_for_plugin
    async with app.bus.subscribe() as q:
        with respx.mock(assert_all_called=True) as router:
            router.get("https://api.github.com/user").mock(return_value=httpx.Response(200, json={"login": "octocat"}))
            integ = await app.integrations.connect("github", {"token": "ghp_secret"})
        events = _drain(q)
    assert integ["connected"] is True and integ["account_label"] == "octocat" and integ["status"] == "connected"
    assert json.loads(keychain["integration:github"]) == {"token": "ghp_secret"}
    assert any(e["type"] == "integration.updated" and e["data"]["id"] == "github" for e in events)
    assert "github" in await app.integrations.connected_plugins()
    assert await app.integrations.get_credentials("github") == {"token": "ghp_secret"}
    assert "github_list_issues" in offered(app)
    row = await app.store.fetchone("SELECT * FROM integrations WHERE id = 'github'")
    assert row["connected"] == 1 and "ghp_secret" not in json.dumps(dict(row))

    with respx.mock() as router:
        router.get("https://api.github.com/repos/octo/hello/issues").mock(return_value=httpx.Response(200, json=[
            {"number": 1, "title": "Bug", "state": "open", "user": {"login": "a"}, "labels": [], "assignees": []},
            {"number": 2, "title": "PR", "state": "open", "pull_request": {}, "user": {"login": "b"}},
        ]))
        res = await app.registry.get("github_list_issues").call(ctx, {"repo": "octo/hello"})
        assert router.calls.last.request.headers["authorization"] == "Bearer ghp_secret"
    assert [i["number"] for i in res["issues"]] == [1]

    out = await app.integrations.disconnect("github")
    assert out["connected"] is False and "integration:github" not in keychain
    assert disabled == ["github"]
    assert "github_list_issues" not in offered(app)


async def test_github_bad_token_reports_error(app):
    with respx.mock() as router:
        router.get("https://api.github.com/user").mock(return_value=httpx.Response(401, json={"message": "Bad credentials"}))
        with pytest.raises(IntegrationError, match="didn't accept"):
            await app.integrations.connect("github", {"token": "nope"})
    integ = await app.integrations.integration("github")
    assert integ["connected"] is False and integ["status"] == "error" and integ["error"]


async def test_slack_auth_test_validation(app, keychain):
    with respx.mock() as router:
        router.get("https://slack.com/api/auth.test").mock(side_effect=[
            httpx.Response(200, json={"ok": False, "error": "invalid_auth"}),
            httpx.Response(200, json={"ok": True, "user": "sarthak", "team": "Existence", "team_id": "T1", "user_id": "U1"}),
        ])
        with pytest.raises(IntegrationError, match="rejected"):
            await app.integrations.connect("slack", {"token": "xoxp-bad"})
        integ = await app.integrations.connect("slack", {"token": "xoxp-good"})
    assert integ["account_label"] == "sarthak @ Existence"
    assert json.loads(keychain["integration:slack"])["token"] == "xoxp-good"
    with pytest.raises(IntegrationError, match="doesn't look like a Slack token"):
        await app.integrations.connect("slack", {"token": "hello"})


async def test_test_endpoint_logic(app):
    assert (await app.integrations.test("weather"))["ok"] is True
    res = await app.integrations.test("notion")
    assert res == {"ok": False, "detail": "Notion isn't connected yet."}


async def test_privacy_filters_normalized_and_persisted(app, config, isolated_home):
    await app.integrations.set_privacy_filters("gmail", {"keywords": [" Salary ", "salary", ""], "emails": "boss@x.com"})
    assert await app.integrations.get_privacy_filters("gmail") == {"keywords": ["Salary"], "emails": ["boss@x.com"], "labels": []}
    row = await app.store.fetchone("SELECT settings FROM integrations WHERE id='gmail'")
    assert json.loads(row["settings"])["privacy_filters"]["keywords"] == ["Salary"]
