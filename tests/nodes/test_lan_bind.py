"""SAFE-15: binding the LAN device listener to all interfaces must be an explicit opt-in."""

from __future__ import annotations

import socket


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _enable_lan(core, *, bind_all: bool) -> None:
    core.config.nodes.lan_enabled = True
    core.config.nodes.lan_bind_all = bind_all
    core.config.nodes.lan_port = _free_port()
    core.config.nodes.mdns_enabled = False


def test_lan_bind_all_defaults_off(config):
    assert config.nodes.lan_bind_all is False


async def test_apply_config_binds_loopback_by_default(nodes_client):
    client = nodes_client()
    core = client.core
    _enable_lan(core, bind_all=False)
    await core.nodes.apply_config()
    try:
        assert core.nodes.lan and core.nodes.lan.running
        assert core.nodes.lan_host == "127.0.0.1"
        status = core.nodes.lan_status()
        assert status["bind_all"] is False  # not exposed beyond this computer
    finally:
        await core.nodes.lan.stop()


async def test_apply_config_requires_opt_in_to_bind_all_interfaces(nodes_client):
    client = nodes_client()
    core = client.core
    _enable_lan(core, bind_all=True)
    await core.nodes.apply_config()
    try:
        assert core.nodes.lan and core.nodes.lan.running
        assert core.nodes.lan_host == "0.0.0.0"
        status = core.nodes.lan_status()
        assert status["bind_all"] is True  # surfaces the exposure warning in doctor and Settings
    finally:
        await core.nodes.lan.stop()


def test_legacy_lan_config_counts_as_bind_all_opt_in():
    from sentient.config.schema import NodesConfig

    legacy = NodesConfig.model_validate({"lan_enabled": True})
    assert legacy.lan_bind_all is True  # saved before the flag existed: listener bound 0.0.0.0, keep working

    explicit = NodesConfig.model_validate({"lan_enabled": True, "lan_bind_all": False})
    assert explicit.lan_bind_all is False  # an explicit choice is never overridden

    fresh = NodesConfig.model_validate({})
    assert fresh.lan_bind_all is False  # new users stay loopback-only


async def test_loopback_only_never_advertises_network_addresses(nodes_client):
    client = nodes_client()
    core = client.core
    _enable_lan(core, bind_all=False)
    await core.nodes.apply_config()
    try:
        status = core.nodes.lan_status()
        assert status["urls"] == [f"wss://127.0.0.1:{core.config.nodes.lan_port}/ws/node"]
        assert status["web_urls"] == [f"https://127.0.0.1:{core.config.nodes.lan_port}/node/"]
    finally:
        await core.nodes.lan.stop()
