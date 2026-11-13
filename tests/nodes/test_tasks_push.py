"""Running tasks reach the devices that subscribe to them: the phone page and the tray (#182, docs/NODES.md §4.7)."""

import time

from sentient.nodes.service import TASKS_COALESCE_S
from tests.nodes.conftest import hello, pair_device

TASKS = [
    {"task_id": "t1", "name": "Book a flight to Goa", "status": "processing"},
    {"task_id": "t2", "name": "Find a hotel", "status": "waiting_for_user"},
    {"task_id": "t3", "name": "Weekly report", "status": "completed"},
]


def _fake_tasks(client, tasks):
    async def listed():
        return tasks

    client.core.tasks.list = listed


def _publish(client, kind, data):
    async def publish():
        client.core.bus.publish(kind, data)

    client.portal.call(publish)


def test_a_subscribed_device_gets_running_tasks_on_connect_and_after_task_events(nodes_client):
    client = nodes_client()
    _, token = pair_device(client)
    _fake_tasks(client, [])
    with client.websocket_connect("/ws/node") as ws:
        ws.send_json(hello(token=token, kind="phone", subscribe=["tasks"]))
        assert ws.receive_json()["type"] == "welcome"
        assert ws.receive_json() == {"type": "tasks", "running": []}

        _fake_tasks(client, TASKS)
        _publish(client, "task.run_progress", {
            "task_id": "t1", "run_id": "r1", "update": {"message": {"type": "tool_call", "tool_name": "web_search"}},
        })
        _publish(client, "task.updated", {"task_id": "t2", "status": "waiting_for_user"})  # same burst, one message
        assert ws.receive_json() == {"type": "tasks", "running": [
            {"id": "t1", "name": "Book a flight to Goa", "status": "processing", "activity": "Using web search"},
            {"id": "t2", "name": "Find a hotel", "status": "waiting_for_user", "activity": None},
        ]}

        _fake_tasks(client, TASKS[1:])
        _publish(client, "task.run_finished", {"task_id": "t1", "run_id": "r1", "status": "completed"})
        assert ws.receive_json()["running"] == [
            {"id": "t2", "name": "Find a hotel", "status": "waiting_for_user", "activity": None},
        ]


def test_a_device_that_did_not_subscribe_gets_no_tasks(nodes_client):
    client = nodes_client()
    _, token = pair_device(client)
    _fake_tasks(client, TASKS)
    with client.websocket_connect("/ws/node") as ws:
        ws.send_json(hello(token=token, subscribe=7))  # glasses; a malformed subscribe is ignored, not fatal
        assert ws.receive_json()["type"] == "welcome"
        _publish(client, "task.updated", {"task_id": "t1", "status": "processing"})
        time.sleep(TASKS_COALESCE_S + 0.3)  # past the window in which a `tasks` message would have gone out
        ws.send_json({"type": "ping", "ts": 1})
        assert ws.receive_json() == {"type": "pong", "ts": 1}  # nothing about tasks came first
