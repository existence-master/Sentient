"""Running tasks reach the devices that subscribe to them: the phone page and the tray (#182, docs/NODES.md §4.7)."""

import asyncio
import time

from sentient.nodes.service import TASKS_COALESCE_S
from tests.nodes.conftest import hello, pair_device

TASKS = [  # TaskService.busy() rows
    {"id": "t1", "name": "Book a flight to Goa", "status": "processing"},
    {"id": "t2", "name": "Find a hotel", "status": "waiting_for_user"},
]


def _fake_tasks(client, tasks):
    async def busy():
        if tasks is None:
            raise RuntimeError("database is locked")
        return tasks

    client.core.tasks.busy = busy


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

        # a try run ends with task.updated, not task.run_finished: its last step goes too
        _publish(client, "task.run_progress", {
            "task_id": "t2", "run_id": "r2", "update": {"message": {"type": "tool_call", "tool_name": "web_search"}},
        })
        _publish(client, "task.updated", {"task_id": "t2", "status": "completed"})
        assert ws.receive_json()["running"][0]["activity"] is None

        # when the tasks can't be read, devices keep the list they have rather than an empty one
        _fake_tasks(client, None)
        _publish(client, "task.updated", {"task_id": "t2", "status": "processing"})
        time.sleep(TASKS_COALESCE_S + 0.3)
        ws.send_json({"type": "ping", "ts": 2})
        assert ws.receive_json() == {"type": "pong", "ts": 2}


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


def test_a_failed_read_is_tried_again_without_another_task_event(nodes_client, monkeypatch):
    monkeypatch.setattr("sentient.nodes.service.TASKS_RETRY_S", 0.1)
    client = nodes_client()
    _, token = pair_device(client)
    _fake_tasks(client, [])
    with client.websocket_connect("/ws/node") as ws:
        ws.send_json(hello(token=token, kind="phone", subscribe=["tasks"]))
        assert ws.receive_json()["type"] == "welcome"
        assert ws.receive_json() == {"type": "tasks", "running": []}

        reads = iter([None, TASKS[1:]])  # the first read fails, the retry works
        client.core.tasks.busy = lambda: _read(next(reads))
        _publish(client, "task.updated", {"task_id": "t2", "status": "waiting_for_user"})
        assert ws.receive_json()["running"] == [
            {"id": "t2", "name": "Find a hotel", "status": "waiting_for_user", "activity": None},
        ]


async def _read(tasks):
    if tasks is None:
        raise RuntimeError("database is locked")
    return tasks


def test_a_timed_out_list_is_sent_once_more_only_while_it_is_the_newest(nodes_client, monkeypatch):
    monkeypatch.setattr("sentient.nodes.service.TASKS_SEND_S", 0.05)
    monkeypatch.setattr("sentient.nodes.service.TASKS_RETRY_S", 0.05)
    client = nodes_client()

    class SlowOnce:
        """A device that misses the deadline on its first send."""
        closed = False

        def __init__(self):
            self.latest_tasks = None
            self.delivered = []

        async def send(self, message):
            if not self.delivered and not getattr(self, "slowed", False):
                self.slowed = True
                await asyncio.sleep(1)
            self.delivered.append(message)

    async def run(newer: bool):
        conn, message = SlowOnce(), {"type": "tasks", "running": []}
        conn.latest_tasks = message
        sending = asyncio.create_task(client.core.nodes._send_tasks(conn, message))
        if newer:
            await asyncio.sleep(0.07)
            conn.latest_tasks = {"type": "tasks", "running": [{"id": "t1"}]}
        await sending
        return conn.delivered

    assert client.portal.call(run, False) == [{"type": "tasks", "running": []}]
    assert client.portal.call(run, True) == []  # the newer list carries the update instead
