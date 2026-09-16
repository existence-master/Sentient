"""Swarm tasks (v2 orchestrate_swarm_task + run_single_item_worker + aggregate).

Pipeline: item extraction (when no items) -> resource manager groups items into
``[{item_indices, worker_prompt, required_tools}]`` -> parallel workers (asyncio,
bounded by ``tasks.swarm_max_agents``), each a ``run_loop`` with only its required
tools -> aggregation -> result generator.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

from sentient.agent.loop import LoopResult
from sentient.llm.provider import parse_json_loose
from sentient.tasks.executor import select_tools
from sentient.tasks.jsonio import complete_json
from sentient.tasks.prompts import (
    ITEM_EXTRACTOR_SYSTEM_PROMPT,
    RESOURCE_MANAGER_SYSTEM_PROMPT,
    SWARM_WORKER_SYSTEM_PROMPT,
    build_tool_catalog,
)

if TYPE_CHECKING:  # pragma: no cover
    from sentient.tasks.service import TaskService

log = logging.getLogger(__name__)


def empty_swarm_details(goal: str, items: list | None = None) -> dict:
    return {
        "goal": goal,
        "items": list(items or []),
        "total_agents": 0,
        "completed_agents": 0,
        "progress_updates": [],
        "aggregated_results": [],
    }


def _as_indices(value: Any, count: int) -> list[int]:
    out: list[int] = []
    for v in value if isinstance(value, list) else []:
        try:
            i = int(v)
        except (TypeError, ValueError):
            continue
        if 0 <= i < count and i not in out:
            out.append(i)
    return out


async def plan_swarm(svc: TaskService, task: dict) -> tuple[list, list[dict], bool]:
    """Returns ``(items, worker_configs, used_fallback)``."""
    app = svc.app
    details = task.get("swarm_details") or {}
    goal = details.get("goal") or task.get("original_prompt") or task.get("description") or ""
    if not goal:
        raise ValueError("Swarm task is missing a goal to extract items from.")
    items = details.get("items") or []
    if not items:
        data = await complete_json(app.llm,
            "planner",
            [{"role": "system", "content": ITEM_EXTRACTOR_SYSTEM_PROMPT}, {"role": "user", "content": goal}],
        )
        items = data.get("items") if isinstance(data, dict) else data
        if not isinstance(items, list) or not items:
            raise ValueError("Item Extractor agent failed to extract a list of items from the goal.")

    catalog = await build_tool_catalog(app)
    system = RESOURCE_MANAGER_SYSTEM_PROMPT.format(available_tools_json=json.dumps(list(catalog.keys())))
    user = f'Goal: "{goal}"\n\nItems (sample of {len(items)} total):\n{json.dumps(items[:5], indent=2, default=str)}'
    data = await complete_json(app.llm,
        "planner", [{"role": "system", "content": system}, {"role": "user", "content": user}]
    )
    raw: Any = data
    if isinstance(data, dict):
        raw = data.get("workers") or data.get("plan") or data.get("configurations") or []
    configs: list[dict] = []
    for c in raw if isinstance(raw, list) else []:
        if not isinstance(c, dict):
            continue
        indices = _as_indices(c.get("item_indices"), len(items))
        prompt = c.get("worker_prompt")
        tools = c.get("required_tools")
        if not indices or not isinstance(prompt, str) or not prompt.strip():
            log.warning("skipping invalid swarm worker configuration: %s", c)
            continue
        configs.append({
            "item_indices": indices,
            "worker_prompt": prompt.strip(),
            "required_tools": [str(t) for t in tools] if isinstance(tools, list) else [],
        })
    if configs:
        return items, configs, False
    fallback = [{
        "item_indices": list(range(len(items))),
        "worker_prompt": f"Overall goal: {goal}\n\nApply this goal to the single item you are given and return the result for that item.",
        "required_tools": [],
    }]
    return items, fallback, True


def parse_worker_output(text: str) -> Any:
    cleaned = (text or "").strip()
    if cleaned.lower() == "null":
        return None
    if cleaned[:1] in {"{", "["} or cleaned.startswith("```"):
        try:
            return parse_json_loose(cleaned)
        except ValueError:
            return cleaned
    return cleaned


async def run_worker(svc: TaskService, task: dict, run_id: str, worker_id: str, item: Any, config: dict) -> Any:
    app = svc.app
    assert app.agent is not None
    task_id = task["id"]
    await svc.swarm_update(task_id, worker_id, "processing", f"Starting work on item: {str(item)[:100]}")
    try:
        tool_names, _, _ = select_tools(app.registry, config.get("required_tools") or [], include_core=False)
        item_context = json.dumps(item, indent=2, default=str)
        messages = [
            {"role": "system", "content": SWARM_WORKER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"**Task:**\n{config['worker_prompt']}\n\n**Input Data for this Task:**\n```json\n{item_context}\n```",
            },
        ]
        ctx = app.agent.tool_context(None, "task")
        ctx.extra.update({"task_id": task_id, "run_id": run_id, "worker_id": worker_id})
        result = LoopResult()
        async for _event in app.agent.run_loop(
            messages,
            ctx,
            result=result,
            role="executor",
            model=task.get("model") or None,
            tool_names=tool_names,
            max_rounds=app.config.tasks.max_tool_rounds,
            use_approvals=False,
            source="task",
        ):
            pass
    except Exception as exc:
        log.warning("swarm worker %s failed: %s", worker_id, exc)
        await svc.swarm_update(task_id, worker_id, "error", f"An error occurred: {exc}")
        return {"error": str(exc), "item": item}
    if result.error:
        await svc.swarm_update(task_id, worker_id, "error", f"An error occurred: {result.error}")
        return {"error": result.error, "item": item}
    final = parse_worker_output(result.text)
    await svc.swarm_update(task_id, worker_id, "completed", f"Finished work. Result: {str(final)[:100]}")
    return final


async def execute_swarm(svc: TaskService, task: dict, run: dict) -> tuple[str, list[Any]]:
    task_id, run_id = task["id"], run["id"]
    items = (task.get("swarm_details") or {}).get("items") or []
    jobs = [
        (config, i)
        for config in (run.get("plan") or [])
        if isinstance(config, dict)
        for i in config.get("item_indices") or []
        if isinstance(i, int) and 0 <= i < len(items)
    ]
    limit = asyncio.Semaphore(svc.app.config.tasks.swarm_max_agents)

    async def one(n: int, config: dict, index: int) -> Any:
        async with limit:
            return await run_worker(svc, task, run_id, f"agent-{n + 1}", items[index], config)

    results = list(await asyncio.gather(*(one(n, c, i) for n, (c, i) in enumerate(jobs))))
    failed = sum(1 for r in results if isinstance(r, dict) and "error" in r)
    status = "completed_with_errors" if failed else "completed"
    await svc.swarm_update(task_id, "aggregator", "aggregating",
                           f"All {len(results)} agents have completed. Generating final report.",
                           aggregated_results=results)
    await svc.progress(task_id, run_id, {
        "type": "info", "content": f"All {len(results)} agents have completed. Generating final report.",
    })
    return status, results
