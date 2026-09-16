"""Charts via QuickChart (Chart.js configs rendered to PNG). No key needed."""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any
from urllib.parse import quote

from sentient import paths
from sentient.integrations.base import IntegrationError, IntegrationPlugin, itool
from sentient.integrations.common import http_client
from sentient.tools.base import Risk, ToolContext

QUICKCHART = "https://quickchart.io/chart"
VALID_TYPES = {"bar", "line", "pie", "doughnut", "radar", "polarArea", "scatter", "bubble", "radialGauge",
               "speedometer", "horizontalBar", "sparkline", "progressBar"}
MAX_GET_URL = 2000


def _config(chart_config: Any) -> dict:
    if isinstance(chart_config, str):
        try:
            chart_config = json.loads(chart_config)
        except json.JSONDecodeError as exc:
            raise IntegrationError("chart_config must be a Chart.js configuration object (valid JSON).") from exc
    if not isinstance(chart_config, dict):
        raise IntegrationError("chart_config must be a Chart.js configuration object.")
    if chart_config.get("type") not in VALID_TYPES:
        raise IntegrationError(f"Chart type must be one of: {', '.join(sorted(VALID_TYPES))}.")
    datasets = (chart_config.get("data") or {}).get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise IntegrationError("chart_config needs data.datasets with at least one dataset.")
    return chart_config


async def chart_url(config: dict, width: int, height: int) -> str:
    encoded = quote(json.dumps(config, separators=(",", ":")))
    url = f"{QUICKCHART}?w={width}&h={height}&bkg=white&c={encoded}"
    if len(url) <= MAX_GET_URL:
        return url
    async with http_client() as http:
        r = await http.post(f"{QUICKCHART}/create", json={"chart": config, "width": width, "height": height,
                                                          "backgroundColor": "white"})
    r.raise_for_status()
    data = r.json()
    if not data.get("success"):
        raise IntegrationError("QuickChart couldn't create the chart.")
    return data["url"]


@itool("charts", "chart_create_url")
async def chart_create_url(ctx: ToolContext, chart_config: dict, width: int = 800, height: int = 450) -> dict:
    """Turn data into a chart image link. `chart_config` is a Chart.js config, e.g.
    {"type": "bar", "data": {"labels": ["Mon", "Tue"], "datasets": [{"label": "Steps", "data": [5000, 7200]}]}}.
    Types: bar, line, pie, doughnut, radar, polarArea, scatter, bubble."""
    url = await chart_url(_config(chart_config), int(width), int(height))
    return {"url": url, "markdown": f"![chart]({url})"}


@itool("charts", "chart_save_image", risk=Risk.write, internal=True)  # saves into Sentient's own files folder
async def chart_save_image(ctx: ToolContext, chart_config: dict, filename: str | None = None, width: int = 800,
                           height: int = 450) -> dict:
    """Render a Chart.js config to a PNG saved in Sentient's files folder (charts/...). Tell the user the file name.
    Same `chart_config` format as chart_create_url."""
    config = _config(chart_config)
    async with http_client(timeout=60) as http:
        r = await http.post(QUICKCHART, json={"chart": config, "width": int(width), "height": int(height),
                                              "format": "png", "backgroundColor": "white"})
    r.raise_for_status()
    if not r.headers.get("content-type", "").startswith("image/"):
        raise IntegrationError("QuickChart didn't return an image. Check the chart configuration.")
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", (filename or "").rsplit(".", 1)[0]).strip("_")
    stem = stem or f"{config['type']}_{datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')}"
    folder = paths.files_dir() / "charts"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{stem}.png"
    target.write_bytes(r.content)
    return {"saved": f"charts/{target.name}", "path": str(target), "bytes": len(r.content)}


class ChartsPlugin(IntegrationPlugin):
    id = "charts"
    display_name = "Charts"
    description = (
        "Turns numbers into bar, line, pie and other charts using QuickChart, as a shareable link or a PNG saved "
        "in Sentient's files folder. Built in, nothing to set up."
    )
    category = "utilities"
    icon = "chart"
    auth_type = "builtin"
    selection_hint = "Use to make a chart or graph image from data."
    tools = [chart_create_url, chart_save_image]


PLUGIN = ChartsPlugin()
