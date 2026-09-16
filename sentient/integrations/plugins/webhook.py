"""Webhooks as a trigger source: a builtin pseudo-integration with no tools.

Its ``triggers`` are computed from the hooks the user created, so the task editor can offer
"When a webhook is called" (source ``webhook``, event = hook id).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sentient.integrations.base import IntegrationPlugin

if TYPE_CHECKING:  # pragma: no cover
    from sentient.integrations.service import IntegrationManager


class WebhookPlugin(IntegrationPlugin):
    id = "webhook"
    display_name = "Webhooks"
    description = (
        "Let other apps, scripts and smart-home gadgets start Sentient tasks by calling a private web address. "
        "Create a webhook, then pick it as the trigger of a task: \"When a webhook is called\"."
    )
    category = "utilities"
    icon = "webhook"
    auth_type = "builtin"
    instructions_md = (
        "1. Create a webhook and give it a name (for example `Build finished`).\n"
        "2. Copy the address and the secret. The secret is shown only once; keep it private.\n"
        "3. In the other app, send a **POST** request to the address with the header "
        "`X-Sentient-Secret: <secret>` and any JSON, form or text body.\n"
        "4. Create a task that runs \"When a webhook is called\" and choose this webhook. The request body is "
        "available to the task as `body`.\n\n"
        "Example: `curl -X POST -H \"X-Sentient-Secret: <secret>\" -H \"Content-Type: application/json\" "
        "-d '{\"status\": \"passed\"}' <address>`"
    )
    tools = []

    async def dynamic_triggers(self, mgr: IntegrationManager) -> list[dict]:
        return [{"event": h["id"], "label": f"When \"{h['name']}\" is called"} for h in await mgr.hooks.list()]


PLUGIN = WebhookPlugin()
