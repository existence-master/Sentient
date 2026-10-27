"""All REST routers. One module per feature package; each module is owned by that package.

core, models, notifications : core (composition root owner)
mcp                         : core (the MCP server's status and token endpoints, #443)
tasks                       : tasks package
integrations                : integrations package
memory, skills, proactivity : memory / proactivity / evolution package
skills_hub                  : skills package (the Skills hub, #210)
voice                       : voice package (also defines the /ws/voice WebSocket)
subagents                   : core
sandbox, browser, channels  : their packages
terminal                    : terminal package
nodes                       : nodes package (also defines /ws/node)
user_model                  : memory package (user model and dreams)
hooks                       : integrations package (inbound webhooks)
migrate                     : core (importing from Hermes)
backups                     : core (backup, restore and snapshots)
timeline                    : core (the activity timeline)
folders                     : folders package (folders and Obsidian vaults as notes)
reminders                   : reminders package (timers and one-off reminders)
openai_api                  : core (the OpenAI-compatible chat API, #211)
"""

from sentient.gateway.routes import (
    backups,
    browser,
    channels,
    core,
    folders,
    hooks,
    integrations,
    mcp,
    memory,
    migrate,
    models,
    nodes,
    notifications,
    openai_api,
    proactivity,
    reminders,
    sandbox,
    skills,
    skills_hub,
    subagents,
    tasks,
    terminal,
    timeline,
    user_model,
    voice,
)

ROUTERS = [
    core.router,
    models.router,
    mcp.router,
    notifications.router,
    tasks.router,
    integrations.router,
    memory.router,
    skills_hub.router,  # before skills: /api/skills/hub/... must not reach /api/skills/{name}
    skills.router,
    proactivity.router,
    voice.router,
    subagents.router,
    sandbox.router,
    terminal.router,
    browser.router,
    nodes.router,
    channels.router,
    user_model.router,
    hooks.router,
    migrate.router,
    backups.router,
    timeline.router,
    folders.router,
    reminders.router,
    openai_api.router,
]
