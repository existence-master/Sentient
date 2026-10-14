"""All REST routers. One module per feature package; each module is owned by that package.

core, models, notifications : core (composition root owner)
tasks                       : tasks package
integrations                : integrations package
memory, skills, proactivity : memory / proactivity / evolution package
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
"""

from sentient.gateway.routes import (
    backups,
    browser,
    channels,
    core,
    hooks,
    integrations,
    memory,
    migrate,
    models,
    nodes,
    notifications,
    proactivity,
    sandbox,
    skills,
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
    notifications.router,
    tasks.router,
    integrations.router,
    memory.router,
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
]
