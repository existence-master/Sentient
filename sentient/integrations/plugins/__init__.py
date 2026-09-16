"""Integration plugin modules, one per service. Order is the order shown in the UI.

Each module exposes ``PLUGIN`` (or ``PLUGINS``), instances of IntegrationPlugin.
"""

PLUGIN_MODULES = [
    # keyless builtins
    "search", "web", "weather", "maps", "news", "charts",
    # Google (OAuth, one shared Desktop client)
    "gmail", "gcalendar", "gdrive", "gdocs", "gsheets", "gslides", "gpeople",
    # any email account with an app password (IMAP push + SMTP)
    "email_imap",
    # tokens / keys
    "github", "slack", "notion", "discord", "trello", "whatsapp",
    # optional keyed alternatives to the keyless builtins
    "alternatives",
    # trigger sources without tools
    "webhook",
]
