"""Sentient - a local-first, configurable, self-evolving personal assistant.

The package is one process: a gateway daemon that owns the agent loop, the
memory, the task scheduler and every tool plugin. Every UI, channel adapter
or device (including future wearables) is a client of that gateway.
"""

import os

__version__ = "3.0.0a0"

# LiteLLM otherwise downloads its model price list from GitHub at runtime: use the copy it ships with, so Sentient
# makes no network call the user didn't ask for (docs/PRIVACY.md). Set before anything imports litellm.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
