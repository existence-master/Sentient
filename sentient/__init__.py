"""Sentient - a local-first, configurable, self-evolving personal assistant.

The package is one process: a gateway daemon that owns the agent loop, the
memory, the task scheduler and every tool plugin. Every UI, channel adapter
or device (including future wearables) is a client of that gateway.
"""

__version__ = "3.0.0a0"
