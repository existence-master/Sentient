"""The activity timeline (docs/API.md section 21, issue #147): one list of what Sentient did and what it will do,
read from the tables the other packages already keep. ``timeline.Timeline`` builds it; ``words`` holds the plain
sentences."""

from sentient.activity.timeline import KINDS, Timeline

__all__ = ["KINDS", "Timeline"]
