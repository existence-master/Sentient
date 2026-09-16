from __future__ import annotations

import json
from pathlib import Path

import yaml

from sentient import paths
from sentient.config.schema import SentientConfig

_HEADER = """# Sentient configuration.
# Every key is optional; missing keys use the defaults shown by `sentient config show --defaults`.
# Secrets (API keys) do NOT go here - run `sentient secrets set <provider>`.
"""


def load_config(path: Path | None = None) -> SentientConfig:
    path = path or paths.config_file()
    if not path.exists():
        return SentientConfig()
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return SentientConfig.model_validate(data)


def save_config(config: SentientConfig, path: Path | None = None) -> Path:
    path = path or paths.config_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = config.model_dump(mode="json")
    with path.open("w", encoding="utf-8") as fh:
        fh.write(_HEADER)
        yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)
    return path


def config_json_schema() -> dict:
    """JSON schema of the config, used by the settings UI to render forms."""
    return SentientConfig.model_json_schema()


def dump_json_schema(path: Path) -> None:
    path.write_text(json.dumps(config_json_schema(), indent=2), encoding="utf-8")
