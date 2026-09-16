from sentient import paths
from sentient.config import config_json_schema, load_config, save_config
from sentient.config.schema import SentientConfig


def test_defaults_load_without_file():
    cfg = load_config()
    assert cfg.gateway.host == "127.0.0.1"
    assert cfg.models.roles.primary.startswith("ollama_chat/")


def test_roundtrip(isolated_home):
    cfg = SentientConfig()
    cfg.assistant.user_name = "Sarthak"
    cfg.models.roles.primary = "anthropic/claude-sonnet-5"
    cfg.models.fallbacks["primary"] = ["ollama_chat/qwen3:8b"]
    path = save_config(cfg)
    assert path == paths.config_file()
    loaded = load_config()
    assert loaded.assistant.user_name == "Sarthak"
    assert loaded.models.fallbacks["primary"] == ["ollama_chat/qwen3:8b"]


def test_schema_has_descriptions_for_ui():
    schema = config_json_schema()
    defs = schema["$defs"]
    assert "description" in defs["ApprovalsConfig"]["properties"]["mode"]
    assert defs["ApprovalsConfig"]["properties"]["mode"]["enum"] == ["off", "ask", "always"]
