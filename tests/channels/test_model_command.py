"""/model in a messaging app (#212): show the current model setup and switch presets with buttons or numbers.

Ollama is never reached: the channel fixtures mock every HTTP call, so local models report "Ollama isn't running".
"""

from __future__ import annotations

from sentient.config.schema import PRESET_CLOUD_MODELS, ModelRoles
from tests.channels.conftest import callback

DEFAULTS = ModelRoles()
ANTHROPIC = PRESET_CLOUD_MODELS["anthropic"]


def keyboard(msg: dict) -> list[dict]:
    return [b for row in msg["reply_markup"]["inline_keyboard"] for b in row]


async def test_model_lists_presets_and_a_button_applies_one(tg):
    app = tg.app
    app.config.models.roles.primary = "openai/gpt-x"
    await tg.pair(42)
    await tg.say(42, "/model")
    msg = tg.api.button_messages()[-1]
    assert "your own setup, chatting with openai/gpt-x" in msg["text"]
    assert "Cloud: Needs a key for Anthropic, OpenAI or OpenRouter, or a ChatGPT sign-in." in msg["text"]
    buttons = keyboard(msg)
    assert [b["text"] for b in buttons] == ["Local only"]  # Cloud and Mixed need a key first
    assert all(len(b["callback_data"].encode()) <= 64 for b in buttons)

    tg.ch.dispatch(callback(42, msg["_id"], buttons[0]["callback_data"]))
    await tg.ch.wait_idle()
    assert app.config.models.active_preset == "Local only"
    assert app.config.models.roles.primary == DEFAULTS.primary
    assert tg.api.sent("answerCallbackQuery")[-1]["text"] == "Switched to Local only."
    edit = [p for p in tg.api.sent("editMessageText") if p["message_id"] == msg["_id"]][-1]
    assert "<b>Switched to Local only</b>" in edit["text"] and "reply_markup" not in edit
    assert "Still needed: Ollama isn't running" in tg.api.screen()[-1]

    await tg.say(42, "/model")
    msg = tg.api.button_messages()[-1]
    assert "Models: <b>Local only</b>" in msg["text"] and "/model undo" in msg["text"]
    assert [b["text"] for b in keyboard(msg)] == ["Local only (current)"]


async def test_model_by_number_name_and_undo(tg, keychain):
    app = tg.app
    keychain["anthropic"] = "sk-ant"
    await tg.pair(42)
    await tg.say(42, "/model 2")  # 1 Local only, 2 Cloud, 3 Mixed
    assert app.config.models.active_preset == "Cloud"
    assert app.config.models.roles.primary == ANTHROPIC["main"]
    assert tg.api.screen()[-1].startswith("Switched to Cloud.")
    await tg.say(42, "/model mixed")
    assert app.config.models.active_preset == "Mixed" and app.config.models.roles.fast == DEFAULTS.fast
    await tg.say(42, "/model undo")
    assert app.config.models.active_preset == "Cloud" and app.config.models.roles.fast == ANTHROPIC["fast"]
    assert tg.api.screen()[-1].startswith("Back to Cloud.")
    await tg.say(42, "/model undo")
    assert "no model switch to undo" in tg.api.screen()[-1]
    await tg.say(42, "/model 9")
    assert "don't have a model setup called '9'" in tg.api.screen()[-1]


async def test_unpaired_chats_cannot_switch_models(tg):
    app = tg.app
    app.config.models.roles.primary = "openai/gpt-x"
    await tg.pair(42)
    await tg.say(42, "/model")
    data = keyboard(tg.api.button_messages()[-1])[0]["callback_data"]
    sent = len(tg.api.sent("sendMessage"))
    await tg.say(7, "/model 1")
    texts = [p["text"] for p in tg.api.sent("sendMessage")[sent:]]
    assert len(texts) == 1 and "isn't paired" in texts[0]
    tg.ch.dispatch(callback(7, 999, data))
    await tg.ch.wait_idle()
    assert "isn't paired" in tg.api.sent("answerCallbackQuery")[-1]["text"]
    assert app.config.models.roles.primary == "openai/gpt-x" and app.config.models.active_preset is None
