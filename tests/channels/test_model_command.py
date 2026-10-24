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

    app.config.models.local_only = True  # already on, so the button has nothing to ask (see the tests below)
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


async def test_model_chat_keeps_a_model_for_this_conversation_only(tg):
    """``/model chat <model>`` (#312): one conversation keeps its own model; the global setup is untouched."""
    app = tg.app
    app.config.models.roles.primary = "openai/gpt-x"
    await tg.pair(42)
    await tg.say(42, "/model chat")
    assert "This chat follows your default model, gpt-x." in tg.api.screen()[-1]
    await tg.say(42, "/model chat lm_studio/big-model")
    assert tg.api.screen()[-1].startswith("This chat now uses big-model.")
    chat = await app.channels.store.chat("telegram", "42")
    session = await app.store.get_session(chat["session_id"])
    assert session["model"] == "lm_studio/big-model" and app.config.models.roles.primary == "openai/gpt-x"
    await tg.say(42, "/model chat")
    assert "This chat uses big-model (lm_studio/big-model)." in tg.api.screen()[-1]

    app.llm.replies = ["answered"]
    await tg.say(42, "hello")
    assert app.llm.calls[-1]["model"] == "lm_studio/big-model"

    await tg.say(42, "/model chat qwen3")
    assert "provider/model" in tg.api.screen()[-1]
    await tg.say(42, "/model chat anthropic/claude-x")  # no key: refused with the reason, nothing changes
    assert "I can't use anthropic/claude-x right now. No Anthropic key is set." in tg.api.screen()[-1]
    assert (await app.store.get_session(chat["session_id"]))["model"] == "lm_studio/big-model"
    await tg.say(42, "/model chat default")
    assert tg.api.screen()[-1] == "This chat follows your default model again, gpt-x."
    assert (await app.store.get_session(chat["session_id"]))["model"] is None


async def test_a_chat_model_that_cant_answer_is_mentioned_once(tg):
    app = tg.app
    app.config.models.experimental_claude_code = False
    await tg.pair(42)
    await tg.say(42, "/model chat")
    chat = await app.channels.store.chat("telegram", "42")
    await app.store.set_session_model(chat["session_id"], "claude-code/sonnet")
    app.llm.replies = ["answered by the default"]
    sent = len(tg.api.screen())
    await tg.say(42, "hello")
    new = tg.api.screen()[sent:]
    notes = [t for t in new if "can't answer right now" in t]
    assert len(notes) == 1 and notes[0].startswith("<i>This chat uses Claude Sonnet")
    assert any("answered by the default" in t for t in new)
    assert app.llm.calls[-1]["model"] is None


async def test_model_says_when_local_only_blocks_a_preset(tg, keychain):
    """#335: with Local only on, cloud presets are listed with that reason, not offered, and refused by name."""
    app = tg.app
    keychain["anthropic"] = "sk-ant"
    app.config.models.local_only = True
    await tg.pair(42)
    await tg.say(42, "/model")
    msg = tg.api.button_messages()[-1]
    assert "Cloud: Local only is on, and it uses cloud models." in msg["text"]
    assert "Add one in Sentient" not in msg["text"]
    assert [b["text"] for b in keyboard(msg)] == ["Local only"]
    await tg.say(42, "/model cloud")
    assert "Local only is on" in tg.api.screen()[-1] and app.config.models.active_preset is None


async def test_local_only_preset_asks_before_turning_the_switch_on(tg):
    """``/model Local only`` (#446) asks first, like the window, and ``yes`` switches the models and Local only together."""
    app = tg.app
    await tg.pair(42)
    await tg.say(42, "/model Local only")
    assert "Turn on Local only too" in tg.api.screen()[-1] and "/model Local only yes" in tg.api.screen()[-1]
    assert app.config.models.active_preset is None and not app.config.models.local_only  # nothing changed yet

    await tg.say(42, "/model Local only yes")
    assert tg.api.screen()[-1].startswith("Switched to Local only, and Local only is on.")
    assert app.config.models.active_preset == "Local only" and app.config.models.local_only

    await tg.say(42, "/model undo")  # undo puts the switch back too
    assert not app.config.models.local_only


async def test_local_only_preset_by_number_asks_and_skips_the_question_when_already_on(tg):
    app = tg.app
    await tg.pair(42)
    await tg.say(42, "/model 1")
    assert "Turn on Local only too" in tg.api.screen()[-1] and not app.config.models.local_only
    app.config.models.local_only = True
    await tg.say(42, "/model Local only")  # the switch is already on: nothing to ask
    assert tg.api.screen()[-1].startswith("Already using Local only.")  # no question, and nothing left to turn on
    assert app.config.models.local_only


async def test_local_only_button_asks_before_turning_the_switch_on(tg):
    """Tapping "Local only" in the /model list asks the same question as ``/model Local only`` (#457)."""
    app = tg.app
    await tg.pair(42)
    await tg.say(42, "/model")
    msg = tg.api.button_messages()[-1]
    tg.ch.dispatch(callback(42, msg["_id"], keyboard(msg)[0]["callback_data"]))
    await tg.ch.wait_idle()
    assert tg.api.sent("answerCallbackQuery")[-1]["text"] == "Turn on Local only too?"
    edit = [p for p in tg.api.sent("editMessageText") if p["message_id"] == msg["_id"]][-1]
    assert "Turn on Local only too?" in edit["text"] and "reply_markup" not in edit
    assert "/model Local only yes" in tg.api.screen()[-1] and "/model Local only no" in tg.api.screen()[-1]
    assert app.config.models.active_preset is None and not app.config.models.local_only  # nothing changed yet

    await tg.say(42, "/model Local only yes")
    assert tg.api.screen()[-1].startswith("Switched to Local only, and Local only is on.")
    assert app.config.models.active_preset == "Local only" and app.config.models.local_only


async def test_local_only_no_cancels(tg):
    """``/model Local only no`` (#457) answers the question by leaving everything as it is."""
    app = tg.app
    await tg.pair(42)
    await tg.say(42, "/model Local only")
    assert "/model Local only no" in tg.api.screen()[-1]
    await tg.say(42, "/model Local only no")
    assert tg.api.screen()[-1] == "Okay, I left your models as they are."
    assert app.config.models.active_preset is None and not app.config.models.local_only
