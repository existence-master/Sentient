# Getting started

This guide takes you from install to a useful assistant in about ten minutes. No account and no technical
background needed.

## 1. Install

- **Windows:** run the Sentient installer. It installs for your user only, needs no administrator rights and
  brings everything it needs. Windows may warn that the app is unrecognised because builds are not signed yet:
  choose **More info → Run anyway**.
- **macOS and Linux:** installers are on the way. Until then, [run from source](DEVELOPING.md).

Your data lives in one folder, `~/.sentient` (on Windows `C:\Users\<you>\.sentient`). Uninstalling the app keeps
it, so you can reinstall without losing anything.

## 2. Choose a brain

When Sentient opens it asks a few questions about you, then which model to use. You have two choices.

**Free and private, on your computer.** Install [Ollama](https://ollama.com/download), then in a terminal:

```bash
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

`qwen3:8b` needs a computer with about 8 GB of graphics memory or 16 GB of RAM. Older small models such as
`qwen3:4b` cannot use tools, so Sentient answers without them.

**Faster, with your own key.** Paste a key from Anthropic, OpenAI, Gemini, OpenRouter or another provider. Keys go
into your system keychain, never into a file.

You can mix both and change your mind any time in **Settings → Models**: for example a cloud model for chat and a
local one for background work. **Test** next to each model checks that it works and can use tools.

## 3. Things to try

- **Chat:** "What's the weather in Pune this weekend?" Watch the tool cards show what Sentient does.
- **Remember:** "My sister Anika is moving to Berlin next month." Ask about it tomorrow.
- **A task:** in **Tasks**, type "Every weekday at 9, summarise my unread email". Sentient drafts a plan; approve
  it and it runs on schedule.
- **A watcher:** "Tell me when the price of this headphone drops below 25,000." Sentient writes a small check that
  runs on its own and only wakes the model when something changes.
- **Voice:** click the microphone, or turn on **Hey Sentient** in Settings → Voice.

## 4. Connect your apps

Open **Integrations**. Each app has a **Connect** button with step-by-step instructions. Web search, weather, maps,
news, web pages and charts work without any setup. Connecting Gmail and Calendar lets Sentient notice things and
suggest actions in **Notifications**; approve one and it becomes a task.

## 5. Take it with you

- **Phone:** open **Devices → Add a device**, turn on *Allow devices on my Wi-Fi*, and scan the QR code. Your phone
  can then show notifications from Sentient, share its location when asked, and talk to it.
- **Telegram:** create a bot with [@BotFather](https://t.me/BotFather), paste its token in **Devices → Messaging
  apps → Telegram**, then send the pairing code to your bot. Only paired chats can talk to Sentient.
- **Smart glasses and your own hardware:** see [NODES.md](NODES.md) and the reference firmware in `firmware/`.

## 6. Stay in control

- Sentient asks before it sends, deletes, buys or runs code. Change how often in **Settings → Approvals & safety**.
- **Memory** and **About you** show everything it remembers and believes about you. Correct or delete anything.
- New skills Sentient writes for itself wait in **Skills → Pending review** until you approve them.
- What leaves your computer, and when, is explained in [PRIVACY.md](PRIVACY.md).

## Troubleshooting

| What you see | What to do |
|---|---|
| "Ollama isn't running" | Start Ollama, then click **Check again**. Or use a cloud key instead. |
| Replies are slow | Local models on a laptop take 15 to 45 seconds with tools. A cloud model answers in a few seconds. Close other apps that use the graphics card. |
| "This model can't use tools" | Pick a model that supports tools, for example `qwen3:8b` or larger. **Test** in Settings → Models tells you. |
| A photo question says the model can't see images | Choose a vision model for the Vision role, for example `ollama_chat/qwen2.5vl:3b`. |
| Something else | Ask in [Discussions](https://github.com/existence-master/Sentient/discussions/categories/q-a) and include the end of `~/.sentient/logs/backend.log`, with personal details removed. |
