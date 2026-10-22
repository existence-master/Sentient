# Verified on a real PC

What has actually been tested end to end, on what hardware, and what has not. Updated with each round of real-model testing. Automated tests (CI) are described in [DEVELOPING.md](DEVELOPING.md).

Tested on 2026-09-15 on a Windows 11 laptop (RTX 4060 with 8 GB, 15 GB RAM) with local Ollama models only:
`qwen3:8b` for every chat role and `nomic-embed-text` for embeddings. Scripts drove the real engine through
its public REST and WebSocket interface, and the desktop screens were captured from the same data.

**Core flows**

| Flow | Result |
|---|---|
| Onboarding, chat with a tool call, recurring task, triggered task | Pass |
| Memory: learn a fact in chat and recall it later | Pass |
| One-off task: plan, approve, run, file created, report, notification | Pass |
| Proactive suggestion from a calendar event | Pass |
| Skill proposed for review after multi-step work | Pass |
| Voice turn with system text-to-speech | Pass. First token in about 6 s |

**New in this release**

| Flow | Result |
|---|---|
| Code execution: the model writes and runs Python that returns a number | Pass |
| Steering: a message sent mid-reply changes the answer | Pass |
| Subagents: a helper runs with its own tools and reports back | Pass |
| Browser: open a local shop page and read a price | Pass |
| Browser: "Place order" asks for approval as a purchase; declining stops it | Pass |
| Devices: pair simulated glasses, show text on them, read their location | Pass |
| User model refresh and nightly consolidation (Pune to Bengaluru contradiction settled) | Pass |
| Skill repair: a failed step produces a fix proposal with the reason | Pass |
| Webhook: wrong secret refused, right secret starts a triggered task | Pass |
| Watcher script job alerts with no model call | Pass |
| Wake word: synthesized "Hey Sentient, what is two plus two?" answered by voice | Pass. About 10 s end to end |
| Sandbox status and a direct run | Pass |

**Also verified on 2026-09-16**

| Check | Result |
|---|---|
| The same 12 flows on cloud models: Claude Sonnet 5 for chat, Claude Haiku 4.5 for background work | Pass, and much faster: most flows finish in 5 to 15 s |
| Smart glasses over the home network: mDNS discovery, TLS with certificate pinning, pairing code, text and notification shown on the device | Pass with the reference device program |
| "What can you see?" through a device camera, using a real webcam frame and a local vision model | Pass with `ollama_chat/qwen2.5vl:3b` |
| A task that has to ask, on `qwen3:8b`: "write a dinner note, ask me Toit or Truffles". The run paused with that question and both options, sent a "needs your answer" notification, and after the answer wrote the right file | Pass |
| Follow-ups deciding and drafting on `qwen3:8b`, with three made-up threads: a client question waiting 4 days, a "sounds good" reply, and your own unanswered request | Pass: replied, skipped, nudged; no placeholders in drafts (3 to 18 s each) |
| Telegram with a real bot and a real phone, on `qwen3:8b`: pairing code, chat, approval buttons before running code (then the Docker sandbox), weather, `/help`, and two voice notes understood by faster-whisper | Pass. Replies to voice notes are text unless `channels.telegram.voice_replies` is on |

**Claude through your own Claude Code (experimental, ADR 0022), verified on 2026-10-10**

Claude Code 2.1.269 signed in to a Claude Max plan (`claude auth status`: `loggedIn true`, `authMethod claude.ai`,
`subscriptionType max`), chat on `claude-code/sonnet`, every other role and memory on local `qwen3:8b`. Fresh
engines driven over REST and `/ws`. No Claude token was printed or stored.

| Check | Result |
|---|---|
| Status and Test: Settings shows the version while the switch is on; Test gives a reply with a tool call | Pass. `version "2.1.269 (Claude Code)"`; Test `ok`, `supports_tools: true`, 12.2 s |
| A plain reply streams | Pass. A short answer comes as one piece after about 13 s; a longer one as 32 pieces between 8.2 s and 9.9 s, done at 11.7 s |
| The exact stream-json shape | Confirmed. `system/init` first (its `tools` list is checked), then `stream_event` text and thinking deltas, the full `assistant` message with any `tool_use` blocks and usage, then `result` |
| Claude Code's own tools are off | Pass. Asked to use its own Bash, PowerShell or Read, it said it only had Sentient's tools and made no call. Every `init` listed only `mcp__sentient__*` tools plus the harmless built-ins |
| A Sentient tool with approval | Pass. `execute_code` asked first (risk `exec`); allowed, it ran in Sentient and streamed its output; declined, nothing ran and the reply said so. `file_write` ran in Sentient's loop without asking, as approvals mode `ask` intends |
| Memory and usage | Pass. Remembered facts reach the reply. Each model round sends `usage` with prompt and completion tokens and no price (the plan pays) |
| Stop everything ends a running reply | Pass. Stopped 2.2 s after a long reply began to stream: the turn ended as cancelled, and the Claude Code process, its launcher and Sentient's tool bridge (4 processes while it answered) were all gone |
| Background work is refused | Pass. Fast and planner roles on `claude-code/` are refused with a plain message, so their fallback model takes over |
| Scratch folders are removed | Pass after the fix for #259: no folder left after each reply or after Stop everything, and a two-hour-old leftover was removed when the engine started |
| Turning the switch on, then the status | Pass after the fix in #246: with the save held back for 1.5 s, the first status still said "off"; the window asked again every 0.4 s and showed the version as soon as the save landed |

Not checked: how this usage appears on the Claude plan's usage page (it needs the owner's claude.ai account).
Chats on `claude-code/` models show no context meter yet (#258).

The automated suite (522 engine tests, desktop typecheck and build) passes.

**Sign in with ChatGPT follow-ups (#241), checked on 2026-10-11 without a ChatGPT account**

The owner's ChatGPT account has no Plus or Pro plan (OpenAI's consent page turned it down), so nothing here signed in
to ChatGPT for real. The window was driven in the built app; the signed-in screens use the dev-only demo sign-in
(`#/...?demo=1&chatgpt=1`, never in a packaged build) with the fictional Maya Rao.

| Check | Result |
|---|---|
| First-run setup, cloud step: a **ChatGPT plan** choice; picking it shows "Use your ChatGPT plan" and **Continue with ChatGPT** | Pass. Real engine, fresh home, dark and light themes |
| The button carries OpenAI's ChatGPT logo (the published sign-in button SVG, unchanged): white logo on black in the light theme, black on white in the dark one | Pass |
| "You're using your ChatGPT plan" shows once after signing in; **Got it** dismisses it for good | Pass in the window (demo sign-in). Engine side: `welcome` false while signed out, true once signed in, false after `POST /api/models/connect/chatgpt/welcome`, through a later sign-out and sign-in (automated) |
| The chat box shows **Using ChatGPT plan** with **Manage usage** while the main model is a `chatgpt/` model | Pass. Seeded Maya home with `chatgpt/gpt-6-astra` as the main model, dark and light themes |
| The fast model rule on a recorded list: OpenAI's published Codex model catalog (11 models, 8 listed) | Pass. Ranked by `priority`: `gpt-6.1-sol` (1), `gpt-6-astra` (2), `gpt-6-sol`, `gpt-6-luna`, then the older ones. Main and fast `chatgpt/gpt-6.1-sol`, the everyday model (the list itself puts the heavier `gpt-6-astra` first); fast jobs send `reasoning: {effort: "low"}` (the lightest the list gives), the main role keeps `medium`. No listed name says "mini" or "nano", so the old guess had nothing to find |
| A user without ChatGPT: real engine start-up on a fresh home | Pass. `GET /api/models/connect/chatgpt` says `signed_in: false`, `welcome: false`; presets unchanged; no errors in the log |
| A chat on `claude-code/sonnet` | Pass. 12.8 s, the right answer |
| A chat on `qwen3:8b` | Pass. 25.5 s, the right answer, 32% of context |

Checked again on 2026-10-22 without a ChatGPT account, after the Connect Claude card (#448) and Claude Code roles (#449)
landed: the first-run cloud step still shows the **ChatGPT plan** choice next to **Anthropic** (whose card is now Connect Claude); a ChatGPT plan
model turned down with 401 or 403 gets the check-up fix "Sign in with ChatGPT again under Settings > Models > Use a
plan you already have." (an account that can't share its plan is asked for a Plus or Pro account); and applying
"Local only" or a preset of your own no longer asks ChatGPT for its model list.

**Owner checklist: Sign in with ChatGPT with real accounts (#241, pending)**

Needs one ChatGPT **Plus** and one ChatGPT **Pro** account. Use the packaged app or `npm run dev` on a fresh data
folder (Settings > Advanced > Data folder shows which). Never paste a token anywhere. Each step says what you should see.

1. **Sign in (Plus).** Settings > Models > Use a plan you already have > **Continue with ChatGPT**. The browser opens
   ChatGPT; sign in and allow Sentient. The app shows "You're signed in with ChatGPT", then "You're using your ChatGPT
   plan" once (Got it closes it for good). The panel shows **Using your ChatGPT plan**, your email and "N models on
   your plan".
2. **The model list.** Settings > Models > Roles > main model: the plan's models are listed as `chatgpt/...`, the
   everyday model first (the one ChatGPT itself offers by default). Note the names and their order on #241: if the
   order looks wrong, the real list may not rank its models (`priority`), and the fast jobs then use the list's order.
3. **Use it.** Press **Use ChatGPT for every job**. The chat box shows **Using ChatGPT plan** with **Manage usage**.
4. **A streamed tool call.** In a new chat: "What's 2 to the power of 20? Use the code tool." The reply streams in
   pieces, the tool call shows (approve it), and the answer is 1048576. Then Settings > Models > **Check my models**:
   Reply, Tool call and Two tool steps pass for the main model, and JSON reply for the background model.
5. **A refresh after an hour.** Leave the app running for at least 65 minutes (the sign-in lasts about an hour), then
   send a chat. It answers with no sign-in prompt, and the panel still shows your email.
6. **Usage limit.** On **Manage usage** (chatgpt.com/settings/usage) set the lowest limit for Sentient, then chat
   until it is reached. The chat says "Usage limit reached. You've used what your ChatGPT plan allows Sentient for
   now..." with the link, and no other paid model is used unless you added a fallback yourself. The check-up's reply
   check fails with that message and no "sign in again" fix. Put the limit back afterwards.
7. **Signed out elsewhere.** On chatgpt.com, remove Sentient's access (or sign out of all devices), then chat. The
   chat says "ChatGPT signed you out. Sign in with ChatGPT again in Settings > Models.", and Check my models gives the
   fix "Sign in with ChatGPT again under Settings > Models > Use a plan you already have."
8. **Switch accounts (Pro).** **Sign out**, then **Continue with ChatGPT** and sign in with the Pro account. It goes
   through on the first try (Sentient reuses the app id ChatGPT gave it the first time), the panel shows the Pro
   email, and the model list is the Pro plan's (step 2 again). The welcome message does not come back.
9. **Not eligible.** If you have a Free account, sign in with it. Either ChatGPT's own page turns it down (the app then
   says "ChatGPT didn't connect" with ChatGPT's reason and Try again), or the first chat says "This ChatGPT account
   can't share its plan with other apps. Plan usage needs ChatGPT Plus or Pro." and Check my models asks you to sign
   in with a Plus or Pro account.

Post the results on #241 (model names are fine; never emails or tokens).

Real-model runs found and fixed these problems, each now covered by a test: a memory update dropped a
city; a task run stopped after announcing its next step; the model passed a whole snapshot line as a browser
element reference, which hid a purchase from the approval check; empty replies after a tool call; scripts
that forgot to import `result`; the tiny wake-word model hearing "Hey" as "He"; and a model claiming a declined
action had happened (declined results now say plainly that nothing was done).

**Not yet verified on real hardware or accounts:**

- Sign in with ChatGPT with a real Plus or Pro account (#241, listed above).
- Integrations with real Google, GitHub, Slack, Notion, Discord, Trello or WhatsApp accounts, and a paired
  Discord chat. Spoken replies to Telegram voice notes.
- A real phone and real glasses hardware. The device protocol was exercised over the network with the
  reference device program, and the camera path with this laptop's webcam. The firmware in `firmware/` has
  never been compiled or flashed.
- The Docker code-execution backend (Docker was not running) and IMAP push against a real mail server.
- A live microphone in voice mode. Wake word and speech were driven with synthesized audio.
- The macOS and Linux installers. The Windows installer is built and verified: a per-user setup that needs no
  admin rights, installs in about a minute, starts the app with its own bundled engine (no Python needed)
  and uninstalls without touching your data. It is not code signed yet, so Windows will warn on first run.

**Known limits:** older `qwen3:4b` pulls cannot call tools in Ollama, so use `qwen3:8b` or larger. Photo
and screen questions need a vision-capable model. With qwen3:8b on this laptop, chat replies with tools take
about 15 to 45 seconds; cloud models are much faster. Running several engines at once on 8 GB of VRAM pushes
Ollama onto the CPU.
