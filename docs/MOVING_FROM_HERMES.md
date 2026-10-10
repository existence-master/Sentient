# Moving from Hermes Agent

Sentient can take over what most people use Hermes Agent for: chat with memory, scheduled jobs, briefs, skills,
messaging apps, a terminal, MCP servers and browser automation. This guide moves you over in an afternoon and lets
you run both side by side until you trust it.

## 1. Install and choose your models

Install Sentient (see [Getting started](GETTING_STARTED.md)) and open it. During setup:

- **Local**: pick a local model (Ollama). Run **Check my models** to confirm it can call tools and fits your graphics
  card.
- **Your Claude plan**: Max and Team plans include API credits. Open **Settings > Models > Use a plan you already
  have** and follow the steps to create an API key from those credits. Sentient cannot sign in with a Claude.ai
  account the way Hermes does; Anthropic does not allow that for other apps.
- **OpenRouter**: **Connect OpenRouter** signs you in with one click, including free models.

Use the model name in the title bar to switch between **Local only**, **Cloud** and **Mixed** at any time, or
`/model` from a chat app.

## 2. Import from Hermes

On the last setup screen, or later in **Settings > Advanced**, choose **Coming from Hermes?** and point it at your
Hermes folder (usually `~/.hermes`). You see everything before anything changes, and you pick what to bring:

| From Hermes | Becomes in Sentient |
|---|---|
| Your own skills | Skills waiting for your review (built-in Hermes skills you never changed are skipped) |
| `MEMORY.md`, `USER.md` | Memories and things Sentient knows about you, marked as imported. They wait in Memory > Review until you approve them (one click removes them all) |
| `SOUL.md` | Sentient's personality, only if you switch it on after seeing both side by side |
| Scheduled jobs | Paused tasks with the same schedule, prompt and delivery (a job that went to WhatsApp goes to your "Message yourself" chat). Each one plans and asks for your approval the first time you turn it on |
| MCP servers | MCP servers, turned off until you enable them. Sign in again, or fill in their keys with **Add values**, where needed |

Keys, tokens, `auth.json` and `.env` are never read.

## 3. Connect the rest

- **WhatsApp**: **Channels > WhatsApp > Connect**, then on your phone open WhatsApp > Linked devices > Link a
  device and scan the code. Talk to Sentient in your "Message yourself" chat. This uses WhatsApp Web, like Hermes,
  and is not WhatsApp's official API, so WhatsApp could limit the account.
- **Telegram and Discord**: **Channels**, connect a bot and pair your chat with a code.
- **MCP servers that need sign-in** (for example Composio or Notion): **Integrations > MCP servers > Sign in**.
  Servers that used headers or environment values show **Add values**: fill them in and the server reconnects.
- **Where each task's results go**: a task's **Details > Send results to** picks the usual paired chats, this
  computer only, or specific chats such as WhatsApp's "Message yourself".
- **Terminal**: off by default. **Settings > Terminal** turns it on, lets you choose the folders it may run in and
  the commands that never need asking (like `git status`). Everything else asks first.
- **Browser profiles**: **Settings > Browser > Profiles**. Create a profile per identity and sign in once, or attach
  to a browser you start yourself with `--remote-debugging-port` and its own `--user-data-dir`. Set a profile on a
  task or in a skill so that job always uses it.

## 4. Briefs

**Notifications > Set up my Daily Brief** creates a morning brief, and its menu adds an evening wrap-up. Both are
normal tasks you can reschedule, and both can be delivered to your paired chats, or only to the chat you pick in
their **Send results to**.

## 5. Approvals: a different habit

Hermes is often run with approvals off. Sentient works better with **lasting rules** instead: in **Settings >
Approvals & safety**, set **Allow** on the apps and tools you trust, **Never** on what it must not touch, and leave
the rest on **Ask**. Purchases always ask, and once a chat or task has read an email or web page, anything that
could send data out asks too.

## 6. Run both for a week

Keep Hermes running while you turn on imported jobs one at a time in Sentient. When a Sentient job has run well for
a few days, pause the Hermes one. Turn Hermes off when nothing is left on it.

## What is different

- Sentient does not use a Claude.ai login; use your plan's API credits or another provider.
- Script jobs report every run that prints something, like in Hermes; Hermes monitor jobs report only changes. A
  script task can switch between the two.
- Hermes features with no Sentient equivalent yet include named assistants and a skills marketplace; see the
  [issue tracker](https://github.com/existence-master/Sentient/issues) for what is planned.
