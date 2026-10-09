# Privacy

Sentient is built so that your assistant's data stays yours. There is no Sentient account and no Sentient server.

## What stays on your computer

Everything Sentient keeps lives in `~/.sentient` on your own disk: chats, memories and what Sentient believes about
you, tasks and their results, skills, files it creates, and settings. Keys and tokens are kept in your operating
system's keychain, not in that folder. **Sentient itself sends no analytics or telemetry.**

To erase everything, quit Sentient and delete `~/.sentient`. Remove saved keys from your system keychain under the
service name `sentient`.

## What leaves your computer, and only when you set it up

| You choose to... | What is sent, and to whom |
|---|---|
| Use a **local model** (Ollama, LM Studio) | Nothing. Prompts stay on your machine. |
| Use a **cloud model** | The conversation, relevant memories and tool results for each request go to that provider (for example Anthropic or OpenAI) under their terms. |
| **Connect an app** (Gmail, Slack, Notion...) | Sentient talks directly to that app's servers with your own login. |
| Use **web search, weather, maps or news** | Your query goes to that service. Web search uses DuckDuckGo by default. |
| Let Sentient **use the browser** | It visits the sites the task needs, in a separate browser profile. You sign in yourself; Sentient never types passwords or card numbers. |
| Pair **Telegram or Discord** | Messages in paired chats pass through Telegram's or Discord's servers. |
| Link **WhatsApp** | Sentient becomes a linked device on your account, like WhatsApp Web. Messages in your "Message yourself" chat (and any chat you pair) pass through WhatsApp's servers, end-to-end encrypted as usual. The link's keys are kept in `~/.sentient/whatsapp`; Disconnect removes them and unlinks Sentient. This is not an official WhatsApp product, so WhatsApp could limit an account that uses it. |
| Pair a **phone or glasses** | Traffic stays on your home network, encrypted. This is off until you turn it on. |
| Create a **webhook** | Whoever has its secret link can start the tasks you attached to it. |

## Safety rails

- Sentient asks before sending, deleting, buying or running code, and purchases always ask, even if you allowed an
  action for the rest of a chat.
- Code that Sentient writes runs without your keys and can only read, not send.
- Work Sentient starts on its own (suggestions, follow-up checks, the nightly memory tidy-up) can only look things up
  and change Sentient's own data, such as its memory and notes. Anything that would send, delete, buy, run code or
  change something outside Sentient is not done, whatever your approval settings say. Sentient may offer it to you
  as a suggestion to approve instead.
- Once Sentient has read something other people wrote (an email, a web page, a message, the event that started a
  task), it asks you before sending, posting, inviting people to an event, typing into a web page, running code,
  or opening a web address on a new site that could carry your data (one with extra details after a `?`, or a very
  long one). Plain addresses and sites it already opened in that chat or task open without asking, so a short,
  clean address is the one way out it doesn't check. This holds even for apps you set to Allow and with approvals
  switched off, and no email or web page can talk it out of asking. In a chat it lasts until you start a new chat; a
  task asks with a notification and waits for your yes.
- Photos and screenshots from your devices ask first.
- One-time codes, sign-in links and password reset links in your email are hidden from the AI, so a tricky email
  can't get them out of Sentient. You still see them when you open the email in your mail app. You can turn this off
  in Settings (Integrations, "Hide one-time codes in email").
- Only chats you paired with a code can talk to Sentient on Telegram or Discord. On WhatsApp only your own
  "Message yourself" chat (and chats you pair with a code) can; Sentient never reads or answers your other chats.

Found a problem? Please report it privately: see [SECURITY.md](../SECURITY.md).
