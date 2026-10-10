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
| Turn on **Claude through your Claude Code** (experimental) | Sentient starts the Claude Code program already on your computer for each chat reply, and Claude Code sends the conversation, relevant memories and tool results to Anthropic under your own Claude Code login and Anthropic's terms. Sentient never reads, copies or keeps your Claude login or anything in your `.claude` folder, leaves API keys and other sign-in settings out so Claude Code always uses the plan you signed in to, and Claude Code's own tools stay off: anything it wants done goes through Sentient's tools and approvals. Only your chats use it, never background work. Off until you turn it on. |
| Use your **Claude plan's API credits** | The same as any Claude model: requests go to Anthropic with your API key. Sentient never asks for your Claude account login. |
| **Connect OpenRouter** | You sign in on openrouter.ai in your browser and OpenRouter hands Sentient a key for your account, kept in your keychain. Requests then go to OpenRouter, which passes them to the company that runs the model you picked. The model list is fetched from openrouter.ai once connected. |
| **Sign in with ChatGPT** | You sign in on chatgpt.com in your browser and allow Sentient to use your ChatGPT Plus or Pro plan. The first time, OpenAI registers Sentient for your account with a random id for this computer (not your name or email). The sign-in is kept in your keychain and renews on its own. Requests then go to OpenAI (`api.openai.com`) and are not stored there (`store: false`). Usage counts against your plan; see or limit it at chatgpt.com/settings/usage. Sign out removes the sign-in and asks OpenAI to cancel it. |
| Use **Nous Portal** | Requests go to Nous Research with your Nous Portal key. |
| **Connect an app** (Gmail, Slack, Notion...) | Sentient talks directly to that app's servers with your own login. |
| Use **web search, weather, maps or news** | Your query goes to that service. Web search uses DuckDuckGo by default. |
| Let Sentient **use the browser** | It visits the sites the task needs, in its own browser profiles (each in `~/.sentient/browser/profiles/<name>`, with its own sign-ins). You sign in yourself; Sentient never types passwords or card numbers. Deleting a profile deletes its folder. |
| Let Sentient **attach to a browser you started** | Sentient connects to that browser's DevTools port, only on this computer (`127.0.0.1` or `localhost`; other addresses are refused). It works in a tab of its own, but can see the titles and addresses of your open tabs. While the port is open, any program on your computer can control that browser, so start it that way only when you need it and with a separate profile. Disconnecting never closes your browser. |
| Pair **Telegram or Discord** | Messages in paired chats pass through Telegram's or Discord's servers. |
| Link **WhatsApp** | Sentient becomes a linked device on your account, like WhatsApp Web. Messages in your "Message yourself" chat (and any chat you pair) pass through WhatsApp's servers, end-to-end encrypted as usual. The link's keys are kept in `~/.sentient/whatsapp`; Disconnect removes them and unlinks Sentient. This is not an official WhatsApp product, so WhatsApp could limit an account that uses it. |
| Pair a **phone or glasses** | Traffic stays on your home network, encrypted. This is off until you turn it on. |
| Create a **webhook** | Whoever has its secret link can start the tasks you attached to it. |
| Turn on the **terminal** | Commands run on your computer as you, so a command you approve can reach the internet (for example `git push`) or change files outside the folders you allowed. Off until you turn it on. |

## Safety rails

- Sentient asks before sending, deleting, buying or running code, and purchases always ask, even if you allowed an
  action for the rest of a chat.
- Code that Sentient writes runs without your keys and can only read, not send.
- Commands on your computer (Settings > Terminal) are off until you turn them on and add a folder. Each command
  shows you the exact command and folder and waits for your yes, unless you listed it as never needing a question or
  set an Allow rule. Your API keys, tokens and passwords are kept out of the command's environment. Some commands are
  never run, whatever you set: formatting disks, shutting down, deleting from the registry or deleting a whole drive
  or home folder. Work Sentient starts on its own, and code it writes, can never run commands.
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
- Memories that come from something other people wrote (an email, a web page, a message, a tool's result), from work
  Sentient did on its own, or from an import (a document, Hermes) wait for you in Memory > Review. Sentient doesn't
  use them anywhere until you approve them, and no email or web page can approve them for you. Ones you don't review
  are deleted after 30 days (Settings > Memory). What you tell Sentient yourself in a chat that hasn't read outside
  content is remembered as before. Summaries of such chats stay out of your other chats and out of the notes
  Sentient keeps about you (MEMORY.md), though you can still read them under Memory > Conversations.
- Photos and screenshots from your devices ask first.
- One-time codes, sign-in links and password reset links in your email are hidden from the AI, so a tricky email
  can't get them out of Sentient. You still see them when you open the email in your mail app. You can turn this off
  in Settings (Integrations, "Hide one-time codes in email").
- Only chats you paired with a code can talk to Sentient on Telegram or Discord. On WhatsApp only your own
  "Message yourself" chat (and chats you pair with a code) can; Sentient never reads or answers your other chats.

Found a problem? Please report it privately: see [SECURITY.md](../SECURITY.md).
