# Changelog

All notable changes to Sentient are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/),
and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Claude through your own Claude Code (experimental):** if Claude Code is installed and signed in on your computer,
  your chats can use Claude through it (Settings > Models, off until you turn it on). Sentient never reads or keeps
  your Claude login. Claude Code's own tools stay off, so anything it wants done goes through Sentient's tools,
  approvals and rules, and Stop everything stops it. It only answers your chats: tasks, suggestions and other
  background work keep using your other models, and it can't be the memory search model. Anthropic may change how
  this is counted or allowed.
- **Models sized for your computer:** Sentient checks how much memory and graphics memory this computer has before
  anything is installed and recommends a local model and how much text it reads at once (for example qwen3:8b reading
  8,192 tokens on an 8 GB graphics card, qwen3:14b reading 16,384 on a 16 GB one). A computer with less than 8 GB of
  memory is pointed to a cloud model, with a small local model offered only for chat. Onboarding shows it as "Recommended
  for this computer" with a one-click download, the "Local only" model setup uses it, and the model check-up names it
  in its fixes, including a shorter context length when a longer one would push the model onto the processor.
- **Context meter:** a small "62% of context" meter next to the message box and on running tasks shows how much of
  what the model reads at once a chat or task is using. From 85% Sentient says so in plain words and suggests a new
  chat or a longer context length. Nothing is blocked.
- **Memory sources on tasks:** a task's result now shows "Used 3 memories" too: what Sentient had in mind while it
  worked, from its instructions and from memory look-ups it read. The list stays with the run when it stops to ask
  you something, after a restart and on Retry, and each one has **This is wrong** like in chat. A chat reply you
  stop now shows its memories right away instead of after a reload.
- **"Never do this" becomes a rule.** When you tell Sentient something like "never delete my emails" or "don't post
  to Slack without asking me", it offers to turn it into a lasting rule: a small card in the chat ("Make this a rule?
  Never: Gmail > Trash") with "Make it a rule" and "Not now". A rule keeps working in every chat, even after a long
  conversation has been summarized, and Settings > Approvals & safety shows it came from your message. Nothing becomes
  a rule without your click, and until you choose, Sentient checks with you before using that tool in the chat.
  Said in Telegram, Discord or WhatsApp, the question comes back there too (on WhatsApp, reply 1 or 2).
- **Memory review:** memories that come from an email, a web page, a message, a tool's result, work Sentient did on its
  own, or an imported document wait for you on the new Review tab of the Memory page, with a count in the sidebar.
  Each one shows where it came from and the text it was taken from; approve it, change the wording and approve, turn
  it down, or approve everything from one place at once. Until you approve one, Sentient doesn't use it in chats,
  tasks, suggestions or what it knows about you, and no email or web page can approve it for you. Ones you don't
  review are let go after 30 days, with a note. Summaries of chats that read outside content stay out of your
  other chats and out of MEMORY.md. What you say yourself in a chat that hasn't read outside content is remembered as
  before.
- **Choose where a task's results go** (Details > Send results to): the usual paired chats, this computer only, or
  just the chats you pick, such as WhatsApp's "Message yourself". Results, failures, questions and plans from that
  task follow the choice, and so does the Daily Brief. Jobs brought over from Hermes keep their delivery: a job that
  went to WhatsApp goes to your "Message yourself" chat (once WhatsApp is linked), Telegram and Discord jobs go to the
  matching paired chat, and `local` jobs stay on this computer.
- **Add values to an MCP server** (Integrations > Custom MCP servers): a server that lists header or environment
  names without values, such as one imported from Hermes, shows **Add values**. Fill them in (they go to your system
  keychain, never to a settings file) and the server reconnects, so there's no need to add it again.
- **Report every run** for script tasks: besides "Tell me when something changes" and "when the script raises an
  alert", a script task can now report its output after every run. Hermes script jobs come over this way, and
  Hermes monitor jobs keep reporting only changes.
- **Coming from Hermes?** Bring your Hermes setup over in one step, from the last onboarding screen or Settings >
  Advanced. Sentient looks inside your Hermes folder (`~/.hermes`, or one you pick) and shows what it found before
  anything changes: your own skills go to Skills for review (Hermes' built-in ones you never changed are left out),
  MEMORY.md and USER.md become memories and things Sentient knows about you once you approve them in Memory > Review
  (with a button to remove them again),
  SOUL.md replaces the personality only if you say so after seeing both side by side, scheduled jobs become paused
  tasks that keep their schedule (resume one and Sentient plans it for you to approve), and MCP servers are added
  turned off. Keys, tokens, sign-ins, `auth.json` and `.env` are never read or copied, so remote servers need a fresh
  sign-in. Jobs Sentient can't run yet (monthly schedules, shell scripts, missing scripts) are listed with the reason.
- **Model presets:** switch every model between local and cloud in one click. Click the model name at the top of the
  window to pick "Local only" (everything on this computer), "Cloud" (your Anthropic, OpenAI or OpenRouter key for
  every job) or "Mixed" (cloud for chat and planning, this computer for background work and memory), go back to a
  recent chat model, or undo the last switch. Save your own setup as a preset under Settings > Models. If a preset
  needs something you don't have yet, the menu offers it right there: download the model with progress, or add the key.
  In Telegram, Discord and WhatsApp, `/model` shows the current setup and switches with a tap or a number. Per-chat
  and per-task model choices still win.
- **Use the AI plans you already pay for** (Settings > Models and the first-run setup): step-by-step help to use the
  monthly API credits included with Claude Max and Team plans (paste the key, test it, then switch every job to Claude
  in one click with the Cloud preset, which you can undo), **Connect OpenRouter** with a browser sign-in instead of copying a key, and
  **Nous Portal** as a provider with an API key. OpenRouter's and Nous Portal's model lists show up in the model
  pickers, with free OpenRouter models marked. Keys stay in your system keychain and removing one disconnects. Pro
  plans don't include API credits, and apps aren't allowed to sign in with a Claude account, so the app says so.
- **Browser profiles:** keep several browser profiles, each with its own sign-ins (for example one for posting on a
  social account and one for shopping), in Settings > Browser: add, rename, delete, and "Open to sign in". A task can
  use a profile (Details > Browser profile), a skill can name one (`browser_profile:` in its frontmatter), and the
  assistant can pick one when it opens a page. Sentient can also **attach to a browser you started yourself** (Brave,
  Chrome or Edge with `--remote-debugging-port`), only on this computer; disconnecting never closes it. Approvals,
  purchase checks, password and code refusals and the allowed and blocked sites work the same in every profile.
  Your existing browser sign-ins move to the "default" profile.
- **WhatsApp:** link Sentient to your WhatsApp by scanning a QR code (Channels > Messaging apps), then talk to it in
  your own "Message yourself" chat: text, voice notes, photos and files, replies as they are written, task results and
  suggestions. When Sentient needs a yes or no it lists numbered options; reply with the number. `/stop`, `/stopall`,
  `/resume`, `/new` and `/help` work there too. Your other chats are never read or answered. This uses the same linking
  as WhatsApp Web through an unofficial app, so WhatsApp could limit an account that uses it; the connect screen says so.
  Sentient's messages in that chat start with its name, so you can tell them from yours. In the installer, spoken
  replies arrive as an audio file rather than a voice note.
- Tasks can pause to ask you a question ("Which of these two flights?") and carry on with your answer. The question
  shows on the task, as a notification with quick answers, and in paired Telegram or Discord chats with buttons;
  replying to the question message there works too. A waiting task keeps waiting across restarts.
- Proper documentation: getting started, privacy, architecture decision records, agent guides, verification report.
- README screenshots and a new banner.
- Lasting rules for apps and tools in Settings > Approvals & safety: **Allow** (go ahead without asking),
  **Ask** (always ask first) or **Never** (Sentient can't use it). Purchases always ask.
- **Follow-ups:** once a day Sentient notices emails waiting on your reply, and your own questions nobody answered,
  in Gmail and IMAP email, and suggests a ready draft. Newsletters, no-reply senders and answered threads are skipped,
  and nothing is sent until you approve it. Settings under Proactivity.
- Sentient stops when it keeps repeating itself: the same tool with the same details giving the same result three
  times. In chat it is first asked to try something else, then it tells you it stopped; a task fails with a clear
  reason and the usual "Task failed" notification. Settings > Approvals & safety.
- When a task reaches one of its limits (40 steps, 30 minutes of work, and on cloud models 2,000,000 tokens or about
  $5 by default) it pauses and asks: "This task has used 40 steps and isn't finished yet. Keep going for another 40,
  or stop here?" Keep going raises that limit for this run; Stop here ends it with a plain message. Time spent waiting
  for your answer doesn't count. Helpers have their own token and spending limits too (1,000,000 tokens and about $2).
- **Memory sources:** under a chat reply, "Used 3 memories" shows what Sentient had in mind when it answered: the
  memories and things it has learned about you that were in front of it, and any it looked up. Each one has
  **This is wrong** to fix or forget it on the spot, and a link to it on the Memory or About you page.
- **Stop everything:** one control that stops every running reply, task, helper, script and browser action at once
  and pauses scheduled tasks, triggers and suggestions until you press Resume, even after a restart. It is in the
  title bar, the tray menu, the shortcut Ctrl+Alt+Shift+S, `/stopall` in paired Telegram and Discord chats, and on
  paired phones. Messages you queued before the stop are not sent. It never asks the model, so nothing can
  talk its way past it.
- **Terminal:** Sentient can run commands on your computer, like git, builds, tests and scripts, in folders you
  allow (PowerShell on Windows, your shell on macOS and Linux). It is off until you turn it on in Settings >
  Terminal and add a folder. Each command shows you the exact command and folder and waits for your OK every time,
  unless it is on your list of commands that never need asking (`git status` and friends) or you set an Allow rule.
  Output shows live in the chat, with a Stop button; commands stop after 3 minutes by default, and Stop everything
  ends them too.
  Formatting disks, shutting down, deleting from the registry and deleting whole drives are never allowed. Your keys
  and passwords are kept out of its commands, and work Sentient starts on its own can't run any.
- **Daily Brief:** a few lines each weekday morning at 07:30 with today's meetings, emails that need you, tasks due
  or waiting for you and the weather, plus headlines on topics you pick. Turn it on in onboarding or with **Set up my
  Daily Brief** in Notifications. It is a normal task you can reschedule, pause or delete, it only reads (it never
  sends or changes anything), and it shows at most 7 lines, each with a link and a "why am I seeing this". Thumbs up
  or down on a line or a section shapes the next briefs (change your mind and the latest rating counts), any section
  can be turned off, and the brief expires at the end of the day. It also arrives in paired Telegram and Discord chats,
  and "read my brief" works in chat and by voice.
- **Evening Brief:** an optional wrap-up at 21:00 every day, set up from the same card: tasks finished or failed today,
  replies and emails sent, files made, what is still waiting for you and tomorrow's first events. Its own task, the
  same 7-line cap, links, read-only rule and end-of-day expiry.
- **Remote MCP servers that need sign-in:** hosted servers such as Notion's or Composio's can now be used. Add a
  remote server and click **Sign in** to approve Sentient in your browser; Sentient refreshes the sign-in on its own and
  **Sign out** forgets it. Servers that take an access token instead get headers (like `Authorization: Bearer ...`)
  whose values are kept in your system keychain. A server that turns Sentient away now shows "Needs sign-in" instead
  of a generic error.
- **Check my models:** a check-up in Settings > Models and in onboarding tests each role's model the way Sentient
  uses it: a short reply, a tool call (and a second tool step for chat and tasks), a JSON reply for background jobs,
  thinking, context length, and whether Ollama runs the model on the graphics card or partly on the processor. Each
  role gets a pass, warning or failure with a plain fix ("qwen3:4b can't call tools reliably; try qwen3:8b") and,
  where the fix is obvious, a button that does it. Nothing changes unless you press it. `sentient doctor --models`
  runs the same check in a terminal.
- **Stuck tasks tell you.** A task that makes no progress for 10 minutes, keeps hitting the same error, or reaches a
  step only you can do (a password, a CAPTCHA) pauses and says why: "Sentient is stuck on 'Book the table': the page
  asks for your password. Open it to help or cancel." Choose Try again, Skip this step or Cancel, or tell it what to
  do. A running task shows when it last did something. Settings > Tasks.
- **Catching up after sleep.** Scheduled tasks missed while the computer was off or asleep run once when Sentient is
  back if they are less than 12 hours late, and are skipped otherwise; never a pile of old runs. One notification
  says what happened ("Caught up after sleep: ran 1, skipped 2"). Each task can choose to always run once or always
  skip. Nothing catches up while Stop everything is on; it happens when you resume.

### Changed
- A task that repeats the same failing step three times now pauses as stuck and asks what to do, instead of failing.

### Fixed
- Local Ollama models now read 8192 tokens by default instead of Ollama's 4096, so long tasks, big inboxes
  and many tools are no longer cut off without warning. Change it with `models.context_length`, per role with
  `models.context_length_per_role`; it never goes above the model's own maximum, and `sentient doctor` shows it.
- A script that Sentient runs now always gets a clear answer when one of its tool requests is refused, instead of a
  dropped connection when the computer is busy.
- What Sentient learns about you reads your messages in the order you sent them, even when two were saved at the
  same instant.

### Security
- Once Sentient has read an email, a web page, a message or anything else other people wrote, it asks before
  sending, posting, inviting people to an event, typing into a web page, running code or opening an address that
  could carry your data to a new site, even for apps set to Allow and with approvals off. The
  approval says why ("Sentient read content from Gmail in this chat, ..."). A chat stays this way until you start a
  new one. A task that read outside content pauses and asks "OK to ...?" instead, then carries on after your yes or
  stops after a no.
- Work Sentient does on its own (proactive checks, the heartbeat, follow-up scans, dreaming and any subagent they
  start) can only look things up and change Sentient's own things. Sending, deleting, buying, running code, changing
  anything outside Sentient or creating tasks is refused in code, even with an Allow rule. Sentient may offer the
  refused action to you as a suggestion to approve. Chats, your tasks and suggestions you approve work as before.
- One-time codes, verification codes, magic sign-in links and password reset links in Gmail and IMAP email are
  replaced with a short placeholder before the AI reads them, in tool results, proactive suggestions and follow-ups.
  Booking, order, ticket and reference codes, dates, prices and ordinary links are left alone, and the email itself
  is unchanged in your mail app.
- Purchases now ask even when approvals are switched off, and code Sentient writes can never make one. A tool set
  to Never while a request waits for your yes, or while a script runs, is blocked right away.

## [3.0.0-alpha.0] - 2026-10-08

Sentient v3: a rewrite as a desktop app that runs on your own computer. No account, no servers, no `.env`.

### Added
- **Desktop app** (Electron + React) that starts a local engine, with onboarding, chat, tasks, memory, integrations,
  skills, notifications, voice mode, devices and settings.
- **Any model for any job:** separate models for chat, background work, planning, running tasks, voice, vision and
  embeddings, local (Ollama, LM Studio) or cloud, with fallbacks and a one-click test.
- **Long-running tasks** carried over from v2 (one-off, recurring, triggered, swarm), plus **watcher tasks** that run
  small scripts without a model, **retry** of failed runs, and changing tasks from chat.
- **Memory** with topics and expiry, an **About you** page (an evolving picture you can correct) and nightly
  **consolidation** that merges duplicates and settles contradictions.
- **Proactivity** that reacts to change feeds (Gmail, Calendar), IMAP push and **webhooks**, plus a quieter heartbeat.
- **Self-improving skills** that Sentient proposes and repairs, always waiting for your review.
- **Devices:** pair phones and smart glasses with a code over an encrypted home-network connection; the desktop is a
  device too. Protocol in `docs/NODES.md`, reference ESP32-S3 firmware in `firmware/`.
- **Messaging:** Telegram and Discord bots with pairing codes, streaming replies and approvals as buttons.
- **Browser control** on your installed Edge or Chrome, and **code execution** in a sandbox that can call tools.
- **Helper agents** that work beside the chat, and **steering** a reply while it is still being written.
- **Wake word** ("Hey Sentient") and talk mode.
- **Windows installer** with the engine bundled; release workflow for Windows, macOS and Linux.

### Changed
- One SQLite file replaces MongoDB, Postgres, Chroma and Redis. Keys live in the OS keychain.
- Contribution model: GitHub flow on `main`, CI without secrets, issue forms and a starter backlog.

### Removed
- The v2 cloud web app and its deployment. It is preserved on the `v2` branch and the `v2-final` tag.

## v2

The cloud web app. Its history is on the [`v2` branch](https://github.com/existence-master/Sentient/tree/v2).

[Unreleased]: https://github.com/existence-master/Sentient/compare/v2-final...main
[3.0.0-alpha.0]: https://github.com/existence-master/Sentient/tree/main
