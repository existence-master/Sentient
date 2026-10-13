# Changelog

All notable changes to Sentient are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/),
and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **See what your models cost, and keep everything on this computer:** Settings > Usage now shows what paid cloud
  models cost today, this week and this month, by model and by job. Models on your Claude or ChatGPT plan show as
  **On your plan** instead of dollars, and local models as free. Set a **monthly budget** there: you get a
  notification at 80%, and at 100% paid cloud models pause until you choose **Keep going this month** (or, if you
  prefer, until next month). The model menu in the title bar has a new **Local only / May use cloud** switch. In
  Local only, nothing is sent to a model outside this computer: not a backup model, not a chat with its own cloud
  model, not tasks, memory search or voice. Anything that would have used the cloud says so in plain words, or a
  local model answers instead.
- **Approve from your phone, and choose how far a yes goes:** when a task needs your OK and you're away from the
  app, the request goes to your paired Telegram, Discord or WhatsApp chat with buttons (or reply with once, task,
  hour, always or no). Only the chat it was sent to can answer it, and a request that waits too long (4 hours by
  default) stops the task instead of being approved late. Every approval now lets you pick **Allow once**, **Allow
  for this chat** or **Allow for this task**, **Allow for 1 hour**, or **Always allow** (which saves an Allow rule).
  Purchases, running code and anything after Sentient read an email or web page still ask every time. Settings >
  Approvals & safety lists what you allowed for a task or for a while, with **Revoke**. Tasks that reach a tool you set to
  Ask now pause and ask you instead of failing.
- **Routines pause while you're away:** when you haven't used Sentient for a week (the app or a chat), or 20 results
  have piled up since you were last around, Sentient asks once whether to keep your recurring tasks and briefs
  running. If you don't come back within a day, it pauses them and tells you in one notification. Any sign of you,
  in the app or a paired chat, keeps them running. **Resume routines** in that notification, in your chat app or on
  the Tasks page brings them all back. Turn on **Keep running while I'm away** on a task that should never pause.
  The day counts and the waiting time are in Settings > Tasks.
- **Sentient never overwrites a skill you edited:** when you change a skill, in Skills or in any text editor, Sentient
  notices and leaves your version alone. When it has a better idea it drafts the change for you to compare and
  approve, and says plainly that your version is untouched. A proposal drafted before your edit warns that approving
  would undo your changes. **Reset to original** puts back the version from before your edits.
- **Skill checks:** Skills now point out problems in plain words: a missing or unreadable header, no description or
  steps, a skill that is too long, links to files that aren't there, tools that aren't installed, a skill that reads
  like a diary of past runs instead of a lesson, and risky instructions (skip approvals, ignore other instructions,
  wipe folders, run a downloaded script, send passwords or keys). They show on each skill, on proposals, on skills
  coming from Hermes, and a skill file that can't load at all is listed with what is wrong. A skill Sentient writes
  with a risky instruction always waits for your review.
- **Bring your memory from ChatGPT, Claude or OpenClaw:** Memory > Import > From another assistant reads a ChatGPT or
  Claude data export (the zip they email you) or your OpenClaw workspace folder, right on your computer. Nothing is
  uploaded and the export is never changed. You first see what's in it (saved memories, conversations, Claude's
  memory and project instructions, OpenClaw's MEMORY.md, USER.md and daily notes) and pick what to bring over. Saved
  memories come over as they are; for conversations and notes your fast AI model picks out lasting facts about you
  in small batches. Everything waits in Memory > Review, and each card says which app, conversation and day it came
  from. Things Sentient already knows are skipped, also across apps, and importing a newer export later only reads
  what's new. You can stop an import, and remove everything from one app in one click.
- **Export your memory:** Memory > ... > Export saves everything Sentient remembers about you as a readable Markdown
  file or a SQLite database file in Sentient's files folder, grouped by topic and saying where each memory came
  from. Nothing leaves your computer.
- **Each chat remembers its model:** pick a model in a chat's message box and that chat keeps it, also after you
  reopen it or restart Sentient. Other chats keep following your default model, and switching the default (or a
  preset) in the title bar only changes those. **Use the default** puts a chat back on the default. The chat list
  shows a small label on chats with their own model. If a chat's model can't answer (Claude Code is turned off or not
  signed in, a local model was removed, a key is missing), the default model answers instead and the reply says so in
  one plain line. Helpers a chat starts use its model too. In Telegram, Discord and WhatsApp, `/model chat <model>`
  does the same for one conversation.
- **Backups and restore:** Settings > Backups backs up everything Sentient keeps (chats, memory, tasks, settings,
  skills and files) to one file, by hand or every day or week. Backups go to Sentient Backups in your Documents
  folder (the one Windows really uses, also when it lives in OneDrive), apart from your data, or to a folder you
  pick. Settings says when that folder syncs to OneDrive, Dropbox, Google Drive or iCloud, since the backups are
  then copied to the cloud too. A backup holds the certificate your devices trust, so keep
  backups somewhere private. Restore one from the list or from
  a file: you see what is in it and confirm, Sentient keeps a snapshot of what you have now, then restarts with the
  backup. Passwords, API keys and sign-ins stay in your computer's keychain and are never in a backup; after a restore
  Sentient lists the ones to sign in to again. Sentient also takes a snapshot on its own before it updates its
  database, before a Hermes import and before a restore, keeps the last five, and any of them can be restored.
- **Recurring tasks remember their last runs:** each run of a recurring task now reads a short note of what its last
  three runs found, sent or decided, plus the task's own notes, so it tells you only what is new instead of repeating
  yesterday's news. Runs keep those notes up to date themselves; see, edit or clear them under **Notes between runs**
  on the task.
- **Watches that only speak up when something important changed:** a check script can now **Tell me when something
  important changes**. Each check is compared with the last one without any AI, so an unchanged page costs nothing
  and stays silent; only when something changed does your fast AI model decide whether it matters, and then you get
  one notification that says what changed. Every watch also keeps a quiet list of its recent checks ("Checked,
  nothing new", "Something changed. You were told") under Runs.
- **Fuzzy times for repeating tasks:** a daily or weekly task can run "in the morning", "afternoon", "evening" or "at
  night" instead of at an exact time; Sentient picks a time inside that part of the day. Say it in your request
  ("every weekday morning") or pick it in the schedule editor.
- **Sources under answers:** when Sentient used the web, your email, a file or a connected app to answer, small
  chips under the reply show what it used: the page, the email's subject, the file or the app, next to the memories
  it had in mind. Click a chip to see the part Sentient read and open the page, email or file. Answers that used
  nothing show no chips.
- **Incognito chats:** start one from the button next to New chat, the command palette or Ctrl+Shift+N
  (Cmd+Shift+N on a Mac). Nothing you say in it is remembered: Sentient saves no memories from it, learns no skills
  from it, leaves it out of what it learns about you and of chat search, and deletes it when you leave it or close
  Sentient (or after 30 minutes without a message). The chat has a tinted header and an Incognito label so you always
  know. It can still use what Sentient already knows about you; turn that off in Settings > General > Conversations.
- **One local model job at a time, your chats first:** a local Ollama model now works on one thing at a time, so
  background work no longer pushes it off the graphics card and slows your chat to a crawl. Your chat always goes
  next; tasks, suggestions, memory upkeep and chat titles wait while you are chatting (and for 30 seconds after), and
  while your laptop runs on battery. A small "Model busy" sign in the title bar says what the model is working on and
  what waits. Change the wait, allow background work on battery or turn the queue off in Settings > Models. Cloud
  models never wait.
- **Read only or Read and write for each connection:** when you connect Gmail, Calendar, GitHub or any other app,
  and on every custom MCP server, pick **Read only** or **Read and write**. Read only lets Sentient look things up
  there but never change, send, delete or run anything: those abilities are hidden from the model and refused if
  it tries anyway, in chats, tasks, helpers and scripts alike, even with an Allow rule or approvals turned off. Switch
  any time from the app's panel in Integrations (or the server's row); it applies at once, no restart. For a server
  that runs other apps' actions by name (Composio's multi-execute), Read only still allows the look-ups and refuses
  the rest, and those look-ups no longer ask for approval. Fewer tools also helps small local models pick the right one.
- **Claude through your own Claude Code (experimental):** if Claude Code is installed and signed in on your computer,
  your chats can use Claude through it (Settings > Models, off until you turn it on). Sentient never reads or keeps
  your Claude login, and leaves API keys and other sign-in settings in your environment out, so Claude Code always
  uses the plan you signed in to. Claude Code's own tools stay off, so anything it wants done goes through Sentient's tools,
  approvals and rules, and Stop everything stops it. It only answers your chats: tasks, suggestions and other
  background work keep using your other models, and it can't be the memory search model. Anthropic may change how
  this is counted or allowed.
- **Browser downloads:** files Sentient downloads while using a website are saved to `downloads/` in your Files
  folder, and the browser result says where. In an attached browser, only downloads from Sentient's own tabs and
  their popups (including `target="_blank"`) are saved, while downloads from other user tabs remain in the browser's
  native Downloads folder. Sentient's own downloads briefly pass through the browser's download folder before being moved to the downloads directory. When attached, if a download cannot be matched to a confirmed browser download record,
  it returns a clear error rather than falsely reporting success.
- **Push to talk and dictation into any app:** hold Ctrl+Alt+Shift+T (Cmd+Option+Shift+T on a Mac) anywhere, speak,
  and let go: what you said goes to Sentient as a chat message, and the answer is read aloud. Press Ctrl+Alt+Shift+D,
  speak, and press it again (or just pause): your words are typed where your cursor is, in any app. A small bar at the
  bottom of the screen shows the microphone is on, with a level meter; Esc cancels. Speech is turned into text on this
  computer. Dictation drops "um" and "uh" and fixes punctuation, or keeps exactly what you said, or lets your fast model
  polish it without ever changing your words (if it does, the tidy version is used). Sentient never types into a
  password box it can recognize, and puts back what you had copied. Change or turn off the shortcuts in Settings >
  General, and pick the cleanup and language in Settings > Voice. Stop everything turns the microphone off.
- **Share your screen with a shortcut:** press Ctrl+Alt+Shift+W (Command+Option+Shift+W on a Mac) to share the
  window you are using, or Ctrl+Alt+Shift+R (Command+Option+Shift+R) to drag over part of the screen. Sentient opens a
  new chat with the picture attached, named after the window, so you can type or say your question and send it. A
  short outline flashes where the picture was taken, nothing is sent until you press send, and the picture goes to
  your vision model when you set one. Both shortcuts are also in the tray menu and can be changed or turned off in
  Settings > General. Since a screen can show other people's text, a chat with a shared picture checks with you
  before sending anything. On a Mac, Sentient explains how to allow Screen Recording the first time.
- **Sign in with ChatGPT** (Settings > Models > Use a plan you already have > ChatGPT plan, or **ChatGPT plan** in
  first-run setup's cloud step): use your ChatGPT Plus or Pro plan for chat, tasks and background work. Click
  **Continue with ChatGPT** (with OpenAI's ChatGPT logo), sign in in your browser and allow Sentient; your plan's
  models then show up in every model picker, and **Use ChatGPT for every job** switches the main and background
  models in one click (the Cloud preset uses your plan when you have no other cloud key). Main and background jobs
  use the everyday model OpenAI ranks first on your plan, background jobs with the lightest thinking it allows, so
  nothing is guessed from model names. After your first sign-in Sentient says once that you're using your ChatGPT
  plan, and while your next message goes to a plan model the chat box shows **Using ChatGPT plan** with a **Manage
  usage** link. The sign-in stays in your system keychain, renews on its own, and **Sign out** removes it. **Manage usage** opens your ChatGPT usage settings, and reaching
  your plan's limit says so plainly. Memory search keeps its own model, because ChatGPT plans don't include one for it.
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
- Editing, approving or importing a skill that needs a particular tool (for example `file_read`) no longer hides it
  from Sentient until the next restart.
- **Watch requests no longer fail when a check script can't be written:** if the AI's check script is rejected twice
  (it loops, sleeps or uses memory, as small local models sometimes do), the task is saved as an ordinary recurring
  task that the AI runs each time, with a short note on the task saying so, instead of ending in an error.
- **Rules from chat work with Claude Code for every model:** when your fast model is Claude Code, "From now on,
  never delete my emails" proposes a rule again. The check now uses the chat's own model or your main model, since
  it is part of answering your message. Apps that chat through `POST /api/chat` also get the "Make this a rule?" card.
- With approvals set to **Never ask**, Sentient no longer stops to ask before just looking something up (a web page,
  a snapshot of the browser, a search) because something you said in the chat sounded like a "never" it offered to
  turn into a rule. Until you answer that suggestion it still checks with you before changing, sending, deleting or
  running anything it covers. It also reads your words more carefully: naming one tool ("don't call browser_open")
  suggests a rule for that tool only, never the whole app, and a limit on the request at hand ("don't cc her this
  time", "in part A") is no longer taken as a lasting rule. Settings now lists what Never ask still checks: purchases,
  apps you set to Ask, sending anything once a chat has read an email or web page, and a change you just told it never
  to make until you answer its suggested rule.
- "Every 10 minutes", "every hour", "every 2 hours" and "every 30 min" now really repeat that often. Before, a local
  model could save these as once a day at 09:00. Repeats are at least 5 minutes apart.
- Check scripts written by the AI that can't work are sent back with the reason and rewritten once: scripts that use
  packages that are not installed, loop forever, wait or sleep, use memory, or raise an alert where Sentient should
  compare results. Scripts you edit yourself are not limited this way.
- A second Sentient data folder (a test or demo profile set with `SENTIENT_HOME`) no longer shares the keychain with
  your own: it gets its own place for API keys, tokens and sign-ins, so it can't use, change or remove yours, and a
  seeded demo can't message anyone through your Telegram bot. Your own data folder (`~/.sentient`) keeps every key
  and sign-in as it is, with nothing to redo. `SENTIENT_KEYCHAIN_NAMESPACE=default` shares yours on purpose. The demo
  seed scripts and `scripts/smoke.mjs` now pause the seeded tasks and turn off messaging deliveries unless given
  `--keep-live`.
- Chats using Claude through your own Claude Code now show the context meter and the "getting long" warning at
  85%: the context length is read from the matching Anthropic model, which LiteLLM knows, instead of nothing.
- **Background work on a laptop that stays on battery:** Sentient's own upkeep (memory, what it learns about you,
  dreams, suggestions, skill review, chat titles) used to wait for the charger, so a laptop used all day on battery
  never ran it. It now runs on battery too, gently: while the battery is above 40%, after you have not chatted for
  five minutes, in short bursts at most every 30 minutes, and never held more than two hours. At 15% or below it
  waits. Tasks you scheduled always run on time, whatever the battery, and so does Dream now unless the battery is
  low. The title bar says "Waiting: on battery" (or
  "Waiting: battery low") while upkeep waits, and every limit is in Settings > Models > Sharing your local model.
- Messages and memories saved at the same moment now always keep their order, and the daily upkeep of MEMORY.md and
  USER.md no longer misses a memory saved while it was reading. Summarizing a long chat no longer skips messages
  saved at the same moment as the last summary.
- When a cloud model says no, chat and tasks now say why in a plain sentence and what to do, instead of showing the
  provider's raw reply. For example: "OpenRouter's free models have reached today's limit. Add credits on
  openrouter.ai, wait until tomorrow, or pick another model in Settings > Models." This covers rate limits, an
  account out of credits, a key that was mistyped or revoked, a free model OpenRouter keeps for coding tools, a model
  that doesn't exist or isn't downloaded, and a provider that is down or too slow. The raw reply stays in the log.
- One long tool result no longer fills a local model's whole context. A result may now take about a quarter of what
  the model reads at once (Settings > General > Conversations), so on an 8,192-token model a 13,000-character Composio
  search no longer triggers "This chat is getting long" after a single step. The model is told plainly when a result
  was cut, and the full result is still saved. Composio's tool search keeps its useful part instead of its first
  characters: the recommended plan, the tool names, which apps are connected and the main tool's settings. A connected
  tool server that sends its answer twice (as text and as data) is now read once.
- Turning on "Use my Claude Code for chats" now shows the Claude Code version straight away, instead of sometimes
  showing nothing until you came back to the page. Settings also says that its replies take a few seconds to start
  and arrive in larger pieces than with your other models.
- Claude Code no longer leaves empty scratch folders behind on Windows. Each reply's folder is removed once Claude
  Code has fully exited, and folders left by earlier runs are cleared when Sentient starts.
- A task that can't be planned because Claude Code is your main model now says so and what to do ("Pick a
  planner model in Settings > Models"), instead of saying the AI model is unavailable.
- A memory waiting for your review now shows the page or message it came from, not the error from a later action
  you declined or that failed.
- Adding a server (MCP) that you already signed in to from another Sentient setup on the same computer, such as a
  fresh install or a test setup, no longer signs you out of it. The sign-in now remembers the server's address, so it
  is kept and reused for the same address and only cleared when the address changes.
- Local models can now use large MCP servers such as Composio, Notion or GitHub. Naming the server in your message
  ("using Composio...") always offers its tools, and a server with more tools than fit offers its most relevant ones,
  keeping its "search" and "run" tools together, instead of none at all.
- A message with a link now offers local models the page reader (`web_fetch`) next to the web browser, and a tool
  you name exactly ("use web_fetch") is always offered (#252).
- Signed-in MCP servers (Composio, for example) no longer ask you to sign in again about an hour after Sentient
  restarts. The sign-in now remembers where the server renews it, so renewing works after a restart; sign-ins saved
  before this fix look that up once on their own.
- Local Ollama models now read 8192 tokens by default instead of Ollama's 4096, so long tasks, big inboxes
  and many tools are no longer cut off without warning. Change it with `models.context_length`, per role with
  `models.context_length_per_role`; it never goes above the model's own maximum, and `sentient doctor` shows it.
- A script that Sentient runs now always gets a clear answer when one of its tool requests is refused, instead of a
  dropped connection when the computer is busy.
- What Sentient learns about you reads your messages in the order you sent them, even when two were saved at the
  same instant.
- The model check-up no longer warns about the graphics card on an 8 GB card running the setup Sentient recommends
  for it (qwen3:8b reading 8,192 tokens at a time). About 90% of the model fits there, which is expected; the only way
  to fit all of it is a context below 8,192 tokens, which cuts off long tasks. A bigger spill, a longer context or a
  different model still gets the warning.
- Saving something new about you no longer overwrites a different memory: "doesn't want files written" is kept next
  to "doesn't want files deleted" instead of replacing it, and so is "doesn't want emails deleted". A memory is only
  changed when the new one is about the same person and the same thing. A vaguer memory that an existing one already
  covers ("your sister lives in a city", "you have a sister") is no longer added, and neither are gaps like "your
  sister's job is unknown".
- Everyday phrases like "If you don't know, say so" no longer make Sentient suggest a Never rule for unrelated
  tools. A suggested rule for a single tool now needs your words to name an action, the app or the tool, so
  "Never write files for me" still suggests blocking file writing.
- Edge's downloads hub no longer replaces the active page or appears as a browser tab when a download starts.

### Security
- Text from web pages, emails, messages and other apps now reaches the AI clearly marked as outside content, with a
  standing rule to treat it as information and never follow instructions hidden in it. A page can't fake the end of
  the marked part. Your own messages are not marked. To compare, turn it off in Settings > General > Conversations
  ("Mark outside content").
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
