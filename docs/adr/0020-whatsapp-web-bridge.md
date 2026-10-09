# 0020. Link WhatsApp as a device with an in-process WhatsApp Web bridge

- **Status:** Accepted
- **Date:** 2026-10-09

## Context

Many people live in WhatsApp, not Telegram or Discord (issue #106). The official WhatsApp Cloud API needs a Meta
business account, a business phone number and a public webhook, which breaks "no servers" and "no accounts" for a
desktop app. The other way in is WhatsApp's own multi-device protocol, the one WhatsApp Web and WhatsApp Desktop use:
the user scans a QR code once and Sentient becomes one of their **Linked devices**. Hermes Agent works this way and
talks to the user in their own "Message yourself" chat.

That protocol is unofficial. WhatsApp's terms do not allow unofficial clients, and accounts that use them can be
limited or banned. For one person talking to their own chat the risk is low, but it is real and the user must hear it
in plain words before linking.

We had to pick a bridge that a non-technical user never installs separately, that works on Windows, macOS and Linux,
and that packages into the frozen engine (ADR 0002). The candidates:

| | neonize (Python, wraps whatsmeow) | Baileys (Node) |
|---|---|---|
| License | Apache-2.0, whatsmeow is MPL-2.0 | MIT |
| Maintenance | Active (0.5.2, September 2026); small team | Very active, large community |
| Runs | Inside the engine: whatsmeow is a Go shared library shipped in the wheel (Windows, macOS, Linux, x64 and arm64) | A second process, run with Electron's own Node (`ELECTRON_RUN_AS_NODE`), talking to the engine over localhost |
| Packaging | PyInstaller collects the library; nothing for the desktop | Ship `node_modules` in the app; the engine needs Electron's path; `python -m sentient serve` alone needs Node |
| Tests | Python fake bridge | Python fake bridge plus a Node side to test |

## Decision

- WhatsApp is a channel (`sentient/channels/whatsapp.py`) on the same base as Telegram and Discord: pairing, commands,
  streaming by editing messages, approvals, task questions, voice notes, files and delivery all come from
  `Channel`. Its default and only mode in this version is **self-chat**: after linking, the "Message yourself" chat is
  paired automatically. Another chat can be paired with a code, as on Telegram. Every other chat, group, status update
  and channel is ignored and never answered, and so are the user's own messages to other people.
- The protocol sits behind a small `WhatsAppBridge` interface. The bridge we ship is **neonize** in
  `sentient/channels/whatsapp_web.py`, imported lazily and installed with the optional `whatsapp` extra, which the
  installer build and CI include. Tests drive the channel through a fake bridge and never touch WhatsApp.
- The linked session (keys, not messages) is whatsmeow's own SQLite file in `~/.sentient/whatsapp`, never in config,
  the main database or the keychain. Disconnect logs out (Sentient disappears from Linked devices) and deletes it.
- WhatsApp has no buttons for personal accounts, so options are numbered ("Reply with a number: 1 Allow, 2 Allow for
  this chat, 3 Deny"). Replying to the message with a number picks that option; a bare number only answers an approval
  that is holding up the reply, so a stray "1" can never approve a plan or a suggestion from hours ago.
- The connect screen says plainly that this is unofficial and that WhatsApp could limit or ban the account.
- The official Cloud API, for people with a business number, is a possible later option, not part of this decision.

## Consequences

- No extra install and no second process: linking works the same in the installer and from source with the extra.
- The engine now loads a Go runtime when WhatsApp is connected. Every Python exception from the library is caught at
  the bridge boundary (the channel shows "needs attention" and keeps reconnecting; event handlers only log), so it
  cannot stop the engine. A hard crash of the Go runtime itself cannot be caught from Python; whatsmeow is widely
  used, so we accept that, and the bridge interface lets us move it to a separate process later without touching the
  channel.
- We depend on a small project tracking a moving, unofficial protocol. When WhatsApp changes, a neonize update is
  needed; old versions stop linking. Keep the dependency current.
- Temporary status lines ("Searching the web...") are off for WhatsApp, because a deleted message leaves "This message
  was deleted" behind. Edits are throttled to every 2 seconds by default.
- Voice replies are OGG/Opus voice notes when PyAV is present (source installs with the voice extra); the installer
  leaves PyAV out, so there a spoken reply is sent as an audio file instead, the same as Telegram.

## Alternatives considered

- **Baileys with Electron's Node.** The most popular library, but it means a second process, shipping and updating
  `node_modules`, and an engine that cannot use WhatsApp on its own. The in-process bridge is simpler for one person.
- **WhatsApp Cloud API.** Official and safe for the account, but needs a Meta business account, a business number and
  a public webhook. Left as a follow-up for people who have those.
- **Driving WhatsApp Web in a browser.** Fragile, heavy, and breaks whenever the page changes.
