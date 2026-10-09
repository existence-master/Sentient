# 0019. Run commands on the host only when turned on, in allowed folders, with approval

- **Status:** Accepted
- **Date:** 2026-10-09

## Context

People moving from Hermes Agent use its local terminal every day: git, builds, tests, package managers, scripts and
file chores. Sentient could only run code in its sandbox ([ADR 0012](0012-code-execution-sandbox.md)), which cannot
change the host, so that work was impossible. A terminal on the host is the most powerful tool Sentient can have:
one command can delete a project, leak a key or turn the computer off. It has to be safe for a non-engineer who
turns it on, and stay safe when a web page or an email tries to talk the model into something.

## Decision

Add a `terminal` package with one tool, `terminal_run(command, cwd?)`, risk `exec`.

- **Off by default.** Settings > Terminal turns it on. Until then the tool is hidden and refuses calls.
- **Allowed folders.** A command starts in the default folder or a `cwd` that, after following links, is inside a
  folder the user added. With no folders, nothing runs. The folder is where a command starts, not a jail: approval is
  what limits what the command does.
- **Approval.** Every command asks, showing the exact command and folder, as any `exec` tool does
  ([ADR 0008](0008-effective-risk-and-approvals.md)). "Allow for this chat" never covers a command: each one asks
  again, so one yes can't turn into a string of commands nobody looked at. Lasting rules work as usual
  ([ADR 0016](0016-lasting-approval-rules.md)): Allow on `terminal` runs commands without asking, Never removes the
  tool. Commands the user lists as never needing a question (default `git status`, `git diff`, `git log`, `ls`, `dir`,
  `pwd`) run without asking, but only as plain commands: no chaining, redirection, substitution, `--output` or flags
  that make git run a configured program (`--ext-diff`, `--textconv`), and with git's `core.fsmonitor` turned off. Their
  effective risk is `read`. A call the terminal refuses anyway is also `read`, so nobody is asked to approve a refusal.
- **Built-in blocklist.** Formatting or wiping disks, shutting down or restarting, deleting from the registry,
  deleting or re-owning a whole drive, system folder or home folder, deleting backups and changing boot settings are
  refused in code, even with an Allow rule or a listed command. It catches slips and obvious injected commands; it
  is a seatbelt, not a security boundary, because a determined command can always be disguised.
- **Who may run commands.** Work nobody asked for never runs one ([ADR 0017](0017-unprompted-work-reads-only.md)):
  its effective risk stays `exec` and the tool refuses it too. Scripts from code execution can never call it, in
  any approvals mode. Task runs and helpers cannot stop to ask yet, so there only listed commands run, unless the user
  set an Allow rule for the terminal or turned approvals off. Under [ADR 0018](0018-untrusted-content-gates-sending.md)
  a command's output counts as outside content (`untrusted_output`), and a command that isn't listed (`exec`) counts
  as able to send data out, so once a chat has read an email, a web page or a command's output, every such command
  asks with the reason shown, even with an Allow rule. Listed commands stay free: they can't carry data out.
- **Process.** Each call is a new PowerShell process on Windows (`pwsh` if installed, else Windows PowerShell) or
  bash, zsh or sh elsewhere, without profiles, with empty input. The environment is the engine's own minus anything
  that looks like a key, token or password, Sentient's own variables and the provider key variables in the config;
  keychain secrets are never added. Output streams to the chat, is trimmed for the model, and is saved to a file when
  long. The timeout (default 180 s), the card's Stop button, cancelling the reply and Stop everything kill the whole
  process tree, using the sandbox's process runner; on macOS and Linux every process carrying the run's marker is
  killed too, so a child that left the process group can't outlive Stop. A process that clears its own
  environment can still escape there; approval is the gate for that.

## Consequences

Coding, devops and file chores that people did with Hermes now work in Sentient, after one yes per command or a rule
the user chose. The new tool widens what Sentient can do on the host, so its defaults stay closed: off, no folders,
and a question for every command that is not a plain listed one. Every future surface that runs tools must keep
scripts and unprompted work away from it. Commands can still read and change anything the user can, outside the
allowed folders too, once approved; the approval card is where people judge that. Long-lived programs (a dev server
started in the background) are stopped when the command ends, and nothing carries over between calls.

## Alternatives considered

Running commands in the sandbox or in Docker: safe, but it cannot touch the user's projects, which is the point.
A persistent shell session per chat: closer to Hermes, but state that carries over (a `cd`, an exported variable)
makes each approval card harder to judge, so each call is a fresh process. A configurable blocklist: a list the
model could talk the user into emptying is weaker than a fixed one; Never rules already let people remove the tool.
Letting tasks ask before each command: the right end state, but tasks cannot pause for approval yet, so they get
listed commands only for now.
