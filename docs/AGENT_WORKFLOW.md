# How AI agents work on Sentient

These are the working rules for AI coding agents on Sentient, whether one assistant helps a contributor or many
agents build in parallel on one computer. They come from running exactly that, and most of them exist because
something went wrong once. People are welcome to follow them too. The product rules (no accounts, keys only in
the keychain, every tool declares a risk) are in [AGENTS.md](../AGENTS.md).

## Before you start

- **One issue, one pull request.** Don't bundle fixes. If you have several issues, finish one (PR open, checks
  green, review handled) before you branch from `main` for the next.
- **Claim the issue first,** so people and agents don't duplicate work. Check that nobody is assigned and that no
  open PR already mentions it (`gh pr list --search "<number>"`). Then claim it: comment `/assign` (see
  [CONTRIBUTING.md](../CONTRIBUTING.md)), or a short comment saying you are on it. If it is taken, pick another
  issue.
- **Branch from `main`.** When several agents work at once, give each its own git worktree.

## Test it for real

Unit tests are not enough. Before a change merges, it has to be tested end to end on a real engine, and you
have to actually see it work.

- **Start your own engine** from your checkout, with its own empty data folder and a random token, on a port you
  checked is free (see [Ports](#ports)):

  ```bash
  PY=.venv/Scripts/python.exe; [ -x "$PY" ] || PY=.venv/bin/python   # Windows, else macOS/Linux
  export SENTIENT_HOME="$(pwd -W 2>/dev/null || pwd)/.test-homes/issue-123"   # new and empty; never ~/.sentient
  unset SENTIENT_KEYCHAIN_NAMESPACE                       # an inherited override would win over the test home
  export SENTIENT_GATEWAY_TOKEN="$("$PY" -c 'import secrets; print(secrets.token_urlsafe(24))')"
  "$PY" -m sentient serve --port 8791
  curl -H "Authorization: Bearer $SENTIENT_GATEWAY_TOKEN" http://127.0.0.1:8791/api/config
  ```

  Drive it over REST and `WS /ws?token=...` as [docs/API.md](API.md) describes. In Git Bash on Windows set
  `MSYS_NO_PATHCONV=1`, or it rewrites `/api/...` arguments into Windows paths. That also stops it translating
  Git Bash paths such as `/c/work/...` for Python, which is why the snippet uses `pwd -W` (it prints
  `C:/work/...`); with plain `$PWD` the engine would quietly use a different folder. `.test-homes/` is ignored by git.
- **Use a real model where the model matters.** Sentient must work on small local models, so test with
  `ollama_chat/qwen3:8b`, and take the [Ollama lock](#the-ollama-lock) first. Always the `ollama_chat/` prefix:
  with `ollama/`, qwen writes its tool calls as plain text. If you also have a cloud model or the Claude Code
  backend, run the test there too. Claude Code answers chats only; keep the background roles (tasks, planner,
  executor) on a local model.
- **For UI changes,** build the app and capture the screens you changed with `desktop/scripts/smoke.mjs` on a
  seeded test home (the fictional persona Maya Rao, see [desktop/AGENTS.md](../desktop/AGENTS.md)). Then look at
  the screenshots yourself. A build that passes is not a screen that looks right.
- **For repository and CI changes,** prove the mechanism on GitHub: a dry-run log, a run on a fork, or a unit
  test of the script when neither is possible.
- **For documentation only,** check what you wrote against the code it describes, run any commands it shows,
  and say so in the PR. No engine or model run is needed when nothing that runs has changed.
- **Stop what you started** (engines, apps, servers) and delete your test homes when you are done.

## Sharing one computer

When several agents (or an agent and a person) use the same machine, these keep them out of each other's way.

### Ports

The person's own Sentient uses its own port (7777 by default, or a free one the app picks). Give every agent a
port of its own, for example one each from 8790 to 8799. Before you start an engine, check that nothing is
listening on the port (`Get-NetTCPConnection -LocalPort 8791` on Windows, `lsof -i :8791` elsewhere). Only send
requests to engines you started, and only stop your own.

### The Ollama lock

Ollama is shared, and when two engines load models at once it falls back to the CPU: everything gets slow and
the results mean nothing. So only one agent talks to the local model at a time.

- The lock is the file `.ollama.lock` at the root of the main checkout (the folder that holds the shared `.git`,
  so every worktree finds the same file). It is ignored by git.
- Take it **before any real model call**, including onboarding and starting a test home with seeded `planning`
  tasks, which call the model as soon as the engine starts. Until you hold it, point the test home's Ollama
  address at a closed port (`api_base` of `models.providers.ollama` and `ollama_chat`, for example
  `http://127.0.0.1:9`).
- If it exists, someone else is using the model: check again every 30 seconds. Write your name in it, delete it
  as soon as your run ends (also when the run fails), and never hold it for more than about 15 minutes at a time.

```bash
LOCK="$(git rev-parse --path-format=absolute --git-common-dir)/../.ollama.lock"
ME="issue-123 $$ $(date -u +%FT%TZ)"
until (set -o noclobber; echo "$ME" > "$LOCK") 2>/dev/null; do sleep 30; done
trap '[ "$(cat "$LOCK" 2>/dev/null)" = "$ME" ] && rm -f "$LOCK"' EXIT   # only ever removes our own lock
# ... the real model run ...
```

- Only delete a lock you hold. If one is much older than 15 minutes, check that whoever it names has really
  stopped (no engine of theirs running, no model loaded in `ollama ps`) before removing it, or ask the person.

Don't pull, delete or reconfigure models, or change the Ollama service's settings, without asking the person.

### The build lock

`npm run build`, `npm run package` and `smoke.mjs` are heavy, and parallel runs trip over each other. Take
`desktop/.build.lock` in the main checkout the same way as the Ollama lock (create it with your name, wait while
it exists, delete it when done).

### Keychain namespaces

Every `SENTIENT_HOME` other than the default `~/.sentient` has its own keychain namespace (`sentient-<hash>`, see
[docs/DEVELOPING.md](DEVELOPING.md#test-homes-and-the-keychain)), so a test home can't read, overwrite or delete
the person's real keys and sign-ins. `SENTIENT_KEYCHAIN_NAMESPACE` overrides that choice, so make sure it is not
set (`unset SENTIENT_KEYCHAIN_NAMESPACE`) before you start a test engine. Never set it to `default` unless the
person asked you to, and remove any test entries you created.

### Seeded test homes

A full seed run and `smoke.mjs` pause every task in the home and turn off messaging apps and deliveries. Keep it
that way: don't pass `--keep-live` unless the test needs it, and if you seed a home some other way, pause its
tasks and turn off its channel deliveries yourself. A demo must never run a task or message anyone.

### Other people's things

- Never connect to a DevTools, CDP or other debugging port you didn't start. The person's own browser may be
  listening on one, signed in to everything. Check the port is free, launch your app with that exact port, and
  confirm the target is your own app before you send anything.
- Never touch the person's real accounts, keychain entries, chats or a browser you didn't launch. Don't connect
  real integrations or send real messages from a test.
- Don't open files in the person's apps unless the test needs it, and close what you opened.

### Disk space

The full suite writes gigabytes of temporary folders, and the default temp folder is often on a small system
drive (a full one fails dozens of tests with "disk is full"). Give pytest a folder on a drive with room, such as
`.pytest-tmp` in your checkout when the checkout has room, and delete it and your old test homes afterwards:

```bash
.venv/Scripts/python.exe -m pytest -q --basetemp=.pytest-tmp   # .pytest-tmp/ and .test-homes/ are ignored by git
```

## The pull request

- A [Conventional Commits](https://www.conventionalcommits.org/) title with an area, and `Closes #123` in the body.
- The commits and the description are the contributor's: no "generated with" footers or AI co-author trailers.
- Fill in the template's **Real test** section with the evidence:
  - what you ran for real (engine from which commit, model id, which screens or endpoints),
  - what you saw: trimmed trace or log lines, API responses, and the result,
  - screenshots or traces **attached to the PR** (drag them into the description), never a path on your computer,
  - anything you could not test, and why.
- The unit tests and lint you ran, with their results.
- A changelog entry in `changes/` when people using Sentient will notice the change (see
  [changes/README.md](../changes/README.md)). Never edit `CHANGELOG.md` itself.
- No keys, tokens, real names or personal data anywhere in the PR.
- After opening it, wait for the checks and the CodeRabbit review. Fix the real findings (two rounds at most, then
  ask a maintainer), and leave merging to a maintainer.

## Someone else's pull request

When you work for the maintainers on a contributor's pull request, do the chores and leave the substance to them
([CONTRIBUTING.md](../CONTRIBUTING.md#we-do-the-chores)). Merging `main`, resolving conflicts, the `changes/`
file, lint fixes, re-running checks and tiny fixes a review asked for are yours to push, as merge commits and
only when "Allow edits by maintainers" is on; never force-push, and comment with exactly what you pushed. Anything
that changes behaviour or design goes back to them as a review. Pull requests labelled `needs update` still need
`main` merged in, because of conflicts or because the bot could not update the branch itself; the `/update` bot
takes the label off once the branch is up to date.

The `/update` bot runs on the built-in `GITHUB_TOKEN`. That token can merge `main` into a branch in this
repository, and into a fork's branch (anyone's fork, the owner's too) when "Allow edits by maintainers" is on.
GitHub refuses it one thing: a merge that brings commits changing `.github/workflows/` into a fork that doesn't
have those commits yet. The token can't be given the `workflows` permission that would allow it, and we don't add
an app or a secret for it. The bot still tries, because a synced fork works; when GitHub refuses, it says why,
points at the **Update branch** button and adds `needs update`. Without conflicts, a maintainer then merges `main`
in with their own credentials. These need the `workflow` scope: `gh auth status` lists the scopes, and
`gh auth refresh -s workflow` adds it. Set `n` to the pull request's number; the head sha makes GitHub refuse the
merge if someone pushed meanwhile:

```bash
n=123
sha=$(gh pr view "$n" -R existence-master/Sentient --json headRefOid --jq .headRefOid)
gh api -X PUT "repos/existence-master/Sentient/pulls/$n/update-branch" -f expected_head_sha="$sha"
```

When GitHub refuses the bot for any reason, its status and message are in the run's log (as a warning) and in
the run's summary.
