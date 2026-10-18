// "/update" comments on pull requests (see .github/workflows/update.yaml and CONTRIBUTING.md).
//
// The PR author, or anyone with write access, comments exactly "/update". The bot then:
//   - already up to date      -> says so, and takes the "needs update" label off
//   - main conflicts with it  -> lists the files, adds "needs update" and says a maintainer will resolve them
//   - it may push the branch  -> merges main in with GitHub's update-branch API (a merge, never a rebase). That
//                                works for forks too, when "Allow edits by maintainers" is on.
//   - it may not (a fork without maintainer edits, or GitHub refuses) -> points the author at GitHub's own
//                                "Update branch" button and adds the label so a maintainer can follow up
// When the branch is pushed again and is up to date, the label comes off by itself (clearLabel).
//
// After its own merge commit the bot restarts the checks, because GitHub treats a commit made with the workflow
// token differently from a person's push: the pull_request runs (CI, Changelog) wait for approval, and
// pull_request_target workflows (the CLA check) don't run at all. So the bot
//   - approves the waiting runs, but only when GitHub had already run the checks on the commit before its merge
//     (otherwise, for example for a first-time contributor, a maintainer still approves them by hand), and then
//   - copies the CLAAssistant result from that commit, which is exactly what the CLA action would decide again:
//     the merge adds no commit of anyone else's (main's commits aren't part of the PR) and its own author,
//     github-actions[bot], is on the CLA allowlist.
// Both happen only for a commit GitHub itself made: verified, committed by web-flow, its first parent the old head.
//
// It only compares the comment with a fixed string, and never checks out or runs the pull request's code. To find
// conflicting files it fetches the two commits' history into an empty bare repository and asks git to merge them
// in memory (git merge-tree): no working tree, no hooks, nothing from the branch is executed.

const LABEL = {
  name: 'needs update',
  color: 'fbca04',
  description: 'Behind main or has conflicts; a maintainer will bring it up to date',
};
// author_association values for people with write access (the PR author may always use the command too).
const WRITERS = ['OWNER', 'MEMBER', 'COLLABORATOR'];
const CLA_CHECK = 'CLAAssistant';
const POLL = { tries: 12, ms: 5000 };

const code = (s) => `\`${s}\``;

const MESSAGES = {
  updated: (base, sha, checks) =>
    `Merged the latest ${code(base)} into this branch (${sha.slice(0, 7)}). Nothing else changed. ` +
    {
      running: 'The checks are running again.',
      held: 'GitHub holds the checks for a commit made by a bot, so a maintainer will start them.',
      noCla: 'The checks are running again, but the CLA check only runs again with your next push.',
    }[checks],
  updating: (base) =>
    `GitHub is merging the latest ${code(base)} into this branch now, and the new commit shows up here in a ` +
    'minute. GitHub holds the checks for a commit made by a bot, so a maintainer will start them.',
  upToDate: (base) => `This branch already has everything from ${code(base)}. Nothing to do.`,
  conflicts: (base, files, canEdit) =>
    `${code(base)} has changes that conflict with this branch` +
    (files && files.length ? ' in:\n\n' + files.map((f) => `- ${code(f)}`).join('\n') + '\n\n' : '. ') +
    `I have labelled it ${code(LABEL.name)}. ` +
    (canEdit
      ? 'A maintainer will resolve the conflicts for you and comment with exactly what they pushed. ' +
        'You are welcome to resolve them yourself instead, if you prefer.'
      : 'A maintainer can resolve them for you once **Allow edits by maintainers** is ticked on this pull ' +
        'request (in the sidebar). Or resolve them yourself, whichever you prefer.'),
  useButton: (base, canEdit, clean) =>
    (canEdit
      ? "I can't push to this branch: GitHub refused the update for this fork. "
      : "I can't push to this branch, because it lives in a fork that doesn't allow edits by maintainers. ") +
    (clean ? `It has no conflicts with ${code(base)}, so click` : 'Click') +
    ' **Update branch** near the bottom of this pull request and GitHub merges it in for you. ' +
    `I have also labelled it ${code(LABEL.name)} so a maintainer can follow up. ` +
    "There's no rush: the merge queue tests your change against the latest main anyway.",
  headMoved: 'This branch changed while I was updating it, so I left it alone. Comment `/update` again to retry.',
  closed: 'This pull request is closed, so there is nothing to update.',
  notAllowed:
    'Only the author of this pull request or a maintainer can use `/update` here. ' +
    'If it needs updating, a maintainer can help.',
  failed: (base) =>
    `Sorry, I could not merge ${code(base)} into this branch. I have labelled it ${code(LABEL.name)} so a ` +
    'maintainer takes a look.',
};

/** True only when the whole comment is "/update" (surrounding spaces and line breaks allowed). */
function parseCommand(body) {
  return String(body || '').trim() === '/update';
}

/** The PR author or someone with write access may update a branch. */
function mayUpdate({ association, commenter, author }) {
  if (WRITERS.includes(String(association || '').toUpperCase())) return true;
  return !!commenter && String(commenter).toLowerCase() === String(author || '').toLowerCase();
}

/** True when the branch lives in another repository (a fork), or the fork is gone. */
function isFork(pr, owner, repo) {
  const name = pr.head && pr.head.repo && pr.head.repo.full_name;
  return !name || name.toLowerCase() !== `${owner}/${repo}`.toLowerCase();
}

/**
 * Decides what to do once the branch's state is known, without touching GitHub.
 * behind: commits on the base branch the PR lacks. conflicts: conflicting files ([] for none, null if unknown).
 * Returns "upToDate" | "conflicts" | "useButton" | "updateBranch".
 */
function decide({ behind, conflicts, fork, canEdit }) {
  if (behind === 0) return 'upToDate';
  if (conflicts && conflicts.length) return 'conflicts';
  // Without "Allow edits by maintainers" nobody but the author can push to a fork's branch.
  if (fork && !canEdit) return 'useButton';
  return 'updateBranch';
}

/** Commits on the base branch (its current tip, by name) that the PR's head doesn't have yet. */
async function behindBy(github, { owner, repo, pr }) {
  const { data } = await github.rest.repos.compareCommitsWithBasehead({
    owner,
    repo,
    basehead: `${pr.head.sha}...${pr.base.ref}`,
    per_page: 1,
  });
  return data.ahead_by;
}

/**
 * Files that conflict when the base branch is merged into the PR, found with `git merge-tree` in a fresh bare
 * repository (only git objects, never a checkout). Returns [] when it merges cleanly.
 */
function gitConflicts({ owner, repo, number, base, dir }) {
  const { execFileSync } = require('node:child_process');
  const fs = require('node:fs');
  if (!/^[A-Za-z0-9._/-]+$/.test(base) || !/^\d+$/.test(String(number))) throw new Error('unexpected ref');
  fs.rmSync(dir, { recursive: true, force: true });
  const git = (...args) =>
    execFileSync('git', ['-C', dir, ...args], {
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'pipe'],
      timeout: 120000,
    });
  execFileSync('git', ['init', '--bare', '--quiet', dir], { stdio: 'ignore' });
  git('remote', 'add', 'origin', `https://github.com/${owner}/${repo}.git`);
  // Fetch commits and trees only; git fetches the few file contents the merge needs on demand.
  git('config', 'remote.origin.promisor', 'true');
  git('config', 'remote.origin.partialclonefilter', 'blob:none');
  git('fetch', '--quiet', '--no-tags', '--filter=blob:none', 'origin',
    `+refs/heads/${base}:refs/heads/base`, `+refs/pull/${number}/head:refs/heads/pr`);
  try {
    git('merge-tree', '--write-tree', '--name-only', '--no-messages', 'pr', 'base');
    return [];
  } catch (err) {
    // Exit code 1 means conflicts: the first line is a tree id, the rest are the conflicting files.
    if (err.status !== 1) throw err;
    const lines = String(err.stdout || '').split('\n').slice(1).map((l) => l.trim()).filter(Boolean);
    return [...new Set(lines)];
  }
}

async function addLabel(github, { owner, repo, issue_number }) {
  try {
    await github.rest.issues.getLabel({ owner, repo, name: LABEL.name });
  } catch (err) {
    if (err.status !== 404) throw err;
    try {
      await github.rest.issues.createLabel({ owner, repo, ...LABEL });
    } catch (e) {
      if (e.status !== 422) throw e; // 422: someone created it meanwhile
    }
  }
  await github.rest.issues.addLabels({ owner, repo, issue_number, labels: [LABEL.name] });
}

async function removeLabel(github, { owner, repo, issue_number }) {
  try {
    await github.rest.issues.removeLabel({ owner, repo, issue_number, name: LABEL.name });
    return true;
  } catch (err) {
    if (err.status === 404) return false; // it wasn't there
    throw err;
  }
}

/** Waits for update-branch's merge commit to appear (the API answers before the merge is done). */
async function waitForNewHead(github, { owner, repo, pull_number, oldSha, sleep, poll }) {
  for (let i = 0; i < poll.tries; i++) {
    await sleep(poll.ms);
    const { data } = await github.rest.pulls.get({ owner, repo, pull_number });
    if (data.head.sha !== oldSha) return data.head.sha;
  }
  return null;
}

/** True when newSha is GitHub's own merge commit on top of oldSha (not something pushed meanwhile). */
async function isOurMerge(github, { owner, repo, oldSha, newSha }) {
  const { data } = await github.rest.repos.getCommit({ owner, repo, ref: newSha });
  const parents = (data.parents || []).map((p) => p.sha);
  return (
    parents.length === 2 &&
    parents[0] === oldSha &&
    !!(data.committer && data.committer.login === 'web-flow') &&
    !!(data.commit && data.commit.verification && data.commit.verification.verified)
  );
}

/**
 * Copies the CLA check's result from the old head to the merge commit (see the top of this file). Called once the
 * approved runs have started: GitHub files a check made with the workflow token under one of that commit's workflow
 * runs, and one filed before the run was approved is dropped when it starts. Returns true when the check is there.
 */
async function carryClaCheck(github, { owner, repo, oldSha, newSha, core }) {
  const latest = async (ref) =>
    (await github.rest.checks.listForRef({ owner, repo, ref, check_name: CLA_CHECK, filter: 'latest' })).data
      .check_runs[0];
  if (await latest(newSha)) {
    core.info('The CLA check is already on the merge commit.');
    return true;
  }
  const old = await latest(oldSha);
  if (!old || old.status !== 'completed') {
    core.info('No finished CLA check on the old head to copy.');
    return false;
  }
  await github.rest.checks.create({
    owner,
    repo,
    name: CLA_CHECK,
    head_sha: newSha,
    status: 'completed',
    conclusion: old.conclusion,
    details_url: old.html_url,
    output: {
      title: `Same as ${oldSha.slice(0, 7)}`,
      summary:
        `Copied from ${oldSha.slice(0, 7)} by /update: this commit only merges the base branch in, so it adds ` +
        'no new contributor.',
    },
  });
  const ok = !!(await latest(newSha));
  core.info(`Copied the CLA check (${old.conclusion}) from ${oldSha.slice(0, 7)}${ok ? '' : ', but GitHub does not list it'}.`);
  return ok;
}

/**
 * Approves the pull_request runs GitHub holds on the merge commit, if it had already run them on the old head.
 * Returns "running" or "held" (a maintainer approves them).
 */
async function approveHeldRuns(github, { owner, repo, oldSha, newSha, sleep, poll, core }) {
  const runs = async (head_sha, status) =>
    (await github.rest.actions.listWorkflowRunsForRepo({ owner, repo, head_sha, event: 'pull_request', status }))
      .data.workflow_runs;
  const before = await runs(oldSha);
  if (!before.length || before.some((r) => r.conclusion === 'action_required')) {
    core.info('The old head\'s checks were not run without approval, so the new ones wait for a maintainer.');
    return 'held';
  }
  let held = [];
  for (let i = 0; i < poll.tries && !held.length; i++) {
    await sleep(poll.ms);
    held = await runs(newSha, 'action_required');
  }
  if (!held.length) return 'running';
  await sleep(poll.ms); // let GitHub finish creating the rest of them
  held = await runs(newSha, 'action_required');
  for (const r of held) {
    await github.rest.actions.approveWorkflowRun({ owner, repo, run_id: r.id });
    core.info(`Approved ${r.name} (${r.id}).`);
  }
  // Wait until they have left the queue, so a check filed under one of them afterwards stays.
  for (let i = 0; i < poll.tries; i++) {
    const waiting = (await runs(newSha)).filter(
      (r) => r.conclusion === 'action_required' || ['queued', 'waiting', 'pending', 'requested'].includes(r.status),
    );
    if (!waiting.length) break;
    await sleep(poll.ms);
  }
  return 'running';
}

/**
 * Entry point for actions/github-script:
 * `await require('./.github/scripts/update.js')({ github, context, core })`.
 * `findConflicts`, `sleep` and `poll` are swapped out by the tests.
 */
async function run({ github, context, core, findConflicts, sleep, poll = POLL, workDir }) {
  const { issue, comment } = context.payload;
  if (!issue || !comment) return core.info('Not a comment.');
  if (!issue.pull_request) return core.info('Comment is on an issue, not a pull request: ignored.');
  if (comment.user.type === 'Bot') return core.info('Comment is from a bot: ignored.');
  if (!parseCommand(comment.body)) return core.info('Not a command.');
  sleep = sleep || ((ms) => new Promise((resolve) => setTimeout(resolve, ms)));

  const { owner, repo } = context.repo;
  const issue_number = issue.number;
  const commenter = comment.user.login;
  const say = (text) => github.rest.issues.createComment({ owner, repo, issue_number, body: `@${commenter} ${text}` });

  if (!mayUpdate({ association: comment.author_association, commenter, author: issue.user && issue.user.login })) {
    core.info(`/update by ${commenter} (${comment.author_association}) on #${issue_number}: not allowed`);
    return say(MESSAGES.notAllowed);
  }

  // Read the pull request now: the payload is a snapshot from when the comment was made.
  const { data: pr } = await github.rest.pulls.get({ owner, repo, pull_number: issue_number });
  if (pr.state !== 'open') return say(MESSAGES.closed);
  const base = pr.base.ref;
  const oldSha = pr.head.sha;
  const fork = isFork(pr, owner, repo);
  const canEdit = !fork || !!pr.maintainer_can_modify;

  const behind = await behindBy(github, { owner, repo, pr });
  let conflicts = null;
  if (behind > 0) {
    try {
      conflicts = await (findConflicts || gitConflicts)({
        owner,
        repo,
        number: issue_number,
        base,
        dir: workDir || `${process.env.RUNNER_TEMP || require('node:os').tmpdir()}/update-merge-check`,
      });
    } catch (err) {
      core.warning(`Could not check for conflicts: ${err.message}`);
    }
  }
  let action = decide({ behind, conflicts, fork, canEdit });
  core.info(
    `/update by ${commenter} on #${issue_number}: behind ${behind}, fork ${fork}, maintainer edits ${canEdit}, ` +
      `conflicts ${conflicts === null ? 'unknown' : conflicts.length} -> ${action}`,
  );

  let newSha = null;
  let checks = 'held';
  if (action === 'updateBranch') {
    try {
      await github.rest.pulls.updateBranch({ owner, repo, pull_number: issue_number, expected_head_sha: oldSha });
      action = 'updated';
    } catch (err) {
      const message = String(err.message || '');
      core.info(`update-branch answered ${err.status}: ${message}`);
      if (err.status === 422 && /expected head sha/i.test(message)) action = 'headMoved';
      else if (err.status === 422 && /conflict/i.test(message)) action = 'conflicts';
      // GitHub may still refuse to push to a fork (for example one owned by an organisation).
      else if (fork && [403, 404, 422].includes(err.status)) action = 'useButton';
      else action = 'failed';
    }
  }
  if (action === 'updated') {
    newSha = await waitForNewHead(github, { owner, repo, pull_number: issue_number, oldSha, sleep, poll });
    if (newSha && (await isOurMerge(github, { owner, repo, oldSha, newSha }))) {
      try {
        checks = await approveHeldRuns(github, { owner, repo, oldSha, newSha, sleep, poll, core });
        if (checks === 'running' && !(await carryClaCheck(github, { owner, repo, oldSha, newSha, core }))) {
          checks = 'noCla';
        }
      } catch (err) {
        core.warning(`Could not restart the checks: ${err.message}`);
      }
    }
  }

  // Without the new head in sight the branch isn't known to be current yet, so the label stays.
  if (action === 'upToDate' || (action === 'updated' && newSha)) await removeLabel(github, { owner, repo, issue_number });
  if (['conflicts', 'useButton', 'failed'].includes(action)) await addLabel(github, { owner, repo, issue_number });

  const reply = {
    upToDate: () => MESSAGES.upToDate(base),
    updated: () => (newSha ? MESSAGES.updated(base, newSha, checks) : MESSAGES.updating(base)),
    conflicts: () => MESSAGES.conflicts(base, conflicts, canEdit),
    useButton: () => MESSAGES.useButton(base, canEdit, Array.isArray(conflicts) && conflicts.length === 0),
    headMoved: () => MESSAGES.headMoved,
    failed: () => MESSAGES.failed(base),
  }[action]();
  core.info(`Result: ${action}${action === 'updated' ? `, checks ${checks}` : ''}`);
  return say(reply);
}

/**
 * On a push to a labelled pull request (pull_request_target, synchronize): take "needs update" off once the
 * branch has everything from the base branch. Says nothing.
 */
async function clearLabel({ github, context, core }) {
  const pr = context.payload.pull_request;
  if (!pr) return core.info('Not a pull request event.');
  if (!(pr.labels || []).some((l) => l.name === LABEL.name)) return core.info('No label to clear.');
  const { owner, repo } = context.repo;
  const behind = await behindBy(github, { owner, repo, pr });
  core.info(`#${pr.number} is ${behind} commit(s) behind ${pr.base.ref}`);
  if (behind === 0 && (await removeLabel(github, { owner, repo, issue_number: pr.number }))) {
    core.info(`Removed "${LABEL.name}".`);
  }
}

module.exports = run;
module.exports.clearLabel = clearLabel;
module.exports.parseCommand = parseCommand;
module.exports.mayUpdate = mayUpdate;
module.exports.isFork = isFork;
module.exports.decide = decide;
module.exports.gitConflicts = gitConflicts;
module.exports.MESSAGES = MESSAGES;
module.exports.LABEL = LABEL;
