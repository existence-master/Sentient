// "/update" comments on pull requests (see .github/workflows/update.yaml and CONTRIBUTING.md).
//
// The PR author, or anyone with write access, comments exactly "/update". The bot then:
//   - already up to date      -> says so, and takes the "needs update" label off
//   - main conflicts with it  -> lists the files, adds "needs update" and says a maintainer will resolve them
//   - it can push the branch  -> merges main in with GitHub's update-branch API (a merge, never a rebase)
//   - it can't push the branch (a fork) -> points the author at GitHub's own "Update branch" button and adds the
//                                label so a maintainer can do it instead
// When the branch is pushed again and is up to date, the label comes off by itself (clearLabel).
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
const POLL = { tries: 12, ms: 5000 };

const code = (s) => `\`${s}\``;

const MESSAGES = {
  updated: (base, sha) =>
    `Merged the latest ${code(base)} into this branch` + (sha ? ` (${sha.slice(0, 7)})` : '') + '. ' +
    'Nothing else changed. GitHub does not start checks for a commit made by a bot, so they run again with ' +
    'your next push.',
  updating: (base) =>
    `GitHub is merging the latest ${code(base)} into this branch now; the new commit shows up here in a minute. ` +
    'GitHub does not start checks for a commit made by a bot, so they run again with your next push.',
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
  useButton: (base) =>
    `I can't push to this branch from here, because it lives in your fork. It has no conflicts with ` +
    `${code(base)}, so click **Update branch** near the bottom of this pull request and GitHub merges it in ` +
    `for you. I have also labelled it ${code(LABEL.name)}, so a maintainer can do it if you'd rather not. ` +
    'You don\'t have to: the merge queue tests your change against the latest main anyway.',
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

/**
 * Entry point for actions/github-script:
 * `await require('./.github/scripts/update.js')({ github, context, core })`.
 * `findConflicts` and `sleep` are swapped out by the tests.
 */
async function run({ github, context, core, findConflicts, sleep, poll = POLL, workDir }) {
  const { issue, comment } = context.payload;
  if (!issue || !comment) return core.info('Not a comment.');
  if (!issue.pull_request) return core.info('Comment is on an issue, not a pull request: ignored.');
  if (comment.user.type === 'Bot') return core.info('Comment is from a bot: ignored.');
  if (!parseCommand(comment.body)) return core.info('Not a command.');

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
  if (action === 'updateBranch') {
    try {
      await github.rest.pulls.updateBranch({ owner, repo, pull_number: issue_number, expected_head_sha: pr.head.sha });
      newSha = await waitForNewHead(github, {
        owner,
        repo,
        pull_number: issue_number,
        oldSha: pr.head.sha,
        sleep: sleep || ((ms) => new Promise((r) => setTimeout(r, ms))),
        poll,
      });
      action = 'updated';
    } catch (err) {
      const message = String(err.message || '');
      core.info(`update-branch answered ${err.status}: ${message}`);
      if (err.status === 422 && /expected head sha/i.test(message)) action = 'headMoved';
      else if (err.status === 422 && /conflict/i.test(message)) action = 'conflicts';
      // The workflow's token may not push to a fork, even with maintainer edits allowed.
      else if (fork && [403, 404, 422].includes(err.status)) action = 'useButton';
      else action = 'failed';
    }
  }

  if (action === 'upToDate' || action === 'updated') await removeLabel(github, { owner, repo, issue_number });
  if (['conflicts', 'useButton', 'failed'].includes(action)) await addLabel(github, { owner, repo, issue_number });

  const reply = {
    upToDate: () => MESSAGES.upToDate(base),
    updated: () => (newSha ? MESSAGES.updated(base, newSha) : MESSAGES.updating(base)),
    conflicts: () => MESSAGES.conflicts(base, conflicts, canEdit),
    useButton: () => MESSAGES.useButton(base),
    headMoved: () => MESSAGES.headMoved,
    failed: () => MESSAGES.failed(base),
  }[action]();
  core.info(`Result: ${action}`);
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
