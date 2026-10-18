// Run with: node --test ".github/scripts/*.test.js"
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
const run = require('./update.js');
const { clearLabel, parseCommand, mayUpdate, isFork, decide, MESSAGES, LABEL } = run;

const OLD = 'a'.repeat(40);
const NEW = 'b'.repeat(40);
const MAIN = 'c'.repeat(40);
const core = { info: () => {}, warning: () => {} };
const noSleep = async () => {};
const poll = { tries: 3, ms: 0 };

function httpError(status, message) {
  return Object.assign(new Error(message), { status });
}

/**
 * A fake GitHub. Options:
 *   pr: overrides for the live pull request; behind: commits on main it lacks
 *   update: what update-branch does ("ok", or an error); merge: the merge commit's metadata
 *   labelExists, oldRuns (pull_request runs on the old head), heldRuns (waiting on the merge commit),
 *   oldCla (the CLA check on the old head), newCla (one already on the merge commit)
 */
function fakeGithub(opts = {}) {
  const calls = [];
  const {
    behind = 3,
    update = 'ok',
    labelExists = true,
    oldRuns = [{ id: 1, name: 'CI', conclusion: 'success' }],
    heldRuns = [{ id: 7, name: 'CI' }, { id: 8, name: 'Changelog' }],
    oldCla = { status: 'completed', conclusion: 'success', html_url: 'https://github.com/x/runs/1' },
    newCla = null,
    merge = { parents: [{ sha: OLD }, { sha: MAIN }], committer: { login: 'web-flow' }, commit: { verification: { verified: true } } },
  } = opts;
  let updated = false;
  let created = null;
  const approved = new Set();
  const pr = {
    state: 'open',
    head: { sha: OLD, repo: { full_name: 'existence-master/Sentient' } },
    base: { ref: 'main' },
    maintainer_can_modify: false,
    ...opts.pr,
  };
  const github = {
    rest: {
      pulls: {
        get: async () => ({ data: { ...pr, head: { ...pr.head, sha: updated ? NEW : pr.head.sha } } }),
        updateBranch: async (args) => {
          calls.push(['updateBranch', args]);
          if (update !== 'ok') throw update;
          updated = true;
          return { status: 202, data: { message: 'Updating pull request branch.' } };
        },
      },
      repos: {
        compareCommitsWithBasehead: async (args) => {
          calls.push(['compare', args]);
          return { data: { ahead_by: behind } };
        },
        getCommit: async () => ({ data: merge }),
      },
      checks: {
        listForRef: async ({ ref }) => ({
          data: { check_runs: [ref === OLD ? oldCla : newCla || (opts.claListed !== false && created)].filter(Boolean) },
        }),
        create: async (args) => {
          calls.push(['createCheck', args]);
          created = { status: 'completed', conclusion: args.conclusion };
        },
      },
      actions: {
        // Held runs wait on the merge commit until approved; then they are in progress.
        listWorkflowRunsForRepo: async ({ head_sha, status }) => {
          if (head_sha === OLD) return { data: { workflow_runs: oldRuns } };
          const now = heldRuns.map((r) =>
            approved.has(r.id) ? { ...r, status: 'in_progress', conclusion: null } : { ...r, status: 'completed', conclusion: 'action_required' },
          );
          return { data: { workflow_runs: status === 'action_required' ? now.filter((r) => r.conclusion === 'action_required') : now } };
        },
        approveWorkflowRun: async (args) => {
          calls.push(['approve', args.run_id]);
          approved.add(args.run_id);
        },
      },
      issues: {
        getLabel: async () => {
          if (!labelExists) throw httpError(404, 'Not Found');
          return { data: LABEL };
        },
        createLabel: async (args) => calls.push(['createLabel', args]),
        addLabels: async (args) => calls.push(['addLabels', args.labels]),
        removeLabel: async (args) => {
          calls.push(['removeLabel', args.name]);
          if (opts.labelOnPr === false) throw httpError(404, 'Label does not exist');
        },
        createComment: async (args) => calls.push(['createComment', args.body]),
      },
    },
  };
  return { github, calls, names: () => calls.map((c) => c[0]) };
}

function context({ body = '/update', login = 'maya-rao', association = 'CONTRIBUTOR', author = 'maya-rao', pr = true, type = 'User' } = {}) {
  return {
    repo: { owner: 'existence-master', repo: 'Sentient' },
    payload: {
      issue: {
        number: 42,
        user: { login: author },
        ...(pr ? { pull_request: { url: 'https://api.github.com/x' } } : {}),
      },
      comment: { body, author_association: association, user: { login, type } },
    },
  };
}

function runWith(fake, ctx = context(), conflicts = []) {
  const findConflicts = async () => {
    if (conflicts instanceof Error) throw conflicts;
    return conflicts;
  };
  return run({ github: fake.github, context: ctx, core, findConflicts, sleep: noSleep, poll });
}

const reply = (fake) => fake.calls.filter((c) => c[0] === 'createComment').map((c) => c[1]);

test('only an exact /update counts', () => {
  assert.equal(parseCommand('/update'), true);
  assert.equal(parseCommand('  /update\r\n'), true);
  assert.equal(parseCommand('/update please'), false);
  assert.equal(parseCommand('please /update'), false);
  assert.equal(parseCommand('/Update'), false);
  assert.equal(parseCommand('/updates'), false);
  assert.equal(parseCommand(''), false);
  assert.equal(parseCommand(undefined), false);
});

test('the author and people with write access may update, nobody else', () => {
  assert.equal(mayUpdate({ association: 'CONTRIBUTOR', commenter: 'Maya-Rao', author: 'maya-rao' }), true);
  for (const association of ['OWNER', 'MEMBER', 'COLLABORATOR']) {
    assert.equal(mayUpdate({ association, commenter: 'ravi', author: 'maya-rao' }), true);
  }
  for (const association of ['CONTRIBUTOR', 'FIRST_TIME_CONTRIBUTOR', 'NONE', undefined]) {
    assert.equal(mayUpdate({ association, commenter: 'ravi', author: 'maya-rao' }), false);
  }
});

test('isFork and decide', () => {
  assert.equal(isFork({ head: { repo: { full_name: 'Existence-Master/sentient' } } }, 'existence-master', 'Sentient'), false);
  assert.equal(isFork({ head: { repo: { full_name: 'maya-rao/Sentient' } } }, 'existence-master', 'Sentient'), true);
  assert.equal(isFork({ head: { repo: null } }, 'existence-master', 'Sentient'), true);
  assert.equal(decide({ behind: 0, conflicts: null, fork: true, canEdit: false }), 'upToDate');
  assert.equal(decide({ behind: 2, conflicts: ['a'], fork: false, canEdit: true }), 'conflicts');
  assert.equal(decide({ behind: 2, conflicts: [], fork: true, canEdit: false }), 'useButton');
  assert.equal(decide({ behind: 2, conflicts: [], fork: true, canEdit: true }), 'updateBranch');
  assert.equal(decide({ behind: 2, conflicts: null, fork: false, canEdit: true }), 'updateBranch');
});

test('comments on issues, from bots, and other text are ignored', async () => {
  for (const ctx of [
    context({ pr: false }),
    context({ type: 'Bot', login: 'coderabbitai[bot]', association: 'NONE' }),
    context({ body: 'can someone /update this' }),
    context({ body: '/update\n\nand also rebase' }),
  ]) {
    const fake = fakeGithub();
    await runWith(fake, ctx);
    assert.deepEqual(fake.calls, []);
  }
});

test('someone else on the pull request is told who can use it, and nothing changes', async () => {
  const fake = fakeGithub();
  await runWith(fake, context({ login: 'ravi', association: 'CONTRIBUTOR' }));
  assert.deepEqual(fake.names(), ['createComment']);
  assert.equal(reply(fake)[0], `@ravi ${MESSAGES.notAllowed}`);
});

test('a branch that is already up to date: says so and clears the label', async () => {
  const fake = fakeGithub({ behind: 0 });
  await runWith(fake);
  assert.deepEqual(fake.names(), ['compare', 'removeLabel', 'createComment']);
  assert.equal(fake.calls[0][1].basehead, `${OLD}...main`);
  assert.equal(reply(fake)[0], `@maya-rao ${MESSAGES.upToDate('main')}`);
});

test('a closed pull request is left alone', async () => {
  const fake = fakeGithub({ pr: { state: 'closed' } });
  await runWith(fake);
  assert.deepEqual(fake.names(), ['createComment']);
  assert.equal(reply(fake)[0], `@maya-rao ${MESSAGES.closed}`);
});

test('a clean same-repo branch: merges main, restarts the checks, clears the label', async () => {
  const fake = fakeGithub();
  await runWith(fake);
  assert.deepEqual(fake.calls.find((c) => c[0] === 'updateBranch')[1], {
    owner: 'existence-master', repo: 'Sentient', pull_number: 42, expected_head_sha: OLD,
  });
  const check = fake.calls.find((c) => c[0] === 'createCheck')[1];
  assert.equal(check.name, 'CLAAssistant');
  assert.equal(check.head_sha, NEW);
  assert.equal(check.conclusion, 'success');
  assert.deepEqual(fake.calls.filter((c) => c[0] === 'approve').map((c) => c[1]), [7, 8]);
  // The CLA check is filed only after the approved runs started, or GitHub drops it when they do.
  assert.ok(fake.names().indexOf('createCheck') > fake.names().lastIndexOf('approve'));
  assert.ok(fake.names().includes('removeLabel'));
  assert.ok(!fake.names().includes('addLabels'));
  assert.equal(reply(fake)[0], `@maya-rao ${MESSAGES.updated('main', NEW, 'running')}`);
  assert.match(reply(fake)[0], /Merged the latest `main` into this branch \(bbbbbbb\)\. Nothing else changed\. The checks are running again\./);
});

test('a fork with maintainer edits on is updated the same way', async () => {
  const fake = fakeGithub({ pr: { head: { sha: OLD, repo: { full_name: 'maya-rao/Sentient' } }, maintainer_can_modify: true } });
  await runWith(fake);
  assert.ok(fake.names().includes('updateBranch'));
  assert.match(reply(fake)[0], /The checks are running again/);
});

test('a fork without maintainer edits: points at the button and labels it, no push attempted', async () => {
  const fake = fakeGithub({ labelExists: false, pr: { head: { sha: OLD, repo: { full_name: 'maya-rao/Sentient' } } } });
  await runWith(fake);
  assert.ok(!fake.names().includes('updateBranch'));
  assert.deepEqual(fake.calls.find((c) => c[0] === 'createLabel')[1], { owner: 'existence-master', repo: 'Sentient', ...LABEL });
  assert.deepEqual(fake.calls.find((c) => c[0] === 'addLabels')[1], ['needs update']);
  assert.equal(reply(fake)[0], `@maya-rao ${MESSAGES.useButton('main')}`);
  assert.match(reply(fake)[0], /click \*\*Update branch\*\*/);
});

test('a fork GitHub refuses to push to falls back to the button', async () => {
  const fake = fakeGithub({
    update: httpError(403, "user doesn't have permission to update head repository"),
    pr: { head: { sha: OLD, repo: { full_name: 'maya-org/Sentient' } }, maintainer_can_modify: true },
  });
  await runWith(fake);
  assert.ok(fake.names().includes('addLabels'));
  assert.equal(reply(fake)[0], `@maya-rao ${MESSAGES.useButton('main')}`);
});

test('conflicts: lists the files, labels it, says a maintainer will resolve them, never pushes', async () => {
  const fake = fakeGithub();
  await runWith(fake, context(), ['changes/456.fixed.md', 'docs/API.md']);
  assert.ok(!fake.names().includes('updateBranch'));
  assert.deepEqual(fake.calls.find((c) => c[0] === 'addLabels')[1], ['needs update']);
  const text = reply(fake)[0];
  assert.match(text, /`main` has changes that conflict with this branch in:\n\n- `changes\/456.fixed.md`\n- `docs\/API.md`\n/);
  assert.match(text, /A maintainer will resolve the conflicts for you and comment with exactly what they pushed/);
});

test('conflicts on a fork without maintainer edits ask for them to be turned on', async () => {
  const fake = fakeGithub({ pr: { head: { sha: OLD, repo: { full_name: 'maya-rao/Sentient' } } } });
  await runWith(fake, context(), ['docs/API.md']);
  assert.match(reply(fake)[0], /once \*\*Allow edits by maintainers\*\* is ticked/);
});

test('when the conflict check fails, update-branch reporting a conflict still gets the label', async () => {
  const fake = fakeGithub({ update: httpError(422, 'merge conflict between base and head') });
  await runWith(fake, context(), new Error('fetch failed'));
  assert.ok(fake.names().includes('updateBranch'));
  assert.ok(fake.names().includes('addLabels'));
  assert.equal(reply(fake)[0], `@maya-rao ${MESSAGES.conflicts('main', null, true)}`);
});

test('a branch that moved during the update is left alone', async () => {
  const fake = fakeGithub({ update: httpError(422, 'expected head sha didn’t match current head ref.') });
  await runWith(fake);
  assert.ok(!fake.names().includes('addLabels'));
  assert.equal(reply(fake)[0], `@maya-rao ${MESSAGES.headMoved}`);
});

test('any other failure on our own repo is labelled for a maintainer', async () => {
  const fake = fakeGithub({ update: httpError(500, 'boom') });
  await runWith(fake);
  assert.ok(fake.names().includes('addLabels'));
  assert.equal(reply(fake)[0], `@maya-rao ${MESSAGES.failed('main')}`);
});

test('checks stay held for a maintainer when the old head never ran without approval', async () => {
  for (const oldRuns of [[], [{ id: 1, name: 'CI', conclusion: 'action_required' }]]) {
    const fake = fakeGithub({ oldRuns });
    await runWith(fake);
    assert.ok(!fake.names().includes('approve'));
    assert.match(reply(fake)[0], /GitHub holds the checks for a commit made by a bot, so a maintainer will start them/);
  }
});

test('nothing is approved or copied for a commit that is not GitHub\'s own merge', async () => {
  for (const merge of [
    { parents: [{ sha: OLD }], committer: { login: 'web-flow' }, commit: { verification: { verified: true } } },
    { parents: [{ sha: MAIN }, { sha: OLD }], committer: { login: 'web-flow' }, commit: { verification: { verified: true } } },
    { parents: [{ sha: OLD }, { sha: MAIN }], committer: { login: 'maya-rao' }, commit: { verification: { verified: true } } },
    { parents: [{ sha: OLD }, { sha: MAIN }], committer: { login: 'web-flow' }, commit: { verification: { verified: false } } },
  ]) {
    const fake = fakeGithub({ merge });
    await runWith(fake);
    assert.ok(!fake.names().includes('approve'));
    assert.ok(!fake.names().includes('createCheck'));
    assert.match(reply(fake)[0], /a maintainer will start them/);
  }
});

test('the CLA check is copied as it was, and not when one is there already or none finished', async () => {
  let fake = fakeGithub({ oldCla: { status: 'completed', conclusion: 'failure', html_url: 'u' } });
  await runWith(fake);
  assert.equal(fake.calls.find((c) => c[0] === 'createCheck')[1].conclusion, 'failure');
  for (const opts of [{ newCla: { status: 'completed', conclusion: 'success' } }, { oldCla: null }, { oldCla: { status: 'in_progress' } }]) {
    fake = fakeGithub(opts);
    await runWith(fake);
    assert.ok(!fake.names().includes('createCheck'));
  }
});

test('a copied CLA check GitHub does not list is reported, not claimed', async () => {
  const fake = fakeGithub({ claListed: false });
  await runWith(fake);
  assert.ok(fake.names().includes('createCheck'));
  assert.equal(reply(fake)[0], `@maya-rao ${MESSAGES.updated('main', NEW, 'noCla')}`);
  assert.match(reply(fake)[0], /the CLA check only runs again with your next push/);
});

test('when the merge commit has not shown up yet, the reply says it is on its way', async () => {
  const fake = fakeGithub();
  fake.github.rest.pulls.get = async () => ({ data: { state: 'open', head: { sha: OLD, repo: { full_name: 'existence-master/Sentient' } }, base: { ref: 'main' } } });
  await runWith(fake);
  assert.equal(reply(fake)[0], `@maya-rao ${MESSAGES.updating('main')}`);
});

test('clearLabel takes the label off once the branch is up to date, and only then', async () => {
  const ctx = (labels) => ({
    repo: { owner: 'existence-master', repo: 'Sentient' },
    payload: { pull_request: { number: 42, labels: labels.map((name) => ({ name })), head: { sha: OLD }, base: { ref: 'main' } } },
  });
  let fake = fakeGithub({ behind: 0 });
  await clearLabel({ github: fake.github, context: ctx(['needs update']), core });
  assert.deepEqual(fake.names(), ['compare', 'removeLabel']);
  fake = fakeGithub({ behind: 2 });
  await clearLabel({ github: fake.github, context: ctx(['needs update']), core });
  assert.deepEqual(fake.names(), ['compare']);
  fake = fakeGithub({ behind: 0 });
  await clearLabel({ github: fake.github, context: ctx(['area: docs']), core });
  assert.deepEqual(fake.names(), []);
});

test('messages are plain: no em-dashes', () => {
  const all = [
    MESSAGES.updated('main', NEW, 'running'), MESSAGES.updated('main', NEW, 'held'), MESSAGES.updated('main', NEW, 'noCla'),
    MESSAGES.updating('main'),
    MESSAGES.upToDate('main'), MESSAGES.conflicts('main', ['a'], true), MESSAGES.conflicts('main', null, false),
    MESSAGES.useButton('main'), MESSAGES.headMoved, MESSAGES.closed, MESSAGES.notAllowed, MESSAGES.failed('main'),
  ];
  for (const m of all) assert.ok(!m.includes('—'), m);
});

// The real conflict finder against a throwaway local repository (no network): a file both sides changed is
// reported, a file only one side changed is not, and a clean merge gives [].
test('gitConflicts lists exactly the conflicting files', (t) => {
  try {
    execFileSync('git', ['merge-tree', '-h'], { stdio: 'ignore' });
  } catch (err) {
    if (err.status !== 129) return t.skip('git is not available');
  }
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'update-test-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const src = path.join(root, 'src');
  const env = { ...process.env, GIT_AUTHOR_NAME: 'Maya Rao', GIT_AUTHOR_EMAIL: 'maya@example.com', GIT_COMMITTER_NAME: 'Maya Rao', GIT_COMMITTER_EMAIL: 'maya@example.com' };
  const g = (...args) => execFileSync('git', ['-C', src, ...args], { env, encoding: 'utf8' });
  const write = (f, s) => fs.writeFileSync(path.join(src, f), s);
  fs.mkdirSync(src);
  g('init', '--quiet', '--initial-branch=main');
  write('a.txt', 'one\n'); write('b.txt', 'one\n'); write('c.txt', 'one\n');
  g('add', '.'); g('commit', '--quiet', '-m', 'start');
  g('switch', '--quiet', '-c', 'feature');
  write('a.txt', 'feature\n'); write('b.txt', 'feature\n');
  g('commit', '--quiet', '-am', 'feature');
  g('update-ref', 'refs/pull/7/head', 'feature');
  g('update-ref', 'refs/pull/8/head', 'main');
  g('switch', '--quiet', 'main');
  write('a.txt', 'main\n'); write('c.txt', 'main\n');
  g('commit', '--quiet', '-am', 'main');

  // gitConflicts fetches from https://github.com/<owner>/<repo>.git; point that at the local repository.
  const prefix = 'https://github.com/';
  const key = 'GIT_CONFIG_COUNT';
  const saved = { [key]: process.env[key], k: process.env.GIT_CONFIG_KEY_0, v: process.env.GIT_CONFIG_VALUE_0 };
  process.env[key] = '1';
  process.env.GIT_CONFIG_KEY_0 = `url.${pathToUrl(root)}/.insteadOf`;
  process.env.GIT_CONFIG_VALUE_0 = prefix;
  t.after(() => {
    for (const [k, v] of [[key, saved[key]], ['GIT_CONFIG_KEY_0', saved.k], ['GIT_CONFIG_VALUE_0', saved.v]]) {
      if (v === undefined) delete process.env[k];
      else process.env[k] = v;
    }
  });
  fs.mkdirSync(path.join(root, 'owner'));
  fs.renameSync(src, path.join(root, 'owner', 'repo.git'));
  const conflicts = run.gitConflicts({ owner: 'owner', repo: 'repo', number: 7, base: 'main', dir: path.join(root, 'check') });
  assert.deepEqual(conflicts, ['a.txt']);
  assert.deepEqual(run.gitConflicts({ owner: 'owner', repo: 'repo', number: 8, base: 'main', dir: path.join(root, 'check') }), []);
  assert.throws(() => run.gitConflicts({ owner: 'owner', repo: 'repo', number: '7; rm', base: 'main', dir: path.join(root, 'check') }));
});

function pathToUrl(p) {
  return 'file://' + (p.startsWith('/') ? '' : '/') + p.replace(/\\/g, '/');
}
