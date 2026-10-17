// Run with: node --test .github/scripts/
const test = require('node:test');
const assert = require('node:assert/strict');
const run = require('./assign.js');
const { parseCommand, decide, MESSAGES } = run;

function fakeGithub({ assignable = true } = {}) {
  const calls = [];
  const github = {
    rest: {
      issues: {
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
  await run({ github, context: context({ body: '/assign' }), core });
  assert.deepEqual(calls[0], ['addAssignees', { owner: 'existence-master', repo: 'Sentient', issue_number: 42, assignees: ['maya-rao'] }]);
  assert.equal(calls[1][0], 'createComment');
  assert.equal(calls[1][1].body, `@maya-rao ${MESSAGES.assigned}`);
  assert.equal(calls.length, 2);
});

test('/assign on a taken issue only replies with who has it', async () => {
  const { github, calls } = fakeGithub();
  await run({ github, context: context({ body: '/assign', assignees: ['ravi'] }), core });
  assert.equal(calls.length, 1);
  assert.equal(calls[0][0], 'createComment');
  assert.match(calls[0][1].body, /^@maya-rao Thanks for offering to help! @ravi is already working on this one/);
});

test('/unassign by the assignee removes them', async () => {
  const { github, calls } = fakeGithub();
  await run({ github, context: context({ body: '/unassign', assignees: ['maya-rao'] }), core });
  assert.deepEqual(calls[0], ['removeAssignees', { owner: 'existence-master', repo: 'Sentient', issue_number: 42, assignees: ['maya-rao'] }]);
  assert.equal(calls[1][1].body, `@maya-rao ${MESSAGES.unassigned}`);
});

test('/unassign by someone else changes nothing', async () => {
  const { github, calls } = fakeGithub();
  await run({ github, context: context({ body: '/unassign', assignees: ['ravi'] }), core });
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
    await run({ github, context: ctx, core });
    assert.equal(calls.length, 0);
  }
});

test('a silently dropped assignment is reported, not claimed', async () => {
  const { github, calls } = fakeGithub({ assignable: false });
  await run({ github, context: context({ body: '/assign' }), core });
  assert.equal(calls[1][1].body, `@maya-rao ${MESSAGES.failed}`);
});

test('the comment text never reaches the API or a reply', async () => {
  const { github, calls } = fakeGithub();
  const body = '/assign\n$(curl evil) ${{ secrets.X }}';
  await run({ github, context: context({ body }), core });
  assert.equal(calls.length, 0);
});
