// Run with: node --test .github/scripts/
const test = require('node:test');
const assert = require('node:assert/strict');
const run = require('./stale.js');
const { BOT, MARK, LABEL, prWaiting, decidePr, decideClaim, claimFacts } = run;

const DAY = 24 * 60 * 60 * 1000;
const T0 = Date.parse('2026-10-01T12:00:00Z');
const at = (d) => new Date(T0 + d * DAY).toISOString();
const user = (login) => ({ login, type: login.endsWith('[bot]') ? 'Bot' : 'User' });
const review = (login, state, d) => ({ user: user(login), state, submitted_at: at(d) });
const commit = (d) => ({ event: 'committed', committer: { date: at(d) }, author: { date: at(d) } });
const comment = (login, d, body = 'hi') => ({ event: 'commented', actor: user(login), user: user(login), created_at: at(d), body });

test('waiting starts at the latest changes-requested review and ignores bot reviews after it', () => {
  const reviews = [review('itsskofficial', 'CHANGES_REQUESTED', 1), review('coderabbitai[bot]', 'COMMENTED', 2)];
  const s = prWaiting({ author: 'maya', reviews, timeline: [commit(0), comment('maya', 0.5)] });
  assert.deepEqual(s, { waiting: true, since: Date.parse(at(1)), reviewer: 'itsskofficial' });
});

test('a commit, a force push, a comment or a review reply from the author stops the wait', () => {
  const reviews = [review('itsskofficial', 'CHANGES_REQUESTED', 1)];
  for (const event of [
    commit(2),
    { event: 'head_ref_force_pushed', actor: user('maya'), created_at: at(2) },
    comment('maya', 2),
    { event: 'reviewed', user: user('maya'), state: 'commented', submitted_at: at(2) },
    { event: 'line-commented', comments: [{ user: user('maya'), created_at: at(2) }] },
  ]) {
    assert.equal(prWaiting({ author: 'maya', reviews, timeline: [commit(0), event] }).waiting, false, event.event);
  }
  // Someone else commenting does not count.
  assert.equal(prWaiting({ author: 'maya', reviews, timeline: [commit(0), comment('ravi', 2)] }).waiting, true);
});

test('an approval or dismissal after the request means not waiting', () => {
  for (const state of ['APPROVED', 'DISMISSED']) {
    const reviews = [review('itsskofficial', 'CHANGES_REQUESTED', 1), review('Kabeer2004', state, 2)];
    assert.equal(prWaiting({ author: 'maya', reviews, timeline: [commit(0)] }).waiting, false);
  }
  assert.equal(prWaiting({ author: 'maya', reviews: [review('coderabbitai[bot]', 'COMMENTED', 1)], timeline: [] }).waiting, false);
});

test('PR timeline: wait, nudge at 7 days, close at 21 only after a week of warning', () => {
  const since = T0;
  const base = { waiting: true, since, nudgedAt: 0, hasLabel: false };
  assert.equal(decidePr({ ...base, now: T0 + 6.9 * DAY }).action, 'wait');
  assert.match(decidePr({ ...base, now: T0 + 6.9 * DAY }).note, /nudge due 2026-10-08/);
  assert.equal(decidePr({ ...base, now: T0 + 7 * DAY }).action, 'nudge');
  assert.equal(decidePr({ ...base, now: T0 + 30 * DAY }).action, 'nudge'); // never closed without a warning
  const nudged = { ...base, nudgedAt: T0 + 7 * DAY, hasLabel: true };
  assert.equal(decidePr({ ...nudged, now: T0 + 20 * DAY }).action, 'wait');
  assert.equal(decidePr({ ...nudged, now: T0 + 21 * DAY }).action, 'close');
  const lateNudge = { ...nudged, nudgedAt: T0 + 20 * DAY };
  assert.equal(decidePr({ ...lateNudge, now: T0 + 22 * DAY }).action, 'wait');
  assert.equal(decidePr({ ...lateNudge, now: T0 + 27 * DAY }).action, 'close');
});

test('the label comes off when the author is active again', () => {
  assert.equal(decidePr({ waiting: false, since: 0, nudgedAt: 0, hasLabel: true, now: T0 }).action, 'unlabel');
  assert.equal(decidePr({ waiting: false, since: 0, nudgedAt: 0, hasLabel: false, now: T0 }).action, 'none');
});

test('claims: only /assign claims, ping at 14 days, unassign at 21 after a week of warning', () => {
  const base = { viaCommand: true, hasOpenPr: false, lastActive: T0, pingedAt: 0 };
  assert.equal(decideClaim({ ...base, viaCommand: false, now: T0 + 40 * DAY }).action, 'none');
  assert.equal(decideClaim({ ...base, hasOpenPr: true, now: T0 + 40 * DAY }).action, 'none');
  assert.equal(decideClaim({ ...base, now: T0 + 13 * DAY }).action, 'wait');
  assert.equal(decideClaim({ ...base, now: T0 + 14 * DAY }).action, 'ping');
  const pinged = { ...base, pingedAt: T0 + 14 * DAY };
  assert.equal(decideClaim({ ...pinged, now: T0 + 20 * DAY }).action, 'wait');
  assert.equal(decideClaim({ ...pinged, now: T0 + 21 * DAY }).action, 'unassign');
});

test('claim facts read the issue timeline', () => {
  const assigned = (actor, d) => ({ event: 'assigned', actor: user(actor), assignee: user('maya'), created_at: at(d) });
  const viaBot = claimFacts([assigned(BOT, 0), comment('maya', 3)], 'maya');
  assert.equal(viaBot.viaCommand, true);
  assert.equal(viaBot.lastActive, Date.parse(at(3)));
  assert.equal(claimFacts([assigned('itsskofficial', 0)], 'maya').viaCommand, false);
  const pr = { event: 'cross-referenced', source: { issue: { pull_request: {}, state: 'open', user: user('maya') } } };
  assert.equal(claimFacts([assigned(BOT, 0), pr], 'maya').hasOpenPr, true);
  const closedPr = { event: 'cross-referenced', source: { issue: { pull_request: {}, state: 'closed', user: user('maya') } } };
  assert.equal(claimFacts([assigned(BOT, 0), closedPr], 'maya').hasOpenPr, false);
  const ping = comment(BOT, 15, `Hi @maya, checking in\n\n${MARK.claimPing}`);
  assert.equal(claimFacts([assigned(BOT, 0), ping], 'maya').pingedAt, Date.parse(at(15)));
  // A ping from before her last comment no longer counts.
  assert.equal(claimFacts([assigned(BOT, 0), ping, comment('maya', 16)], 'maya').pingedAt, 0);
});

// A fake GitHub with one PR per situation and one claimed issue.
function fakeGithub() {
  const writes = [];
  const pulls = [
    { number: 1, user: user('maya'), draft: false, labels: [] }, // waiting 8 days: nudge
    { number: 2, user: user('itsskofficial'), draft: false, labels: [] }, // maintainer: skipped
    { number: 3, user: user('ravi'), draft: true, labels: [] }, // draft: skipped
    { number: 4, user: user('ravi'), draft: false, labels: [{ name: LABEL.name }] }, // pushed since: unlabel
    { number: 5, user: user('dependabot[bot]'), draft: false, labels: [] }, // bot: skipped
    { number: 6, user: user('lena'), draft: false, labels: [{ name: LABEL.name }] }, // nudged long ago: close
  ];
  const reviews = {
    1: [review('itsskofficial', 'CHANGES_REQUESTED', 0)],
    2: [review('Kabeer2004', 'CHANGES_REQUESTED', 0)],
    3: [review('itsskofficial', 'CHANGES_REQUESTED', 0)],
    4: [review('itsskofficial', 'CHANGES_REQUESTED', 0)],
    5: [review('itsskofficial', 'CHANGES_REQUESTED', 0)],
    6: [review('itsskofficial', 'CHANGES_REQUESTED', -14)],
  };
  const timelines = {
    1: [commit(-1)],
    4: [commit(-1), commit(2)],
    6: [commit(-15), comment(BOT, -7, `Hi @lena\n\n${MARK.nudge}`)],
    10: [{ event: 'assigned', actor: user(BOT), assignee: user('omar'), created_at: at(-7) }],
    11: [{ event: 'assigned', actor: user('itsskofficial'), assignee: user('omar'), created_at: at(-30) }],
  };
  const issues = [
    { number: 10, assignees: [user('omar')] }, // claimed with /assign 15 days ago: ping
    { number: 11, assignees: [user('omar')] }, // assigned by a maintainer: left alone
    { number: 12, assignees: [user('Kabeer2004')] }, // maintainer: skipped
    { number: 6, assignees: [user('lena')], pull_request: {} }, // a PR in the issues list: skipped
  ];
  const record = (name) => async (args) => writes.push([name, args]);
  const github = {
    paginate: async (fn, args) => fn(args),
    graphql: async () => ({ repository: { pullRequest: { closingIssuesReferences: { nodes: [
      { number: 20, assignees: { nodes: [{ login: 'lena' }] } },
      { number: 21, assignees: { nodes: [{ login: 'someone-else' }] } },
    ] } } } }),
    rest: {
      pulls: {
        list: async () => pulls,
        listReviews: async ({ pull_number }) => reviews[pull_number] || [],
        update: record('pulls.update'),
      },
      issues: {
        listEventsForTimeline: async ({ issue_number }) => timelines[issue_number] || [],
        listForRepo: async () => issues,
        createLabel: record('createLabel'),
        addLabels: record('addLabels'),
        removeLabel: record('removeLabel'),
        removeAssignees: record('removeAssignees'),
        createComment: record('createComment'),
      },
    },
  };
  return { github, writes };
}

const context = { repo: { owner: 'existence-master', repo: 'Sentient' } };
const quietCore = { info: () => {} };
const NOW = T0 + 8 * DAY;

test('a dry run decides everything and changes nothing', async () => {
  const { github, writes } = fakeGithub();
  const lines = [];
  const rows = await run({ github, context, core: { info: (l) => lines.push(l) }, dryRun: true, now: NOW });
  assert.deepEqual(writes, []);
  const actions = Object.fromEntries(rows.map((r) => [`${r[0]} ${r[1]}`, r[3]]));
  assert.deepEqual(actions, {
    'PR #1': 'nudge',
    'PR #3': 'none',
    'PR #4': 'unlabel',
    'PR #6': 'close',
    'issue #10': 'ping',
    'issue #11': 'none',
  });
  assert.ok(lines.some((l) => l.startsWith('[dry run] would nudge PR #1 (maya)')));
});

test('a live run labels, nudges, closes, unassigns and pings', async () => {
  const { github, writes } = fakeGithub();
  await run({ github, context, core: quietCore, dryRun: false, now: T0 + 14 * DAY });
  const names = writes.map((w) => `${w[0]} ${w[1].issue_number || w[1].pull_number}`);
  assert.deepEqual(names, [
    'createLabel undefined',
    'addLabels 1',
    'createComment 1',
    'removeLabel 4',
    'removeAssignees 20',
    'createComment 6',
    'pulls.update 6',
    'createComment 10',
  ]);
  const nudge = writes.find((w) => w[0] === 'createComment' && w[1].issue_number === 1)[1].body;
  assert.match(nudge, /^Hi @maya, just a friendly check-in/);
  assert.ok(nudge.includes(MARK.nudge));
  const close = writes.find((w) => w[0] === 'createComment' && w[1].issue_number === 6)[1].body;
  assert.match(close, /You can reopen it any time/);
  assert.match(close, /unassigned you from #20/);
  assert.deepEqual(writes.find((w) => w[0] === 'removeAssignees')[1].assignees, ['lena']);
  assert.match(writes.at(-1)[1].body, /^Hi @omar, checking in on this one/);
});
