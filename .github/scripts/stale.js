// Daily housekeeping for contributor pull requests and issue claims (see .github/workflows/stale.yaml).
//
// Pull requests (not maintainers', not drafts) whose latest review asks for changes, with no new commit or
// comment from the author since:
//   7 days  -> label "waiting on author" and one friendly nudge
//   21 days -> close with a kind note (only if the nudge went out at least 7 days earlier) and unassign the
//              author from the issues the PR closes
//   author pushes or comments -> the label comes off
// Issues claimed with /assign by someone who isn't a maintainer, with no open PR from them and no comment from
// them since:
//   14 days -> a friendly ping
//   21 days -> unassigned with a note (only if the ping went out at least 7 days earlier)
//
// With dryRun every decision is logged and nothing on GitHub changes.

const MAINTAINERS = ['itsskofficial', 'kabeer2004', 'dependabot', 'dependabot[bot]'];
const BOT = 'github-actions[bot]';
const LABEL = { name: 'waiting on author', color: 'fef2c0', description: 'A maintainer asked for changes; waiting for the author' };
const DAY = 24 * 60 * 60 * 1000;
const LIMITS = { nudge: 7, close: 21, claimPing: 14, claimDrop: 21, grace: 7 };
const MARK = {
  nudge: '<!-- sentient-bot:waiting-on-author -->',
  close: '<!-- sentient-bot:closed-waiting-on-author -->',
  claimPing: '<!-- sentient-bot:claim-ping -->',
  claimDrop: '<!-- sentient-bot:claim-dropped -->',
};

const MESSAGES = {
  nudge: (author) =>
    `Hi @${author}, just a friendly check-in. A maintainer asked for a few changes here about a week ago. ` +
    'Whenever you are ready, push an update or reply with any questions, and we are happy to help. ' +
    'If you no longer have time for it, that is completely fine too, just let us know.\n\n' + MARK.nudge,
  close: (author, issues) =>
    `Hi @${author}, we are closing this for now because it has been waiting on changes for three weeks. ` +
    'Thank you for the work you put into it! You can reopen it any time, or comment here and we will reopen it ' +
    'for you.' +
    (issues.length
      ? ` We have unassigned you from ${issues.map((n) => `#${n}`).join(', ')} so someone else can pick it up; ` +
        'comment `/assign` there if you want it back while it is still free.'
      : '') +
    '\n\n' + MARK.close,
  claimPing: (user) =>
    `Hi @${user}, checking in on this one. Are you still working on it? A quick comment or a draft PR is ` +
    'perfect. If you have moved on, comment `/unassign` so someone else can pick it up. No pressure either way.' +
    '\n\n' + MARK.claimPing,
  claimDrop: (user) =>
    `Hi @${user}, we have unassigned you because we have not heard back in a while, so others can pick this up. ` +
    'Thanks for your interest! If you want it back, comment `/assign` while it is still free.' +
    '\n\n' + MARK.claimDrop,
};

const lower = (s) => String(s || '').toLowerCase();
const isMaintainer = (login) => MAINTAINERS.includes(lower(login));
const isBot = (user) => !user || user.type === 'Bot' || lower(user.login).endsWith('[bot]');
const time = (s) => (s ? new Date(s).getTime() : 0);
const days = (ms) => Math.floor((ms / DAY) * 10) / 10;
const dateOnly = (ms) => new Date(ms).toISOString().slice(0, 10);
const actorOf = (e) => lower((e.actor && e.actor.login) || (e.user && e.user.login));

/** The latest review that approved, dismissed or asked for changes, from someone other than the author or a bot. */
function latestDecisiveReview(reviews, author) {
  const decisive = reviews
    .filter((r) => ['APPROVED', 'CHANGES_REQUESTED', 'DISMISSED'].includes(r.state))
    .filter((r) => !isBot(r.user) && lower(r.user.login) !== lower(author))
    .sort((a, b) => time(a.submitted_at) - time(b.submitted_at));
  return decisive[decisive.length - 1] || null;
}

/** The last time the author pushed or commented, from the PR timeline. Any new commit or a reopen counts. */
function lastAuthorActivity(timeline, author) {
  const me = lower(author);
  let last = 0;
  for (const e of timeline) {
    let at = 0;
    if (e.event === 'committed') at = time((e.committer && e.committer.date) || (e.author && e.author.date));
    else if (e.event === 'head_ref_force_pushed' && actorOf(e) === me) at = time(e.created_at);
    else if (e.event === 'reopened') at = time(e.created_at); // reopened by anyone: it is active again
    else if (e.event === 'commented' && actorOf(e) === me) at = time(e.created_at);
    else if (e.event === 'reviewed' && actorOf(e) === me) at = time(e.submitted_at);
    else if (e.event === 'line-commented') {
      for (const c of e.comments || []) if (lower(c.user && c.user.login) === me) at = Math.max(at, time(c.created_at));
    }
    last = Math.max(last, at);
  }
  return last;
}

/** When the bot last posted a comment carrying `mark` after `since` (0 if never). */
function botCommentAfter(timeline, mark, since, mention) {
  let at = 0;
  for (const e of timeline) {
    if (e.event !== 'commented' || actorOf(e) !== BOT) continue;
    const body = String(e.body || '');
    if (!body.includes(mark) || (mention && !lower(body).includes(`@${lower(mention)}`))) continue;
    if (time(e.created_at) > since) at = Math.max(at, time(e.created_at));
  }
  return at;
}

/** Whether a PR counts as waiting on its author, and since when. */
function prWaiting({ author, reviews, timeline }) {
  const review = latestDecisiveReview(reviews, author);
  if (!review || review.state !== 'CHANGES_REQUESTED') return { waiting: false, since: 0, reviewer: null };
  const since = time(review.submitted_at);
  const waiting = lastAuthorActivity(timeline, author) <= since;
  return { waiting, since, reviewer: review.user.login };
}

/** Pure decision for one PR. Returns { action, note }. Actions: none, unlabel, wait, nudge, close. */
function decidePr({ waiting, since, nudgedAt, hasLabel, now }) {
  if (!waiting) return hasLabel ? { action: 'unlabel', note: 'author is active again' } : { action: 'none', note: '' };
  const waited = (now - since) / DAY;
  if (!nudgedAt) {
    if (waited >= LIMITS.nudge) return { action: 'nudge', note: `waiting ${days(now - since)} days` };
    return { action: 'wait', note: `waiting ${days(now - since)} days; nudge due ${dateOnly(since + LIMITS.nudge * DAY)}` };
  }
  const closeAt = Math.max(since + LIMITS.close * DAY, nudgedAt + LIMITS.grace * DAY);
  if (now >= closeAt) return { action: 'close', note: `waiting ${days(now - since)} days, nudged ${dateOnly(nudgedAt)}` };
  return { action: 'wait', note: `waiting ${days(now - since)} days, nudged; close due ${dateOnly(closeAt)}` };
}

/** Pure decision for one claim. Actions: none, wait, ping, unassign. */
function decideClaim({ viaCommand, hasOpenPr, lastActive, pingedAt, now }) {
  if (!viaCommand) return { action: 'none', note: 'not claimed with /assign' };
  if (hasOpenPr) return { action: 'none', note: 'has an open pull request' };
  const quiet = (now - lastActive) / DAY;
  if (!pingedAt) {
    if (quiet >= LIMITS.claimPing) return { action: 'ping', note: `quiet ${days(now - lastActive)} days` };
    return { action: 'wait', note: `quiet ${days(now - lastActive)} days; ping due ${dateOnly(lastActive + LIMITS.claimPing * DAY)}` };
  }
  const dropAt = Math.max(lastActive + LIMITS.claimDrop * DAY, pingedAt + LIMITS.grace * DAY);
  if (now >= dropAt) return { action: 'unassign', note: `quiet ${days(now - lastActive)} days, pinged ${dateOnly(pingedAt)}` };
  return { action: 'wait', note: `quiet ${days(now - lastActive)} days, pinged; unassign due ${dateOnly(dropAt)}` };
}

/** Facts about one assignee of an issue, from the issue timeline. */
function claimFacts(timeline, user) {
  const me = lower(user);
  let assignedAt = 0;
  let viaCommand = false;
  let lastComment = 0;
  let hasOpenPr = false;
  for (const e of timeline) {
    if (e.event === 'assigned' && lower(e.assignee && e.assignee.login) === me && time(e.created_at) >= assignedAt) {
      assignedAt = time(e.created_at);
      viaCommand = actorOf(e) === BOT;
    } else if (e.event === 'commented' && actorOf(e) === me) {
      lastComment = Math.max(lastComment, time(e.created_at));
    } else if (e.event === 'cross-referenced') {
      const src = e.source && e.source.issue;
      if (src && src.pull_request && src.state === 'open' && lower(src.user && src.user.login) === me) hasOpenPr = true;
    }
  }
  const lastActive = Math.max(assignedAt, lastComment);
  return { viaCommand, hasOpenPr, lastActive, pingedAt: botCommentAfter(timeline, MARK.claimPing, lastActive, user) };
}

async function linkedIssues(github, owner, repo, number) {
  const q = `query($owner:String!,$repo:String!,$n:Int!){repository(owner:$owner,name:$repo){pullRequest(number:$n){
    closingIssuesReferences(first:20){nodes{number assignees(first:20){nodes{login}}}}}}}`;
  const res = await github.graphql(q, { owner, repo, n: number });
  return res.repository.pullRequest.closingIssuesReferences.nodes.map((i) => ({
    number: i.number,
    assignees: i.assignees.nodes.map((a) => a.login),
  }));
}

/** Entry point for actions/github-script. */
async function run({ github, context, core, dryRun = true, now = Date.now() }) {
  const { owner, repo } = context.repo;
  const rows = [];
  const say = (kind, number, who, action, note) => {
    rows.push([kind, `#${number}`, who, action, note]);
    core.info(`${dryRun && !['none', 'wait'].includes(action) ? '[dry run] would ' : ''}${action} ${kind} #${number} (${who}): ${note}`);
  };
  const write = async (fn) => {
    if (!dryRun) await fn();
  };
  core.info(dryRun ? 'Dry run: nothing on GitHub will change.' : 'Live run.');
  const timelineOf = (issue_number) =>
    github.paginate(github.rest.issues.listEventsForTimeline, { owner, repo, issue_number, per_page: 100 });

  let labelReady = false;
  const ensureLabel = async () => {
    if (labelReady || dryRun) return;
    try {
      await github.rest.issues.createLabel({ owner, repo, ...LABEL });
    } catch (err) {
      if (err.status !== 422) throw err; // 422: it already exists
    }
    labelReady = true;
  };

  const pulls = await github.paginate(github.rest.pulls.list, { owner, repo, state: 'open', per_page: 100 });
  for (const pr of pulls) {
    const author = pr.user.login;
    if (isMaintainer(author) || isBot(pr.user)) continue;
    if (pr.draft) {
      say('PR', pr.number, author, 'none', 'draft');
      continue;
    }
    const [reviews, timeline] = await Promise.all([
      github.paginate(github.rest.pulls.listReviews, { owner, repo, pull_number: pr.number, per_page: 100 }),
      timelineOf(pr.number),
    ]);
    const state = prWaiting({ author, reviews, timeline });
    const hasLabel = (pr.labels || []).some((l) => lower(l.name) === LABEL.name);
    const nudgedAt = state.waiting ? botCommentAfter(timeline, MARK.nudge, state.since) : 0;
    const { action, note } = decidePr({ ...state, nudgedAt, hasLabel, now });
    let reason = note || 'no open request for changes';
    if (state.reviewer) {
      const asked = `changes asked by ${state.reviewer} on ${dateOnly(state.since)}`;
      reason = state.waiting ? `${note} (${asked})` : `author pushed or replied since ${asked}`;
    }
    say('PR', pr.number, author, action, reason);
    const issue_number = pr.number;
    if (action === 'unlabel') {
      await write(() => github.rest.issues.removeLabel({ owner, repo, issue_number, name: LABEL.name }));
    } else if (action === 'nudge') {
      await write(async () => {
        await ensureLabel();
        await github.rest.issues.addLabels({ owner, repo, issue_number, labels: [LABEL.name] });
        await github.rest.issues.createComment({ owner, repo, issue_number, body: MESSAGES.nudge(author) });
      });
    } else if (action === 'close') {
      const issues = (await linkedIssues(github, owner, repo, pr.number)).filter((i) =>
        i.assignees.some((a) => lower(a) === lower(author)),
      );
      if (issues.length) core.info(`  linked issues ${author} would be unassigned from: ${issues.map((i) => `#${i.number}`).join(', ')}`);
      await write(async () => {
        for (const i of issues) {
          await github.rest.issues.removeAssignees({ owner, repo, issue_number: i.number, assignees: [author] });
        }
        await github.rest.issues.createComment({ owner, repo, issue_number, body: MESSAGES.close(author, issues.map((i) => i.number)) });
        await github.rest.pulls.update({ owner, repo, pull_number: pr.number, state: 'closed' });
      });
    }
  }

  const assigned = await github.paginate(github.rest.issues.listForRepo, { owner, repo, state: 'open', assignee: '*', per_page: 100 });
  for (const issue of assigned) {
    if (issue.pull_request) continue;
    const people = (issue.assignees || []).filter((a) => !isMaintainer(a.login) && !isBot(a));
    if (!people.length) continue;
    const timeline = await timelineOf(issue.number);
    for (const person of people) {
      const user = person.login;
      const { action, note } = decideClaim({ ...claimFacts(timeline, user), now });
      say('issue', issue.number, user, action, note);
      const issue_number = issue.number;
      if (action === 'ping') {
        await write(() => github.rest.issues.createComment({ owner, repo, issue_number, body: MESSAGES.claimPing(user) }));
      } else if (action === 'unassign') {
        await write(async () => {
          await github.rest.issues.removeAssignees({ owner, repo, issue_number, assignees: [user] });
          await github.rest.issues.createComment({ owner, repo, issue_number, body: MESSAGES.claimDrop(user) });
        });
      }
    }
  }

  if (core.summary) {
    await core.summary
      .addHeading(dryRun ? 'Stale check (dry run, nothing changed)' : 'Stale check')
      .addTable([
        [{ data: 'Kind', header: true }, { data: 'Number', header: true }, { data: 'Who', header: true },
          { data: 'Action', header: true }, { data: 'Why', header: true }],
        ...rows,
      ])
      .write();
  }
  return rows;
}

module.exports = run;
Object.assign(module.exports, {
  MAINTAINERS, BOT, LABEL, LIMITS, MARK, MESSAGES,
  latestDecisiveReview, lastAuthorActivity, botCommentAfter, prWaiting, decidePr, decideClaim, claimFacts,
});
