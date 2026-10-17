// Handles "/assign" and "/unassign" comments on issues (see .github/workflows/assign.yaml).
// It only compares the comment text with fixed strings: nothing from the comment is ever run.

const MESSAGES = {
  assigned: 'Assigned to you. Open a PR within a week or comment if you need more time.',
  alreadyYours: 'This one is already assigned to you. Open a PR within a week or comment if you need more time.',
  taken: (owners) =>
    `Thanks for offering to help! ${owners} is already working on this one. ` +
    'Have a look at the issues labeled `good first issue` or `help wanted` for something free, ' +
    'or comment here if you would like to team up.',
  unassigned: 'Unassigned you. Thanks for letting us know! The issue is free for someone else now.',
  notYours: (owners) =>
    `Only the person working on this (${owners}) can use \`/unassign\` here. ` +
    'If something is stuck, a maintainer can help.',
  notAssigned: 'Nobody is assigned to this issue, so there is nothing to unassign. Comment `/assign` to take it.',
  closed: 'This issue is closed, so there is nothing to pick up here. Thanks for offering!',
  failed: 'Sorry, I could not assign you. A maintainer will take a look.',
};

/** Returns "assign", "unassign" or null. The whole comment must be the command. */
function parseCommand(body) {
  const text = String(body || '').trim();
  if (text === '/assign') return 'assign';
  if (text === '/unassign') return 'unassign';
  return null;
}

function mentionAll(logins) {
  return logins.map((login) => `@${login}`).join(', ');
}

/**
 * Decides what to do, without touching GitHub.
 * Returns { action: "assign" | "unassign" | "none", reply: string | null }.
 */
function decide({ command, commenter, assignees, issueState }) {
  const owners = assignees.map((a) => a.toLowerCase());
  const me = commenter.toLowerCase();
  if (issueState !== 'open') return { action: 'none', reply: MESSAGES.closed };
  if (command === 'assign') {
    if (owners.includes(me)) return { action: 'none', reply: MESSAGES.alreadyYours };
    if (assignees.length > 0) return { action: 'none', reply: MESSAGES.taken(mentionAll(assignees)) };
    return { action: 'assign', reply: MESSAGES.assigned };
  }
  if (command === 'unassign') {
    if (owners.includes(me)) return { action: 'unassign', reply: MESSAGES.unassigned };
    if (assignees.length > 0) return { action: 'none', reply: MESSAGES.notYours(mentionAll(assignees)) };
    return { action: 'none', reply: MESSAGES.notAssigned };
  }
  return { action: 'none', reply: null };
}

/** Entry point for actions/github-script: `await require('./.github/scripts/assign.js')({ github, context, core })`. */
async function run({ github, context, core }) {
  const { issue, comment } = context.payload;
  if (!issue || !comment) return core.info('Not an issue comment.');
  if (issue.pull_request) return core.info('Comment is on a pull request: ignored.');
  if (comment.user.type === 'Bot') return core.info('Comment is from a bot: ignored.');
  const command = parseCommand(comment.body);
  if (!command) return core.info('Not a command.');

  const { owner, repo } = context.repo;
  const issue_number = issue.number;
  const commenter = comment.user.login;
  // The payload is a snapshot from when the comment was made; read the issue now so two quick claims see each other.
  const { data: current } = await github.rest.issues.get({ owner, repo, issue_number });
  const assignees = (current.assignees || []).map((a) => a.login);
  const { action, reply } = decide({ command, commenter, assignees, issueState: current.state });
  let text = reply;

  if (action === 'assign') {
    const { data } = await github.rest.issues.addAssignees({ owner, repo, issue_number, assignees: [commenter] });
    // GitHub silently drops people it can't assign, so check that it worked.
    const now = (data.assignees || []).map((a) => a.login.toLowerCase());
    if (!now.includes(commenter.toLowerCase())) text = MESSAGES.failed;
  } else if (action === 'unassign') {
    await github.rest.issues.removeAssignees({ owner, repo, issue_number, assignees: [commenter] });
  }
  core.info(`/${command} by ${commenter} on #${issue_number}: ${action}`);
  if (text) await github.rest.issues.createComment({ owner, repo, issue_number, body: `@${commenter} ${text}` });
}

module.exports = run;
module.exports.parseCommand = parseCommand;
module.exports.decide = decide;
module.exports.MESSAGES = MESSAGES;
