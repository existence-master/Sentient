/**
 * Dev-only demo data for desktop agent B screens, used to design and screenshot them before the engine
 * packages land. Enabled by `demo=1` in the hash route (`#/about?demo=1`) and never in a packaged build.
 *
 * - apiB calls answer from here (see api-b.ts).
 * - installLeapDemo() also adds (never replaces) clearly-labelled demo rows with demo-only ids to caches owned
 *   by lib/api.ts (a script job task, a skill repair proposal, leap notifications) so the Tasks, Skills and
 *   notification screens can be captured before those engine features exist.
 * - Packaged builds load from app.asar and can never enable it.
 */
import type { QueryClient } from '@tanstack/react-query'
import { qk } from '@/hooks/queryKeys'
import type { Notification, NotificationList, SkillDiff, SkillsList, Task } from '@/lib/types'
import type { Dream, FeedStatus, Hook, HookCreated, Insight, InsightDimension, InsightStatus, SandboxResult, ScriptJob, TaskWithScript, UserModel } from './types-b'

let enabled: boolean | null = null

export function isLeapDemo(): boolean {
  if (enabled !== null) return enabled
  try {
    const packaged = window.location.href.includes('app.asar')
    const flag = /[?&]demo=1\b/.test(window.location.hash) || sessionStorage.getItem('sentient.leapDemo') === '1'
    enabled = !packaged && flag
    if (enabled) sessionStorage.setItem('sentient.leapDemo', '1')
  } catch {
    enabled = false
  }
  return enabled
}

const now = Date.now()
const ago = (minutes: number) => new Date(now - minutes * 60_000).toISOString()
const H = 60
const D = 24 * H

// ---------------------------------------------------------------------------- user model
let insights: Insight[] = [
  ins('i1', 'preferences', 'Prefers short, direct answers with the key point first, and details only when asked.', 0.92, 'confirmed', 'inferred', [
    ['feedback', 'You said: “Just give me the answer, skip the preamble.”', 3 * D],
    ['message', '“tl;dr first please”', 9 * D]
  ]),
  ins('i2', 'preferences', 'Likes working with dark themes and quiet, minimal interfaces.', 0.64, 'active', 'inferred', [['fact', 'Switched Sentient, VS Code and Figma to dark mode', 12 * D]]),
  ins('i3', 'communication', 'Writes to investors formally, but keeps messages to the team casual and brief.', 0.78, 'active', 'inferred', [
    ['message', '“Hi Priya, thank you for the thoughtful notes on the deck.”', 2 * D],
    ['message', '“yo, pushing the fix tonight 👍”', 1 * D]
  ]),
  ins('i4', 'goals', 'Wants to ship the Sentient desktop beta before the end of September.', 0.88, 'confirmed', 'user', [['summary', 'Planning chat about the September beta milestone', 5 * D]]),
  ins('i5', 'goals', 'Is training for a 10 km run in December.', 0.55, 'active', 'inferred', [['fact', 'Logged 4 runs this week, longest 6.2 km', 2 * D]]),
  ins('i6', 'routines', 'Does deep work early, usually from 6 to 10 in the morning, and keeps meetings after lunch.', 0.81, 'active', 'inferred', [
    ['summary', 'Most focused sessions start between 6 and 7 AM', 4 * D],
    ['fact', 'Declined three morning meetings last week', 6 * D]
  ]),
  ins('i7', 'relationships', 'Calls his mother most Sundays and plans family events well ahead.', 0.7, 'active', 'inferred', [['message', '“Remind me to call Mom on Sunday evening”', 7 * D]]),
  ins('i8', 'values', 'Cares about privacy and prefers tools that keep data on his own computer.', 0.9, 'confirmed', 'inferred', [
    ['message', '“I don’t want my notes going to someone else’s server.”', 20 * D],
    ['fact', 'Chose local models for memory and voice', 14 * D]
  ]),
  ins('i9', 'work_style', 'Thinks in checklists and likes plans broken into small steps he can approve.', 0.74, 'active', 'inferred', [['summary', 'Asked for step-by-step plans in 6 recent tasks', 3 * D]]),
  ins('i10', 'dislikes', 'Does not like being interrupted with notifications during focus hours.', 0.42, 'disputed', 'inferred', [['feedback', 'Dismissed 4 suggestions sent before 10 AM', 2 * D]]),
  ins('i11', 'context', 'Is a founder in Pune building a local-first AI assistant with a small team.', 0.95, 'confirmed', 'user', [['fact', 'Onboarding: professional context', 30 * D]])
]

function ins(id: string, dimension: InsightDimension, statement: string, confidence: number, status: InsightStatus, source: 'inferred' | 'user', ev: Array<[string, string, number]>): Insight {
  return {
    id,
    dimension,
    statement,
    confidence,
    status,
    source,
    evidence: ev.map(([kind, quote, minutes], i) => ({ kind, ref: `${id}-${i}`, quote, at: ago(minutes) })),
    created_at: ago(40 * D),
    updated_at: ago(ev[0]?.[2] ?? D)
  }
}

let questions = [
  { id: 'q1', question: 'Should I keep suggestions quiet until 10 AM so your mornings stay free for deep work?', insight_id: 'i10', created_at: ago(5 * H) },
  { id: 'q2', question: 'You mentioned a 10 km run in December. Would you like me to plan training around your calendar?', insight_id: 'i5', created_at: ago(26 * H) }
]

const SUMMARY =
  'Sarthak is a founder in Pune building a private, local-first assistant, and the September beta is front of mind. ' +
  'He does his best thinking early in the morning, likes answers short and plans in small, approvable steps, and ' +
  'protects his data and his focus. Outside work he is training for a December 10 km and stays close to his family.'

// ---------------------------------------------------------------------------- dreams
let dreams: Dream[] = [
  dream('d1', 9 * H, 'schedule', { facts_reviewed: 146, merged: 6, contradictions_resolved: 2, promoted: 3, expired: 5, insights_updated: 4 },
    'Last night I went through **146 memories** from the past week.\n\n' +
      'I noticed I had written down your Thursday investor call three different ways, so I merged them into one. ' +
      'You told me in April that you live in Mumbai, but everything since says **Pune**, so I kept Pune and let the old note go.\n\n' +
      'Your morning runs have become a habit, so I moved “training for a 10 km” from a passing mention into something I keep in mind. ' +
      'I also let five short-term reminders fade now that their dates have passed.\n\n' +
      'One thing I am unsure about: you dismissed several suggestions early in the day. I added a question for you rather than guessing.'),
  dream('d2', D + 9 * H, 'manual', { facts_reviewed: 38, merged: 1, contradictions_resolved: 0, promoted: 1, expired: 2, insights_updated: 1 },
    'You asked me to tidy up after importing your resume. I found **38 new memories**, merged a duplicate about your ' +
      'time at Existence, and noted that you now lead a small team.'),
  dream('d3', 2 * D + 9 * H, 'schedule', { facts_reviewed: 121, merged: 3, contradictions_resolved: 1, promoted: 2, expired: 7, insights_updated: 2 },
    'A quiet night. I merged three notes about the Goa trip, settled when your sister’s birthday is (it is the 14th, not the 4th), ' +
      'and cleared seven reminders that were done.'),
  { ...dream('d4', 3 * D + 9 * H, 'schedule', {}, ''), status: 'error', error: 'The memory model was not reachable, so I will try again tonight.' }
]

function dream(id: string, minutesAgo: number, trigger: 'schedule' | 'manual', stats: Dream['stats'], journal: string): Dream {
  return { id, started_at: ago(minutesAgo), finished_at: ago(minutesAgo - 4), status: 'completed', trigger, stats, journal_md: journal, error: null }
}

// ---------------------------------------------------------------------------- hooks
let hooks: Hook[] = [
  { id: 'hk_7f3a91', name: 'Shopify orders', url: '/hooks/hk_7f3a91', created_at: ago(12 * D), last_called_at: ago(38), calls: 214 },
  { id: 'hk_22c0de', name: 'Home Assistant', url: '/hooks/hk_22c0de', created_at: ago(4 * D), last_called_at: ago(2 * D), calls: 9 },
  { id: 'hk_b81e44', name: 'Zapier: new form entry', url: '/hooks/hk_b81e44', created_at: ago(3 * H), last_called_at: null, calls: 0 }
]

const feed = (source: string, display_name: string, kind: string, status: string, last: number, last_error: string | null = null): FeedStatus => ({
  source,
  display_name,
  kind,
  connected: true,
  active: status === 'ok',
  status,
  last_sync_at: ago(last),
  last_success_at: ago(status === 'ok' ? last : 3 * H),
  last_error,
  note: null,
  failures: last_error ? 2 : 0,
  next_attempt_at: null,
  emitted: 12
})

export const DEMO_FEEDS: FeedStatus[] = [
  feed('gmail', 'Gmail', 'gmail_history', 'ok', 1),
  feed('gcalendar', 'Google Calendar', 'calendar_sync_token', 'ok', 1),
  feed('email_imap', 'Email (IMAP)', 'imap_idle', 'error', 4, 'The mail server closed the connection. Trying again in a minute.')
]

// ---------------------------------------------------------------------------- script jobs
export const DEMO_SCRIPT_TASK_ID = 'demo-script-watch'

const SCRIPT_CODE = `"""Watch the price of the Sony WH-1000XM6 and alert below the target."""
from sentient_tools import tools, result

TARGET = 25_000  # rupees
URL = "https://www.amazon.in/dp/B0DX4Q6ZR1"

page = tools.web_fetch(url=URL)
text = page.get("text", "")

price = None
for line in text.splitlines():
    if "₹" in line and "M.R.P" not in line:
        digits = "".join(ch for ch in line if ch.isdigit())
        if digits:
            price = int(digits[:6])
            break

if price is None:
    raise RuntimeError("Couldn't find the price on the page")

print(f"Current price: ₹{price:,}")
result({
    "alert": price < TARGET,
    "price": price,
    "message": f"The Sony WH-1000XM6 is now ₹{price:,}, below your ₹{TARGET:,} target.",
})
`

function scriptTask(id: string, patch: Partial<TaskWithScript>): TaskWithScript {
  const base: TaskWithScript = {
    task_id: id,
    name: 'Tell me when the Sony headphones drop below ₹25,000',
    description: 'Check the Amazon price of the Sony WH-1000XM6 every morning and tell me when it drops below ₹25,000.',
    status: 'active',
    priority: 1,
    assignee: 'ai',
    task_type: 'single',
    schedule: { type: 'recurring', frequency: 'daily', time: '09:00', timezone: 'Asia/Kolkata' },
    plan: [{ tool: 'web', description: 'Fetch the product page and read the current price' }],
    runs: [],
    chat_history: [],
    clarifying_questions: [],
    swarm_details: null,
    enabled: true,
    model: null,
    original_context: { source: 'chat' },
    error: null,
    next_execution_at: new Date(now + 14 * H * 60_000).toISOString(),
    last_execution_at: ago(10 * H),
    created_at: ago(6 * D),
    updated_at: ago(10 * H),
    script: {
      code: SCRIPT_CODE,
      condition: 'alert',
      then: 'notify',
      last_result: { alert: false, price: 26_490, message: 'The Sony WH-1000XM6 is now ₹26,490, below your ₹25,000 target.' },
      last_run_at: ago(10 * H),
      last_error: null
    },
    ...patch
  }
  ;(base as { task_type: string }).task_type = 'script'
  return base
}

const scriptTasks: TaskWithScript[] = [
  scriptTask(DEMO_SCRIPT_TASK_ID, {}),
  scriptTask('demo-script-status', {
    name: 'Watch the Sentient status page for changes',
    description: 'Every hour, check status.existence.technology and run a task to summarise what changed.',
    schedule: { type: 'recurring', frequency: 'daily', time: '08:00', timezone: 'Asia/Kolkata' },
    status: 'approval_pending',
    last_execution_at: null,
    script: {
      code: 'from sentient_tools import tools, result\n\npage = tools.web_fetch(url="https://status.existence.technology")\nresult(page.get("text", "")[:2000])\n',
      condition: 'changed',
      then: 'run',
      last_result: null,
      last_run_at: null,
      last_error: null
    }
  })
]

// ---------------------------------------------------------------------------- skills repair
const REPAIR_NAME = 'invoice-filing'
const REPAIR_CURRENT = `---
name: invoice-filing
description: Save vendor invoices from Gmail to Drive and log them in the expenses sheet.
---
# Invoice filing

## Procedure
1. Search Gmail for the invoice email and download the PDF attachment.
2. Upload it to Finance/Invoices in Google Drive.
3. Add vendor, amount and due date to the Expenses sheet.

## Pitfalls
- Some vendors send a link instead of an attachment.
`
const REPAIR_PROPOSED = `---
name: invoice-filing
description: Save vendor invoices from Gmail to Drive and log them in the expenses sheet.
---
# Invoice filing

## Procedure
1. Search Gmail for the invoice email and download the PDF attachment.
2. Upload it to Finance/Invoices/<year> in Google Drive, creating the year folder if it is missing.
3. Add vendor, amount and due date to the Expenses sheet.

## Pitfalls
- Some vendors send a link instead of an attachment.
- Drive returns "File not found" when the year folder does not exist yet. Create it first, then upload.
`

// ---------------------------------------------------------------------------- notifications
function note(id: string, kind: string, title: string, message: string, payload: Record<string, unknown>, minutes: number, task_id: string | null = null): Notification {
  return { id, kind: kind as Notification['kind'], title, message, payload: payload as Notification['payload'], task_id, read: false, created_at: ago(minutes) }
}

const demoNotes: Notification[] = [
  note('demo-n-script', 'task', 'Price alert: Sony WH-1000XM6', 'The Sony WH-1000XM6 is now **₹24,490**, below your ₹25,000 target.', { event: 'script_alert', message: 'The Sony WH-1000XM6 is now ₹24,490, below your ₹25,000 target.', result: { price: 24490 } }, 12, DEMO_SCRIPT_TASK_ID),
  note('demo-n-sub', 'info', 'Your helper finished', 'I compared the five CRMs you listed. **HubSpot** and **Zoho** fit best for a team of five.', { event: 'subagent_completed', subagent_id: 'sa_1', session_id: 'demo-session', goal: 'Compare five CRMs for a small team' }, 55),
  note('demo-n-dream', 'info', 'I tidied up my memory overnight', 'I reviewed 146 memories, merged 6 duplicates and settled 2 contradictions.', { event: 'dream_completed', dream_id: 'd1', stats: dreams[0].stats }, 9 * H),
  note('demo-n-repair', 'skill', 'A fix for invoice-filing', 'Filing an invoice failed because a Drive folder was missing. I drafted a fix to **invoice-filing**.', { skill: REPAIR_NAME, action: 'patch', origin: { repair: true, task_id: 'demo-trigger-mail' } }, 3 * H)
]

// ---------------------------------------------------------------------------- mutable demo API
const clone = <T>(v: T): T => JSON.parse(JSON.stringify(v)) as T

export const demo = {
  userModel: (): UserModel => clone({ summary: SUMMARY, updated_at: ago(2 * H), insights, questions }),
  addInsight: (statement: string, dimension: string): Insight => {
    const i: Insight = { id: `i${Date.now()}`, dimension, statement, confidence: 1, status: 'confirmed', source: 'user', evidence: [], created_at: new Date().toISOString(), updated_at: new Date().toISOString() }
    insights = [...insights, i]
    return i
  },
  patchInsight: (id: string, patch: { statement?: string; status?: InsightStatus }): Insight => {
    insights = insights.map((i) => (i.id === id ? { ...i, ...patch, updated_at: new Date().toISOString() } : i))
    return insights.find((i) => i.id === id) as Insight
  },
  deleteInsight: (id: string) => {
    insights = insights.filter((i) => i.id !== id)
    return { ok: true }
  },
  dropQuestion: (id: string) => {
    questions = questions.filter((q) => q.id !== id)
    return { ok: true }
  },
  dreams: (): Dream[] => clone(dreams),
  runDream: (): Dream => {
    const d: Dream = { id: `d${Date.now()}`, started_at: new Date().toISOString(), finished_at: null, status: 'running', trigger: 'manual', stats: {}, journal_md: '', error: null }
    dreams = [d, ...dreams]
    return d
  },
  hooks: (): Hook[] => clone(hooks),
  createHook: (name: string): HookCreated => {
    const id = `hk_${Math.random().toString(16).slice(2, 8)}`
    const h: Hook = { id, name, url: `/hooks/${id}`, created_at: new Date().toISOString(), last_called_at: null, calls: 0 }
    hooks = [h, ...hooks]
    return { ...h, secret: `whsec_${Math.random().toString(36).slice(2)}${Math.random().toString(36).slice(2)}` }
  },
  deleteHook: (id: string) => {
    hooks = hooks.filter((h) => h.id !== id)
    return { ok: true }
  },
  scriptTest: (_taskId: string): SandboxResult => ({
    ok: true,
    backend: 'process',
    stdout: 'Current price: ₹24,490\n',
    stderr: '',
    result: { alert: true, price: 24_490, message: 'The Sony WH-1000XM6 is now ₹24,490, below your ₹25,000 target.' },
    files_created: [],
    tool_calls: 1,
    duration_ms: 1840,
    error: null
  }),
  patchScript: (taskId: string, script: Partial<ScriptJob>): Task => {
    const t = scriptTasks.find((x) => x.task_id === taskId) ?? scriptTasks[0]
    t.script = { ...(t.script as ScriptJob), ...script }
    return clone(t)
  }
}

// ---------------------------------------------------------------------------- cache injection
export function installLeapDemo(qc: QueryClient): () => void {
  if (!isLeapDemo()) return () => {}
  const frozen = { staleTime: Infinity, retry: false } as const

  for (const t of scriptTasks) {
    qc.setQueryDefaults(qk.tasks.detail(t.task_id), frozen)
    qc.setQueryData(qk.tasks.detail(t.task_id), clone(t))
  }
  const patchTasks = (list: Task[] | undefined) => {
    if (!list) return list
    const missing = scriptTasks.filter((s) => !list.some((t) => t.task_id === s.task_id))
    return missing.length ? [...missing.map(clone), ...list] : list
  }
  const patchSkills = (list: SkillsList | undefined) => {
    // additive only: never touch a real skill that happens to share the demo name
    if (!list || list.pending.some((s) => s.name === REPAIR_NAME) || list.active.some((s) => s.name === REPAIR_NAME)) return list
    qc.setQueryDefaults(qk.skills.diff(REPAIR_NAME), frozen)
    qc.setQueryData<SkillDiff>(qk.skills.diff(REPAIR_NAME), { current: REPAIR_CURRENT, proposed: REPAIR_PROPOSED })
    const repair = {
      name: REPAIR_NAME,
      description: 'Save vendor invoices from Gmail to Drive and log them in the expenses sheet.',
      author: 'assistant',
      state: 'pending_review',
      tags: ['finance'],
      requires_tools: ['gmail', 'gdrive', 'gsheets'],
      version: 2,
      use_count: 14,
      view_count: 20,
      patch_count: 1,
      last_used_at: ago(5 * H),
      created_by_review: true,
      reason: 'The last run of “File vendor invoices to Drive” failed: Drive said “File not found” because the 2026 folder did not exist. This fix creates the year folder first.',
      origin: { repair: true, task_id: 'demo-trigger-mail' },
      proposed_at: ago(3 * H)
    } as unknown as SkillsList['pending'][number]
    const active = [{ ...repair, state: 'active', reason: null, origin: null } as typeof repair, ...list.active]
    return { ...list, active, pending: [repair, ...list.pending] }
  }
  const patchNotes = (list: NotificationList | undefined) => {
    if (!list || list.notifications.some((n) => n.id.startsWith('demo-n-'))) return list
    return { notifications: [...demoNotes, ...list.notifications], unread: list.unread + demoNotes.length }
  }

  const apply = () => {
    const t = qc.getQueryData<Task[]>(qk.tasks.all)
    const tn = patchTasks(t)
    if (tn !== t) qc.setQueryData(qk.tasks.all, tn)
    const s = qc.getQueryData<SkillsList>(qk.skills.all)
    const sn = patchSkills(s)
    if (sn !== s) qc.setQueryData(qk.skills.all, sn)
    const n = qc.getQueryData<NotificationList>(qk.notifications)
    const nn = patchNotes(n)
    if (nn !== n) qc.setQueryData(qk.notifications, nn)
  }
  let busy = false
  const unsub = qc.getQueryCache().subscribe((e) => {
    if (busy || e.type !== 'updated') return
    busy = true
    try {
      apply()
    } finally {
      busy = false
    }
  })
  return unsub
}
