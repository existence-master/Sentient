// #182 The tray's running tasks, from the engine's `tasks` message (docs/NODES.md §4.7). Pure, so tests run without Electron.

export interface RunningTask {
  id: string
  name: string
  status: string
  activity: string | null
}

// name and step are clipped apart, so a long name never hides that a task waits for an answer
const MAX_NAME = 30
const MAX_DOING = 28

/** Read the engine's `tasks` message; malformed entries are dropped, and anything else reads as none running. */
export function parseRunningTasks(msg: unknown): RunningTask[] {
  const running = (msg as { running?: unknown } | null)?.running
  if (!Array.isArray(running)) return []
  return running
    .filter((t): t is Record<string, unknown> => !!t && typeof t === 'object' && typeof t.id === 'string')
    .map((t) => ({
      id: t.id as string,
      name: typeof t.name === 'string' && t.name.trim() ? t.name : 'Task',
      status: typeof t.status === 'string' ? t.status : 'processing',
      activity: typeof t.activity === 'string' && t.activity ? t.activity : null
    }))
}

function clip(text: string, max: number): string {
  return text.length <= max ? text : `${text.slice(0, max - 1)}…`
}

/** One menu line: the task's name, then what it is doing or that it waits for an answer. */
export function runningTaskLabel(t: RunningTask): string {
  const doing = t.status === 'waiting_for_user' ? 'Waiting for your answer' : (t.activity ?? 'Running')
  return `${clip(t.name, MAX_NAME)}: ${clip(doing, MAX_DOING)}`
}

/** The tooltip's note on running tasks, or '' when none run. */
export function runningTasksTooltip(tasks: RunningTask[]): string {
  if (tasks.length === 0) return ''
  return tasks.length === 1 ? '1 task running' : `${tasks.length} tasks running`
}
