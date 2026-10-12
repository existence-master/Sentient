import { IconPlayerPlay, IconPlaneDeparture } from '@tabler/icons-react'
import { toast } from 'sonner'
import { Alert, Button } from '@/components/ui'
import { useAway, useAwayActions } from '@/hooks/tasks'
import { errorMessage } from '@/lib/api'
import type { Task } from '@/lib/types'
import { relativeTime } from '@/lib/utils'

/** Resume routines paused while you were away (#163), with a toast saying how many came back. */
export function useResumeRoutines() {
  const { resume } = useAwayActions()
  const run = () =>
    resume.mutate(undefined, {
      onSuccess: (r) =>
        toast.success(r.resumed ? `Resumed ${r.resumed} ${r.resumed === 1 ? 'routine' : 'routines'}` : 'Nothing left to resume', {
          description: r.resumed ? 'They run on their usual schedule again.' : 'You already turned them back on.'
        }),
      onError: (e) => toast.error("Couldn't resume your routines", { description: errorMessage(e) })
    })
  return { run, pending: resume.isPending }
}

/** On the Tasks page: the routines Sentient paused while you were away, and one button to bring them back. */
export function AwayBanner({ tasks }: { tasks: Task[] }) {
  const away = useAway()
  const resume = useResumeRoutines()
  const data = away.data
  if (!data || data.state !== 'paused') return null
  const ids = new Set(data.task_ids)
  const still = tasks.filter((t) => ids.has(t.task_id) && !t.enabled)
  if (!still.length) return null
  const names = still.slice(0, 3).map((t) => t.name || 'Untitled task')
  const more = still.length > 3 ? ` and ${still.length - 3} more` : ''
  return (
    <Alert
      tone="warning"
      icon={<IconPlaneDeparture />}
      title={`${still.length} ${still.length === 1 ? 'routine is' : 'routines are'} paused because you were away`}
      action={
        <Button size="sm" variant="primary" leftIcon={<IconPlayerPlay size={14} />} loading={resume.pending} onClick={resume.run}>
          Resume {still.length === 1 ? 'it' : 'all'}
        </Button>
      }
      className="mt-4"
    >
      {names.join(', ')}
      {more}. Paused {data.since ? relativeTime(data.since) : 'recently'}, so results didn't pile up while you weren't reading them.
    </Alert>
  )
}
