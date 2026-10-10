import { IconArrowUpRight, IconMessages } from '@tabler/icons-react'
import { useMemo } from 'react'
import { useNavigate } from 'react-router'
import { Alert, Button, EmptyState, Skeleton } from '@/components/ui'
import { useSessions } from '@/hooks/core'
import { useMemorySummaries } from '@/hooks/memory'
import { errorMessage } from '@/lib/api'
import { formatTime, parseDate } from '@/lib/utils'
import { recentRelative } from './meta'

function range(start: string, end: string): string {
  const s = parseDate(start)
  const e = parseDate(end)
  if (!s || !e) return ''
  const date = (d: Date) => d.toLocaleDateString(undefined, { weekday: 'short', month: 'short', day: 'numeric' })
  if (s.toDateString() === e.toDateString()) return `${date(s)} · ${formatTime(start)} – ${formatTime(end)}`
  return `${date(s)} ${formatTime(start)} – ${date(e)} ${formatTime(end)}`
}

export function ConversationsTab() {
  const summaries = useMemorySummaries(100)
  const sessions = useSessions()
  const navigate = useNavigate()
  const titles = useMemo(() => new Map((sessions.data ?? []).map((s) => [s.id, s.title])), [sessions.data])

  if (summaries.isLoading) {
    return (
      <div className="space-y-3">
        {[0, 1, 2].map((i) => (
          <Skeleton key={i} className="h-28 rounded-xl" />
        ))}
      </div>
    )
  }
  if (summaries.isError) {
    return (
      <Alert tone="danger" title="Couldn't load conversation memories">
        {errorMessage(summaries.error)}
      </Alert>
    )
  }
  if (!summaries.data?.length) {
    return (
      <EmptyState
        icon={<IconMessages />}
        title="No conversation memories yet"
        description="After a chat has been quiet for a while, Sentient writes a short first-person summary of it so it can recall what you talked about."
      />
    )
  }

  return (
    <div className="space-y-2.5">
      <p className="max-w-2xl text-sm text-fg-muted">
        Sentient keeps short summaries of older conversations. It searches them when you ask about something you discussed before, like
        “what did we decide about the trek?”.
      </p>
      <ol className="relative space-y-3 pt-2">
        {summaries.data.map((s) => {
          const title = (s.session_id && titles.get(s.session_id)) || 'Conversation'
          return (
            <li key={s.id} className="group rounded-xl border border-border bg-surface p-4 transition-colors hover:border-border-strong">
              <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
                <div className="flex size-7 items-center justify-center rounded-lg bg-accent/10 text-accent-text">
                  <IconMessages size={15} />
                </div>
                <div className="min-w-0 flex-1">
                  <div className="truncate text-sm font-medium text-fg">{title}</div>
                  <div className="text-xs text-fg-subtle">
                    {range(s.start_at, s.end_at)} {recentRelative(s.end_at) && <span className="text-fg-faint">· {recentRelative(s.end_at)}</span>}
                  </div>
                </div>
                {s.session_id && titles.has(s.session_id) && (
                  <Button size="sm" variant="ghost" rightIcon={<IconArrowUpRight size={14} />} onClick={() => navigate(`/chat/${s.session_id}`)}>
                    Open chat
                  </Button>
                )}
              </div>
              <p className="selectable mt-3 border-l-2 border-accent/30 pl-3 text-sm leading-relaxed text-fg-muted">{s.content}</p>
              {s.untrusted && (
                <p className="mt-2 text-xs text-fg-subtle">This chat read content from {s.untrusted}, so Sentient keeps this summary out of your other chats.</p>
              )}
            </li>
          )
        })}
      </ol>
    </div>
  )
}
