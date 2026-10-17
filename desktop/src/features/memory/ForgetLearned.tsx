/**
 * "Also forget what Sentient learned from it" (#153): the option shown when disconnecting an app, an MCP server, a
 * messaging app, a device or a folder. It says beforehand how much would go ("Forget 42 memories and 310 saved
 * emails"); with the box left clear nothing is deleted.
 */
import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { qk } from '@/hooks/queryKeys'
import { api } from '@/lib/api'
import type { ForgetCounts } from '@/lib/types'
import { cn } from '@/lib/utils'

const count = (n: number, one: string, many: string) => `${n.toLocaleString()} ${n === 1 ? one : many}`

/** "42 memories, 3 chats with 120 messages and 310 saved emails", or '' when it taught Sentient nothing. */
export function forgetParts(c: ForgetCounts): string {
  const parts: string[] = []
  if (c.memories) parts.push(count(c.memories, 'memory', 'memories'))
  if (c.chats) parts.push(`${count(c.chats, 'chat', 'chats')} with ${count(c.messages, 'message', 'messages')}`)
  if (c.items) parts.push(c.item_noun === 'emails' ? count(c.items, 'saved email', 'saved emails') : count(c.items, 'saved item', 'saved items'))
  if (c.results) parts.push(count(c.results, 'result in your chats', 'results in your chats'))
  if (c.summaries) parts.push(count(c.summaries, 'chat summary', 'chat summaries'))
  if (parts.length < 2) return parts.join('')
  return `${parts.slice(0, -1).join(', ')} and ${parts[parts.length - 1]}`
}

/** The toast line after a forget: "Forgot 42 memories and 310 saved emails." */
export function forgottenMessage(c?: ForgetCounts | null): string | undefined {
  if (!c) return undefined
  const what = forgetParts(c)
  return what ? `Forgot ${what}.` : 'There was nothing to forget.'
}

/** The box's state, cleared each time the dialog opens. */
export function useForgetChoice() {
  const [forget, setForget] = useState(false)
  return { forget, setForget, reset: () => setForget(false) }
}

export function ForgetLearned({
  source,
  open,
  checked,
  onCheckedChange,
  extra
}: {
  /** app:<id>, channel:<id>, device:<id> or folder:<id> */
  source: string
  /** count only while the dialog is open */
  open: boolean
  checked: boolean
  onCheckedChange: (checked: boolean) => void
  /** what removing it deletes anyway, said under the count (a folder's index of notes) */
  extra?: string
}) {
  const preview = useQuery({
    queryKey: qk.memories.forgetPreview(source),
    queryFn: () => api.memories.forgetPreview(source),
    enabled: open && !!source,
    staleTime: 0,
    retry: false
  })
  const c = preview.data
  const what = c ? forgetParts(c) : ''
  const nothing = !!c && !what
  let line: string
  if (preview.isPending) line = 'Counting what Sentient learned from it...'
  else if (preview.isError || !c) line = "Couldn't count what Sentient learned from it. Ticking the box still forgets all of it."
  else if (nothing) line = "Sentient hasn't learned anything from it yet."
  else line = `Forget ${what}.`

  return (
    <label
      className={cn(
        'flex cursor-pointer items-start gap-2.5 rounded-xl border px-3 py-2.5 transition-colors',
        checked ? 'border-danger/40 bg-danger/5' : 'border-border bg-surface hover:bg-hover',
        nothing && 'cursor-default opacity-70'
      )}
    >
      <input
        type="checkbox"
        className="mt-0.5 size-4 shrink-0 cursor-pointer accent-danger"
        checked={checked && !nothing}
        disabled={nothing}
        onChange={(e) => onCheckedChange(e.target.checked)}
        aria-describedby={`forget-${source}`}
      />
      <span className="min-w-0">
        <span className="block text-sm font-medium text-fg">Also forget what Sentient learned from it</span>
        <span id={`forget-${source}`} className="mt-0.5 block text-xs text-fg-muted">
          {line}
          {c && c.shared > 0 && ` ${count(c.shared, 'memory', 'memories')} you also shared elsewhere ${c.shared === 1 ? 'stays' : 'stay'}.`}
          {extra && ` ${extra}`}
          {!checked && !nothing && ' Leave it clear to keep everything.'}
        </span>
      </span>
    </label>
  )
}
