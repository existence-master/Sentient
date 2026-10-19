/**
 * Tasks by email (#159): forward or copy an email to your Sentient address and Sentient plans a task from it.
 * Settings live in `tasks.email` (PATCH /api/config); what happened to the latest emails comes from
 * `GET /api/tasks/email`.
 */
import { IconArrowRight, IconBan, IconCircleCheck, IconMailForward } from '@tabler/icons-react'
import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router'
import { Alert, Badge, Button, FormRow, FormSection, Input, Skeleton, StatusDot, Switch } from '@/components/ui'
import { useConfigEditor } from '@/hooks/config'
import { useEmailTasks } from '@/hooks/tasks'
import { isNotImplemented } from '@/lib/api'
import type { EmailTaskLogEntry, EmailTasksConfig } from '@/lib/types'
import { relativeTime } from '@/lib/utils'

const KEYWORDS = 'tasks by email forward copy plus address sentient address who can send allowed senders label folder spoofed'
const EMPTY: EmailTasksConfig = { enabled: false, address: '', label: '', allowed_senders: [] }
const APP_LABELS: Record<string, string> = { gmail: 'Gmail', email_imap: 'Email (IMAP)' }

/** Comma, semicolon or space separated addresses, lower-cased and without repeats. */
function parseAddresses(text: string): string[] {
  return Array.from(new Set(text.split(/[\s,;]+/).map((a) => a.trim().toLowerCase()).filter((a) => a.includes('@'))))
}

/** A text field that saves when you leave it or press Enter, so a half-typed address is never saved. */
function CommitInput({ value, onCommit, ...props }: { value: string; onCommit: (v: string) => void } & Omit<React.ComponentProps<typeof Input>, 'value' | 'onChange'>) {
  const [draft, setDraft] = useState(value)
  useEffect(() => setDraft(value), [value])
  const commit = () => {
    if (draft.trim() !== value) onCommit(draft.trim())
  }
  return <Input {...props} value={draft} onChange={(e) => setDraft(e.target.value)} onBlur={commit} onKeyDown={(e) => e.key === 'Enter' && commit()} />
}

export function EmailTasksSection({ query }: { query?: string }) {
  const navigate = useNavigate()
  const { config, patch } = useConfigEditor()
  const status = useEmailTasks()
  const cfg: EmailTasksConfig = { ...EMPTY, ...((config?.tasks?.email as EmailTasksConfig | undefined) ?? {}) }
  const q = (query ?? '').trim().toLowerCase()
  if (q && !KEYWORDS.includes(q)) return null

  const s = status.data
  const missing = status.isError && isNotImplemented(status.error)
  const suggested = s?.suggested_address ?? ''
  const own = s?.own_addresses ?? []
  const set = (p: Partial<EmailTasksConfig>) => patch({ tasks: { email: p } }, { immediate: true })
  const noAccount = !!s && s.watching.length === 0

  return (
    <FormSection
      title="Tasks by email"
      description="Forward or copy an email to your Sentient address and Sentient plans a task from it. The plan waits for your OK, and anything it would send because of the email asks you first."
    >
      <FormRow label="Take tasks by email" description={cfg.enabled ? (s?.active ? 'On. Email to your Sentient address becomes a task.' : 'Pick your Sentient address below to start.') : 'Off. Turn it on to pick an address.'}>
        <Switch
          checked={cfg.enabled}
          disabled={missing}
          onCheckedChange={(on) => set(on && !cfg.address && suggested ? { enabled: true, address: suggested } : { enabled: on })}
        />
      </FormRow>
      {cfg.enabled && (
        <>
          {noAccount && (
            <div className="px-4 py-3">
              <Alert tone="warning" title="Connect your email first">
                New email reaches Sentient through Gmail or Email (IMAP). Connect one in Integrations.
                <Button size="sm" variant="ghost" className="ml-1" rightIcon={<IconArrowRight size={14} />} onClick={() => navigate('/integrations')}>
                  Integrations
                </Button>
              </Alert>
            </div>
          )}
          <FormRow
            label="Your Sentient address"
            stack
            description={
              <>
                Email sent or copied to this address becomes a task. Most email services deliver an address with +something to your normal inbox, so{' '}
                {suggested ? <b className="font-medium text-fg-muted">{suggested}</b> : 'you+sentient@gmail.com'} reaches you without a new account.
              </>
            }
          >
            <div className="flex w-full items-center gap-2">
              <CommitInput value={cfg.address} placeholder={suggested || 'you+sentient@gmail.com'} onCommit={(v) => set({ address: v.toLowerCase() })} aria-label="Your Sentient address" />
              {suggested && cfg.address !== suggested && (
                <Button size="sm" variant="secondary" onClick={() => set({ address: suggested })}>
                  Use {suggested}
                </Button>
              )}
            </div>
          </FormRow>
          <FormRow label="Label or folder" description="Optional. Email with this Gmail label, or in this email folder, becomes a task too. Over IMAP, also add the folder to the ones the account watches.">
            <CommitInput value={cfg.label} placeholder="Sentient" onCommit={(v) => set({ label: v })} aria-label="Label or folder" className="w-56" />
          </FormRow>
          <FormRow
            label="Who can send tasks"
            stack
            description={
              <>
                Your own addresses can always send tasks{own.length ? ` (${own.join(', ')})` : ''}. Add others here. Sentient checks the real email address, never the name shown next to it.
              </>
            }
          >
            <CommitInput
              value={cfg.allowed_senders.join(', ')}
              placeholder="a.colleague@example.com, a.friend@example.com"
              onCommit={(v) => set({ allowed_senders: parseAddresses(v) })}
              aria-label="Who can send tasks"
            />
          </FormRow>
          <RecentEmails entries={s?.recent ?? []} loading={status.isLoading} onOpen={(id) => navigate(`/tasks?task=${encodeURIComponent(id)}`)} />
        </>
      )}
    </FormSection>
  )
}

function RecentEmails({ entries, loading, onOpen }: { entries: EmailTaskLogEntry[]; loading: boolean; onOpen: (taskId: string) => void }) {
  return (
    <div className="px-4 py-3.5">
      <div className="mb-2 flex items-center gap-2 text-sm font-medium text-fg">
        <IconMailForward size={15} className="text-fg-subtle" />
        Latest emails to your Sentient address
      </div>
      {loading ? (
        <Skeleton className="h-12" />
      ) : !entries.length ? (
        <p className="text-xs text-fg-subtle">Nothing yet. Forward an email to your Sentient address to try it.</p>
      ) : (
        <ul className="divide-y divide-border rounded-lg border border-border">
          {entries.slice(0, 8).map((e, i) => {
            const made = e.outcome === 'task'
            const shown = e.from.replace(/\s+/g, ' ').trim()
            const really = e.sender && !shown.toLowerCase().endsWith(`<${e.sender}>`) && shown.toLowerCase() !== e.sender
            return (
              <li key={`${e.at}-${i}`} className="flex items-start gap-2.5 px-3 py-2.5">
                {made ? <IconCircleCheck size={16} className="mt-0.5 shrink-0 text-success" /> : <IconBan size={16} className="mt-0.5 shrink-0 text-danger" />}
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-sm text-fg">
                    <span className="truncate">{shown || 'Unknown sender'}</span>
                    {e.sender && (really || !made) && <Badge size="xs" tone={made ? 'neutral' : 'danger'}>really {e.sender}</Badge>}
                    <span className="text-2xs text-fg-faint">
                      {APP_LABELS[e.app] ?? e.app} · {relativeTime(e.at)}
                    </span>
                  </div>
                  <div className="truncate text-xs text-fg-subtle">{e.subject || '(no subject)'}</div>
                  <div className="mt-0.5 flex items-center gap-1.5 text-xs">
                    <StatusDot tone={made ? 'success' : 'danger'} />
                    <span className={made ? 'text-fg-muted' : 'text-danger'}>{made ? 'Made a task' : `Ignored. ${e.reason ?? ''}`}</span>
                  </div>
                </div>
                {made && e.task_id && (
                  <Button size="sm" variant="ghost" rightIcon={<IconArrowRight size={14} />} onClick={() => onOpen(e.task_id as string)}>
                    Open
                  </Button>
                )}
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}
