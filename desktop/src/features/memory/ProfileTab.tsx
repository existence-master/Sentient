import { IconArrowRight, IconBulb, IconCheck, IconDatabase, IconFileText, IconMessages, IconPencil, IconSettings, IconUserHeart, IconX } from '@tabler/icons-react'
import { useState } from 'react'
import { useNavigate } from 'react-router'
import { toast } from 'sonner'
import { Alert, Button, Card, Markdown, Skeleton, Textarea } from '@/components/ui'
import { useMemoryActions, useWorkspace } from '@/hooks/memory'
import { errorMessage } from '@/lib/api'
import type { WorkspaceFileId } from '@/lib/types'

const TIERS = [
  {
    icon: IconFileText,
    title: 'Your profile',
    body: 'Two short notes Sentient reads before every reply: who you are (USER.md) and the big picture of your projects and commitments (MEMORY.md). You can edit both.'
  },
  {
    icon: IconDatabase,
    title: 'Facts',
    body: 'Small, single facts learned from chats, documents and setup. Sentient looks up the relevant ones for each message. Plans and temporary things expire on their own.'
  },
  {
    icon: IconMessages,
    title: 'Conversation memories',
    body: 'Summaries of older chats, so Sentient can remember what you talked about weeks ago without re-reading everything.'
  }
]

export function ProfileTab() {
  const ws = useWorkspace()
  const navigate = useNavigate()

  return (
    <div className="space-y-6">
      <section className="rounded-2xl border border-border bg-elevated/50 p-5">
        <div className="flex items-center gap-2 text-sm font-semibold text-fg">
          <IconBulb size={16} className="text-accent-text" /> How Sentient remembers
        </div>
        <div className="mt-4 grid gap-3 md:grid-cols-3">
          {TIERS.map((t, i) => (
            <div key={t.title} className="rounded-xl border border-border bg-surface p-3.5">
              <div className="flex items-center gap-2">
                <span className="flex size-6 items-center justify-center rounded-md bg-accent/12 font-mono text-2xs font-semibold text-accent-text">{i + 1}</span>
                <t.icon size={15} className="text-fg-muted" />
                <span className="text-sm font-medium text-fg">{t.title}</span>
              </div>
              <p className="mt-2 text-xs leading-relaxed text-fg-subtle">{t.body}</p>
            </div>
          ))}
        </div>
        <p className="mt-3 text-xs text-fg-subtle">Everything stays on this computer. Nothing here is shared unless a tool you approve sends it.</p>
      </section>

      {ws.isLoading ? (
        <div className="grid gap-4 xl:grid-cols-2">
          <Skeleton className="h-72 rounded-xl" />
          <Skeleton className="h-72 rounded-xl" />
        </div>
      ) : ws.isError ? (
        <Alert tone="danger" title="Couldn't load your profile">
          {errorMessage(ws.error)}
        </Alert>
      ) : (
        <div className="grid items-start gap-4 xl:grid-cols-2">
          <ProfileDoc which="user" title="About you" file="USER.md" content={ws.data?.user ?? ''} hint="Sentient adds what it learns under “Learned”." />
          <ProfileDoc which="memory" title="Long-term notes" file="MEMORY.md" content={ws.data?.memory ?? ''} hint="Sentient keeps this short and refreshes it daily." />
        </div>
      )}

      <button
        type="button"
        onClick={() => navigate('/about')}
        className="group flex w-full items-center gap-3 rounded-xl border border-accent/25 bg-accent/[0.06] px-4 py-3 text-left transition-colors hover:bg-accent/10"
      >
        <span className="flex size-8 items-center justify-center rounded-lg bg-accent/15 text-accent-text">
          <IconUserHeart size={17} />
        </span>
        <span className="min-w-0 flex-1">
          <span className="block text-sm font-medium text-fg">See how Sentient understands you</span>
          <span className="block text-xs text-fg-subtle">Your preferences, goals and style, with the reasons behind each one. Correct anything that isn’t you.</span>
        </span>
        <IconArrowRight size={16} className="text-accent-text transition-transform group-hover:translate-x-0.5" />
      </button>

      <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-border bg-surface px-4 py-3">
        <div className="text-sm text-fg-muted">Want to change Sentient’s personality and tone (SOUL.md)?</div>
        <Button size="sm" variant="secondary" leftIcon={<IconSettings size={14} />} onClick={() => navigate('/settings/personality')}>
          Open Personality settings
        </Button>
      </div>
    </div>
  )
}

function ProfileDoc({ which, title, file, content, hint }: { which: WorkspaceFileId; title: string; file: string; content: string; hint: string }) {
  const { writeWorkspace } = useMemoryActions()
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(content)
  const dirty = draft !== content

  const save = () =>
    writeWorkspace.mutate(
      { which, content: draft },
      {
        onSuccess: () => {
          toast.success(`${file} saved`)
          setEditing(false)
        },
        onError: (e) => toast.error(`Couldn't save ${file}`, { description: errorMessage(e) })
      }
    )

  return (
    <Card className="overflow-hidden">
      <div className="flex h-11 items-center gap-2 border-b border-border px-4">
        <span className="text-sm font-semibold text-fg">{title}</span>
        <span className="font-mono text-2xs text-fg-subtle">{file}</span>
        <div className="flex-1" />
        {editing ? (
          <>
            <Button size="xs" variant="ghost" leftIcon={<IconX size={13} />} onClick={() => (setEditing(false), setDraft(content))}>
              Cancel
            </Button>
            <Button size="xs" variant="primary" leftIcon={<IconCheck size={13} />} disabled={!dirty} loading={writeWorkspace.isPending} onClick={save}>
              Save
            </Button>
          </>
        ) : (
          <Button size="xs" variant="ghost" leftIcon={<IconPencil size={13} />} onClick={() => (setDraft(content), setEditing(true))}>
            Edit
          </Button>
        )}
      </div>
      {editing ? (
        <Textarea
          autoFocus
          autoGrow
          minHeight={280}
          maxHeight={640}
          value={draft}
          spellCheck={false}
          onChange={(e) => setDraft(e.target.value)}
          className="rounded-none border-0 bg-transparent px-4 py-3.5 font-mono text-[12.5px] shadow-none hover:border-0 focus:border-0 focus:ring-0"
        />
      ) : content.trim() ? (
        <div className="max-h-[520px] overflow-y-auto px-5 py-4">
          <Markdown className="!text-sm [&_h1]:!text-[1.3em]">{content}</Markdown>
        </div>
      ) : (
        <div className="px-5 py-8 text-center text-sm text-fg-subtle">Empty. Sentient fills this in as it learns.</div>
      )}
      <div className="border-t border-border px-4 py-2 text-2xs text-fg-subtle">{hint}</div>
    </Card>
  )
}
