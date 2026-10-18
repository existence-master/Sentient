import { IconBriefcase, IconCheck, IconFolder, IconHome, IconInfinity } from '@tabler/icons-react'
import { useState, type ReactNode } from 'react'
import { Input, Popover, PopoverContent, PopoverTrigger, Tooltip } from '@/components/ui'
import { ScopeIcon, scopeLabel } from '@/features/memory/bits'
import { useMemoryScopes } from '@/hooks/memory'
import { cn } from '@/lib/utils'

/**
 * Which memories this chat uses (#155): Personal, Work or a project. Recall reads that scope plus the memories kept
 * for every chat, and new memories from the chat are saved there.
 *
 * `folder` (#392): the same picker for an added folder in Settings > Folders, with Every chat first (its default) and
 * no counts, since a folder's notes aren't memories.
 */
export function ChatScopeMenu({
  value,
  onChange,
  className,
  folder
}: {
  value: string | null | undefined
  onChange: (v: string) => void
  className?: string
  /** The folder's name: this picks where its notes are used. */
  folder?: string
}) {
  const [open, setOpen] = useState(false)
  const [custom, setCustom] = useState('')
  const scopes = useMemoryScopes()
  const fallback = folder ? 'always' : 'personal'
  const current = value || fallback
  const projects = (scopes.data ?? []).filter((s) => !s.built_in).map((s) => s.name)
  if (!['always', 'personal', 'work', ...projects].includes(current)) projects.unshift(current)
  const count = (name: string) => (folder ? undefined : scopes.data?.find((s) => s.name === name)?.count)

  const pick = (v: string) => {
    if (v !== current) onChange(v)
    setOpen(false)
    setCustom('')
  }
  const tooltip = folder
    ? current === 'always'
      ? `Notes in ${folder} are used in every chat`
      : `Notes in ${folder} are used in ${scopeLabel(current)} chats`
    : `This chat uses your ${scopeLabel(current)} memories`

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <Tooltip content={tooltip}>
        <PopoverTrigger asChild>
          <button
            type="button"
            aria-label={folder ? `Where notes in ${folder} are used` : 'Memories for this chat'}
            data-chat-scope={folder ? undefined : current}
            data-folder-scope={folder ? current : undefined}
            className={cn(
              'flex h-8 max-w-48 shrink-0 items-center gap-1.5 rounded-lg px-2 text-xs font-medium transition-colors',
              current === fallback ? 'text-fg-subtle hover:bg-hover hover:text-fg' : 'bg-accent/12 text-accent-text hover:bg-accent/18',
              className
            )}
          >
            <ScopeIcon scope={current} />
            <span className="truncate">{folder ? `Used in ${current === 'always' ? 'every chat' : scopeLabel(current)}` : scopeLabel(current)}</span>
          </button>
        </PopoverTrigger>
      </Tooltip>
      <PopoverContent align={folder ? 'start' : 'end'} className="w-72 p-1.5">
        <div className="px-2.5 pb-1.5 pt-1 text-xs text-fg-subtle">
          {folder
            ? 'Which chats use the notes in this folder. A Work folder only comes up in work chats, unless you ask about it.'
            : 'Which memories this chat uses. New ones it learns are kept there too. Memories set to every chat are always used.'}
        </div>
        {folder && (
          <Option active={current === 'always'} onClick={() => pick('always')} icon={<IconInfinity size={15} />}>
            Every chat
          </Option>
        )}
        <Option active={current === 'personal'} onClick={() => pick('personal')} icon={<IconHome size={15} />} hint={count('personal')}>
          Personal
        </Option>
        <Option active={current === 'work'} onClick={() => pick('work')} icon={<IconBriefcase size={15} />} hint={count('work')}>
          Work
        </Option>
        {projects.length > 0 && <div className="mx-2.5 mb-1 mt-2 text-2xs font-medium uppercase tracking-wide text-fg-faint">Projects</div>}
        <div className="max-h-44 overflow-y-auto">
          {projects.map((p) => (
            <Option key={p} active={current === p} onClick={() => pick(p)} icon={<IconFolder size={15} />} hint={count(p)}>
              {p}
            </Option>
          ))}
        </div>
        <form
          className="mt-1.5 border-t border-border p-1.5 pt-2"
          onSubmit={(e) => {
            e.preventDefault()
            const name = custom.trim().replace(/\s+/g, ' ').slice(0, 40)
            if (name && name.toLowerCase() !== 'always') pick(name)
          }}
        >
          <Input size="sm" value={custom} onChange={(e) => setCustom(e.target.value)} placeholder="New project, e.g. Kestrel launch" aria-label="New project" />
        </form>
      </PopoverContent>
    </Popover>
  )
}

function Option({ active, onClick, icon, hint, children }: { active: boolean; onClick: () => void; icon: ReactNode; hint?: number; children: ReactNode }) {
  return (
    <button type="button" onClick={onClick} className="flex h-8 w-full items-center gap-2 rounded-lg px-2.5 text-left text-sm text-fg hover:bg-active">
      <span className="shrink-0 text-fg-subtle">{icon}</span>
      <span className="min-w-0 flex-1 truncate">{children}</span>
      {hint !== undefined && <span className="text-2xs tabular-nums text-fg-subtle">{hint}</span>}
      <IconCheck size={14} className={cn('shrink-0 text-accent-text', !active && 'invisible')} />
    </button>
  )
}
