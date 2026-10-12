import { IconCheck, IconCpu, IconX } from '@tabler/icons-react'
import { useState, type ReactNode } from 'react'
import { Input, Popover, PopoverContent, PopoverTrigger, Tooltip } from '@/components/ui'
import { useConfig } from '@/hooks/core'
import { useLocalModels } from '@/hooks/models'
import { chatModelLabel, localModelValue, looksLikeEmbedding, modelShortName, ROLE_META } from '@/lib/models'
import { recentChatModels, rememberChatModel } from '@/lib/recentModels'
import type { RoleName } from '@/lib/types'
import { cn } from '@/lib/utils'

const CLAUDE_CODE = ['claude-code/sonnet', 'claude-code/opus']

/**
 * The chat's own model (#312): kept for every message in this chat, also after reopening it or restarting Sentient.
 * `value` null follows the default model, which the title bar menu changes. Role models, Claude Code (when turned
 * on), recent and installed local models, or any `provider/model`.
 */
export function ChatModelMenu({ value, onChange }: { value: string | null | undefined; onChange: (v: string | null) => void }) {
  const [open, setOpen] = useState(false)
  const [custom, setCustom] = useState('')
  const config = useConfig()
  const local = useLocalModels()
  const roles = config.data?.models.roles
  const primary = roles?.primary ?? undefined

  const seen = new Set<string>(primary ? [primary] : [])
  const fresh = (m: string | null | undefined): m is string => !!m && !seen.has(m) && !!seen.add(m)
  const claudeOptions = config.data?.models.experimental_claude_code ? CLAUDE_CODE.filter(fresh) : []
  const recentOptions = recentChatModels().filter(fresh).slice(0, 4)
  const roleOptions = (['fast', 'voice', 'planner', 'executor', 'vision'] as RoleName[])
    .map((r) => ({ role: r, model: roles?.[r] }))
    .filter((x): x is { role: RoleName; model: string } => fresh(x.model))
  const localOptions = (local.data?.ollama.models ?? [])
    .filter((m) => !(m.is_embedding ?? looksLikeEmbedding(m.name)))
    .map((m) => localModelValue('ollama', m, false))
    .filter(fresh)
  const other = value && !seen.has(value) ? value : null // typed in, or no longer listed

  const pick = (v: string | null) => {
    if (v) rememberChatModel(v)
    onChange(v === primary ? null : v)
    setOpen(false)
    setCustom('')
  }

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <Tooltip content={value ? `This chat uses ${value}` : `This chat uses your default model${primary ? `, ${modelShortName(primary)}` : ''}`}>
        <PopoverTrigger asChild>
          <button
            type="button"
            aria-label="Model for this chat"
            data-chat-model={value ?? ''}
            className={cn(
              'flex h-8 max-w-48 items-center gap-1.5 rounded-lg px-2 text-xs font-medium transition-colors',
              value ? 'bg-accent/12 text-accent-text hover:bg-accent/18' : 'text-fg-subtle hover:bg-hover hover:text-fg'
            )}
          >
            <IconCpu size={15} />
            <span className="truncate">{value ? chatModelLabel(value) : 'Default'}</span>
            {value && (
              <span
                role="button"
                aria-label="Use the default model"
                onClick={(e) => {
                  e.stopPropagation()
                  onChange(null)
                }}
                className="-mr-0.5 flex rounded p-0.5 hover:bg-accent/20"
              >
                <IconX size={11} />
              </span>
            )}
          </button>
        </PopoverTrigger>
      </Tooltip>
      <PopoverContent align="start" side="top" className="w-80 p-1.5">
        <div className="px-2.5 pb-1.5 pt-1 text-xs text-fg-subtle">Model for this chat. It stays picked when you come back.</div>
        <Option active={!value} onClick={() => pick(null)} hint={modelShortName(primary)}>
          Use the default
        </Option>
        {other && (
          <Option active onClick={() => pick(other)}>
            <span className="font-mono text-xs">{chatModelLabel(other)}</span>
          </Option>
        )}
        {claudeOptions.map((m) => (
          <Option key={m} active={value === m} onClick={() => pick(m)} hint="Claude Code">
            {chatModelLabel(m)}
          </Option>
        ))}
        {recentOptions.length > 0 && (
          <>
            <Heading>Recent</Heading>
            {recentOptions.map((m) => (
              <Option key={m} active={value === m} onClick={() => pick(m)}>
                <span className="font-mono text-xs">{modelShortName(m)}</span>
              </Option>
            ))}
          </>
        )}
        {roleOptions.map((r) => (
          <Option key={r.role} active={value === r.model} onClick={() => pick(r.model)} hint={modelShortName(r.model)}>
            {ROLE_META[r.role].label}
          </Option>
        ))}
        {localOptions.length > 0 && (
          <>
            <Heading>Installed locally</Heading>
            <div className="max-h-44 overflow-y-auto">
              {localOptions.map((m) => (
                <Option key={m} active={value === m} onClick={() => pick(m)}>
                  <span className="font-mono text-xs">{modelShortName(m)}</span>
                </Option>
              ))}
            </div>
          </>
        )}
        <form
          className="mt-1.5 border-t border-border p-1.5 pt-2"
          onSubmit={(e) => {
            e.preventDefault()
            if (custom.trim()) pick(custom.trim())
          }}
        >
          <Input size="sm" value={custom} onChange={(e) => setCustom(e.target.value)} placeholder="Any model, e.g. openai/gpt-5-mini" className="font-mono" />
        </form>
      </PopoverContent>
    </Popover>
  )
}

function Heading({ children }: { children: ReactNode }) {
  return <div className="mx-2.5 mb-1 mt-2 text-2xs font-medium uppercase tracking-wide text-fg-faint">{children}</div>
}

function Option({ active, onClick, hint, children }: { active: boolean; onClick: () => void; hint?: string; children: ReactNode }) {
  return (
    <button type="button" onClick={onClick} className="flex h-8 w-full items-center gap-2 rounded-lg px-2.5 text-left text-sm text-fg hover:bg-active">
      <span className="min-w-0 flex-1 truncate">{children}</span>
      {hint && <span className="truncate font-mono text-2xs text-fg-subtle">{hint}</span>}
      <IconCheck size={14} className={cn('shrink-0 text-accent-text', !active && 'invisible')} />
    </button>
  )
}
