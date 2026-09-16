import { IconCheck, IconCpu, IconX } from '@tabler/icons-react'
import { useState, type ReactNode } from 'react'
import { Input, Popover, PopoverContent, PopoverTrigger, Tooltip } from '@/components/ui'
import { useConfig } from '@/hooks/core'
import { useLocalModels } from '@/hooks/models'
import { localModelValue, looksLikeEmbedding, modelShortName, ROLE_META } from '@/lib/models'
import type { RoleName } from '@/lib/types'
import { cn } from '@/lib/utils'

/** Per-message model override: role models, installed local models, or any `provider/model`. */
export function ModelOverride({ value, onChange }: { value: string | undefined; onChange: (v: string | undefined) => void }) {
  const [open, setOpen] = useState(false)
  const [custom, setCustom] = useState('')
  const config = useConfig()
  const local = useLocalModels()
  const roles = config.data?.models.roles

  const roleOptions = (['fast', 'voice', 'planner', 'executor', 'vision'] as RoleName[])
    .map((r) => ({ role: r, model: roles?.[r] }))
    .filter((x): x is { role: RoleName; model: string } => !!x.model && x.model !== roles?.primary)
  const localOptions = (local.data?.ollama.models ?? [])
    .filter((m) => !(m.is_embedding ?? looksLikeEmbedding(m.name)))
    .map((m) => localModelValue('ollama', m, false))

  const pick = (v: string | undefined) => {
    onChange(v)
    setOpen(false)
  }

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <Tooltip content={value ? `This message uses ${value}` : 'Model for this message'}>
        <PopoverTrigger asChild>
          <button
            type="button"
            className={cn(
              'flex h-8 max-w-44 items-center gap-1.5 rounded-lg px-2 text-xs font-medium transition-colors',
              value ? 'bg-accent/12 text-accent-text hover:bg-accent/18' : 'text-fg-subtle hover:bg-hover hover:text-fg'
            )}
          >
            <IconCpu size={15} />
            <span className="truncate">{value ? modelShortName(value) : 'Auto'}</span>
            {value && (
              <span
                role="button"
                aria-label="Clear model override"
                onClick={(e) => {
                  e.stopPropagation()
                  onChange(undefined)
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
        <div className="px-2.5 pb-1.5 pt-1 text-xs text-fg-subtle">Model for this message</div>
        <Option active={!value} onClick={() => pick(undefined)} hint={modelShortName(roles?.primary)}>
          Default (primary)
        </Option>
        {roleOptions.map((r) => (
          <Option key={r.role} active={value === r.model} onClick={() => pick(r.model)} hint={modelShortName(r.model)}>
            {ROLE_META[r.role].label}
          </Option>
        ))}
        {localOptions.length > 0 && (
          <>
            <div className="mx-2.5 mb-1 mt-2 text-2xs font-medium uppercase tracking-wide text-fg-faint">Installed locally</div>
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

function Option({ active, onClick, hint, children }: { active: boolean; onClick: () => void; hint?: string; children: ReactNode }) {
  return (
    <button type="button" onClick={onClick} className="flex h-8 w-full items-center gap-2 rounded-lg px-2.5 text-left text-sm text-fg hover:bg-active">
      <span className="min-w-0 flex-1 truncate">{children}</span>
      {hint && <span className="truncate font-mono text-2xs text-fg-subtle">{hint}</span>}
      <IconCheck size={14} className={cn('shrink-0 text-accent-text', !active && 'invisible')} />
    </button>
  )
}
