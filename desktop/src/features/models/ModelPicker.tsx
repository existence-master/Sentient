import { IconRefresh } from '@tabler/icons-react'
import { useMemo } from 'react'
import { toast } from 'sonner'
import { Combobox } from '@/components/ui'
import { useConfig } from '@/hooks/core'
import { useClaudeCodeStatus, useLocalModels, useModelCatalog, useProviders } from '@/hooks/models'
import { allowedForRole, isClaudeCode } from '@/lib/claudeCode'
import { buildModelOptions, chatModelLabel, isLocalModel, modelShortName, providerLabel, providerOf, ROLE_META, withCatalog } from '@/lib/models'
import type { RoleName } from '@/lib/types'
import { cn } from '@/lib/utils'

export interface ModelPickerProps {
  value: string | null | undefined
  onChange: (value: string) => void
  embedding?: boolean
  /** Optional roles: show a "use primary" choice that maps to ''. */
  noneLabel?: string
  /** Restrict suggestions to one provider id (onboarding cloud step). */
  provider?: string
  /** The role this model is for. Claude Code is offered only for roles that answer chats, and never taken for others (#442). */
  role?: RoleName
  /** Offer Claude Code (when it is on this computer) even while its switch is off: picking it turns it on (onboarding). */
  offerClaudeCode?: boolean
  className?: string
  size?: 'sm' | 'md'
  id?: string
}

/** Model combobox: installed local models + provider suggestions + any free text. */
export function ModelPicker({ value, onChange, embedding, noneLabel, provider, role, offerClaudeCode, className, size, id }: ModelPickerProps) {
  const local = useLocalModels()
  const providers = useProviders()
  const config = useConfig()
  const claudeOn = !!config.data?.models.experimental_claude_code
  const claude = useClaudeCodeStatus(claudeOn)
  const accepting = claude.data?.roles
  const claudeHere = !embedding && allowedForRole('claude-code/', role, accepting)
  // Live model lists for connected providers whose models change often. Never fetched for a provider without a key.
  const connected = (id: string) => !!providers.data?.find((p) => p.id === id)?.key_set && (!provider || provider === id)
  const openrouter = useModelCatalog('openrouter', connected('openrouter'))
  const nous = useModelCatalog('nous', connected('nous'))
  const chatgpt = useModelCatalog('chatgpt', connected('chatgpt'))

  const groups = useMemo(() => {
    let all = buildModelOptions(local.data, providers.data, { embedding })
    all = withCatalog(all, { id: 'openrouter', label: providerLabel('openrouter') }, openrouter.data, { embedding })
    all = withCatalog(all, { id: 'nous', label: providerLabel('nous') }, nous.data, { embedding })
    all = withCatalog(all, { id: 'chatgpt', label: 'Using your ChatGPT plan' }, chatgpt.data, { embedding })
    if (provider) all = all.filter((g) => g.id === provider || (provider === 'ollama_chat' && g.id === 'ollama'))
    // Claude Code only answers chats: never listed for a role that refuses it (#442)
    all = all.map((g) => ({ ...g, options: g.options.filter((o) => claudeHere || !isClaudeCode(o.value)) })).filter((g) => g.options.length)
    const claudeModels = claude.data?.installed && (claudeOn || offerClaudeCode) && claudeHere ? claude.data.models : []
    if (claudeModels.length) {
      // right after the models on this computer, before the providers that need a key
      const at = all.filter((g) => g.id === 'ollama' || g.id === 'lm_studio').length
      all.splice(at, 0, {
        id: 'claude-code',
        label: 'Claude through your Claude Code (chats only)',
        options: claudeModels.map((m) => ({ value: m, label: chatModelLabel(m), hint: 'Experimental' }))
      })
    }
    return all
  }, [local.data, providers.data, embedding, provider, openrouter.data, nous.data, chatgpt.data, claude.data, claudeOn, offerClaudeCode, claudeHere])

  const pick = (next: string) => {
    if (!embedding && !claudeHere && isClaudeCode(next)) {
      // typed by hand: Claude Code would refuse this role's work, so it is not taken
      const what = role === 'fast' ? 'the background model' : role ? `the ${ROLE_META[role].label.toLowerCase()} model` : 'used here'
      toast.error(`Claude Code can't be ${what}`, {
        description: 'Claude Code only answers your chats. Pick another model for this.'
      })
      return
    }
    onChange(next)
  }

  return (
    <Combobox
      id={id}
      value={value ?? ''}
      onChange={pick}
      groups={groups}
      size={size}
      className={cn('font-normal', className)}
      placeholder={embedding ? 'Choose an embedding model' : 'Choose a model'}
      searchPlaceholder="Search models or type provider/model…"
      noneOption={noneLabel ? { label: noneLabel, value: '' } : undefined}
      renderValue={(v) => {
        const p = providerOf(v)
        return (
          <span className="flex min-w-0 items-center gap-2">
            {p && (
              <span
                className={cn(
                  'shrink-0 rounded px-1.5 py-px text-2xs font-medium',
                  isLocalModel(v) ? 'bg-success/12 text-success' : 'bg-info/12 text-info'
                )}
              >
                {providerLabel(p)}
              </span>
            )}
            <span className="truncate font-mono text-xs">{modelShortName(v)}</span>
          </span>
        )
      }}
      footer={
        <button
          type="button"
          onClick={() => void local.refetch()}
          className="flex h-8 w-full items-center gap-2 rounded-lg px-2.5 text-xs text-fg-subtle hover:bg-hover hover:text-fg"
        >
          <IconRefresh size={13} className={cn(local.isFetching && 'animate-spin')} />
          {local.data?.ollama.reachable ? 'Refresh installed models' : 'Ollama not detected · check again'}
        </button>
      }
    />
  )
}
