import { IconRefresh } from '@tabler/icons-react'
import { useMemo } from 'react'
import { Combobox } from '@/components/ui'
import { useLocalModels, useModelCatalog, useProviders } from '@/hooks/models'
import { buildModelOptions, isLocalModel, modelShortName, providerLabel, providerOf, withCatalog } from '@/lib/models'
import { cn } from '@/lib/utils'

export interface ModelPickerProps {
  value: string | null | undefined
  onChange: (value: string) => void
  embedding?: boolean
  /** Optional roles: show a "use primary" choice that maps to ''. */
  noneLabel?: string
  /** Restrict suggestions to one provider id (onboarding cloud step). */
  provider?: string
  className?: string
  size?: 'sm' | 'md'
  id?: string
}

/** Model combobox: installed local models + provider suggestions + any free text. */
export function ModelPicker({ value, onChange, embedding, noneLabel, provider, className, size, id }: ModelPickerProps) {
  const local = useLocalModels()
  const providers = useProviders()
  // Live model lists for connected providers whose models change often. Never fetched for a provider without a key.
  const connected = (id: string) => !!providers.data?.find((p) => p.id === id)?.key_set && (!provider || provider === id)
  const openrouter = useModelCatalog('openrouter', connected('openrouter'))
  const nous = useModelCatalog('nous', connected('nous'))

  const groups = useMemo(() => {
    let all = buildModelOptions(local.data, providers.data, { embedding })
    all = withCatalog(all, { id: 'openrouter', label: providerLabel('openrouter') }, openrouter.data, { embedding })
    all = withCatalog(all, { id: 'nous', label: providerLabel('nous') }, nous.data, { embedding })
    return provider ? all.filter((g) => g.id === provider || (provider === 'ollama_chat' && g.id === 'ollama')) : all
  }, [local.data, providers.data, embedding, provider, openrouter.data, nous.data])

  return (
    <Combobox
      id={id}
      value={value ?? ''}
      onChange={onChange}
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
