import {
  IconArrowDown,
  IconArrowUp,
  IconBrain,
  IconChevronRight,
  IconCloud,
  IconCpu,
  IconDeviceDesktop,
  IconDots,
  IconExternalLink,
  IconEye,
  IconKey,
  IconListCheck,
  IconMicrophone,
  IconPlus,
  IconRefresh,
  IconRoute,
  IconTrash,
  IconBolt,
  type Icon
} from '@tabler/icons-react'
import { useQueryClient } from '@tanstack/react-query'
import { AnimatePresence, motion } from 'motion/react'
import { useEffect, useRef, useState } from 'react'
import { toast } from 'sonner'
import {
  Alert,
  Badge,
  Button,
  Card,
  ConfirmDialog,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuTrigger,
  IconButton,
  Input,
  SegmentedControl,
  Skeleton,
  Slider,
  StatusDot,
  Switch
} from '@/components/ui'
import { ModelPicker } from '@/features/models/ModelPicker'
import { ModelTest } from '@/features/models/ModelTest'
import { OllamaPull } from '@/features/models/OllamaPull'
import { ProviderKeyDialog } from '@/features/models/ProviderKeyDialog'
import { useConfigEditor } from '@/hooks/config'
import { useDeleteSecret, useLocalModels, useProviders, useSecrets, useSetFallbacks, useSetRoles } from '@/hooks/models'
import { qk } from '@/hooks/queryKeys'
import { api, errorMessage } from '@/lib/api'
import { getBridge } from '@/lib/bridge'
import { localModelValue, looksLikeEmbedding, modelShortName, REASONING_LEVELS, ROLE_META } from '@/lib/models'
import { ROLE_NAMES, type Provider, type RoleName, type SentientConfig } from '@/lib/types'
import { cn, formatBytes } from '@/lib/utils'
import type { SectionProps } from '../SettingsPage'

const ROLE_ICON: Record<RoleName, Icon> = {
  primary: IconCpu,
  fast: IconBolt,
  voice: IconMicrophone,
  planner: IconRoute,
  executor: IconListCheck,
  embedding: IconBrain,
  vision: IconEye
}

export function ModelsSection({ query }: SectionProps) {
  const { config } = useConfigEditor()
  const q = query.trim().toLowerCase()
  const roles = ROLE_NAMES.filter((r) => !q || `${ROLE_META[r].label} ${ROLE_META[r].description} ${r} model role`.toLowerCase().includes(q))

  if (!config) {
    return (
      <div className="space-y-3">
        {[0, 1, 2].map((i) => (
          <Skeleton key={i} className="h-40 rounded-xl" />
        ))}
      </div>
    )
  }

  return (
    <div className="space-y-10">
      <section className="space-y-3">
        <SectionTitle title="Roles" description="Sentient uses different models for different jobs. Mix local and cloud freely." />
        {roles.map((r) => (
          <RoleCard key={r} role={r} config={config} />
        ))}
      </section>
      <ProvidersPanel />
      <OllamaPanel />
    </div>
  )
}

function SectionTitle({ title, description, actions }: { title: string; description?: string; actions?: React.ReactNode }) {
  return (
    <div className="flex items-end justify-between gap-4 px-1">
      <div>
        <h3 className="text-sm font-semibold text-fg">{title}</h3>
        {description && <p className="mt-0.5 text-xs text-fg-subtle">{description}</p>}
      </div>
      {actions}
    </div>
  )
}

// ---------------------------------------------------------------------------- role card
function RoleCard({ role, config }: { role: RoleName; config: SentientConfig }) {
  const meta = ROLE_META[role]
  const RIcon = ROLE_ICON[role]
  const qc = useQueryClient()
  const { patch } = useConfigEditor()
  const setRoles = useSetRoles()
  const value = config.models.roles[role]
  const effort = (config.models.reasoning[role] as string | undefined) ?? ''
  const temperature = config.models.temperature[role]
  const fallbacks = config.models.fallbacks[role] ?? []
  const [showFallbacks, setShowFallbacks] = useState(fallbacks.length > 0)

  const changeModel = (v: string) => {
    if (meta.required && !v) {
      toast.message(`${meta.label} needs a model`)
      return
    }
    setRoles.mutate(
      { [role]: v || null },
      {
        onSuccess: () => toast.success(`${meta.label} model updated`, { description: v ? modelShortName(v) : `Uses ${ROLE_META[meta.fallsBackTo ?? 'primary'].label.toLowerCase()}` }),
        onError: (e) => toast.error("Couldn't change the model", { description: errorMessage(e) })
      }
    )
  }

  const setTemperature = async (t: number | null) => {
    if (t !== null) {
      patch({ models: { temperature: { [role]: t } } })
      return
    }
    const { [role]: _removed, ...rest } = config.models.temperature
    void _removed
    const next = { ...config, models: { ...config.models, temperature: rest } }
    qc.setQueryData(qk.config, next)
    try {
      await api.config.put(next)
    } catch (e) {
      toast.error("Couldn't reset temperature", { description: errorMessage(e) })
      void qc.invalidateQueries({ queryKey: qk.config })
    }
  }

  return (
    <Card className="overflow-hidden">
      <div className="flex items-start gap-3 px-4 pt-4">
        <div className="flex size-9 shrink-0 items-center justify-center rounded-lg border border-border bg-elevated text-accent-text">
          <RIcon size={18} />
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold text-fg">{meta.label}</span>
            {!meta.required && <Badge size="xs">Optional</Badge>}
          </div>
          <p className="mt-0.5 text-xs leading-relaxed text-fg-subtle">{meta.description}</p>
        </div>
      </div>

      <div className="space-y-3.5 px-4 pb-4 pt-3.5">
        <div className="flex flex-wrap items-center gap-2.5">
          <ModelPicker
            value={value}
            onChange={changeModel}
            embedding={meta.embedding}
            noneLabel={meta.required ? undefined : `Use ${ROLE_META[meta.fallsBackTo ?? 'primary'].label.toLowerCase()} (${modelShortName(config.models.roles.primary)})`}
            className="min-w-72 flex-1"
          />
          <ModelTest model={value || config.models.roles.primary} role={role} embedding={meta.embedding} className="shrink-0" />
        </div>

        {!meta.embedding && (
          <div className="flex flex-wrap items-center gap-x-8 gap-y-3 border-t border-border pt-3.5">
            <div className="flex items-center gap-3">
              <span className="text-xs text-fg-muted">Reasoning</span>
              <SegmentedControl
                size="sm"
                value={effort || 'default'}
                onChange={(v) => v !== 'default' && patch({ models: { reasoning: { [role]: v } } }, { immediate: true })}
                options={[...REASONING_LEVELS.map((l) => ({ value: l, label: l === 'none' ? 'Off' : l[0].toUpperCase() + l.slice(1) }))]}
              />
            </div>
            <div className="flex min-w-64 flex-1 items-center gap-3">
              <span className="text-xs text-fg-muted">Temperature</span>
              <Switch size="sm" checked={temperature !== undefined} onCheckedChange={(on) => void setTemperature(on ? 0.7 : null)} aria-label="Custom temperature" />
              {temperature !== undefined ? (
                <>
                  <Slider value={temperature} min={0} max={2} step={0.05} onValueChange={(t) => void setTemperature(Math.round(t * 100) / 100)} className="max-w-48" aria-label="Temperature" />
                  <span className="w-8 text-xs tabular-nums text-fg-muted">{temperature.toFixed(2)}</span>
                </>
              ) : (
                <span className="text-xs text-fg-subtle">Model default</span>
              )}
            </div>
          </div>
        )}

        <div className="border-t border-border pt-3">
          <button type="button" onClick={() => setShowFallbacks((s) => !s)} className="flex items-center gap-1.5 text-xs text-fg-muted hover:text-fg">
            <IconChevronRight size={13} className={cn('transition-transform', showFallbacks && 'rotate-90')} />
            Fallbacks
            <span className="text-fg-subtle">{fallbacks.length ? `(${fallbacks.length})` : '· tried in order if this model fails'}</span>
          </button>
          <AnimatePresence initial={false}>
            {showFallbacks && (
              <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
                <FallbackEditor role={role} chain={fallbacks} embedding={meta.embedding} />
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </Card>
  )
}

function FallbackEditor({ role, chain, embedding }: { role: RoleName; chain: string[]; embedding?: boolean }) {
  const qc = useQueryClient()
  const setFallbacks = useSetFallbacks()
  const [items, setItems] = useState<string[]>(chain)
  const timer = useRef<number | undefined>(undefined)

  useEffect(() => setItems(chain), [chain])

  const commit = (next: string[]) => {
    setItems(next)
    qc.setQueryData<SentientConfig>(qk.config, (old) => (old ? { ...old, models: { ...old.models, fallbacks: { ...old.models.fallbacks, [role]: next.filter(Boolean) } } } : old))
    window.clearTimeout(timer.current)
    timer.current = window.setTimeout(() => {
      setFallbacks.mutate({ [role]: next.filter(Boolean) }, { onError: (e) => toast.error("Couldn't save fallbacks", { description: errorMessage(e) }) })
    }, 500)
  }

  const move = (i: number, d: -1 | 1) => {
    const next = items.slice()
    ;[next[i], next[i + d]] = [next[i + d], next[i]]
    commit(next)
  }

  return (
    <div className="space-y-2 pt-3">
      {items.map((m, i) => (
        <div key={`${i}-${m}`} className="flex items-center gap-2">
          <span className="w-5 text-center text-xs tabular-nums text-fg-subtle">{i + 1}</span>
          <ModelPicker size="sm" embedding={embedding} value={m} onChange={(v) => commit(items.map((x, j) => (j === i ? v : x)))} className="flex-1" />
          <IconButton size="sm" label="Move up" disabled={i === 0} icon={<IconArrowUp size={14} />} onClick={() => move(i, -1)} />
          <IconButton size="sm" label="Move down" disabled={i === items.length - 1} icon={<IconArrowDown size={14} />} onClick={() => move(i, 1)} />
          <IconButton size="sm" label="Remove" icon={<IconTrash size={14} />} onClick={() => commit(items.filter((_, j) => j !== i))} />
        </div>
      ))}
      <Button size="xs" variant="ghost" leftIcon={<IconPlus size={13} />} onClick={() => setItems([...items, ''])} className="ml-5">
        Add fallback
      </Button>
    </div>
  )
}

// ---------------------------------------------------------------------------- providers
function ProvidersPanel() {
  const providers = useProviders()
  const secrets = useSecrets()
  const [keyFor, setKeyFor] = useState<Provider | null>(null)
  const [removeFor, setRemoveFor] = useState<Provider | null>(null)
  const deleteSecret = useDeleteSecret()

  return (
    <section className="space-y-3">
      <SectionTitle title="Providers" description="API keys are stored in your system keychain. Local servers need no key." />
      {providers.isLoading ? (
        <Skeleton className="h-64 rounded-xl" />
      ) : providers.isError ? (
        <Alert tone="danger" title="Couldn't load providers">
          {errorMessage(providers.error)}
        </Alert>
      ) : (
        <Card className="divide-y divide-border">
          {(providers.data ?? []).map((p) => (
            <ProviderRow
              key={p.id}
              provider={p}
              source={secrets.data?.find((s) => s.name === p.id && (s.kind ?? 'provider') === 'provider')?.source ?? null}
              onKey={() => setKeyFor(p)}
              onRemove={() => setRemoveFor(p)}
            />
          ))}
        </Card>
      )}
      <ProviderKeyDialog provider={keyFor} open={!!keyFor} onOpenChange={(o) => !o && setKeyFor(null)} />
      <ConfirmDialog
        open={!!removeFor}
        onOpenChange={(o) => !o && setRemoveFor(null)}
        title={`Remove ${removeFor?.label} key?`}
        description="Models from this provider will stop working until you add a key again."
        confirmLabel="Remove key"
        onConfirm={async () => {
          if (removeFor) await deleteSecret.mutateAsync(removeFor.id)
          toast.success('Key removed')
        }}
      />
    </section>
  )
}

function ProviderRow({ provider: p, source, onKey, onRemove }: { provider: Provider; source: 'keychain' | 'env' | null; onKey: () => void; onRemove: () => void }) {
  const { config, patch } = useConfigEditor()
  const [open, setOpen] = useState(false)
  const local = p.kind === 'local'
  const base = config?.models.providers[p.id]?.api_base ?? ''

  return (
    <div>
      <div className="flex items-center gap-3 px-4 py-3">
        <div className={cn('flex size-8 shrink-0 items-center justify-center rounded-lg border border-border', local ? 'text-success' : 'text-info')}>
          {local ? <IconDeviceDesktop size={16} /> : <IconCloud size={16} />}
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 text-sm font-medium text-fg">
            {p.label}
            {local ? (
              <Badge size="xs" tone="success">
                Local
              </Badge>
            ) : p.key_set ? (
              <Badge size="xs" tone="success">
                Key set{source ? ` · ${source}` : ''}
              </Badge>
            ) : (
              <Badge size="xs" tone="warning">
                No key
              </Badge>
            )}
          </div>
          <div className="truncate font-mono text-2xs text-fg-subtle">{local ? base || 'default address' : p.suggested.slice(0, 3).map(modelShortName).join(' · ')}</div>
        </div>
        <IconButton size="sm" label="Docs" icon={<IconExternalLink size={14} />} onClick={() => void getBridge().openExternal(p.docs_url)} />
        {!local && (
          <Button size="sm" variant={p.key_set ? 'ghost' : 'secondary'} leftIcon={<IconKey size={13} />} onClick={onKey}>
            {p.key_set ? 'Replace key' : 'Add key'}
          </Button>
        )}
        {!local && source === 'keychain' && <IconButton size="sm" label="Remove key" icon={<IconTrash size={14} />} onClick={onRemove} />}
        <IconButton size="sm" label="Base URL" active={open} icon={<IconChevronRight size={14} className={cn('transition-transform', open && 'rotate-90')} />} onClick={() => setOpen((o) => !o)} />
      </div>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
            <div className="flex items-center gap-3 px-4 pb-3.5 pl-15">
              <span className="w-20 shrink-0 text-xs text-fg-muted">Base URL</span>
              <Input
                size="sm"
                value={base}
                placeholder={local ? 'http://localhost:11434' : 'Leave empty for the official API'}
                className="font-mono"
                onChange={(e) => {
                  const v = e.target.value.trim() || null
                  patch({ models: { providers: p.id === 'ollama_chat' ? { ollama_chat: { api_base: v }, ollama: { api_base: v } } : { [p.id]: { api_base: v } } } })
                }}
              />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

// ---------------------------------------------------------------------------- ollama
function OllamaPanel() {
  const local = useLocalModels()
  const setRoles = useSetRoles()
  const ollama = local.data?.ollama
  const models = ollama?.models ?? []

  const useFor = (role: RoleName, name: string, embedding: boolean) =>
    setRoles.mutate(
      { [role]: localModelValue('ollama', { name }, embedding) },
      {
        onSuccess: () => toast.success(`${ROLE_META[role].label} now uses ${name}`),
        onError: (e) => toast.error("Couldn't change the model", { description: errorMessage(e) })
      }
    )

  return (
    <section className="space-y-3">
      <SectionTitle
        title="Local models"
        description="Models installed in Ollama on this computer."
        actions={
          <Button size="xs" variant="ghost" leftIcon={<IconRefresh size={13} className={cn(local.isFetching && 'animate-spin')} />} onClick={() => void local.refetch()}>
            Refresh
          </Button>
        }
      />
      <Card>
        <div className="flex items-center gap-2.5 border-b border-border px-4 py-3 text-sm">
          <StatusDot tone={local.isLoading ? 'neutral' : ollama?.reachable ? 'success' : 'danger'} />
          <span className="font-medium text-fg">{local.isLoading ? 'Looking for Ollama…' : ollama?.reachable ? 'Ollama is running' : "Ollama isn't running"}</span>
          {ollama?.reachable && <span className="text-xs text-fg-subtle">{models.length} installed</span>}
          <div className="flex-1" />
          {!ollama?.reachable && !local.isLoading && (
            <Button size="xs" variant="secondary" leftIcon={<IconExternalLink size={12} />} onClick={() => void getBridge().openExternal('https://ollama.com/download')}>
              Install Ollama
            </Button>
          )}
        </div>
        {local.isLoading ? (
          <div className="space-y-2 p-4">
            <Skeleton className="h-8" />
            <Skeleton className="h-8" />
          </div>
        ) : (
          ollama?.reachable &&
          models.length > 0 && (
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-2xs uppercase tracking-wide text-fg-subtle">
                  <th className="px-4 py-2 font-medium">Model</th>
                  <th className="px-2 py-2 font-medium">Size</th>
                  <th className="px-2 py-2 font-medium">Family</th>
                  <th className="px-2 py-2 text-right font-medium">On disk</th>
                  <th className="w-10" />
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {models.map((m) => {
                  const embedding = m.is_embedding ?? looksLikeEmbedding(m.name)
                  return (
                    <tr key={m.name} className="hover:bg-hover">
                      <td className="px-4 py-2">
                        <span className="font-mono text-xs text-fg">{m.name}</span>
                        {embedding && (
                          <Badge size="xs" tone="info" className="ml-2">
                            embedding
                          </Badge>
                        )}
                      </td>
                      <td className="px-2 py-2 text-xs text-fg-muted">{m.parameter_size ?? '-'}</td>
                      <td className="px-2 py-2 text-xs text-fg-muted">{m.family || '-'}</td>
                      <td className="px-2 py-2 text-right text-xs tabular-nums text-fg-muted">{formatBytes(m.size)}</td>
                      <td className="px-2 py-1.5">
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <IconButton size="xs" label="Use for…" tooltip={false} icon={<IconDots size={14} />} />
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end">
                            <DropdownMenuLabel>Use for</DropdownMenuLabel>
                            {(embedding ? (['embedding'] as RoleName[]) : (['primary', 'fast', 'voice', 'planner', 'executor', 'vision'] as RoleName[])).map((r) => (
                              <DropdownMenuItem key={r} onSelect={() => useFor(r, m.name, embedding)}>
                                {ROLE_META[r].label}
                              </DropdownMenuItem>
                            ))}
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          )
        )}
        {ollama?.reachable && (
          <div className="border-t border-border p-4">
            <div className="mb-2.5 text-xs font-medium text-fg-muted">Download a model</div>
            <OllamaPull installed={models.map((m) => m.name)} />
          </div>
        )}
      </Card>
    </section>
  )
}
