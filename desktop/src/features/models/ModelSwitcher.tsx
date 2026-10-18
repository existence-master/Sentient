import { IconArrowBackUp, IconCheck, IconKey, IconSettings, IconStethoscope } from '@tabler/icons-react'
import { useEffect, useState, type ReactNode } from 'react'
import { useNavigate } from 'react-router'
import { toast } from 'sonner'
import { Popover, PopoverContent, PopoverTrigger, SegmentedControl, Spinner, Tooltip } from '@/components/ui'
import { useConfig } from '@/hooks/core'
import { useModelPresets, useSetLocalOnly, useSetRoles } from '@/hooks/models'
import { errorMessage } from '@/lib/api'
import { modelShortName } from '@/lib/models'
import { recentChatModels, rememberChatModel } from '@/lib/recentModels'
import { cn } from '@/lib/utils'
import { LocalOnlyPresetConfirm } from './LocalOnlyPresetConfirm'
import { PresetFixes } from './PresetFixes'
import { ProviderKeyDialog } from './ProviderKeyDialog'
import { usePresetSwitch } from './usePresetSwitch'

/** Cloud providers the built-in Cloud and Mixed presets can use, in the order they are tried. */
const PRESET_CLOUD = [
  { id: 'anthropic', label: 'Anthropic' },
  { id: 'openai', label: 'OpenAI' },
  { id: 'openrouter', label: 'OpenRouter' }
]

/**
 * The title bar model menu: Local only or May use cloud (#148), presets (a check on the active one), recent chat models, the model check-up and the
 * full Models settings. Missing pieces (a download, a key) are offered right in the menu.
 */
export function ModelSwitcher({ primary, tip, children }: { primary: string | undefined; tip: string; children: ReactNode }) {
  const [open, setOpen] = useState(false)
  const navigate = useNavigate()
  const presets = useModelPresets()
  const setRoles = useSetRoles()
  const config = useConfig()
  const setLocalOnly = useSetLocalOnly()
  const localOnly = !!config.data?.models.local_only
  const sw = usePresetSwitch()
  const [recent, setRecent] = useState<string[]>(() => recentChatModels())

  useEffect(() => {
    rememberChatModel(primary)
    setRecent(recentChatModels())
  }, [primary])

  const list = presets.data
  // a missing key can be fixed here; a preset Local only blocks is fixed with the switch above
  const locked = list?.presets.filter((p) => !p.available && p.blocked_by !== 'local_only') ?? []
  const blocked = list?.presets.some((p) => p.blocked_by === 'local_only') ?? false
  const others = recent.filter((m) => m !== primary).slice(0, 4)

  const go = (path: string) => {
    setOpen(false)
    navigate(path)
  }
  const chatWith = (model: string) =>
    setRoles.mutate(
      { primary: model },
      {
        onSuccess: () => toast.success(`Chatting with ${modelShortName(model)}`),
        onError: (e) => toast.error("Couldn't change the model", { description: errorMessage(e) })
      }
    )
  const chooseWhere = (value: string) => {
    const on = value === 'local'
    if (on === localOnly) return
    setLocalOnly.mutate(on, {
      onSuccess: () => toast.success(on ? 'Local only: nothing goes to a cloud model' : 'Cloud models may be used again'),
      onError: (e) => toast.error("Couldn't change where AI runs", { description: errorMessage(e) })
    })
  }
  const addKey = (id: string) => {
    setOpen(false)
    sw.addKey(id)
  }

  return (
    <>
      <Popover open={open} onOpenChange={setOpen}>
        <Tooltip content={tip} side="bottom">
          <PopoverTrigger asChild>{children}</PopoverTrigger>
        </Tooltip>
        <PopoverContent align="end" side="bottom" className="no-drag max-h-[min(640px,80vh)] w-84 overflow-y-auto p-1.5">
          <Heading>Where AI runs</Heading>
          <div className="px-1.5 pb-1">
            <SegmentedControl
              size="sm"
              fullWidth
              aria-label="Where AI runs"
              value={localOnly ? 'local' : 'cloud'}
              onChange={chooseWhere}
              options={[
                { value: 'local', label: 'Local only', disabled: !config.data || setLocalOnly.isPending },
                { value: 'cloud', label: 'May use cloud', disabled: !config.data || setLocalOnly.isPending }
              ]}
            />
          </div>
          <p className="px-2.5 pb-1 text-2xs leading-relaxed text-fg-subtle">
            {localOnly
              ? 'Nothing is sent to a model outside this computer, not even a backup model or a chat with its own model.'
              : 'Cloud models you set up may be used. Paid ones count toward your monthly budget.'}
          </p>
          <Divider />
          <Heading>Presets</Heading>
          {presets.isLoading && (
            <div className="flex items-center gap-2 px-2.5 py-2 text-xs text-fg-subtle">
              <Spinner size={12} /> Loading…
            </div>
          )}
          {presets.isError && <div className="px-2.5 py-2 text-xs text-fg-subtle">Presets aren't available right now.</div>}
          {list?.presets.map((p) => (
            <Row
              key={p.name}
              active={p.active}
              disabled={!p.available || sw.busy}
              busy={sw.pending === p.name}
              hint={
                p.blocked_by === 'local_only'
                  ? 'Uses cloud models, so Local only blocks it'
                  : p.available
                    ? p.active && list.modified
                      ? 'Changed since you picked it'
                      : (p.description ?? summary(p.roles.primary))
                    : (p.reason ?? undefined)
              }
              onClick={() => sw.apply(p.name) && setOpen(false)}
            >
              {p.name}
            </Row>
          ))}
          {locked.length > 0 && (
            <div className="flex flex-wrap items-center gap-1 px-2.5 pb-1.5 pt-1 text-2xs text-fg-subtle">
              <IconKey size={12} className="shrink-0" />
              <span>Add a key:</span>
              {PRESET_CLOUD.map((c) => (
                <button key={c.id} type="button" onClick={() => addKey(c.id)} className="rounded px-1 text-accent-text hover:underline">
                  {c.label}
                </button>
              ))}
            </div>
          )}
          {blocked && (
            <p className="px-2.5 pb-1 pt-1 text-2xs leading-relaxed text-fg-subtle">
              Local only is on, so presets with cloud models can't be picked. Choose May use cloud above to use them.
            </p>
          )}
          {list?.can_undo && (
            <Row icon={<IconArrowBackUp size={15} />} disabled={sw.busy} onClick={sw.undo} hint={list.undo_preset ?? undefined}>
              Undo last switch
            </Row>
          )}
          <p className="px-2.5 pb-1 pt-1.5 text-2xs leading-relaxed text-fg-subtle">
            This changes the default. A chat with its own model, picked in its message box, keeps it.
          </p>

          {sw.missing.length > 0 && (
            <>
              <Divider />
              <Heading>Still needed</Heading>
              <PresetFixes className="px-1 pb-1" missing={sw.missing} pull={sw.pull} onPull={(n) => void sw.pull.pull(n)} onAddKey={addKey} />
            </>
          )}

          {others.length > 0 && (
            <>
              <Divider />
              <Heading>Recent models for chat</Heading>
              {others.map((m) => (
                <Row key={m} active={false} disabled={setRoles.isPending} onClick={() => chatWith(m)}>
                  <span className="font-mono text-xs">{modelShortName(m)}</span>
                </Row>
              ))}
            </>
          )}

          <Divider />
          <Row icon={<IconStethoscope size={15} />} onClick={() => go('/settings/models?check=1')}>
            Check my models
          </Row>
          <Row icon={<IconSettings size={15} />} onClick={() => go('/settings/models')}>
            Set up models…
          </Row>
        </PopoverContent>
      </Popover>
      <ProviderKeyDialog provider={sw.keyFor} open={!!sw.keyFor} onOpenChange={(o) => !o && sw.setKeyFor(null)} />
      <LocalOnlyPresetConfirm sw={sw} />
    </>
  )
}

function summary(primary: string | null | undefined): string | undefined {
  return primary ? `Chat: ${modelShortName(primary)}` : undefined
}

function Heading({ children }: { children: ReactNode }) {
  return <div className="px-2.5 pb-1 pt-1.5 text-2xs font-medium uppercase tracking-wide text-fg-faint">{children}</div>
}

function Divider() {
  return <div className="mx-1 my-1.5 border-t border-border" />
}

function Row({
  active,
  disabled,
  busy,
  hint,
  icon,
  onClick,
  children
}: {
  active?: boolean
  disabled?: boolean
  busy?: boolean
  hint?: string
  icon?: ReactNode
  onClick: () => void
  children: ReactNode
}) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      className="flex w-full items-center gap-2.5 rounded-lg px-2.5 py-1.5 text-left text-sm text-fg hover:bg-active disabled:cursor-default disabled:hover:bg-transparent"
    >
      {icon && <span className="shrink-0 text-fg-muted">{icon}</span>}
      <span className={cn('min-w-0 flex-1', disabled && !busy && 'opacity-60')}>
        <span className="block truncate">{children}</span>
        {hint && <span className="block truncate text-2xs text-fg-subtle">{hint}</span>}
      </span>
      {busy ? (
        <Spinner size={13} />
      ) : (
        active !== undefined && <IconCheck size={14} className={cn('shrink-0 text-accent-text', !active && 'invisible')} />
      )}
    </button>
  )
}
