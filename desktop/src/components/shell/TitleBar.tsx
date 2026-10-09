import { IconChevronDown, IconLayoutSidebar, IconSearch } from '@tabler/icons-react'
import { useNavigate } from 'react-router'
import { Logo } from '@/components/brand/Logo'
import { IconButton, Kbd, StatusDot, Tooltip, type Tone } from '@/components/ui'
import { ModelSwitcher } from '@/features/models/ModelSwitcher'
import { useBootstrap } from '@/hooks/core'
import { useModelPresets } from '@/hooks/models'
import { getBridge } from '@/lib/bridge'
import { modelShortName } from '@/lib/models'
import { cn, modKey } from '@/lib/utils'
import { useConnection } from '@/stores/connection'
import { useUI } from '@/stores/ui'
import { StopButton } from './StopControls'

/** Height matches the native title bar overlay (electron/main/window.ts). */
export const TITLEBAR_HEIGHT = 40

export function useWindowControlsInset(): { left: number; right: number } {
  const bridge = getBridge()
  if (!bridge.isDesktop) return { left: 0, right: 0 }
  if (bridge.platform === 'darwin') return { left: 76, right: 0 }
  return { left: 0, right: 140 }
}

export function StatusPill({ className }: { className?: string }) {
  const backend = useConnection((s) => s.backend)
  const socket = useConnection((s) => s.socket)
  const { data } = useBootstrap()
  const presets = useModelPresets()
  const navigate = useNavigate()
  const primary = data?.models.primary
  const preset = presets.data?.modified ? null : presets.data?.active

  let tone: Tone = 'success'
  let label = modelShortName(primary) || 'Ready'
  let tip = `Engine ready${preset ? ` · ${preset}` : ''} · chatting with ${primary ?? 'unknown'}. Click to switch models.`
  const ready = backend.state === 'ready'
  if (!ready) {
    tone = backend.state === 'failed' ? 'danger' : 'warning'
    label = backend.state === 'failed' ? 'Engine stopped' : 'Restarting engine…'
    tip = backend.message ?? label
  } else if (socket !== 'open') {
    tone = 'warning'
    label = 'Reconnecting…'
    tip = 'Live connection to the engine dropped; reconnecting.'
  }

  const pill = (
    <button
      type="button"
      onClick={ready ? undefined : () => navigate('/settings/models')}
      className={cn(
        'no-drag flex h-6.5 max-w-64 items-center gap-2 rounded-full border border-border bg-surface/60 px-2.5 text-xs text-fg-muted transition-colors hover:border-border-strong hover:text-fg data-[state=open]:border-border-strong data-[state=open]:text-fg',
        className
      )}
    >
      <StatusDot tone={tone} pulse={tone === 'warning'} />
      {ready && preset && <span className="shrink-0 text-fg-subtle">{preset}</span>}
      <span className="truncate font-medium">{label}</span>
      {ready && <IconChevronDown size={12} className="-ml-0.5 shrink-0 text-fg-subtle" />}
    </button>
  )

  if (!ready) {
    return (
      <Tooltip content={tip} side="bottom">
        {pill}
      </Tooltip>
    )
  }
  return (
    <ModelSwitcher primary={primary} tip={tip}>
      {pill}
    </ModelSwitcher>
  )
}

export function TitleBar({ minimal = false }: { minimal?: boolean }) {
  const inset = useWindowControlsInset()
  const toggleSidebar = useUI((s) => s.toggleSidebar)
  const collapsed = useUI((s) => s.sidebarCollapsed)
  const setPaletteOpen = useUI((s) => s.setPaletteOpen)

  return (
    <div
      className="drag relative flex shrink-0 items-center gap-2 bg-bg pl-2"
      style={{ height: TITLEBAR_HEIGHT, paddingLeft: inset.left || 8, paddingRight: inset.right || 10 }}
    >
      {!minimal && (
        <IconButton
          size="sm"
          label={collapsed ? 'Show sidebar' : 'Hide sidebar'}
          shortcut={`${modKey}+B`}
          side="bottom"
          icon={<IconLayoutSidebar size={16} />}
          onClick={toggleSidebar}
        />
      )}
      <div className="flex items-center gap-2 pl-1">
        <Logo size={18} />
        <span className="text-sm font-semibold tracking-tight text-fg">Sentient</span>
      </div>

      {!minimal && (
        <div className="pointer-events-none absolute inset-x-0 flex justify-center">
          <button
            type="button"
            onClick={() => setPaletteOpen(true)}
            className="no-drag pointer-events-auto flex h-7 w-[min(360px,34vw)] items-center gap-2 rounded-lg border border-border bg-surface/70 px-2.5 text-sm text-fg-subtle transition-colors hover:border-border-strong hover:text-fg-muted"
          >
            <IconSearch size={14} />
            <span className="flex-1 text-left">Search or jump to…</span>
            <span className="flex items-center gap-0.5">
              <Kbd>{modKey}</Kbd>
              <Kbd>K</Kbd>
            </span>
          </button>
        </div>
      )}

      <div className="flex-1" />
      {!minimal && (
        <div className="relative flex items-center gap-2">
          <StopButton />
          <StatusPill />
        </div>
      )}
    </div>
  )
}
