import {
  IconBellRinging,
  IconCamera,
  IconDots,
  IconExternalLink,
  IconMapPin,
  IconPencil,
  IconScreenshot,
  IconShieldLock,
  IconSpeakerphone,
  IconTextSize,
  IconTrash,
  IconX,
  type Icon
} from '@tabler/icons-react'
import { AnimatePresence, motion } from 'motion/react'
import { useState } from 'react'
import { toast } from 'sonner'
import {
  Badge,
  Button,
  ConfirmDialog,
  Dialog,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
  Tooltip
} from '@/components/ui'
import { errorMessage } from '@/lib/api'
import { getBridge } from '@/lib/bridge'
import type { DeviceInvokeResult, DeviceNode } from '@/lib/types'
import { cn } from '@/lib/utils'
import { DesktopPrivacy } from './DesktopPrivacy'
import { useDeviceActions } from './hooks'
import { BatteryPill, capabilityChips, DeviceTile, platformLabel, presence } from './meta'

interface TestAction {
  capability: string
  label: string
  icon: Icon
  params?: Record<string, unknown>
  done?: (name: string) => string
}

const TESTS: TestAction[] = [
  { capability: 'camera.photo', label: 'Take photo', icon: IconCamera },
  { capability: 'screen.capture', label: 'Capture screen', icon: IconScreenshot },
  {
    capability: 'notify.show',
    label: 'Show notification',
    icon: IconBellRinging,
    params: { title: 'Sentient', text: 'Hello from Sentient. This is a test notification.' },
    done: (n) => `Sent a notification to ${n}`
  },
  { capability: 'display.text', label: 'Show text', icon: IconTextSize, params: { text: 'Hello from Sentient' }, done: (n) => `Sent text to ${n}` },
  { capability: 'speak', label: 'Speak', icon: IconSpeakerphone, params: { text: 'Hi, this is Sentient. Your device is working.' }, done: (n) => `${n} is speaking` },
  { capability: 'location.get', label: 'Get location', icon: IconMapPin }
]

/** Plain-language reason for a failed test (§13 invoke `code`). */
function invokeFailure(res: DeviceInvokeResult, node: DeviceNode): string {
  const err = res.error
  const code = res.code ?? (typeof err === 'object' && err ? err.code : undefined)
  const message = typeof err === 'string' ? err : err?.message
  switch (code) {
    case 'offline':
      return `${node.name} is offline right now.`
    case 'unsupported':
      return `${node.name} can't do that.`
    case 'timeout':
      return `${node.name} didn't answer in time.`
    case 'not_allowed':
      return 'Turn on "Let Sentient see my screen when I ask" below, then try again.'
    default:
      return message || 'The device did not answer.'
  }
}

type Outcome = { kind: 'image'; src: string; label: string } | { kind: 'location'; lat: number; lon: number; accuracy?: number; label?: string }

export function DeviceCard({ node }: { node: DeviceNode }) {
  const actions = useDeviceActions()
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(node.name)
  const [confirm, setConfirm] = useState(false)
  const [busy, setBusy] = useState<string | null>(null)
  const [outcome, setOutcome] = useState<Outcome | null>(null)
  const [zoom, setZoom] = useState(false)
  const isDesktop = node.node_id === 'desktop' || node.kind === 'desktop'
  const tests = TESTS.filter((t) => node.capabilities.includes(t.capability))
  const chips = capabilityChips(node.capabilities)

  const commit = () => {
    const name = draft.trim()
    setEditing(false)
    if (name && name !== node.name) {
      actions.rename.mutate({ id: node.node_id, name }, { onError: (e) => toast.error("Couldn't rename the device", { description: errorMessage(e) }) })
    }
  }

  const run = (t: TestAction) => {
    setBusy(t.capability)
    actions.invoke.mutate(
      { id: node.node_id, capability: t.capability, params: t.params },
      {
        onSettled: () => setBusy(null),
        onError: (e) => toast.error(`${t.label} didn't work`, { description: errorMessage(e) }),
        onSuccess: (res: DeviceInvokeResult) => {
          if (!res.ok) {
            toast.error(`${t.label} didn't work`, { description: invokeFailure(res, node) })
            return
          }
          const d = res.data ?? {}
          if (typeof d.base64 === 'string') {
            setOutcome({ kind: 'image', src: `data:${d.mime ?? 'image/jpeg'};base64,${d.base64}`, label: t.capability === 'screen.capture' ? 'Screenshot' : 'Photo' })
          } else if (typeof d.lat === 'number' && typeof d.lon === 'number') {
            setOutcome({ kind: 'location', lat: d.lat, lon: d.lon, accuracy: d.accuracy_m, label: d.label })
          } else {
            toast.success(t.done?.(node.name) ?? `${t.label} worked`)
          }
        }
      }
    )
  }

  return (
    <motion.div layout initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} className="flex flex-col rounded-2xl border border-border bg-surface p-4">
      <div className="flex items-start gap-3.5">
        <DeviceTile kind={node.kind} online={node.online} />
        <div className="min-w-0 flex-1">
          {editing ? (
            <input
              autoFocus
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onBlur={commit}
              onKeyDown={(e) => {
                if (e.key === 'Enter') commit()
                if (e.key === 'Escape') setEditing(false)
              }}
              aria-label="Device name"
              className="h-7 w-full rounded-md border border-accent/50 bg-field px-2 text-sm font-medium text-fg outline-none ring-3 ring-accent/15"
            />
          ) : (
            <div className="flex min-w-0 items-center gap-2">
              <span className="truncate text-md font-medium text-fg">{node.name}</span>
              {isDesktop && <Badge size="xs">This computer</Badge>}
            </div>
          )}
          <div className="mt-0.5 flex items-center gap-1.5 text-xs text-fg-subtle">
            {platformLabel(node.platform) && <span>{platformLabel(node.platform)}</span>}
            {platformLabel(node.platform) && <span className="text-fg-faint">·</span>}
            <span className={cn(node.online && 'text-success')}>{presence(node)}</span>
            {node.online && node.connection && !isDesktop && (
              <>
                <span className="text-fg-faint">·</span>
                <span>{node.connection === 'lan' ? 'On your Wi-Fi' : 'On this computer'}</span>
              </>
            )}
            {node.kind === 'glasses' && node.worn !== null && node.worn !== undefined && node.online && (
              <>
                <span className="text-fg-faint">·</span>
                <span>{node.worn ? 'Being worn' : 'Not being worn'}</span>
              </>
            )}
          </div>
        </div>
        <BatteryPill node={node} />
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button type="button" aria-label="Device actions" className="flex size-7 items-center justify-center rounded-md text-fg-subtle hover:bg-hover hover:text-fg">
              <IconDots size={16} />
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem
              icon={<IconPencil />}
              onSelect={() => {
                setDraft(node.name)
                setTimeout(() => setEditing(true), 0)
              }}
            >
              Rename
            </DropdownMenuItem>
            {!isDesktop && (
              <>
                <DropdownMenuSeparator />
                <DropdownMenuItem icon={<IconTrash />} danger onSelect={() => setConfirm(true)}>
                  Remove
                </DropdownMenuItem>
              </>
            )}
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      {chips.length > 0 && (
        <div className="mt-3.5 flex flex-wrap gap-1.5">
          {chips.map((c) => (
            <span key={c.label} className="inline-flex h-6 items-center gap-1.5 rounded-full bg-active/70 px-2.5 text-xs text-fg-muted">
              <c.icon size={12} className="text-fg-subtle" />
              {c.label}
            </span>
          ))}
        </div>
      )}

      {tests.length > 0 && (
        <div className="mt-3.5 flex flex-wrap gap-1.5 border-t border-border pt-3.5">
          <span className="mr-1 self-center text-2xs font-medium uppercase tracking-wide text-fg-faint">Try it</span>
          {tests.map((t) => {
            const button = (
              <Button
                key={t.capability}
                size="xs"
                variant="outline"
                disabled={!node.online || (busy !== null && busy !== t.capability)}
                loading={busy === t.capability}
                leftIcon={<t.icon size={13} />}
                onClick={() => run(t)}
              >
                {t.label}
              </Button>
            )
            return node.online ? (
              button
            ) : (
              <Tooltip key={t.capability} content={`${node.name} is offline`}>
                <span>{button}</span>
              </Tooltip>
            )
          })}
        </div>
      )}

      <AnimatePresence initial={false}>
        {outcome && (
          <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
            <div className="relative mt-3 flex items-start gap-3 rounded-xl border border-border bg-sunken/50 p-2.5">
              {outcome.kind === 'image' ? (
                <>
                  <button type="button" onClick={() => setZoom(true)} className="overflow-hidden rounded-lg border border-border" title="View larger">
                    <img src={outcome.src} alt={outcome.label} className="h-32 max-w-60 object-cover" />
                  </button>
                  <div className="pt-1 text-xs text-fg-subtle">
                    <div className="text-sm font-medium text-fg">{outcome.label} from {node.name}</div>
                    Just now. It isn&apos;t saved anywhere.
                  </div>
                </>
              ) : (
                <>
                  <div className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-accent/10 text-accent-text">
                    <IconMapPin size={18} />
                  </div>
                  <div className="min-w-0 flex-1 text-xs text-fg-subtle">
                    <div className="text-sm font-medium text-fg">{outcome.label ?? 'Current location'}</div>
                    {outcome.lat.toFixed(4)}, {outcome.lon.toFixed(4)}
                    {outcome.accuracy ? ` · within ${Math.round(outcome.accuracy)} m` : ''}
                    <div>
                      <button
                        type="button"
                        onClick={() => void getBridge().openExternal(`https://www.google.com/maps?q=${outcome.lat},${outcome.lon}`)}
                        className="mt-1 inline-flex items-center gap-1 text-accent-text hover:underline"
                      >
                        Open in Maps <IconExternalLink size={11} />
                      </button>
                    </div>
                  </div>
                </>
              )}
              <button
                type="button"
                aria-label="Dismiss"
                onClick={() => setOutcome(null)}
                className="absolute right-1.5 top-1.5 flex size-6 items-center justify-center rounded-md text-fg-subtle hover:bg-hover hover:text-fg"
              >
                <IconX size={13} />
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {isDesktop && (
        <div className="mt-3.5 border-t border-border pt-3.5">
          <div className="mb-2.5 flex items-center gap-1.5 text-2xs font-medium uppercase tracking-wide text-fg-faint">
            <IconShieldLock size={12} /> Privacy
          </div>
          <DesktopPrivacy />
        </div>
      )}

      <ConfirmDialog
        open={confirm}
        onOpenChange={setConfirm}
        title={`Remove ${node.name}?`}
        description="Sentient won't be able to reach it any more. To use it again, you'll need to pair it with a new code."
        confirmLabel="Remove device"
        onConfirm={async () => {
          try {
            await actions.remove.mutateAsync(node.node_id)
            toast.success(`${node.name} was removed`)
          } catch (e) {
            toast.error("Couldn't remove the device", { description: errorMessage(e) })
          }
        }}
      />
      <Dialog open={zoom && outcome?.kind === 'image'} onOpenChange={setZoom} size="xl" title={outcome?.kind === 'image' ? `${outcome.label} from ${node.name}` : ''}>
        {outcome?.kind === 'image' && <img src={outcome.src} alt="" className="max-h-[70vh] w-full rounded-lg object-contain" />}
      </Dialog>
    </motion.div>
  )
}
