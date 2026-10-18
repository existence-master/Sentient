/**
 * #170 Always-visible microphone and camera indicator in the title bar (the tray shows the same, electron/main/tray.ts).
 *
 * Shown whenever a microphone or camera is live, on this computer or on a paired phone or glasses (docs/API.md
 * section 24). Clicking it lists which device and what for, each with Turn off, plus devices whose camera is switched
 * off on the device itself.
 */
import { IconCamera, IconCameraOff, IconMicrophone } from '@tabler/icons-react'
import { useEffect, useState } from 'react'
import { toast } from 'sonner'
import { Button, Popover, PopoverContent, PopoverTrigger } from '@/components/ui'
import { useSensors, useStopSensor } from '@/hooks/core'
import { errorMessage } from '@/lib/api'
import type { SensorState, SensorUse } from '@/lib/types'
import { formatTime } from '@/lib/utils'
import { useConnection } from '@/stores/connection'
import { useUI } from '@/stores/ui'

export function sensorLabel(s: Pick<SensorState, 'mic' | 'camera'>): string {
  if (s.mic && s.camera) return 'Mic and camera on'
  return s.mic ? 'Mic on' : 'Camera on'
}

function sensorName(u: SensorUse): string {
  return `${u.sensor === 'mic' ? 'Microphone' : 'Camera'} on ${u.device === 'This computer' ? 'this computer' : u.device}`
}

function SensorRow({ use, asked, onStop, busy }: { use: SensorUse; asked: boolean; onStop: () => void; busy: boolean }) {
  const Icon = use.sensor === 'mic' ? IconMicrophone : IconCamera
  return (
    <li className="flex items-center gap-3 px-3 py-2.5">
      <span className="flex size-8 shrink-0 items-center justify-center rounded-full bg-danger/10 text-danger">
        <Icon size={16} />
      </span>
      <div className="min-w-0 flex-1">
        <div className="truncate text-sm font-medium text-fg">{sensorName(use)}</div>
        <div className="truncate text-xs text-fg-subtle">
          {use.purpose} · since {formatTime(use.since)}
        </div>
      </div>
      {use.can_stop &&
        (asked ? (
          <span className="shrink-0 text-xs text-fg-subtle">Asked to turn off</span>
        ) : (
          <Button size="xs" variant="danger" loading={busy} onClick={onStop}>
            Turn off
          </Button>
        ))}
    </li>
  )
}

/** Title bar indicator: hidden while no microphone or camera is live. */
export function SensorIndicator() {
  const ready = useConnection((s) => s.backend.state === 'ready')
  const { data } = useSensors()
  const open = useUI((s) => s.sensorsOpen)
  const setOpen = useUI((s) => s.setSensorsOpen)
  const stop = useStopSensor()
  const [asked, setAsked] = useState<Set<string>>(new Set())
  const [stopping, setStopping] = useState<string | null>(null)

  const live = ready && !!data && (data.mic || data.camera)
  useEffect(() => {
    if (!live && open) setOpen(false) // closed with the last one, so it doesn't pop open by itself next time
  }, [live, open, setOpen])

  if (!live || !data) return null
  const label = sensorLabel(data)
  const devices = [...new Set(data.active.map((u) => u.device))]
  const where = devices.length === 1 ? (devices[0] === 'This computer' ? 'this computer' : devices[0]) : `${devices.length} devices`

  const turnOff = (id?: string) => {
    setStopping(id ?? 'all')
    stop.mutate(id, {
      onSuccess: (r) => setAsked((prev) => new Set([...prev, ...r.asked])),
      onError: (err) => toast.error("Couldn't turn it off", { description: errorMessage(err) }),
      onSettled: () => setStopping(null)
    })
  }

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button
          type="button"
          data-testid="sensor-indicator"
          aria-label={`${label}: ${where}. Show details`}
          title={`${label}: ${where}`}
          className="no-drag flex h-6.5 max-w-52 items-center gap-1.5 rounded-full border border-danger/40 bg-danger/10 px-2.5 text-xs font-medium text-danger transition-colors hover:bg-danger/18 data-[state=open]:bg-danger/18"
        >
          <span className="relative flex size-2 shrink-0">
            <span className="absolute inline-flex size-full animate-ping rounded-full bg-danger opacity-60" />
            <span className="relative inline-flex size-2 rounded-full bg-danger" />
          </span>
          {data.mic && <IconMicrophone size={13} className="shrink-0" />}
          {data.camera && <IconCamera size={13} className="shrink-0" />}
          <span className="truncate">{label}</span>
        </button>
      </PopoverTrigger>
      <PopoverContent align="start" className="w-[22rem] p-0">
        <div className="border-b border-border px-3 py-2.5">
          <div className="text-sm font-semibold text-fg">Live right now</div>
          <div className="text-xs text-fg-subtle">
            Sentient shows this whenever a microphone or camera is on, on this computer or on your devices.
          </div>
        </div>
        <ul className="divide-y divide-border" data-testid="sensor-list">
          {data.active.map((u) => (
            <SensorRow key={u.id} use={u} asked={asked.has(u.id)} busy={stopping === u.id} onStop={() => turnOff(u.id)} />
          ))}
          {data.camera_off.map((d) => (
            <li key={`off-${d.device_id}`} className="flex items-center gap-3 px-3 py-2.5">
              <span className="flex size-8 shrink-0 items-center justify-center rounded-full bg-hover text-fg-muted">
                <IconCameraOff size={16} />
              </span>
              <div className="min-w-0 flex-1">
                <div className="truncate text-sm font-medium text-fg">Camera switched off on {d.device}</div>
                <div className="text-xs text-fg-subtle">Only the device itself can turn it back on.</div>
              </div>
            </li>
          ))}
        </ul>
        {data.active.filter((u) => u.can_stop).length > 1 && (
          <div className="flex justify-end border-t border-border px-3 py-2">
            <Button size="xs" variant="danger" loading={stopping === 'all'} onClick={() => turnOff()}>
              Turn off all
            </Button>
          </div>
        )}
      </PopoverContent>
    </Popover>
  )
}
