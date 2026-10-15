/** Which devices Sentient may see and use, for integrations with an entity picker (Home Assistant, #175). */
import { IconAlertTriangle, IconLock, IconSearch } from '@tabler/icons-react'
import { useMemo, useState } from 'react'
import { toast } from 'sonner'
import { Alert, Badge, Input, SegmentedControl, Skeleton, Switch } from '@/components/ui'
import { useIntegrationActions, useIntegrationEntities } from '@/hooks/integrations'
import { errorMessage } from '@/lib/api'
import type { EntityChoice, Integration, IntegrationEntity } from '@/lib/types'
import { cn, humanize } from '@/lib/utils'

const KIND: Record<string, string> = {
  light: 'Light',
  switch: 'Switch',
  input_boolean: 'Switch',
  fan: 'Fan',
  climate: 'Thermostat',
  water_heater: 'Water heater',
  lock: 'Lock',
  cover: 'Blind or door',
  alarm_control_panel: 'Alarm',
  binary_sensor: 'Sensor',
  sensor: 'Sensor',
  scene: 'Scene',
  script: 'Script',
  media_player: 'Speaker or TV',
  camera: 'Camera',
  person: 'Person',
  automation: 'Automation'
}

/** Locks and alarms always ask before Sentient changes them, whatever is chosen here. */
const ALWAYS_ASKS = new Set(['lock', 'alarm_control_panel'])

export function kindLabel(domain: string): string {
  return KIND[domain] ?? humanize(domain)
}

export function EntityPicker({ integration: i }: { integration: Integration }) {
  const query = useIntegrationEntities(i.id, i.connected)
  const actions = useIntegrationActions()
  const [search, setSearch] = useState('')
  const data = query.data

  const groups = useMemo(() => {
    const words = search.toLowerCase().split(/\s+/).filter(Boolean)
    const out = new Map<string, IntegrationEntity[]>()
    for (const e of data?.entities ?? []) {
      const hay = `${e.name} ${e.entity_id} ${e.area} ${kindLabel(e.domain)}`.toLowerCase()
      if (words.some((w) => !hay.includes(w))) continue
      const room = e.area || 'No room'
      out.set(room, [...(out.get(room) ?? []), e])
    }
    return [...out.entries()].sort(([a], [b]) => (a === 'No room' ? 1 : b === 'No room' ? -1 : a.localeCompare(b)))
  }, [data, search])

  if (!i.connected) return <p className="text-xs text-fg-subtle">Connect {i.display_name} to choose which devices Sentient may use.</p>
  if (query.isLoading) return <Skeleton className="h-40 w-full rounded-xl" />
  if (query.error || !data)
    return (
      <Alert tone="danger" icon={<IconAlertTriangle />} title="Couldn't load your devices">
        {errorMessage(query.error)}
      </Alert>
    )

  const all = data.entities.map((e) => e.entity_id)
  const allowed = data.entities.filter((e) => e.allowed).length
  const save = (choice: EntityChoice) =>
    actions.setEntities.mutate({ id: i.id, choice }, { onError: (e) => toast.error("Couldn't save your choice", { description: errorMessage(e) }) })
  const toggle = (id: string, on: boolean) => {
    const current = data.mode === 'all' ? all : data.picked
    save({ mode: 'picked', picked: on ? Array.from(new Set([...current, id])) : current.filter((x) => x !== id) })
  }

  return (
    <div className="space-y-3">
      <SegmentedControl
        fullWidth
        aria-label="Which devices Sentient may use"
        value={data.mode}
        onChange={(mode) => save({ mode, picked: mode === 'picked' && data.mode === 'all' ? all : data.picked })}
        options={[
          { value: 'all', label: 'Every device' },
          { value: 'picked', label: 'Only the ones I pick' }
        ]}
      />
      <p className="text-xs leading-relaxed text-fg-subtle">
        {data.mode === 'all'
          ? `Sentient can use all ${data.entities.length} devices, and new ones you add later. Turn one off below to choose.`
          : `Sentient can use ${allowed} of ${data.entities.length} devices. It can't see the others at all.`}{' '}
        Locks, alarms and garage doors always ask you first.
      </p>
      {data.entities.length > 8 && (
        <Input size="sm" leftIcon={<IconSearch />} placeholder="Find a device or room" value={search} onChange={(e) => setSearch(e.target.value)} />
      )}
      <div className="max-h-80 overflow-y-auto rounded-xl border border-border bg-surface">
        {groups.length === 0 && <p className="px-3.5 py-3 text-xs text-fg-subtle">No devices match.</p>}
        {groups.map(([room, items]) => (
          <div key={room}>
            <div className="sticky top-0 z-[1] border-b border-border bg-sunken/90 px-3.5 py-1.5 text-2xs font-medium uppercase tracking-wide text-fg-subtle backdrop-blur">
              {room}
            </div>
            <ul className="divide-y divide-border">
              {items.map((e) => (
                <li key={e.entity_id} className={cn('flex items-center gap-3 px-3.5 py-2', !e.allowed && 'opacity-60')}>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-1.5">
                      <span className="truncate text-sm font-medium text-fg">{e.name}</span>
                      {ALWAYS_ASKS.has(e.domain) && (
                        <Badge size="xs" tone="danger" icon={<IconLock />}>
                          Always asks
                        </Badge>
                      )}
                    </div>
                    <div className="truncate text-xs text-fg-subtle">
                      {kindLabel(e.domain)}
                      {e.state ? ` · ${humanize(e.state).toLowerCase()}` : ''}
                    </div>
                  </div>
                  <Switch
                    size="sm"
                    aria-label={`Let Sentient use ${e.name}`}
                    checked={e.allowed}
                    disabled={actions.setEntities.isPending}
                    onCheckedChange={(on) => toggle(e.entity_id, on)}
                  />
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </div>
  )
}
