import { IconDeviceDesktop, IconDeviceMobile, IconDevices, IconEyeglass2, IconMessages, IconPlus } from '@tabler/icons-react'
import { motion } from 'motion/react'
import { useEffect, useState } from 'react'
import { useLocation, useNavigate, useParams } from 'react-router'
import { Alert, Button, PageHeader, Skeleton, Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui'
import { MessagingApps } from '@/features/channels/MessagingApps'
import { AddDeviceDialog } from '@/features/devices/AddDeviceDialog'
import { DeviceCard } from '@/features/devices/DeviceCard'
import { devicesUnavailable, useDevices } from '@/features/devices/hooks'
import { errorMessage } from '@/lib/api'

type Tab = 'devices' | 'messaging'

/** Devices (`/devices`) and messaging apps (`/devices/messaging`). */
export function DevicesPage() {
  const params = useParams()
  const navigate = useNavigate()
  const location = useLocation()
  const tab: Tab = params.tab === 'messaging' ? 'messaging' : 'devices'
  const [adding, setAdding] = useState(false)
  const addNonce = (location.state as { add?: number } | null)?.add

  // "Add a device" from the command palette, or `?add=1`.
  useEffect(() => {
    if (addNonce || /(^|[?&])add=1\b/.test(location.search)) setAdding(true)
  }, [addNonce, location.search])

  return (
    <div className="h-full overflow-y-auto">
      <PageHeader
        icon={<IconDevices />}
        title="Devices"
        description="Your phone, glasses and this computer. Sentient can see, hear and reach you through them, and you can message it from the apps you already use."
        actions={
          tab === 'devices' && (
            <Button variant="primary" leftIcon={<IconPlus size={16} />} onClick={() => setAdding(true)}>
              Add a device
            </Button>
          )
        }
      />
      <Tabs value={tab} onValueChange={(v) => navigate(v === 'messaging' ? '/devices/messaging' : '/devices', { replace: true })}>
        <div className="px-8">
          <TabsList>
            <TabsTrigger value="devices">
              <IconDevices size={15} /> Devices
            </TabsTrigger>
            <TabsTrigger value="messaging">
              <IconMessages size={15} /> Messaging apps
            </TabsTrigger>
          </TabsList>
        </div>
        <div className="max-w-6xl px-8 pb-12 pt-6">
          <TabsContent value="devices">
            <DevicesTab onAdd={() => setAdding(true)} />
          </TabsContent>
          <TabsContent value="messaging">
            <MessagingApps />
          </TabsContent>
        </div>
      </Tabs>
      <AddDeviceDialog open={adding} onOpenChange={setAdding} />
    </div>
  )
}

function DevicesTab({ onAdd }: { onAdd: () => void }) {
  const { data, isLoading, error } = useDevices()

  if (isLoading) {
    return (
      <div className="grid gap-4 xl:grid-cols-2">
        {[0, 1].map((i) => (
          <Skeleton key={i} className="h-48 rounded-2xl" />
        ))}
      </div>
    )
  }
  if (error && devicesUnavailable(error)) {
    return <VisionHero onAdd={onAdd} unavailable />
  }
  if (error) return <Alert tone="danger" title="Couldn't load your devices">{errorMessage(error)}</Alert>

  const devices = [...(data ?? [])].sort((a, b) => Number(b.node_id === 'desktop') - Number(a.node_id === 'desktop') || Number(b.online) - Number(a.online))
  const others = devices.filter((d) => d.kind !== 'desktop')

  return (
    <div className="space-y-6">
      {!others.length && <VisionHero onAdd={onAdd} />}
      {devices.length > 0 && (
        <div className="grid items-start gap-4 xl:grid-cols-2">
          {devices.map((d) => (
            <DeviceCard key={d.node_id} node={d} />
          ))}
        </div>
      )}
    </div>
  )
}

const IDEAS = [
  { icon: IconDeviceMobile, title: 'Your phone', text: 'Send a photo of a receipt, share where you are, and get reminders when you are out.' },
  { icon: IconEyeglass2, title: 'Smart glasses', text: 'Ask about what you are looking at and see short answers right in front of you.' },
  { icon: IconDeviceDesktop, title: 'Another computer', text: 'Use its webcam and speaker, so Sentient can help from the other room too.' }
]

function VisionHero({ onAdd, unavailable }: { onAdd: () => void; unavailable?: boolean }) {
  return (
    <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} className="relative overflow-hidden rounded-2xl border border-border bg-elevated/50 p-7">
      <div
        aria-hidden
        className="pointer-events-none absolute -right-20 -top-24 size-80 rounded-full opacity-[0.1] blur-3xl"
        style={{ background: 'radial-gradient(circle, var(--accent), transparent 70%)' }}
      />
      <div className="relative flex flex-wrap items-center gap-6">
        <div className="flex -space-x-3">
          {[IconDeviceMobile, IconEyeglass2].map((I, i) => (
            <div key={i} className="flex size-14 items-center justify-center rounded-2xl border border-border-strong bg-surface text-accent-text shadow-soft" style={{ transform: `rotate(${i ? 6 : -6}deg)` }}>
              <I size={26} stroke={1.5} />
            </div>
          ))}
        </div>
        <div className="min-w-0 flex-1">
          <h2 className="text-lg font-semibold tracking-tight text-fg">Take Sentient with you</h2>
          <p className="mt-1 max-w-xl text-sm text-fg-muted">
            {unavailable
              ? "Pairing devices isn't available in this version of Sentient's engine yet. Here's what it will let you do."
              : 'Pair a device on your Wi-Fi in under a minute. Everything stays between your devices and this computer.'}
          </p>
        </div>
        {!unavailable && (
          <Button variant="primary" leftIcon={<IconPlus size={16} />} onClick={onAdd}>
            Add a device
          </Button>
        )}
      </div>
      <div className="relative mt-6 grid gap-3 md:grid-cols-3">
        {IDEAS.map((f) => (
          <div key={f.title} className="flex gap-3 rounded-xl border border-border bg-surface p-3.5">
            <div className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-accent/10 text-accent-text">
              <f.icon size={17} />
            </div>
            <div>
              <div className="text-sm font-medium text-fg">{f.title}</div>
              <div className="mt-0.5 text-xs leading-relaxed text-fg-subtle">{f.text}</div>
            </div>
          </div>
        ))}
      </div>
    </motion.div>
  )
}
