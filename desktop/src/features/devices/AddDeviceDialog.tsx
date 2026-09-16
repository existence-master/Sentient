import { IconCircleCheckFilled, IconDeviceMobile, IconEyeglass2, IconRefresh, IconShieldLock, IconWifi } from '@tabler/icons-react'
import { useQueryClient } from '@tanstack/react-query'
import { motion } from 'motion/react'
import QRCode from 'qrcode'
import { useEffect, useMemo, useRef, useState } from 'react'
import { toast } from 'sonner'
import { Alert, Button, Dialog, SegmentedControl, Skeleton, Spinner, Switch } from '@/components/ui'
import { CopyChip } from '@/features/integrations/InstructionsGuide'
import { api, errorMessage } from '@/lib/api'
import { getBridge } from '@/lib/bridge'
import type { DevicePairing } from '@/lib/types'
import { cn } from '@/lib/utils'
import { deviceKeys, useDeviceActions, useDevices, useLanInfo } from './hooks'
import { previewMode } from './preview'

function useNow(ms = 1000) {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const t = window.setInterval(() => setNow(Date.now()), ms)
    return () => window.clearInterval(t)
  }, [ms])
  return now
}

/** `wss://192.168.1.24:7443/ws/node` or `https://192.168.1.24:7443` -> `https://192.168.1.24:7443` */
function httpsBase(url: string): string {
  return url.replace(/^wss:/, 'https:').replace(/^ws:/, 'http:').replace(/\/ws\/node\/?$/, '').replace(/\/+$/, '')
}

export function AddDeviceDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      size="lg"
      title="Add a device"
      description="Pair your phone, smart glasses or another computer so Sentient can see, hear and reach you through it."
    >
      {open && <AddDeviceBody close={() => onOpenChange(false)} />}
    </Dialog>
  )
}

function AddDeviceBody({ close }: { close: () => void }) {
  const qc = useQueryClient()
  const { pairing } = useDeviceActions()
  const devices = useDevices()
  const lan = useLanInfo()
  const [info, setInfo] = useState<DevicePairing | null>(null)
  const [qr, setQr] = useState<string | null>(null)
  const [kind, setKind] = useState<'phone' | 'other'>('phone')
  const [savingLan, setSavingLan] = useState(false)
  const known = useRef<Set<string> | null>(null)
  const now = useNow()

  const newCode = () =>
    pairing.mutate(undefined, {
      onSuccess: (p) => setInfo(p),
      onError: (e) => toast.error("Couldn't make a pairing code", { description: errorMessage(e) })
    })

  useEffect(() => {
    newCode()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (!info) return
    // A phone camera opens `web_url` directly in the browser; the sentient:// link is for native apps.
    void QRCode.toDataURL(info.web_url || info.qr, { margin: 1, width: 440, errorCorrectionLevel: 'M', color: { dark: '#111114', light: '#ffffff' } })
      .then(setQr)
      .catch(() => setQr(null))
  }, [info])

  // Remember which devices existed when the dialog opened, to notice the new one arriving (node.updated).
  if (known.current === null && devices.data) known.current = new Set(devices.data.map((d) => d.node_id))
  const arrived = useMemo(() => devices.data?.find((d) => known.current && !known.current.has(d.node_id)), [devices.data])

  const lanEnabled = info?.lan_enabled ?? lan.data?.enabled ?? false
  const lanRunning = lan.data?.running ?? lanEnabled
  const urls = (info?.urls.length ? info.urls : (lan.data?.urls ?? [])).map(httpsBase)
  const base = urls[0]
  const phoneLink = info?.web_url || (base ? `${base}/node/` : undefined)
  const fingerprint = info?.fingerprint ?? lan.data?.fingerprint ?? null
  const onLan = !!phoneLink && /^https:/.test(phoneLink)
  const remaining = info ? Math.max(0, new Date(info.expires_at).getTime() - now) : 0
  const expired = !!info && remaining <= 0
  const mm = Math.floor(remaining / 60000)
  const ss = String(Math.floor((remaining % 60000) / 1000)).padStart(2, '0')

  const setLan = async (enabled: boolean) => {
    setSavingLan(true)
    try {
      if (!previewMode()) await api.config.patch({ nodes: { lan_enabled: enabled } })
      await qc.invalidateQueries({ queryKey: deviceKeys.lan })
      newCode()
    } catch (e) {
      toast.error("Couldn't change that setting", { description: errorMessage(e) })
    } finally {
      setSavingLan(false)
    }
  }

  if (arrived) {
    return (
      <motion.div initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} className="flex flex-col items-center gap-3 py-10 text-center">
        <motion.span initial={{ scale: 0.4 }} animate={{ scale: 1 }} transition={{ type: 'spring', stiffness: 380, damping: 18 }}>
          <IconCircleCheckFilled size={52} className="text-success" />
        </motion.span>
        <div className="text-md font-semibold text-fg">{arrived.name} is connected</div>
        <div className="max-w-sm text-sm text-fg-muted">Try it from the device card, or ask Sentient something like &ldquo;what am I looking at?&rdquo;</div>
        <Button className="mt-3" variant="secondary" onClick={close}>
          Done
        </Button>
      </motion.div>
    )
  }

  return (
    <div className="space-y-4 pb-2">
      <div className="grid gap-5 sm:grid-cols-[220px_1fr]">
        <div className="flex flex-col items-center">
          <div className="relative size-[220px] overflow-hidden rounded-2xl border border-border bg-white p-2.5 shadow-soft">
            {qr ? <img src={qr} alt="Pairing QR code" className={cn('size-full', expired && 'opacity-15 blur-[2px]')} /> : <Skeleton className="size-full" />}
            {expired && (
              <div className="absolute inset-0 flex items-center justify-center">
                <Button size="sm" variant="primary" leftIcon={<IconRefresh size={14} />} onClick={newCode} loading={pairing.isPending}>
                  New code
                </Button>
              </div>
            )}
          </div>
          <div className="mt-3 flex gap-1.5" aria-label={info ? `Pairing code ${info.code}` : 'Pairing code'}>
            {(info?.code ?? '      ').split('').map((d, i) => (
              <span
                key={i}
                className={cn(
                  'flex h-10 w-8 items-center justify-center rounded-lg border border-border-strong bg-sunken font-mono text-xl font-semibold text-fg',
                  i === 2 && 'mr-2',
                  expired && 'text-fg-faint line-through'
                )}
              >
                {d.trim() || <Spinner size={12} className="text-fg-faint" />}
              </span>
            ))}
          </div>
          <div className={cn('mt-2 text-xs tabular-nums', expired ? 'text-danger' : remaining < 60_000 ? 'text-warning' : 'text-fg-subtle')}>
            {!info ? 'Making a code…' : expired ? 'This code expired' : `Expires in ${mm}:${ss}`}
          </div>
          {info && !expired && (
            <button type="button" onClick={newCode} className="mt-0.5 text-2xs text-fg-subtle hover:text-fg">
              Get a new code
            </button>
          )}
        </div>

        <div className="min-w-0 space-y-3">
          <SegmentedControl
            size="sm"
            fullWidth
            value={kind}
            onChange={setKind}
            options={[
              { value: 'phone', label: 'Phone', icon: <IconDeviceMobile size={14} /> },
              { value: 'other', label: 'Glasses or a computer', icon: <IconEyeglass2 size={14} /> }
            ]}
          />
          {kind === 'phone' ? (
            <ol className="space-y-2.5 text-sm text-fg-muted">
              <Step n={1}>Connect your phone to the same Wi-Fi as this computer.</Step>
              <Step n={2}>Point your phone&apos;s camera at the code, and open the link it shows.</Step>
              <Step n={3}>
                Or type this address into your phone&apos;s browser:{' '}
                {onLan && phoneLink ? <CopyChip value={phoneLink.replace(/#.*$/, '')} /> : <span className="text-fg-subtle">(turn on Wi-Fi pairing below)</span>}
              </Step>
              <Step n={4}>Enter the 6-digit code when asked. That&apos;s it.</Step>
              {phoneLink && !onLan && (
                <li className="pl-7">
                  <button type="button" onClick={() => void getBridge().openExternal(phoneLink)} className="text-xs text-accent-text hover:underline">
                    Try the device app on this computer first
                  </button>
                </li>
              )}
            </ol>
          ) : (
            <ol className="space-y-2.5 text-sm text-fg-muted">
              <Step n={1}>Make sure the glasses companion or the other computer is on the same Wi-Fi.</Step>
              <Step n={2}>
                On that computer, with Sentient installed, run:
                <div className="mt-1.5">
                  <CopyChip className="px-2 py-1" value={info ? `sentient node --url "${info.qr}"` : 'sentient node --url ...'} />
                </div>
              </Step>
              <Step n={3}>It uses that computer&apos;s webcam, speaker and keyboard as a device. Glasses apps that support Sentient ask for the same address and code.</Step>
            </ol>
          )}
        </div>
      </div>

      <div className="rounded-xl border border-border bg-elevated/40 p-3.5">
        <div className="flex items-start gap-3">
          <IconWifi size={17} className="mt-0.5 shrink-0 text-fg-subtle" />
          <div className="min-w-0 flex-1">
            <div className="text-sm font-medium text-fg">Allow devices on my Wi-Fi to connect</div>
            <p className="mt-0.5 text-xs leading-relaxed text-fg-subtle">
              Sentient listens on your home network so your phone and glasses can reach this computer. Only devices you pair with a code can
              connect, and nothing goes through the internet.
            </p>
          </div>
          <Switch checked={lanEnabled} disabled={savingLan || lan.isLoading} onCheckedChange={(v) => void setLan(v)} aria-label="Allow devices on my Wi-Fi to connect" />
        </div>
        {!lanEnabled && (
          <Alert tone="warning" className="mt-3">
            Your phone can&apos;t reach this computer until this is on.
          </Alert>
        )}
        {lanEnabled && lan.data?.error && (
          <Alert tone="danger" className="mt-3">
            {lan.data.error}
          </Alert>
        )}
        {lanEnabled && !lanRunning && !lan.data?.error && (
          <div className="mt-3 flex items-center gap-2 text-xs text-fg-subtle">
            <Spinner size={12} /> Getting ready on your Wi-Fi…
          </div>
        )}
        {lanEnabled && fingerprint && (
          <div className="mt-3 flex items-start gap-2.5 rounded-lg border border-border bg-surface px-3 py-2.5">
            <IconShieldLock size={15} className="mt-0.5 shrink-0 text-fg-subtle" />
            <div className="min-w-0 text-xs leading-relaxed text-fg-subtle">
              Your phone may warn that the connection isn&apos;t private. That&apos;s expected: this computer made its own security certificate. It
              should match:
              <div className="mt-1 break-all font-mono text-2xs text-fg-muted">{fingerprint}</div>
            </div>
          </div>
        )}
      </div>

      <div className="flex items-center justify-center gap-2 text-xs text-fg-subtle">
        <Spinner size={12} /> Waiting for your device to connect…
      </div>
    </div>
  )
}

function Step({ n, children }: { n: number; children: React.ReactNode }) {
  return (
    <li className="flex gap-2.5">
      <span className="mt-px flex size-5 shrink-0 items-center justify-center rounded-full border border-border-strong bg-elevated text-2xs font-semibold tabular-nums text-fg-muted">
        {n}
      </span>
      <span className="min-w-0 flex-1 leading-relaxed">{children}</span>
    </li>
  )
}
