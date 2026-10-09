import { IconAlertTriangle, IconCircleCheckFilled, IconQrcode } from '@tabler/icons-react'
import { motion } from 'motion/react'
import QRCode from 'qrcode'
import { useEffect, useState } from 'react'
import { toast } from 'sonner'
import { Alert, Button, Dialog, Skeleton, Spinner } from '@/components/ui'
import { InstructionsGuide } from '@/features/integrations/InstructionsGuide'
import { errorMessage } from '@/lib/api'
import type { Channel } from '@/lib/types'
import { useChannelActions, useChannels } from './hooks'
import { ChannelTile } from './meta'

/** Link WhatsApp by scanning a QR code with the phone (§14). The code arrives on the Channel through `channel.updated`. */
export function WhatsAppLinkDialog({ channel, onClose }: { channel: Channel | null; onClose: () => void }) {
  const actions = useChannelActions()
  const { data } = useChannels()
  const [started, setStarted] = useState(false)
  const [image, setImage] = useState<string | null>(null)
  const live = data?.find((c) => c.id === channel?.id) ?? channel
  const qr = live?.status === 'linking' ? (live.qr ?? null) : null

  useEffect(() => {
    setStarted(live?.status === 'linking')
    setImage(null)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [channel?.id])

  useEffect(() => {
    if (!qr) {
      setImage(null)
      return
    }
    void QRCode.toDataURL(qr, { margin: 1, width: 440, errorCorrectionLevel: 'L', color: { dark: '#111114', light: '#ffffff' } })
      .then(setImage)
      .catch(() => setImage(null))
  }, [qr])

  const start = () => {
    if (!channel) return
    setStarted(true)
    actions.connect.mutate(
      { id: channel.id, fields: {} },
      { onError: (e) => toast.error("Couldn't start linking WhatsApp", { description: errorMessage(e) }) }
    )
  }

  const linked = started && live?.status === 'connected'
  const failed = live?.status === 'error'

  return (
    <Dialog
      open={!!channel}
      onOpenChange={(o) => !o && onClose()}
      size="lg"
      title={
        <span className="flex items-center gap-3">
          <ChannelTile channel="whatsapp" size={34} />
          Link WhatsApp
        </span>
      }
    >
      {live &&
        (linked ? (
          <motion.div initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} className="flex flex-col items-center gap-3 py-10 text-center">
            <IconCircleCheckFilled size={52} className="text-success" />
            <div className="text-md font-semibold text-fg">WhatsApp is linked{live.account_label ? ` to ${live.account_label}` : ''}</div>
            <p className="max-w-sm text-sm text-fg-muted">Open your &ldquo;Message yourself&rdquo; chat in WhatsApp and say hi. Sentient answers there.</p>
            <Button className="mt-2" variant="secondary" onClick={onClose}>
              Done
            </Button>
          </motion.div>
        ) : (
          <div className="grid gap-5 pb-1 sm:grid-cols-[220px_1fr]">
            <div className="flex flex-col items-center gap-3">
              <div className="relative flex size-[220px] items-center justify-center overflow-hidden rounded-2xl border border-border bg-white p-2.5 shadow-soft">
                {image ? (
                  <img src={image} alt="QR code to link WhatsApp" className="size-full" />
                ) : started && !failed ? (
                  <Skeleton className="size-full" />
                ) : (
                  <IconQrcode size={64} className="text-[#c9c9cf]" />
                )}
              </div>
              {!started || failed ? (
                <Button variant="primary" onClick={start} loading={actions.connect.isPending}>
                  {started ? 'Get a new code' : 'Show QR code'}
                </Button>
              ) : (
                <span className="flex items-center gap-2 text-xs text-fg-subtle">
                  <Spinner size={12} /> {image ? 'Waiting for your phone…' : 'Getting a code…'}
                </span>
              )}
            </div>
            <div className="min-w-0 space-y-4">
              {failed && live.error && <Alert tone="danger">{live.error}</Alert>}
              <div className="rounded-xl border border-border bg-surface p-4">
                <InstructionsGuide markdown={live.setup.instructions_md} />
              </div>
              <div className="flex gap-2.5 rounded-xl border border-warning/30 bg-warning/5 px-3.5 py-3 text-sm text-fg-muted">
                <IconAlertTriangle size={17} className="mt-0.5 shrink-0 text-warning" />
                <span>
                  WhatsApp doesn&apos;t officially allow assistants on personal accounts. Linking is the same as WhatsApp Web, but WhatsApp could limit or
                  ban an account that uses an unofficial app. It is rare, and it is your call.
                </span>
              </div>
            </div>
          </div>
        ))}
    </Dialog>
  )
}
