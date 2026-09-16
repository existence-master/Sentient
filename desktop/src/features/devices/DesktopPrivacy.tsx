import { IconCamera, IconScreenshot } from '@tabler/icons-react'
import { useEffect, useState } from 'react'
import { ConfirmDialog, Switch } from '@/components/ui'
import { getBridge } from '@/lib/bridge'
import type { DevicePrivacy } from '@/types/bridge'

const ROWS = [
  {
    key: 'screen' as const,
    icon: IconScreenshot,
    label: 'Let Sentient see my screen when I ask',
    description: 'Takes one screenshot, only when you ask about something on your screen.',
    confirmTitle: 'Let Sentient see your screen when you ask?',
    confirmBody: 'Sentient takes a single screenshot each time you ask about your screen, and you always see a notice when it does. You can turn this off here at any time.'
  },
  {
    key: 'camera' as const,
    icon: IconCamera,
    label: 'Let Sentient use my camera when I ask',
    description: 'Takes one photo, only when you ask. The camera light comes on for a moment.',
    confirmTitle: 'Let Sentient use your camera when you ask?',
    confirmBody: 'Sentient takes a single photo each time you ask for one, and you always see a notice when it does. You can turn this off here at any time.'
  }
]

/** Privacy switches for this computer as a device. Unset means "ask me the first time". */
export function DesktopPrivacy() {
  const bridge = getBridge()
  const [privacy, setPrivacy] = useState<DevicePrivacy>({ screen: null, camera: null })
  const [confirm, setConfirm] = useState<(typeof ROWS)[number] | null>(null)

  useEffect(() => {
    void bridge.getDevicePrivacy().then(setPrivacy)
  }, [bridge])

  const save = async (patch: Partial<DevicePrivacy>) => setPrivacy(await bridge.setDevicePrivacy(patch))

  return (
    <div className="space-y-3">
      {ROWS.map((row) => {
        const value = privacy[row.key]
        return (
          <div key={row.key} className="flex items-start gap-3">
            <row.icon size={16} className="mt-0.5 shrink-0 text-fg-subtle" />
            <div className="min-w-0 flex-1">
              <div className="text-sm text-fg">{row.label}</div>
              <div className="text-xs text-fg-subtle">
                {row.description}
                {value === null && ' Sentient will ask you the first time.'}
              </div>
            </div>
            <Switch
              size="sm"
              disabled={!bridge.isDesktop}
              checked={value === true}
              onCheckedChange={(on) => (on ? setConfirm(row) : void save({ [row.key]: false }))}
              aria-label={row.label}
            />
          </div>
        )
      })}
      <ConfirmDialog
        open={!!confirm}
        onOpenChange={(o) => !o && setConfirm(null)}
        title={confirm?.confirmTitle}
        description={confirm?.confirmBody}
        confirmLabel="Allow"
        cancelLabel="Not now"
        tone="primary"
        onConfirm={async () => {
          if (confirm) await save({ [confirm.key]: true })
        }}
      />
    </div>
  )
}
