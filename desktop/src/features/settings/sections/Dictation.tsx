import { IconArrowRight, IconKeyboard } from '@tabler/icons-react'
import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router'
import { Alert, Button, FormRow, FormSection, Shortcut } from '@/components/ui'
import { useDictationStatus } from '@/features/voice/dictation'
import { formatAccelerator } from '@/lib/accelerator'
import { getBridge } from '@/lib/bridge'
import type { ShortcutInfo } from '@/types/bridge'
import { SchemaForm } from '../SchemaForm'
import type { SectionProps } from '../SettingsPage'

/** Shown in this section; the generic voice form leaves them out. */
export const DICTATION_KEYS = [
  'voice.dictation.cleanup',
  'voice.dictation.language',
  'voice.dictation.stop_after_silence_s',
  'voice.dictation.speak_replies'
]

const MODES = [
  {
    id: 'pushToTalk',
    label: 'Push to talk',
    how: (keys: string) => `Hold ${keys}, speak, and let go. What you said goes to Sentient as a chat message.`
  },
  {
    id: 'dictate',
    label: 'Dictate into any app',
    how: (keys: string) =>
      `Press ${keys}, speak, then press it again or pause. Sentient types your words where your cursor is, and never into a password box.`
  }
] as const

/** Push to talk and dictation (#169). The shortcuts are changed in Settings > General with the other shortcuts. */
export function DictationSection({ query }: SectionProps) {
  const navigate = useNavigate()
  const bridge = getBridge()
  const status = useDictationStatus((s) => s.status)
  const [shortcuts, setShortcuts] = useState<ShortcutInfo[]>([])
  const q = query.trim().toLowerCase()
  const matches = !q || 'push to talk dictation dictate type any app shortcut hotkey keys microphone hold'.includes(q)

  useEffect(() => {
    if (!bridge.isDesktop) return
    bridge
      .getShortcuts()
      .then(setShortcuts)
      .catch(() => setShortcuts([]))
  }, [bridge])

  const platform = bridge.platform
  const mac = platform === 'darwin'

  return (
    <div className="space-y-6">
      {matches && (
        <FormSection
          title="Push to talk and dictation"
          description="Talk to Sentient or type with your voice in any app. Your speech is turned into text on this computer."
          actions={
            bridge.isDesktop && (
              <Button size="sm" variant="ghost" rightIcon={<IconArrowRight size={13} />} onClick={() => navigate('/settings/general')}>
                Change shortcuts
              </Button>
            )
          }
        >
          {MODES.map((m) => {
            const s = shortcuts.find((x) => x.id === m.id)
            const keys = s?.accelerator ? formatAccelerator(s.accelerator, platform) : ''
            return (
              <FormRow
                key={m.id}
                label={m.label}
                description={
                  !bridge.isDesktop
                    ? 'Available in the desktop app.'
                    : keys
                      ? m.how(keys)
                      : 'Turned off. Give it a shortcut in Settings > General to use it.'
                }
                error={s?.taken && s.accelerator ? 'Another app was already using this shortcut when Sentient started. Pick a different one.' : undefined}
              >
                {keys ? <Shortcut keys={keys} /> : <span className="text-sm text-fg-subtle">Off</span>}
              </FormRow>
            )
          })}
          <div className="flex items-center gap-2 px-4 py-2.5 text-xs text-fg-muted">
            <IconKeyboard size={14} className="text-fg-subtle" />
            While the microphone is on, a small bar at the bottom of your screen shows it. Press Esc to cancel.
          </div>
        </FormSection>
      )}
      {matches && mac && (status?.accessibility === false || (status?.microphone && status.microphone !== 'granted')) && (
        <Alert
          tone="info"
          title="Two permissions on your Mac"
          action={
            <div className="flex flex-col gap-1.5">
              {status?.microphone !== 'granted' && (
                <Button size="sm" onClick={() => void bridge.dictation.openPermissionSettings('microphone')}>
                  Microphone settings
                </Button>
              )}
              {status?.accessibility === false && (
                <Button size="sm" onClick={() => void bridge.dictation.openPermissionSettings('accessibility')}>
                  Accessibility settings
                </Button>
              )}
            </div>
          }
        >
          macOS asks before an app can hear you or type for you. In System Settings, open Privacy & Security, then turn
          Sentient on under Microphone (to listen) and Accessibility (to type your words into other apps). Until then,
          dictated words are put on the clipboard for you to paste.
        </Alert>
      )}
      {matches && platform === 'linux' && (
        <Alert tone="info" title="Typing into other apps on Linux">
          Sentient uses xdotool, or wtype on Wayland, to paste your words. If neither is installed, your words are put on
          the clipboard for you to paste. Linux can’t tell Sentient when a password box has focus, so check where your
          cursor is before you dictate.
        </Alert>
      )}
      <SchemaForm section="voice" title="Dictation" include={DICTATION_KEYS} filter={query} />
    </div>
  )
}
