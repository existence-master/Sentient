import { IconKeyboard } from '@tabler/icons-react'
import { useState } from 'react'
import { Alert, Button, FormRow, FormSection, Shortcut, Switch } from '@/components/ui'
import { useConfigEditor } from '@/hooks/config'
import { useDictationStatus } from '@/features/voice/dictation'
import { getBridge } from '@/lib/bridge'
import { acceleratorFromKeyPress, formatAccelerator } from '@/lib/shortcuts'
import type { DictationShortcutStatus } from '@/types/bridge'
import { SchemaForm } from '../SchemaForm'
import type { SectionProps } from '../SettingsPage'

/** Shown in this section; the generic voice form leaves them out. */
export const DICTATION_KEYS = [
  'voice.dictation.push_to_talk',
  'voice.dictation.push_to_talk_shortcut',
  'voice.dictation.dictate',
  'voice.dictation.dictate_shortcut',
  'voice.dictation.cleanup',
  'voice.dictation.language',
  'voice.dictation.stop_after_silence_s',
  'voice.dictation.speak_replies'
]

const DETAIL_KEYS = DICTATION_KEYS.slice(4)

/** Click, then press the new keys. Esc keeps the old shortcut. */
function ShortcutButton({ value, onChange, label }: { value: string; onChange: (accelerator: string) => void; label: string }) {
  const platform = getBridge().platform
  const [recording, setRecording] = useState(false)
  const [hint, setHint] = useState<string | null>(null)
  if (recording) {
    return (
      <Button
        size="sm"
        variant="secondary"
        autoFocus
        aria-label={`Press the new keys for ${label}`}
        onBlur={() => {
          setRecording(false)
          setHint(null)
        }}
        onKeyDown={(e) => {
          e.preventDefault()
          e.stopPropagation()
          if (e.key === 'Escape') {
            setRecording(false)
            setHint(null)
            return
          }
          const acc = acceleratorFromKeyPress(e, platform)
          if (acc === null) return // only modifiers so far
          if (!acc) {
            setHint('Hold Ctrl, Alt or Shift too')
            return
          }
          setRecording(false)
          setHint(null)
          onChange(acc)
        }}
      >
        {hint ?? 'Press the new keys…'}
      </Button>
    )
  }
  return (
    <Button size="sm" variant="ghost" aria-label={`Change the shortcut for ${label}`} onClick={() => setRecording(true)}>
      <Shortcut keys={formatAccelerator(value, platform)} />
    </Button>
  )
}

function Problem({ status }: { status?: DictationShortcutStatus }) {
  if (!status?.enabled || status.registered || !status.problem) return null
  return <span className="text-danger">{status.problem}</span>
}

export function DictationSection({ query }: SectionProps) {
  const { config, setValue } = useConfigEditor()
  const status = useDictationStatus((s) => s.status)
  const bridge = getBridge()
  const d = config?.voice?.dictation
  const q = query.trim().toLowerCase()
  const matches = !q || 'push to talk dictation dictate type any app shortcut hotkey keys microphone hold'.includes(q)
  if (!d) return null // an older engine without dictation settings

  const platform = bridge.platform
  const talkKeys = formatAccelerator(d.push_to_talk_shortcut, platform)
  const dictateKeys = formatAccelerator(d.dictate_shortcut, platform)
  const mac = platform === 'darwin'
  const set = (key: string, value: unknown) => setValue(`voice.dictation.${key}`, value, { immediate: true })

  return (
    <div className="space-y-6">
      {matches && (
        <FormSection
          title="Push to talk and dictation"
          description="Talk to Sentient or type with your voice in any app. Your speech is turned into text on this computer."
        >
          <FormRow
            label="Push to talk"
            description={
              <>
                Hold {talkKeys}, speak, and let go. What you said goes to Sentient as a chat message.{' '}
                <Problem status={status?.talk} />
              </>
            }
          >
            <div className="flex items-center gap-2">
              <ShortcutButton label="push to talk" value={d.push_to_talk_shortcut} onChange={(acc) => set('push_to_talk_shortcut', acc)} />
              <Switch checked={d.push_to_talk} onCheckedChange={(on) => set('push_to_talk', on)} aria-label="Push to talk" />
            </div>
          </FormRow>
          <FormRow
            label="Dictate into any app"
            description={
              <>
                Press {dictateKeys}, speak, then press it again or pause. Sentient types your words where your cursor
                is, and never into a password box. <Problem status={status?.dictate} />
              </>
            }
          >
            <div className="flex items-center gap-2">
              <ShortcutButton label="dictation" value={d.dictate_shortcut} onChange={(acc) => set('dictate_shortcut', acc)} />
              <Switch checked={d.dictate} onCheckedChange={(on) => set('dictate', on)} aria-label="Dictate into any app" />
            </div>
          </FormRow>
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
      <SchemaForm section="voice" title="Dictation" include={DETAIL_KEYS} filter={query} />
    </div>
  )
}
