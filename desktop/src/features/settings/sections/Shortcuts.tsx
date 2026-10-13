import { useCallback, useEffect, useState } from 'react'
import { Button, FormRow, FormSection, Shortcut, Switch } from '@/components/ui'
import { acceleratorFromKey, formatAccelerator } from '@/lib/accelerator'
import { getBridge } from '@/lib/bridge'
import type { ShortcutId, ShortcutInfo } from '@/types/bridge'

/** Global shortcuts (#172, #169, #173): change, turn off or reset. They work while Sentient is in the tray. Ask about
 *  selected text also gets its own on/off switch here. */
export function ShortcutSettings() {
  const bridge = getBridge()
  const [items, setItems] = useState<ShortcutInfo[] | null>(null)
  const [recording, setRecording] = useState<ShortcutId | null>(null)
  const [errors, setErrors] = useState<Partial<Record<ShortcutId, string>>>({})

  useEffect(() => {
    if (!bridge.isDesktop) return
    bridge
      .getShortcuts()
      .then(setItems)
      .catch(() => setItems([]))
  }, [bridge])

  const apply = useCallback(
    async (id: ShortcutId, accelerator: string | null) => {
      try {
        const r = await bridge.setShortcut(id, accelerator)
        setItems(r.shortcuts)
        setErrors((e) => ({ ...e, [id]: r.ok ? undefined : r.error }))
      } catch {
        setErrors((e) => ({ ...e, [id]: "Couldn't change the shortcut. Please try again." }))
      }
    },
    [bridge]
  )

  // While recording, the next key press with modifiers becomes the shortcut; Esc on its own cancels.
  useEffect(() => {
    if (!recording) return
    const onKey = (e: KeyboardEvent) => {
      e.preventDefault()
      e.stopPropagation()
      if (e.key === 'Escape' && !e.ctrlKey && !e.altKey && !e.metaKey && !e.shiftKey) {
        setRecording(null)
        return
      }
      const accelerator = acceleratorFromKey(e, bridge.platform)
      if (!accelerator) return
      setRecording(null)
      void apply(recording, accelerator)
    }
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  }, [recording, apply, bridge.platform])

  const askInfo = items?.find((s) => s.id === 'askSelection')
  const askKeys = askInfo?.accelerator ? formatAccelerator(askInfo.accelerator, bridge.platform) : ''

  return (
    <>
      {bridge.isDesktop && askInfo && (
        <FormSection
          title="Ask about selected text"
          description="Select text in any app, press the shortcut, and ask Sentient to explain, summarize, translate or reply. Nothing is read until you press it."
        >
          <FormRow
            label="Ask about selected text"
            description={
              askKeys
                ? `Press ${askKeys} with text selected. Sentient copies only that text, puts your clipboard back, and shows you what it copied. The text is treated as outside content, so Sentient asks before sending anything.`
                : 'Turned off. Turn it on to use it with its shortcut.'
            }
            error={errors.askSelection}
          >
            <div className="flex items-center gap-2">
              {askKeys && <Shortcut keys={askKeys} />}
              <Switch
                aria-label="Ask about selected text"
                checked={!!askInfo.accelerator}
                onCheckedChange={(on) => void apply('askSelection', on ? null : '')}
              />
            </div>
          </FormRow>
        </FormSection>
      )}
      <FormSection
        title="Keyboard shortcuts"
        description="These work anywhere, also while Sentient is in the tray. Nothing is captured until you press one: a picture of your screen, the microphone for talking and dictating, or the text you selected."
      >
        {!bridge.isDesktop ? (
          <FormRow label="Share your screen" description="Available in the desktop app.">
            <span className="text-sm text-fg-subtle">Off</span>
          </FormRow>
        ) : (
          (items ?? []).map((s) => {
            const error =
              errors[s.id] ??
              (s.taken && s.accelerator
                ? 'Another app was already using this shortcut when Sentient started. Pick a different one.'
                : undefined)
            return (
              <FormRow key={s.id} label={s.label} description={s.description} error={error}>
                <div className="flex flex-wrap items-center gap-2">
                  {recording === s.id ? (
                    <span role="status" className="text-sm text-accent-text">
                      Press the new keys. Esc cancels.
                    </span>
                  ) : s.accelerator ? (
                    <Shortcut keys={formatAccelerator(s.accelerator, bridge.platform)} />
                  ) : (
                    <span className="text-sm text-fg-subtle">Off</span>
                  )}
                  <Button size="sm" variant="outline" onClick={() => setRecording(recording === s.id ? null : s.id)}>
                    {recording === s.id ? 'Cancel' : 'Change'}
                  </Button>
                  {s.accelerator && recording !== s.id && (
                    <Button size="sm" variant="ghost" onClick={() => void apply(s.id, '')}>
                      Turn off
                    </Button>
                  )}
                  {s.accelerator !== s.defaultAccelerator && recording !== s.id && (
                    <Button size="sm" variant="ghost" onClick={() => void apply(s.id, null)}>
                      Use default
                    </Button>
                  )}
                </div>
              </FormRow>
            )
          })
        )}
      </FormSection>
    </>
  )
}
