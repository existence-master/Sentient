import { useEffect, useRef } from 'react'
import { isEditableTarget, isMac } from '@/lib/utils'

/**
 * Keyboard shortcut. `combo` like "mod+k", "mod+shift+n", "escape", "mod+,".
 * `mod` is Ctrl on Windows/Linux and Cmd on macOS.
 */
export function useHotkey(
  combo: string,
  handler: (e: KeyboardEvent) => void,
  opts: { enabled?: boolean; allowInInputs?: boolean } = {}
) {
  const ref = useRef(handler)
  ref.current = handler
  const { enabled = true, allowInInputs = true } = opts

  useEffect(() => {
    if (!enabled) return
    const parts = combo.toLowerCase().split('+')
    const key = parts[parts.length - 1]
    const needMod = parts.includes('mod')
    const needShift = parts.includes('shift')
    const needAlt = parts.includes('alt')
    const onKey = (e: KeyboardEvent) => {
      const mod = isMac ? e.metaKey : e.ctrlKey
      if (needMod !== mod || needShift !== e.shiftKey || needAlt !== e.altKey) return
      if (e.key.toLowerCase() !== key && e.code.toLowerCase() !== `key${key}`) return
      if (!allowInInputs && !needMod && isEditableTarget(e.target)) return
      e.preventDefault()
      ref.current(e)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [combo, enabled, allowInInputs])
}
