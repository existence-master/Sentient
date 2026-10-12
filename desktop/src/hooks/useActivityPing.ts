import { useEffect } from 'react'
import { api } from '@/lib/api'
import { isDemoMode } from '@/lib/demo'

/** At most one ping every few minutes: the engine only needs to know you were around today. */
const EVERY_MS = 5 * 60 * 1000

/**
 * Tell the engine the window is in use (#163), so routines don't pause while you are around. A ping goes out when
 * the window gets focus or becomes visible, and on a click or key press, at most once every five minutes. Nothing
 * is sent while the window sits unused in the background.
 */
export function useActivityPing() {
  useEffect(() => {
    if (isDemoMode()) return
    let last = 0
    const ping = () => {
      if (document.visibilityState !== 'visible' || !document.hasFocus()) return
      const now = Date.now()
      if (now - last < EVERY_MS) return
      last = now
      api.activity().catch(() => {
        last = 0 // try again on the next sign of use
      })
    }
    const onVisible = () => document.visibilityState === 'visible' && ping()
    ping()
    window.addEventListener('focus', ping)
    window.addEventListener('pointerdown', ping, { passive: true })
    window.addEventListener('keydown', ping)
    document.addEventListener('visibilitychange', onVisible)
    return () => {
      window.removeEventListener('focus', ping)
      window.removeEventListener('pointerdown', ping)
      window.removeEventListener('keydown', ping)
      document.removeEventListener('visibilitychange', onVisible)
    }
  }, [])
}
