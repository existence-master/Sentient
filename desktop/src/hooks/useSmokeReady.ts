import { useIsFetching, useIsMutating } from '@tanstack/react-query'
import { useEffect, useRef } from 'react'
import { useLocation } from 'react-router'
import { getBridge } from '@/lib/bridge'

/**
 * Tells the shell (smoke screenshot hook) that the current route rendered and nothing
 * is loading. Fires once per route after `idleMs` of quiet, fonts loaded.
 */
export function useSmokeReady(ready = true, idleMs = 900) {
  const fetching = useIsFetching()
  const mutating = useIsMutating()
  const location = useLocation()
  const fired = useRef<string | null>(null)

  useEffect(() => {
    if (!ready || fetching > 0 || mutating > 0) return
    const key = location.pathname + location.search
    if (fired.current === key) return
    const t = window.setTimeout(async () => {
      await document.fonts?.ready
      fired.current = key
      getBridge().readyForScreenshot()
    }, idleMs)
    return () => window.clearTimeout(t)
  }, [ready, fetching, mutating, location.pathname, location.search, idleMs])
}
