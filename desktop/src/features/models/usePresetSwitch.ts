import { useState } from 'react'
import { toast } from 'sonner'
import { useApplyPreset, useOllamaPull, useProviders, useUndoPreset } from '@/hooks/models'
import { errorMessage } from '@/lib/api'
import type { PresetApplyResult, Provider } from '@/lib/types'

/**
 * Switching model presets with what goes with it: a toast with Undo, the list of anything still missing, a
 * download in progress and the key dialog. Shared by the title bar menu and Settings > Models.
 */
export function usePresetSwitch() {
  const applyPreset = useApplyPreset()
  const undoPreset = useUndoPreset()
  const pull = useOllamaPull()
  const providers = useProviders()
  const [result, setResult] = useState<PresetApplyResult | null>(null)
  const [keyFor, setKeyFor] = useState<Provider | null>(null)

  const undo = () =>
    undoPreset.mutate(undefined, {
      onSuccess: (r) => {
        setResult(r.missing.length ? r : null)
        toast.success(r.preset ? `Back to ${r.preset}` : 'Back to your previous models')
      },
      onError: (e) => toast.error("Couldn't undo", { description: errorMessage(e) })
    })

  const apply = (name: string) =>
    applyPreset.mutate(name, {
      onSuccess: (r) => {
        setResult(r.missing.length ? r : null)
        if (!r.changed.length) toast.message(`Already using ${r.preset}`)
        else
          toast.success(`Switched to ${r.preset}`, {
            description: r.missing.length ? 'Some models still need setting up. The fixes are listed with the presets.' : undefined,
            action: { label: 'Undo', onClick: undo }
          })
      },
      onError: (e) => toast.error("Couldn't switch models", { description: errorMessage(e) })
    })

  const addKey = (id: string) => {
    const p = providers.data?.find((x) => x.id === id)
    setKeyFor(p ?? { id, label: id, kind: 'cloud', key_required: true, key_set: false, api_base: null, docs_url: '', suggested: [] })
  }

  /** Drop fixed items: a finished download, or a key that is now set. */
  const missing = (result?.missing ?? []).filter((m) => {
    if (m.action?.kind === 'pull_model') return !(pull.done && pull.name === m.action.name)
    if (m.action?.kind === 'add_key') {
      const id = m.action.provider
      return !providers.data?.some((p) => p.id === id && p.key_set)
    }
    return true
  })

  return {
    apply,
    undo,
    busy: applyPreset.isPending || undoPreset.isPending,
    pending: applyPreset.isPending ? applyPreset.variables : null,
    missing,
    clearMissing: () => setResult(null),
    pull,
    addKey,
    keyFor,
    setKeyFor
  }
}
