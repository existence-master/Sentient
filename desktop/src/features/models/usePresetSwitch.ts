import { useState } from 'react'
import { toast } from 'sonner'
import { useConfig } from '@/hooks/core'
import { useApplyPreset, useOllamaPull, useProviders, useUndoPreset } from '@/hooks/models'
import { errorMessage } from '@/lib/api'
import type { PresetApplyResult, Provider } from '@/lib/types'

/** The built-in preset that shares its name with the Local only switch (`LOCAL_PRESET` in config/schema.py). */
export const LOCAL_PRESET = 'Local only'

/**
 * Switching model presets with what goes with it: a toast with Undo, the list of anything still missing, a
 * download in progress and the key dialog. Shared by the title bar menu and Settings > Models.
 *
 * Picking the built-in Local only preset while the Local only switch is off asks first (`askLocal`), then switches
 * the models and turns the switch on in one step (#335). Render `LocalOnlyPresetConfirm` next to the presets.
 */
export function usePresetSwitch() {
  const applyPreset = useApplyPreset()
  const undoPreset = useUndoPreset()
  const pull = useOllamaPull()
  const providers = useProviders()
  const config = useConfig()
  const [result, setResult] = useState<PresetApplyResult | null>(null)
  const [keyFor, setKeyFor] = useState<Provider | null>(null)
  const [askLocal, setAskLocal] = useState<string | null>(null)

  const undo = () =>
    undoPreset.mutate(undefined, {
      onSuccess: (r) => {
        setResult(r.missing.length ? r : null)
        toast.success(r.preset ? `Back to ${r.preset}` : 'Back to your previous models')
      },
      onError: (e) => toast.error("Couldn't undo", { description: errorMessage(e) })
    })

  const run = (name: string, localOnly = false) =>
    applyPreset.mutateAsync({ name, localOnly }).then(
      (r) => {
        setResult(r.missing.length ? r : null)
        const turnedOn = localOnly && r.local_only
        if (!r.changed.length && !turnedOn) toast.message(`Already using ${r.preset}`)
        else
          toast.success(`Switched to ${r.preset}`, {
            description:
              [turnedOn && 'Local only is on: nothing goes to a cloud model.', r.missing.length && 'Some models still need setting up. The fixes are listed with the presets.']
                .filter(Boolean)
                .join(' ') || undefined,
            action: { label: 'Undo', onClick: undo }
          })
      },
      (e) => {
        toast.error("Couldn't switch models", { description: errorMessage(e) })
      }
    )

  /** Apply a preset. Returns true when it asks first instead (Local only with the switch off), so a menu can close. */
  const apply = (name: string): boolean => {
    if (name === LOCAL_PRESET && !config.data?.models.local_only) { // asks too while the setting is still loading
      setAskLocal(name)
      return true
    }
    void run(name)
    return false
  }

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
    /** The preset waiting for the Local only confirmation, and its two answers. */
    askLocal,
    confirmLocal: async () => {
      if (askLocal) await run(askLocal, true)
      setAskLocal(null)
    },
    cancelLocal: () => setAskLocal(null),
    busy: applyPreset.isPending || undoPreset.isPending,
    pending: applyPreset.isPending ? (applyPreset.variables?.name ?? null) : null,
    missing,
    clearMissing: () => setResult(null),
    pull,
    addKey,
    keyFor,
    setKeyFor
  }
}
