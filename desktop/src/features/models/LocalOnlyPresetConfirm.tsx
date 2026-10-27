import { ConfirmDialog } from '@/components/ui'
import type { usePresetSwitch } from './usePresetSwitch'

/** "Turn on Local only too?" before the Local only preset switches the models and the switch together (#335). */
export function LocalOnlyPresetConfirm({ sw }: { sw: ReturnType<typeof usePresetSwitch> }) {
  return (
    <ConfirmDialog
      open={!!sw.askLocal}
      onOpenChange={(o) => !o && sw.cancelLocal()}
      tone="primary"
      title="Turn on Local only too?"
      description="Every model will run on this computer, and the Local only switch makes sure nothing is sent to a cloud model, not even a backup model or a chat with its own model. You can turn it off in this menu."
      confirmLabel="Switch and turn on"
      onConfirm={sw.confirmLocal}
    />
  )
}
