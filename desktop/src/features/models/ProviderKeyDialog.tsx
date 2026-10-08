import { IconExternalLink, IconEye, IconEyeOff, IconLock } from '@tabler/icons-react'
import { useState } from 'react'
import { toast } from 'sonner'
import { Button, Dialog, Field, IconButton, Input } from '@/components/ui'
import { useSetSecret } from '@/hooks/models'
import { errorMessage } from '@/lib/api'
import { getBridge } from '@/lib/bridge'
import type { Provider } from '@/lib/types'

/** Masked API key entry. The key goes straight to the OS keychain and is never shown again. */
export function ProviderKeyDialog({
  provider,
  open,
  onOpenChange,
  onSaved
}: {
  provider: Pick<Provider, 'id' | 'label' | 'docs_url'> | null
  open: boolean
  onOpenChange: (open: boolean) => void
  onSaved?: () => void
}) {
  const [value, setValue] = useState('')
  const [show, setShow] = useState(false)
  const save = useSetSecret()

  const submit = () => {
    if (!provider || !value.trim()) return
    save.mutate(
      { name: provider.id, value: value.trim() },
      {
        onSuccess: () => {
          toast.success(`${provider.label} key saved`, { description: 'Stored in your system keychain.' })
          setValue('')
          onOpenChange(false)
          onSaved?.()
        },
        onError: (e) => toast.error("Couldn't save the key", { description: errorMessage(e) })
      }
    )
  }

  return (
    <Dialog
      open={open}
      onOpenChange={(o) => {
        if (!o) setValue('')
        onOpenChange(o)
      }}
      title={`${provider?.label ?? 'Provider'} API key`}
      description="Paste your key. It's stored in your operating system's keychain, never in files or logs."
      footer={
        <>
          <Button variant="ghost" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button variant="primary" loading={save.isPending} disabled={!value.trim()} onClick={submit}>
            Save key
          </Button>
        </>
      }
    >
      <form
        onSubmit={(e) => {
          e.preventDefault()
          submit()
        }}
        className="space-y-3"
      >
        <Field label="API key" htmlFor="provider-key">
          <Input
            id="provider-key"
            autoFocus
            type={show ? 'text' : 'password'}
            autoComplete="off"
            value={value}
            onChange={(e) => setValue(e.target.value)}
            placeholder="sk-…"
            leftIcon={<IconLock />}
            className="font-mono text-xs"
            rightSlot={
              <IconButton size="xs" tooltip={false} label={show ? 'Hide key' : 'Show key'} icon={show ? <IconEyeOff size={14} /> : <IconEye size={14} />} onClick={() => setShow((s) => !s)} />
            }
          />
        </Field>
        {provider?.docs_url && (
          <button
            type="button"
            onClick={() => void getBridge().openExternal(provider.docs_url)}
            className="flex items-center gap-1.5 text-xs text-accent-text hover:underline"
          >
            Get a {provider.label} key <IconExternalLink size={12} />
          </button>
        )}
      </form>
    </Dialog>
  )
}
