/** Webhooks (docs/API.md §16): private links other apps call to start your tasks. */
import { IconAlertTriangle, IconCheck, IconCopy, IconEye, IconEyeOff, IconListCheck, IconPlus, IconTrash, IconWebhook } from '@tabler/icons-react'
import { useState } from 'react'
import { useNavigate } from 'react-router'
import { toast } from 'sonner'
import { Alert, Badge, Button, ConfirmDialog, Dialog, EmptyState, Field, IconButton, Input, Skeleton } from '@/components/ui'
import { absoluteHookUrl, errorMessage, isNotImplemented } from '@/lib/api'
import { useHookActions, useHooks } from '@/hooks/automations'
import type { Hook, HookCreated } from '@/lib/types'
import { cn, copyText, relativeTime } from '@/lib/utils'

export function WebhooksSection({ id }: { id?: string }) {
  const hooks = useHooks()
  const { remove } = useHookActions()
  const [creating, setCreating] = useState(false)
  const [deleting, setDeleting] = useState<Hook | null>(null)
  const missing = hooks.isError && isNotImplemented(hooks.error)

  return (
    <section id={id} className="scroll-mt-6">
      <div className="mb-3 flex flex-wrap items-end gap-3">
        <div className="min-w-0 flex-1">
          <h2 className="flex items-center gap-2 text-md font-semibold text-fg">
            <IconWebhook size={17} className="text-accent-text" /> Webhooks
          </h2>
          <p className="mt-0.5 max-w-2xl text-sm text-fg-subtle">
            A private link other apps can call to start one of your tasks. Handy for online stores, forms, smart home devices and tools like Zapier.
          </p>
        </div>
        <Button variant="secondary" leftIcon={<IconPlus size={15} />} disabled={missing || hooks.isLoading} onClick={() => setCreating(true)}>
          Create webhook
        </Button>
      </div>

      {hooks.isLoading ? (
        <Skeleton className="h-28 rounded-xl" />
      ) : hooks.isError ? (
        missing ? (
          <div className="rounded-xl border border-dashed border-border">
            <EmptyState compact icon={<IconWebhook />} title="Webhooks arrive with the next engine update" description="Once they’re here, you can let other apps start your tasks by calling a private link." />
          </div>
        ) : (
          <Alert tone="danger" title="Couldn’t load your webhooks" action={<Button size="sm" onClick={() => void hooks.refetch()}>Retry</Button>}>
            {errorMessage(hooks.error)}
          </Alert>
        )
      ) : !hooks.data?.length ? (
        <div className="rounded-xl border border-dashed border-border">
          <EmptyState
            compact
            icon={<IconWebhook />}
            title="No webhooks yet"
            description="Create one, paste its link into the other app, then make a task that starts when it’s called."
            action={
              <Button size="sm" variant="primary" leftIcon={<IconPlus size={14} />} onClick={() => setCreating(true)}>
                Create webhook
              </Button>
            }
          />
        </div>
      ) : (
        <div className="divide-y divide-border overflow-hidden rounded-xl border border-border bg-surface">
          {hooks.data.map((h) => (
            <HookRow key={h.id} hook={h} onDelete={() => setDeleting(h)} />
          ))}
        </div>
      )}

      <CreateHookDialog open={creating} onOpenChange={setCreating} />
      <ConfirmDialog
        open={!!deleting}
        onOpenChange={(o) => !o && setDeleting(null)}
        title={`Delete “${deleting?.name ?? ''}”?`}
        description="Apps that call this link will stop reaching Sentient, and tasks that start from it won’t run. This can’t be undone."
        confirmLabel="Delete webhook"
        onConfirm={async () => {
          if (!deleting) return
          try {
            await remove.mutateAsync(deleting.id)
            toast.success('Webhook deleted')
          } catch (e) {
            toast.error('Couldn’t delete it', { description: errorMessage(e) })
          }
        }}
      />
    </section>
  )
}

function CopyButton({ text, label }: { text: string; label: string }) {
  const [copied, setCopied] = useState(false)
  return (
    <IconButton
      size="xs"
      label={copied ? 'Copied' : label}
      icon={copied ? <IconCheck size={13} className="text-success" /> : <IconCopy size={13} />}
      onClick={() =>
        void copyText(text).then((ok) => {
          if (!ok) return
          setCopied(true)
          window.setTimeout(() => setCopied(false), 1400)
        })
      }
    />
  )
}

function HookRow({ hook, onDelete }: { hook: Hook; onDelete: () => void }) {
  const navigate = useNavigate()
  const url = absoluteHookUrl(hook.url)
  return (
    <div className="group flex flex-wrap items-center gap-x-3.5 gap-y-2 px-4 py-3">
      <span className="flex size-9 shrink-0 items-center justify-center rounded-xl border border-border bg-elevated text-accent-text">
        <IconWebhook size={18} />
      </span>
      <div className="min-w-0 flex-1 basis-60">
        <div className="flex items-center gap-2">
          <span className="truncate text-sm font-medium text-fg">{hook.name}</span>
          {hook.calls === 0 && <Badge size="xs">Waiting for its first call</Badge>}
        </div>
        <div className="mt-0.5 flex min-w-0 items-center gap-1">
          <code className="selectable truncate font-mono text-2xs text-fg-subtle">{url}</code>
          <CopyButton text={url} label="Copy link" />
        </div>
      </div>
      <div className="w-40 shrink-0 text-right text-xs">
        <div className="tabular-nums text-fg-muted">{hook.calls === 1 ? 'Called once' : hook.calls ? `Called ${hook.calls.toLocaleString()} times` : 'Never called'}</div>
        <div className="text-fg-subtle">{hook.last_called_at ? `Last ${relativeTime(hook.last_called_at)}` : `Made ${relativeTime(hook.created_at)}`}</div>
      </div>
      <Button
        size="xs"
        variant="ghost"
        leftIcon={<IconListCheck size={13} />}
        onClick={() => navigate(`/tasks?compose=1&prompt=${encodeURIComponent(`When my “${hook.name}” webhook is called, `)}`)}
      >
        Use in a task
      </Button>
      <IconButton size="sm" label="Delete webhook" icon={<IconTrash size={15} />} onClick={onDelete} />
    </div>
  )
}

const SUGGESTED = ['Online store orders', 'Contact form', 'Smart home', 'Zapier']

function CreateHookDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (o: boolean) => void }) {
  const { create } = useHookActions()
  const navigate = useNavigate()
  const [name, setName] = useState('')
  const [created, setCreated] = useState<HookCreated | null>(null)
  const [reveal, setReveal] = useState(false)

  const close = () => {
    onOpenChange(false)
    window.setTimeout(() => {
      setName('')
      setCreated(null)
      setReveal(false)
    }, 200)
  }

  const submit = () => {
    const n = name.trim()
    if (!n) return
    create.mutate(n, { onSuccess: setCreated, onError: (e) => toast.error('Couldn’t create the webhook', { description: errorMessage(e) }) })
  }

  if (created) {
    const url = absoluteHookUrl(created.url)
    const masked = reveal ? created.secret : '•'.repeat(Math.min(32, created.secret.length))
    const curl = (secret: string) =>
      `curl -X POST "${url}" \\\n  -H "X-Sentient-Secret: ${secret}" \\\n  -H "Content-Type: application/json" \\\n  -d '{"order": "1042", "total": 2499}'`
    return (
      <Dialog
        open={open}
        onOpenChange={(o) => !o && close()}
        size="lg"
        modalLock
        title={`“${created.name}” is ready`}
        description="Paste the link into the other app. Keep the secret private, like a password."
        footer={
          <>
            <Button variant="ghost" leftIcon={<IconListCheck size={15} />} onClick={() => (close(), navigate(`/tasks?compose=1&prompt=${encodeURIComponent(`When my “${created.name}” webhook is called, `)}`))}>
              Use in a task
            </Button>
            <Button variant="primary" onClick={close}>
              Done
            </Button>
          </>
        }
      >
        <div className="space-y-4">
          <CopyField label="Link" value={url} copy={url} />
          <CopyField
            label="Secret"
            value={masked}
            copy={created.secret}
            extra={<IconButton size="xs" label={reveal ? 'Hide secret' : 'Show secret'} icon={reveal ? <IconEyeOff size={13} /> : <IconEye size={13} />} onClick={() => setReveal((r) => !r)} />}
          />
          <Alert tone="warning" icon={<IconAlertTriangle />} title="Copy the secret now">
            For your safety, Sentient won’t show it again. If you lose it, delete this webhook and make a new one.
          </Alert>
          <div>
            <div className="mb-1.5 flex items-center justify-between">
              <span className="text-sm font-medium text-fg">Try it from a terminal</span>
              <CopyButton text={curl(created.secret)} label="Copy command" />
            </div>
            <pre className="selectable overflow-x-auto rounded-xl border border-border bg-sunken px-3.5 py-3 font-mono text-[12px] leading-relaxed text-fg">{curl(masked)}</pre>
            <p className="mt-2 text-xs text-fg-subtle">
              The other app sends the secret in the <code className="font-mono">X-Sentient-Secret</code> header. A good call gets back{' '}
              <code className="font-mono">{'{"ok": true}'}</code>; a wrong secret gets 401, an unknown link 404, a body over the size limit 413 and broken JSON 400.
              This link works on this computer; apps on the internet need a tunnel to reach it.
            </p>
          </div>
        </div>
      </Dialog>
    )
  }

  return (
    <Dialog
      open={open}
      onOpenChange={(o) => !o && close()}
      title="Create a webhook"
      description="Give it a name you’ll recognise. You’ll get a private link and a secret to paste into the other app."
      footer={
        <>
          <Button variant="ghost" onClick={close}>
            Cancel
          </Button>
          <Button variant="primary" loading={create.isPending} disabled={!name.trim()} onClick={submit}>
            Create
          </Button>
        </>
      }
    >
      <div className="space-y-3">
        <Field label="Name">
          <Input autoFocus value={name} placeholder="e.g. Online store orders" onChange={(e) => setName(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && submit()} />
        </Field>
        <div className="flex flex-wrap gap-1.5">
          {SUGGESTED.map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => setName(s)}
              className={cn('rounded-full border px-2.5 py-1 text-xs transition-colors', name === s ? 'border-accent/40 bg-accent/12 text-fg' : 'border-border text-fg-muted hover:border-border-strong hover:text-fg')}
            >
              {s}
            </button>
          ))}
        </div>
      </div>
    </Dialog>
  )
}

function CopyField({ label, value, copy, extra }: { label: string; value: string; copy: string; extra?: React.ReactNode }) {
  return (
    <div>
      <div className="mb-1.5 text-sm font-medium text-fg">{label}</div>
      <div className="flex items-center gap-1 rounded-lg border border-border bg-field py-1 pl-3 pr-1">
        <code className="selectable min-w-0 flex-1 truncate font-mono text-xs text-fg">{value}</code>
        {extra}
        <CopyButton text={copy} label={`Copy ${label.toLowerCase()}`} />
      </div>
    </div>
  )
}
