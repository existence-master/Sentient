import {
  IconAlertTriangle,
  IconChevronDown,
  IconEye,
  IconEyeOff,
  IconKey,
  IconPlus,
  IconRefresh,
  IconServer2,
  IconTerminal2,
  IconTrash,
  IconWorld,
  IconX
} from '@tabler/icons-react'
import { AnimatePresence, motion } from 'motion/react'
import { useState } from 'react'
import { toast } from 'sonner'
import {
  Alert,
  Badge,
  Button,
  ConfirmDialog,
  Dialog,
  Field,
  IconButton,
  Input,
  SegmentedControl,
  Skeleton,
  Spinner,
  StatusDot,
  type Tone
} from '@/components/ui'
import { useMcpActions, useMcpServers } from '@/hooks/integrations'
import { errorMessage, isNotImplemented } from '@/lib/api'
import type { McpServer } from '@/lib/types'
import { cn } from '@/lib/utils'
import { BrandIcon } from './BrandIcon'
import { plainDescription, RISK_LABEL, riskOf, toolLabel } from './meta'

const STATUS: Record<string, { tone: Tone; label: string }> = {
  connected: { tone: 'success', label: 'Connected' },
  connecting: { tone: 'warning', label: 'Connecting…' },
  error: { tone: 'danger', label: 'Error' },
  disconnected: { tone: 'neutral', label: 'Disconnected' },
  disabled: { tone: 'neutral', label: 'Turned off' }
}

/** Shell-like argument split: respects "double" and 'single' quotes. */
export function splitArgs(raw: string): string[] {
  const out: string[] = []
  for (const m of raw.matchAll(/"([^"]*)"|'([^']*)'|(\S+)/g)) out.push(m[1] ?? m[2] ?? m[3] ?? '')
  return out
}

export function McpServersSection({ id }: { id?: string }) {
  const servers = useMcpServers()
  const [adding, setAdding] = useState(false)
  const anyConnecting = (servers.data ?? []).some((s) => s.status === 'connecting')

  return (
    <section id={id} className="scroll-mt-6">
      <div className="mb-3 flex items-end gap-4">
        <div className="min-w-0 flex-1">
          <h2 className="flex items-center gap-2 text-md font-semibold text-fg">
            Custom MCP servers
            {servers.data && servers.data.length > 0 && <span className="text-sm font-normal text-fg-subtle">{servers.data.length}</span>}
            {anyConnecting && <Spinner size={13} className="text-fg-subtle" />}
          </h2>
          <p className="mt-0.5 text-sm text-fg-subtle">
            Plug in any Model Context Protocol server to give Sentient new tools. For developers and power users.
          </p>
        </div>
        <Button variant="secondary" leftIcon={<IconPlus size={15} />} onClick={() => setAdding(true)}>
          Add server
        </Button>
      </div>

      {servers.isLoading ? (
        <Skeleton className="h-24 rounded-xl" />
      ) : servers.isError ? (
        <Alert tone={isNotImplemented(servers.error) ? 'info' : 'danger'} title="Couldn't load MCP servers">
          {errorMessage(servers.error)}
        </Alert>
      ) : !servers.data?.length ? (
        <button
          type="button"
          onClick={() => setAdding(true)}
          className="flex w-full items-center gap-4 rounded-xl border border-dashed border-border-strong bg-surface/50 px-5 py-5 text-left transition-colors hover:bg-hover"
        >
          <BrandIcon id="mcp" size={40} />
          <div className="min-w-0 flex-1">
            <div className="text-sm font-medium text-fg">No custom servers yet</div>
            <div className="text-xs text-fg-subtle">Run a local command (npx, uvx, a script) or connect to a remote MCP URL.</div>
          </div>
          <IconPlus size={16} className="text-fg-subtle" />
        </button>
      ) : (
        <ul className="space-y-2.5">
          {servers.data.map((s) => (
            <McpServerRow key={s.name} server={s} />
          ))}
        </ul>
      )}

      <AddMcpDialog open={adding} onOpenChange={setAdding} />
    </section>
  )
}

function McpServerRow({ server: s }: { server: McpServer }) {
  const { test, remove } = useMcpActions()
  const [confirm, setConfirm] = useState(false)
  const [open, setOpen] = useState(false)
  const status = STATUS[s.status] ?? { tone: 'neutral' as Tone, label: s.status }
  const target = s.transport === 'http' ? s.url : [s.command, ...(s.args ?? [])].filter(Boolean).join(' ')

  return (
    <li className={cn('overflow-hidden rounded-xl border bg-surface', s.status === 'error' ? 'border-danger/30' : 'border-border')}>
      <div className="flex items-center gap-3 px-4 py-3">
        <span className="flex size-9 shrink-0 items-center justify-center rounded-lg border border-border bg-elevated text-fg-muted">
          {s.transport === 'http' ? <IconWorld size={18} /> : <IconTerminal2 size={18} />}
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="truncate text-sm font-semibold text-fg">{s.name}</span>
            <Badge size="xs">{s.transport === 'http' ? 'Remote' : 'Local command'}</Badge>
            <span className="flex items-center gap-1.5 text-xs text-fg-muted">
              {s.status === 'connecting' ? <Spinner size={11} /> : <StatusDot tone={status.tone} />}
              {status.label}
            </span>
          </div>
          <div className="mt-0.5 truncate font-mono text-xs text-fg-subtle" title={target ?? ''}>
            {target}
          </div>
        </div>
        <Button
          size="sm"
          variant="ghost"
          leftIcon={<IconRefresh size={14} />}
          loading={test.isPending && test.variables === s.name}
          onClick={() =>
            test.mutate(s.name, {
              onSuccess: (r) =>
                r.ok
                  ? toast.success(`${s.name} works`, { description: `${r.tools.length} tool${r.tools.length === 1 ? '' : 's'} found` })
                  : toast.error(`${s.name} didn't respond`, { description: r.error }),
              onError: (e) => toast.error('Test failed', { description: errorMessage(e) })
            })
          }
        >
          Test
        </Button>
        <IconButton size="sm" label="Remove server" icon={<IconTrash size={15} />} onClick={() => setConfirm(true)} />
      </div>

      {s.status === 'error' && s.error && (
        <div className="px-4 pb-3">
          <Alert tone="danger" icon={<IconAlertTriangle />} className="py-2.5">
            <span className="font-mono text-xs">{s.error}</span>
          </Alert>
        </div>
      )}

      <div className="flex flex-wrap items-center gap-1.5 border-t border-border bg-sunken/30 px-4 py-2.5">
        {s.env_keys.map((k) => (
          <Badge key={k} size="xs" icon={<IconKey />}>
            {k}
          </Badge>
        ))}
        {s.tools.length ? (
          <button type="button" onClick={() => setOpen((o) => !o)} className="flex items-center gap-1 text-xs font-medium text-fg-muted hover:text-fg">
            <IconChevronDown size={13} className={cn('transition-transform', open && 'rotate-180')} />
            {s.tools.length} tool{s.tools.length === 1 ? '' : 's'} discovered
          </button>
        ) : (
          <span className="text-xs text-fg-subtle">{s.status === 'connected' ? 'No tools exposed' : 'Tools appear once the server connects'}</span>
        )}
      </div>

      <AnimatePresence initial={false}>
        {open && s.tools.length > 0 && (
          <motion.ul initial={{ height: 0 }} animate={{ height: 'auto' }} exit={{ height: 0 }} className="divide-y divide-border overflow-hidden border-t border-border">
            {s.tools.map((t) => {
              const risk = RISK_LABEL[riskOf(t.risk)]
              return (
                <li key={t.name} className="flex items-start gap-3 px-4 py-2">
                  <div className="min-w-0 flex-1">
                    <div className="text-sm text-fg">{toolLabel(t.mcp_name || t.name)}</div>
                    {t.description && <div className="text-xs text-fg-subtle">{plainDescription(t.description)}</div>}
                  </div>
                  <Badge size="xs" tone={risk.tone}>
                    {risk.label}
                  </Badge>
                </li>
              )
            })}
          </motion.ul>
        )}
      </AnimatePresence>

      <ConfirmDialog
        open={confirm}
        onOpenChange={setConfirm}
        title={`Remove ${s.name}?`}
        description="The server is stopped and its tools disappear from Sentient. Saved environment values are deleted from your keychain."
        confirmLabel="Remove"
        onConfirm={async () => {
          try {
            await remove.mutateAsync(s.name)
            toast.success(`${s.name} removed`)
          } catch (e) {
            toast.error("Couldn't remove the server", { description: errorMessage(e) })
          }
        }}
      />
    </li>
  )
}

interface EnvRow {
  id: number
  key: string
  value: string
  show: boolean
}

function AddMcpDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  const { add } = useMcpActions()
  const [name, setName] = useState('')
  const [transport, setTransport] = useState<'stdio' | 'http'>('stdio')
  const [command, setCommand] = useState('')
  const [args, setArgs] = useState('')
  const [url, setUrl] = useState('')
  const [env, setEnv] = useState<EnvRow[]>([])
  const [touched, setTouched] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const reset = () => {
    setName('')
    setTransport('stdio')
    setCommand('')
    setArgs('')
    setUrl('')
    setEnv([])
    setTouched(false)
    setError(null)
  }

  const nameInvalid = !/^[A-Za-z0-9][A-Za-z0-9 _-]{0,40}$/.test(name.trim())
  const targetInvalid = transport === 'stdio' ? !command.trim() : !/^https?:\/\/\S+$/.test(url.trim())

  const submit = () => {
    setTouched(true)
    if (nameInvalid || targetInvalid) return
    setError(null)
    const envObj = Object.fromEntries(env.filter((r) => r.key.trim()).map((r) => [r.key.trim(), r.value]))
    add.mutate(
      transport === 'stdio'
        ? { name: name.trim(), transport, command: command.trim(), args: splitArgs(args), env: envObj }
        : { name: name.trim(), transport, url: url.trim(), env: envObj },
      {
        onSuccess: (srv) => {
          if (srv.status === 'error') toast.warning(`${srv.name} was added but couldn't start`, { description: srv.error ?? undefined })
          else if (srv.status === 'connected') toast.success(`${srv.name} is connected`, { description: `${srv.tools.length} tools available` })
          else toast.success(`${srv.name} added`, { description: 'Still connecting in the background.' })
          reset()
          onOpenChange(false)
        },
        onError: (e) => setError(errorMessage(e))
      }
    )
  }

  return (
    <Dialog
      open={open}
      onOpenChange={(o) => {
        if (add.isPending) return
        if (!o) reset()
        onOpenChange(o)
      }}
      size="md"
      modalLock
      title={
        <span className="flex items-center gap-3">
          <BrandIcon id="mcp" size={32} />
          Add an MCP server
        </span>
      }
      description="Sentient starts the server, discovers its tools and keeps it running."
      footer={
        <>
          {add.isPending && (
            <span className="mr-auto flex items-center gap-2 text-xs text-fg-subtle">
              <Spinner size={12} /> Connecting, this can take up to 15 seconds…
            </span>
          )}
          <Button variant="ghost" disabled={add.isPending} onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button variant="primary" loading={add.isPending} onClick={submit}>
            Add server
          </Button>
        </>
      }
    >
      <form
        className="space-y-4"
        onSubmit={(e) => {
          e.preventDefault()
          submit()
        }}
      >
        <Field label="Name" htmlFor="mcp-name" error={touched && nameInvalid ? 'Use letters, numbers, spaces, - or _' : undefined} description="Shown in Sentient and used to name its tools.">
          <Input id="mcp-name" autoFocus value={name} placeholder="e.g. linear" invalid={touched && nameInvalid} onChange={(e) => setName(e.target.value)} />
        </Field>
        <SegmentedControl
          fullWidth
          value={transport}
          onChange={setTransport}
          options={[
            { value: 'stdio', label: 'Local command', icon: <IconTerminal2 size={14} /> },
            { value: 'http', label: 'Remote URL', icon: <IconWorld size={14} /> }
          ]}
        />
        {transport === 'stdio' ? (
          <div className="grid gap-3 sm:grid-cols-[1fr_1.4fr]">
            <Field label="Command" htmlFor="mcp-cmd" error={touched && targetInvalid ? 'Required' : undefined}>
              <Input id="mcp-cmd" className="font-mono text-xs" value={command} placeholder="npx" invalid={touched && targetInvalid} onChange={(e) => setCommand(e.target.value)} />
            </Field>
            <Field label="Arguments" htmlFor="mcp-args" optional description='Separated by spaces. Quote values with spaces.'>
              <Input id="mcp-args" className="font-mono text-xs" value={args} placeholder="-y @acme/mcp-server" onChange={(e) => setArgs(e.target.value)} />
            </Field>
          </div>
        ) : (
          <Field label="Server URL" htmlFor="mcp-url" error={touched && targetInvalid ? 'Enter an http(s) URL' : undefined}>
            <Input id="mcp-url" className="font-mono text-xs" value={url} placeholder="https://mcp.example.com/mcp" invalid={touched && targetInvalid} onChange={(e) => setUrl(e.target.value)} />
          </Field>
        )}

        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm font-medium text-fg">Environment variables</div>
              <div className="text-xs text-fg-subtle">API keys and tokens the server needs. Values go to your system keychain.</div>
            </div>
            <Button size="xs" variant="ghost" leftIcon={<IconPlus size={12} />} onClick={() => setEnv((rows) => [...rows, { id: Date.now(), key: '', value: '', show: false }])}>
              Add
            </Button>
          </div>
          {env.map((row) => (
            <div key={row.id} className="flex items-center gap-2">
              <Input
                size="sm"
                className="font-mono"
                wrapperClassName="w-[40%]"
                placeholder="API_KEY"
                value={row.key}
                onChange={(e) => setEnv((rows) => rows.map((r) => (r.id === row.id ? { ...r, key: e.target.value.toUpperCase().replace(/\s+/g, '_') } : r)))}
              />
              <Input
                size="sm"
                className="font-mono"
                type={row.show ? 'text' : 'password'}
                placeholder="value"
                value={row.value}
                onChange={(e) => setEnv((rows) => rows.map((r) => (r.id === row.id ? { ...r, value: e.target.value } : r)))}
                rightSlot={
                  <button
                    type="button"
                    aria-label={row.show ? 'Hide value' : 'Show value'}
                    onClick={() => setEnv((rows) => rows.map((r) => (r.id === row.id ? { ...r, show: !r.show } : r)))}
                    className="flex size-5 items-center justify-center text-fg-subtle hover:text-fg"
                  >
                    {row.show ? <IconEyeOff size={13} /> : <IconEye size={13} />}
                  </button>
                }
              />
              <IconButton size="sm" label="Remove variable" icon={<IconX size={14} />} onClick={() => setEnv((rows) => rows.filter((r) => r.id !== row.id))} />
            </div>
          ))}
        </div>

        {error && (
          <Alert tone="danger" icon={<IconAlertTriangle />} title="Couldn't add the server">
            {error}
          </Alert>
        )}
        <button type="submit" hidden />
      </form>
    </Dialog>
  )
}

export { IconServer2 as McpIcon }
