import {
  IconAlertTriangle,
  IconChevronDown,
  IconEye,
  IconEyeOff,
  IconKey,
  IconLogin2,
  IconLogout,
  IconPlayerPlay,
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
import type { McpAuth, McpServer } from '@/lib/types'
import { cn } from '@/lib/utils'
import { BrandIcon } from './BrandIcon'
import { plainDescription, RISK_LABEL, riskOf, toolLabel } from './meta'

const STATUS: Record<string, { tone: Tone; label: string }> = {
  connected: { tone: 'success', label: 'Connected' },
  connecting: { tone: 'warning', label: 'Connecting…' },
  needs_sign_in: { tone: 'warning', label: 'Needs sign-in' },
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
  const { test, remove, signIn, signOut, setEnabled } = useMcpActions()
  const [confirm, setConfirm] = useState(false)
  const [open, setOpen] = useState(false)
  const [editingValues, setEditingValues] = useState(false)
  const missing = s.missing_values ?? []
  const valueKeys = s.transport === 'http' ? s.header_keys : s.env_keys
  const status = STATUS[s.status] ?? { tone: 'neutral' as Tone, label: s.status }
  const target = s.transport === 'http' ? s.url : [s.command, ...(s.args ?? [])].filter(Boolean).join(' ')
  const canSignIn = s.transport === 'http' && (s.status === 'needs_sign_in' || (s.auth === 'oauth' && !s.signed_in))
  const startSignIn = () =>
    signIn.mutate(s.name, {
      onSuccess: () => toast.info('Finish signing in with your browser', { description: `Sentient connects to ${s.name} once you approve it.` }),
      onError: (e) => toast.error("Couldn't start the sign-in", { description: errorMessage(e) })
    })

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
        {missing.length > 0 ? (
          <Button size="sm" variant="primary" leftIcon={<IconKey size={14} />} onClick={() => setEditingValues(true)}>
            Add values
          </Button>
        ) : (
          valueKeys.length > 0 && <IconButton size="sm" label="Change saved values" icon={<IconKey size={15} />} onClick={() => setEditingValues(true)} />
        )}
        {!s.enabled && missing.length === 0 && (
          <Button
            size="sm"
            variant="primary"
            leftIcon={<IconPlayerPlay size={14} />}
            loading={setEnabled.isPending && setEnabled.variables?.name === s.name}
            onClick={() =>
              setEnabled.mutate(
                { name: s.name, enabled: true },
                { onError: (e) => toast.error(`Couldn't turn on ${s.name}`, { description: errorMessage(e) }) }
              )
            }
          >
            Turn on
          </Button>
        )}
        {s.enabled && canSignIn && (
          <Button size="sm" variant="primary" leftIcon={<IconLogin2 size={14} />} loading={signIn.isPending && signIn.variables === s.name} onClick={startSignIn}>
            {s.signing_in ? 'Sign in again' : 'Sign in'}
          </Button>
        )}
        {s.signed_in && !canSignIn && (
          <Button
            size="sm"
            variant="ghost"
            leftIcon={<IconLogout size={14} />}
            loading={signOut.isPending && signOut.variables === s.name}
            onClick={() =>
              signOut.mutate(s.name, {
                onSuccess: () => toast.success(`Signed out of ${s.name}`),
                onError: (e) => toast.error("Couldn't sign out", { description: errorMessage(e) })
              })
            }
          >
            Sign out
          </Button>
        )}
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

      {s.signing_in ? (
        <div className="px-4 pb-3">
          <Alert tone="info" icon={<Spinner size={14} />} className="py-2.5">
            Waiting for you to approve Sentient in your browser.
          </Alert>
        </div>
      ) : (
        s.status === 'needs_sign_in' &&
        s.error && (
          <div className="px-4 pb-3">
            <Alert tone="warning" icon={<IconAlertTriangle />} className="py-2.5">
              {s.error}
            </Alert>
          </div>
        )
      )}

      {missing.length > 0 && (
        <div className="px-4 pb-3">
          <Alert tone="warning" icon={<IconKey />} className="py-2.5">
            {missing.length === 1 ? `${missing[0]} has no value yet.` : `${missing.join(', ')} have no values yet.`} Add {missing.length === 1 ? 'it' : 'them'} so the server can connect.
          </Alert>
        </div>
      )}

      {s.status === 'error' && s.error && (
        <div className="px-4 pb-3">
          <Alert tone="danger" icon={<IconAlertTriangle />} className="py-2.5">
            <span className="font-mono text-xs">{s.error}</span>
          </Alert>
        </div>
      )}

      <div className="flex flex-wrap items-center gap-1.5 border-t border-border bg-sunken/30 px-4 py-2.5">
        {s.env_keys.map((k) => (
          <Badge key={k} size="xs" icon={<IconKey />} tone={missing.includes(k) ? 'warning' : undefined}>
            {k}
          </Badge>
        ))}
        {s.header_keys.map((k) => (
          <Badge key={`h-${k}`} size="xs" icon={<IconKey />} tone={missing.includes(k) ? 'warning' : undefined}>
            {k}
          </Badge>
        ))}
        {s.signed_in && (
          <Badge size="xs" tone="success" icon={<IconLogin2 />}>
            Signed in
          </Badge>
        )}
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

      <McpValuesDialog server={s} keys={valueKeys} open={editingValues} onOpenChange={setEditingValues} />

      <ConfirmDialog
        open={confirm}
        onOpenChange={setConfirm}
        title={`Remove ${s.name}?`}
        description="The server is stopped and its tools disappear from Sentient. Saved values and sign-ins are deleted from your keychain."
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

/** "Add values": fill in the header or environment values a server lists by name. They go to the keychain. */
function McpValuesDialog({ server: s, keys, open, onOpenChange }: { server: McpServer; keys: string[]; open: boolean; onOpenChange: (open: boolean) => void }) {
  const { setValues } = useMcpActions()
  const [values, setVals] = useState<Record<string, string>>({})
  const [shown, setShown] = useState<Record<string, boolean>>({})
  const [error, setError] = useState<string | null>(null)
  const missing = s.missing_values ?? []
  const filled = Object.values(values).some((v) => v.trim())
  const kind = s.transport === 'http' ? 'header' : 'setting'

  const close = (o: boolean) => {
    if (setValues.isPending) return
    if (!o) {
      setVals({})
      setShown({})
      setError(null)
    }
    onOpenChange(o)
  }

  const save = () => {
    setError(null)
    setValues.mutate(
      { name: s.name, values, enable: !s.enabled },
      {
        onSuccess: (srv) => {
          if (srv.status === 'connected') toast.success(`${srv.name} is connected`, { description: `${srv.tools.length} tools available` })
          else if (srv.status === 'needs_sign_in') toast.warning(`${srv.name} didn't accept these values`, { description: srv.error ?? undefined })
          else if (srv.status === 'error') toast.warning(`Saved, but ${srv.name} couldn't start`, { description: srv.error ?? undefined })
          else toast.success('Values saved', { description: srv.missing_values?.length ? `Still missing: ${srv.missing_values.join(', ')}` : undefined })
          close(false)
        },
        onError: (e) => setError(errorMessage(e))
      }
    )
  }

  return (
    <Dialog
      open={open}
      onOpenChange={close}
      size="md"
      modalLock
      title={`Values for ${s.name}`}
      description={`Each ${kind} is saved in your system keychain, never in a settings file. Leave a box empty to keep what is saved.`}
      footer={
        <>
          {setValues.isPending && (
            <span className="mr-auto flex items-center gap-2 text-xs text-fg-subtle">
              <Spinner size={12} /> Connecting, this can take up to 15 seconds…
            </span>
          )}
          <Button variant="ghost" disabled={setValues.isPending} onClick={() => close(false)}>
            Cancel
          </Button>
          <Button variant="primary" loading={setValues.isPending} disabled={!filled && s.enabled} onClick={save}>
            {s.enabled ? 'Save and reconnect' : 'Save and turn on'}
          </Button>
        </>
      }
    >
      <form
        className="space-y-3"
        onSubmit={(e) => {
          e.preventDefault()
          save()
        }}
      >
        {keys.map((k) => (
          <Field key={k} label={<span className="font-mono text-xs">{k}</span>} htmlFor={`mcp-value-${k}`} description={missing.includes(k) ? 'No value yet' : 'A value is saved'}>
            <Input
              id={`mcp-value-${k}`}
              size="sm"
              className="font-mono"
              type={shown[k] ? 'text' : 'password'}
              autoComplete="off"
              placeholder={missing.includes(k) ? (s.transport === 'http' && k.toLowerCase() === 'authorization' ? 'Bearer your-token' : 'value') : 'Keep the saved value'}
              value={values[k] ?? ''}
              onChange={(e) => setVals((v) => ({ ...v, [k]: e.target.value }))}
              rightSlot={
                <button
                  type="button"
                  aria-label={shown[k] ? 'Hide value' : 'Show value'}
                  onClick={() => setShown((x) => ({ ...x, [k]: !x[k] }))}
                  className="flex size-5 items-center justify-center text-fg-subtle hover:text-fg"
                >
                  {shown[k] ? <IconEyeOff size={13} /> : <IconEye size={13} />}
                </button>
              }
            />
          </Field>
        ))}
        {error && (
          <Alert tone="danger" icon={<IconAlertTriangle />} title="Couldn't save the values">
            {error}
          </Alert>
        )}
        <button type="submit" hidden />
      </form>
    </Dialog>
  )
}

interface SecretRow {
  id: number
  key: string
  value: string
  show: boolean
}

const AUTH_HELP: Record<McpAuth, string> = {
  none: 'The server is open, or you will sign in later if it asks.',
  headers: 'For servers that give you an access token or API key. Values go to your system keychain.',
  oauth: 'Your browser opens so you can approve Sentient. Sentient keeps you signed in.'
}

function SecretRows({
  title,
  description,
  rows,
  setRows,
  keyPlaceholder,
  valuePlaceholder,
  normalizeKey
}: {
  title: string
  description: string
  rows: SecretRow[]
  setRows: (update: (rows: SecretRow[]) => SecretRow[]) => void
  keyPlaceholder: string
  valuePlaceholder: string
  normalizeKey: (key: string) => string
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-sm font-medium text-fg">{title}</div>
          <div className="text-xs text-fg-subtle">{description}</div>
        </div>
        <Button size="xs" variant="ghost" leftIcon={<IconPlus size={12} />} onClick={() => setRows((r) => [...r, { id: Date.now(), key: '', value: '', show: false }])}>
          Add
        </Button>
      </div>
      {rows.map((row) => (
        <div key={row.id} className="flex items-center gap-2">
          <Input
            size="sm"
            className="font-mono"
            wrapperClassName="w-[40%]"
            placeholder={keyPlaceholder}
            value={row.key}
            onChange={(e) => setRows((r) => r.map((x) => (x.id === row.id ? { ...x, key: normalizeKey(e.target.value) } : x)))}
          />
          <Input
            size="sm"
            className="font-mono"
            type={row.show ? 'text' : 'password'}
            placeholder={valuePlaceholder}
            value={row.value}
            onChange={(e) => setRows((r) => r.map((x) => (x.id === row.id ? { ...x, value: e.target.value } : x)))}
            rightSlot={
              <button
                type="button"
                aria-label={row.show ? 'Hide value' : 'Show value'}
                onClick={() => setRows((r) => r.map((x) => (x.id === row.id ? { ...x, show: !x.show } : x)))}
                className="flex size-5 items-center justify-center text-fg-subtle hover:text-fg"
              >
                {row.show ? <IconEyeOff size={13} /> : <IconEye size={13} />}
              </button>
            }
          />
          <IconButton size="sm" label="Remove" icon={<IconX size={14} />} onClick={() => setRows((r) => r.filter((x) => x.id !== row.id))} />
        </div>
      ))}
    </div>
  )
}

const toObject = (rows: SecretRow[]) => Object.fromEntries(rows.filter((r) => r.key.trim()).map((r) => [r.key.trim(), r.value]))

function AddMcpDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  const { add, signIn } = useMcpActions()
  const [name, setName] = useState('')
  const [transport, setTransport] = useState<'stdio' | 'http'>('stdio')
  const [command, setCommand] = useState('')
  const [args, setArgs] = useState('')
  const [url, setUrl] = useState('')
  const [auth, setAuth] = useState<McpAuth>('none')
  const [env, setEnv] = useState<SecretRow[]>([])
  const [headers, setHeaders] = useState<SecretRow[]>([])
  const [touched, setTouched] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const reset = () => {
    setName('')
    setTransport('stdio')
    setCommand('')
    setArgs('')
    setUrl('')
    setAuth('none')
    setEnv([])
    setHeaders([])
    setTouched(false)
    setError(null)
  }

  const nameInvalid = !/^[A-Za-z0-9][A-Za-z0-9 _-]{0,40}$/.test(name.trim())
  const targetInvalid = transport === 'stdio' ? !command.trim() : !/^https?:\/\/\S+$/.test(url.trim())
  const headersInvalid = transport === 'http' && auth === 'headers' && !headers.some((r) => r.key.trim() && r.value.trim())

  const submit = () => {
    setTouched(true)
    if (nameInvalid || targetInvalid || headersInvalid) return
    setError(null)
    add.mutate(
      transport === 'stdio'
        ? { name: name.trim(), transport, command: command.trim(), args: splitArgs(args), env: toObject(env) }
        : { name: name.trim(), transport, url: url.trim(), auth, headers: auth === 'headers' ? toObject(headers) : {} },
      {
        onSuccess: (srv) => {
          if (srv.auth === 'oauth' && !srv.signed_in) {
            signIn.mutate(srv.name, {
              onSuccess: () => toast.info(`${srv.name} was added`, { description: 'Finish signing in with your browser.' }),
              onError: (e) => toast.warning(`${srv.name} was added but the sign-in didn't start`, { description: errorMessage(e) })
            })
          } else if (srv.status === 'needs_sign_in') toast.warning(`${srv.name} asks you to sign in`, { description: srv.error ?? 'Use Sign in on the server.' })
          else if (srv.status === 'error') toast.warning(`${srv.name} was added but couldn't start`, { description: srv.error ?? undefined })
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
          <>
            <div className="grid gap-3 sm:grid-cols-[1fr_1.4fr]">
              <Field label="Command" htmlFor="mcp-cmd" error={touched && targetInvalid ? 'Required' : undefined}>
                <Input id="mcp-cmd" className="font-mono text-xs" value={command} placeholder="npx" invalid={touched && targetInvalid} onChange={(e) => setCommand(e.target.value)} />
              </Field>
              <Field label="Arguments" htmlFor="mcp-args" optional description='Separated by spaces. Quote values with spaces.'>
                <Input id="mcp-args" className="font-mono text-xs" value={args} placeholder="-y @acme/mcp-server" onChange={(e) => setArgs(e.target.value)} />
              </Field>
            </div>
            <SecretRows
              title="Environment variables"
              description="API keys and tokens the server needs. Values go to your system keychain."
              rows={env}
              setRows={setEnv}
              keyPlaceholder="API_KEY"
              valuePlaceholder="value"
              normalizeKey={(k) => k.toUpperCase().replace(/\s+/g, '_')}
            />
          </>
        ) : (
          <>
            <Field label="Server URL" htmlFor="mcp-url" error={touched && targetInvalid ? 'Enter an http(s) URL' : undefined}>
              <Input id="mcp-url" className="font-mono text-xs" value={url} placeholder="https://mcp.example.com/mcp" invalid={touched && targetInvalid} onChange={(e) => setUrl(e.target.value)} />
            </Field>
            <Field label="Sign-in" description={AUTH_HELP[auth]}>
              <SegmentedControl
                fullWidth
                size="sm"
                value={auth}
                onChange={(v) => {
                  setAuth(v)
                  if (v === 'headers' && !headers.length) setHeaders([{ id: Date.now(), key: 'Authorization', value: '', show: false }])
                }}
                options={[
                  { value: 'none', label: 'None' },
                  { value: 'oauth', label: 'Sign in with browser', icon: <IconLogin2 size={14} /> },
                  { value: 'headers', label: 'Access token', icon: <IconKey size={14} /> }
                ]}
              />
            </Field>
            {auth === 'headers' && (
              <>
                <SecretRows
                  title="Headers"
                  description="Sent with every request, for example Authorization with the value Bearer and your token."
                  rows={headers}
                  setRows={setHeaders}
                  keyPlaceholder="Authorization"
                  valuePlaceholder="Bearer your-token"
                  normalizeKey={(k) => k.replace(/\s+/g, '-')}
                />
                {touched && headersInvalid && <p className="text-xs text-danger">Add a header with a name and a value.</p>}
              </>
            )}
          </>
        )}

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
