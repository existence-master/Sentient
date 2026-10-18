import { api } from '@/lib/api'
import {
  IconAlertTriangle,
  IconArrowRight,
  IconBook2,
  IconCircleCheckFilled,
  IconEye,
  IconEyeOff,
  IconKey,
  IconListNumbers,
  IconLock,
  IconWorld
} from '@tabler/icons-react'
import { AnimatePresence, motion } from 'motion/react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { toast } from 'sonner'
import { Alert, Button, Dialog, Field, Input, Spinner } from '@/components/ui'
import { upsertIntegration, useIntegrationActions, useIntegrations } from '@/hooks/integrations'
import { errorMessage } from '@/lib/api'
import type { ConnectionAccess, Integration, OAuthStart } from '@/lib/types'
import { cn } from '@/lib/utils'
import { useQueryClient } from '@tanstack/react-query'
import { AccessChoice, hasChangingTools } from './AccessChoice'
import { BrandIcon } from './BrandIcon'
import { CopyChip, ExternalLinkChip, InstructionsGuide, openExternal } from './InstructionsGuide'
import { isGoogle } from './meta'

type Phase = { kind: 'form' } | { kind: 'browser'; start: OAuthStart } | { kind: 'done'; label: string | null }

export function ConnectDialog({
  integration,
  open,
  onOpenChange
}: {
  integration: Integration | null
  open: boolean
  onOpenChange: (open: boolean) => void
}) {
  return (
    <Dialog
      open={open && !!integration}
      onOpenChange={onOpenChange}
      size="lg"
      modalLock
      title={integration ? <ConnectTitle integration={integration} /> : 'Connect'}
      description={integration?.description}
    >
      {integration && <ConnectBody key={integration.id} integration={integration} close={() => onOpenChange(false)} />}
    </Dialog>
  )
}

function ConnectTitle({ integration }: { integration: Integration }) {
  return (
    <span className="flex items-center gap-3">
      <BrandIcon id={integration.id} icon={integration.icon} size={34} />
      <span>
        {integration.connected ? 'Reconnect' : 'Connect'} {integration.display_name}
      </span>
    </span>
  )
}

function ConnectBody({ integration, close }: { integration: Integration; close: () => void }) {
  const qc = useQueryClient()
  const { connect } = useIntegrationActions()
  const list = useIntegrations()
  const live = list.data?.find((x) => x.id === integration.id) ?? integration
  const fields = live.setup.fields
  const oauth = live.auth_type === 'oauth'
  const googleClientSaved = isGoogle(live.id) && fields.length > 0 && fields.every((f) => !f.required)

  const [values, setValues] = useState<Record<string, string>>({})
  const [touched, setTouched] = useState(false)
  const [reveal, setReveal] = useState<Record<string, boolean>>({})
  const [formError, setFormError] = useState<string | null>(live.status === 'error' ? live.error : null)
  const [phase, setPhase] = useState<Phase>({ kind: 'form' })
  const [guideOpen, setGuideOpen] = useState(!googleClientSaved)
  const [access, setAccess] = useState<ConnectionAccess>(live.access ?? 'read_write')
  const canChooseAccess = live.access !== undefined && hasChangingTools(live.tools)
  const errorAtStart = useRef<string | null>(null)
  const phaseRef = useRef<Phase>(phase)
  phaseRef.current = phase

  const cancelBrowserFlow = () => {
    void api.integrations
      .cancel(live.id)
      .then((i) => upsertIntegration(qc, i))
      .catch(() => undefined)
  }

  // Closing the dialog mid sign-in abandons the flow so the card doesn't stay "connecting".
  useEffect(() => {
    return () => {
      if (phaseRef.current.kind === 'browser') void api.integrations.cancel(integration.id).catch(() => undefined)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const missing = useMemo(() => fields.filter((f) => f.required && !values[f.key]?.trim()).map((f) => f.key), [fields, values])

  // Browser flows finish out of band: watch the live integration (integration.updated keeps the cache fresh).
  useEffect(() => {
    if (phase.kind !== 'browser') return
    if (live.connected && live.status === 'connected') {
      setPhase({ kind: 'done', label: live.account_label })
    } else if (live.status === 'error' && live.error && live.error !== errorAtStart.current) {
      setFormError(live.error)
      setPhase({ kind: 'form' })
    }
  }, [phase.kind, live.connected, live.status, live.error, live.account_label])

  useEffect(() => {
    if (phase.kind !== 'done') return
    toast.success(`${live.display_name} is connected${phase.label ? ` as ${phase.label}` : ''}`)
    const t = window.setTimeout(close, 1600)
    return () => window.clearTimeout(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase.kind])

  const submit = () => {
    setTouched(true)
    if (missing.length) return
    setFormError(null)
    errorAtStart.current = live.status === 'error' ? live.error : null
    const payload = Object.fromEntries(Object.entries(values).map(([k, v]) => [k, v.trim()]).filter(([, v]) => v))
    connect.mutate(
      { id: live.id, fields: payload, access: canChooseAccess ? access : undefined },
      {
        onSuccess: (res) => {
          if ('auth_url' in res) {
            setPhase({ kind: 'browser', start: res })
            // Fallback when `integration.updated` is delayed: refetch once the window regains focus.
            const refresh = () => void qc.invalidateQueries({ queryKey: ['integrations'] })
            window.addEventListener('focus', refresh, { once: true })
          } else {
            upsertIntegration(qc, res)
            setPhase({ kind: 'done', label: res.account_label })
          }
        },
        onError: (e) => setFormError(errorMessage(e))
      }
    )
  }

  if (phase.kind === 'done') {
    return (
      <motion.div initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} className="flex flex-col items-center gap-3 py-10 text-center">
        <motion.span initial={{ scale: 0.4 }} animate={{ scale: 1 }} transition={{ type: 'spring', stiffness: 380, damping: 18 }}>
          <IconCircleCheckFilled size={52} className="text-success" />
        </motion.span>
        <div className="text-md font-semibold text-fg">{live.display_name} is connected</div>
        {phase.label && <div className="text-sm text-fg-muted">Signed in as {phase.label}</div>}
        <Button className="mt-3" variant="secondary" onClick={close}>
          Done
        </Button>
      </motion.div>
    )
  }

  if (phase.kind === 'browser') {
    const { start } = phase
    return (
      <div className="flex flex-col items-center gap-4 py-6 text-center">
        {start.user_code ? (
          <>
            <div className="text-sm text-fg-muted">Enter this code on GitHub to finish connecting:</div>
            <div className="flex items-center gap-3">
              <span className="rounded-xl border border-border-strong bg-sunken px-5 py-3 font-mono text-3xl font-semibold tracking-[0.2em] text-fg">
                {start.user_code}
              </span>
            </div>
            <CopyChip value={start.user_code} className="h-7 px-2.5 text-sm" />
            <Button variant="primary" rightIcon={<IconArrowRight size={15} />} onClick={() => openExternal(start.auth_url)}>
              Open {new URL(start.auth_url).hostname.replace(/^www\./, '')}
            </Button>
          </>
        ) : (
          <>
            <div className="relative flex size-16 items-center justify-center">
              <span className="absolute inset-0 animate-ping rounded-full bg-accent/15" />
              <span className="relative flex size-14 items-center justify-center rounded-full border border-accent/30 bg-accent/10 text-accent-text">
                <IconWorld size={26} />
              </span>
            </div>
            <div>
              <div className="text-md font-semibold text-fg">Finish signing in in your browser</div>
              <p className="mx-auto mt-1 max-w-sm text-sm text-fg-muted">
                A sign-in page opened in your browser. Pick your account and allow access, then come back here. This window updates by itself.
              </p>
            </div>
            <Button variant="secondary" size="sm" onClick={() => openExternal(start.auth_url)}>
              Open the sign-in page again
            </Button>
          </>
        )}
        <div className="flex items-center gap-2 text-xs text-fg-subtle">
          <Spinner size={13} /> Waiting for {live.display_name}…
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => {
            cancelBrowserFlow()
            setPhase({ kind: 'form' })
          }}
        >
          Cancel
        </Button>
      </div>
    )
  }

  const hasGuide = !!live.setup.instructions_md?.trim()

  return (
    <form
      className="space-y-4"
      onSubmit={(e) => {
        e.preventDefault()
        submit()
      }}
    >
      {googleClientSaved && (
        <Alert tone="success" icon={<IconCircleCheckFilled />} title="Your Google client is already saved">
          Leave the fields empty and continue to sign in with Google.
        </Alert>
      )}

      {hasGuide && (
        <div className="overflow-hidden rounded-xl border border-border bg-sunken/40">
          <button
            type="button"
            onClick={() => setGuideOpen((o) => !o)}
            className="flex w-full items-center gap-2 px-3.5 py-2.5 text-left text-sm font-medium text-fg hover:bg-hover"
            aria-expanded={guideOpen}
          >
            <IconListNumbers size={16} className="text-accent-text" />
            <span className="flex-1">Step-by-step guide</span>
            <span className="text-xs font-normal text-fg-subtle">{guideOpen ? 'Hide' : 'Show'}</span>
          </button>
          <AnimatePresence initial={false}>
            {guideOpen && (
              <motion.div initial={{ height: 0 }} animate={{ height: 'auto' }} exit={{ height: 0 }} className="overflow-hidden">
                <div className="max-h-[300px] overflow-y-auto border-t border-border px-4 py-3.5">
                  <InstructionsGuide markdown={live.setup.instructions_md} />
                  {live.setup.docs_url && (
                    <div className="mt-3 flex items-center gap-1.5 text-xs text-fg-subtle">
                      <IconBook2 size={13} /> Official docs <ExternalLinkChip url={live.setup.docs_url} />
                    </div>
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}

      {fields.length > 0 && (
        <div className={cn('grid gap-3', fields.length > 1 && 'sm:grid-cols-2')}>
          {fields.map((f, i) => {
            const invalid = touched && missing.includes(f.key)
            const shown = reveal[f.key]
            return (
              <Field
                key={f.key}
                htmlFor={`field-${f.key}`}
                label={f.label}
                optional={!f.required}
                error={invalid ? 'Required' : undefined}
                description={f.help}
                className={cn(fields.length > 2 && fields.length % 2 === 1 && i === fields.length - 1 && 'sm:col-span-2')}
              >
                <Input
                  id={`field-${f.key}`}
                  autoFocus={i === 0 && !googleClientSaved}
                  type={f.secret && !shown ? 'password' : 'text'}
                  autoComplete="off"
                  invalid={invalid}
                  value={values[f.key] ?? ''}
                  placeholder={(f as { placeholder?: string }).placeholder || undefined}
                  leftIcon={f.secret ? <IconLock /> : f.key.includes('key') || f.key.includes('token') ? <IconKey /> : undefined}
                  onChange={(e) => setValues((v) => ({ ...v, [f.key]: e.target.value }))}
                  rightSlot={
                    f.secret ? (
                      <button
                        type="button"
                        aria-label={shown ? 'Hide value' : 'Show value'}
                        onClick={() => setReveal((r) => ({ ...r, [f.key]: !r[f.key] }))}
                        className="flex size-6 items-center justify-center rounded text-fg-subtle hover:text-fg"
                      >
                        {shown ? <IconEyeOff size={14} /> : <IconEye size={14} />}
                      </button>
                    ) : undefined
                  }
                />
              </Field>
            )
          })}
        </div>
      )}

      {canChooseAccess && (
        <Field label="What Sentient may do">
          <AccessChoice value={access} onChange={setAccess} app={live.display_name} />
        </Field>
      )}

      {formError && (
        <Alert tone="danger" icon={<IconAlertTriangle />} title={live.status === 'error' && formError === live.error ? 'Last attempt failed' : "Couldn't connect"}>
          {formError}
        </Alert>
      )}

      <div className="-mx-5 -mb-3 flex items-center gap-2 border-t border-border px-5 py-3">
        <span className="mr-auto flex items-center gap-1.5 text-xs text-fg-subtle">
          <IconLock size={13} /> Secrets are stored in your system keychain
        </span>
        <Button variant="ghost" onClick={close}>
          Cancel
        </Button>
        <Button type="submit" variant="primary" loading={connect.isPending} rightIcon={oauth ? <IconArrowRight size={15} /> : undefined}>
          {oauth ? 'Continue in browser' : connect.isPending ? 'Checking…' : 'Connect'}
        </Button>
      </div>
    </form>
  )
}
