import {
  IconAlertTriangle,
  IconArrowUpRight,
  IconBolt,
  IconChevronDown,
  IconCircleCheck,
  IconPlugConnected,
  IconPlugConnectedX,
  IconPlus,
  IconShieldLock,
  IconSparkles,
  IconStethoscope,
  IconUserCircle
} from '@tabler/icons-react'
import { useEffect, useRef, useState, type ReactNode } from 'react'
import { useNavigate } from 'react-router'
import { toast } from 'sonner'
import { Alert, Badge, Button, ConfirmDialog, Sheet, Spinner, StatusDot, Tooltip } from '@/components/ui'
import { useIntegrationActions } from '@/hooks/integrations'
import { errorMessage } from '@/lib/api'
import type { Integration, IntegrationTestResult } from '@/lib/types'
import { cn, formatNumber } from '@/lib/utils'
import { BrandIcon } from './BrandIcon'
import { categoryLabel, plainDescription, RISK_LABEL, riskOf, statusMeta, toolLabel, TRIGGER_HINT } from './meta'
import { PrivacyFiltersEditor } from './PrivacyFiltersEditor'

export function IntegrationDrawer({
  integration,
  all,
  open,
  focus,
  onOpenChange,
  onConnect,
  onOpenOther
}: {
  integration: Integration | null
  all: Integration[]
  open: boolean
  focus?: string | null
  onOpenChange: (open: boolean) => void
  onConnect: (id: string) => void
  onOpenOther: (id: string) => void
}) {
  return (
    <Sheet
      open={open && !!integration}
      onOpenChange={onOpenChange}
      width={540}
      title={integration?.display_name ?? 'Integration'}
      description={integration ? categoryLabel(integration.category) : undefined}
    >
      {integration && (
        <DrawerBody key={integration.id} integration={integration} all={all} focus={focus} onConnect={onConnect} onOpenOther={onOpenOther} />
      )}
    </Sheet>
  )
}

function Section({ id, icon, title, description, children, actions }: { id?: string; icon: ReactNode; title: string; description?: ReactNode; children: ReactNode; actions?: ReactNode }) {
  return (
    <section id={id} className="scroll-mt-4 border-t border-border px-5 py-5">
      <div className="mb-3 flex items-start gap-2.5">
        <span className="mt-0.5 flex text-fg-subtle [&_svg]:size-4">{icon}</span>
        <div className="min-w-0 flex-1">
          <h3 className="text-sm font-semibold text-fg">{title}</h3>
          {description && <p className="mt-0.5 text-xs leading-relaxed text-fg-subtle">{description}</p>}
        </div>
        {actions}
      </div>
      {children}
    </section>
  )
}

function DrawerBody({
  integration: i,
  all,
  focus,
  onConnect,
  onOpenOther
}: {
  integration: Integration
  all: Integration[]
  focus?: string | null
  onConnect: (id: string) => void
  onOpenOther: (id: string) => void
}) {
  const navigate = useNavigate()
  const actions = useIntegrationActions()
  const [confirm, setConfirm] = useState(false)
  const [test, setTest] = useState<IntegrationTestResult | null>(null)
  const [showAllTools, setShowAllTools] = useState(false)
  const root = useRef<HTMLDivElement>(null)

  const builtin = i.auth_type === 'builtin'
  const status = statusMeta(i)
  const upgrades = all.filter((x) => x.alternative_for === i.id)
  const replaces = i.alternative_for ? all.find((x) => x.id === i.alternative_for) : undefined
  const counts = i.tools.reduce<Record<string, number>>((acc, t) => {
    const r = riskOf(t.risk)
    acc[r] = (acc[r] ?? 0) + 1
    return acc
  }, {})
  const tools = showAllTools ? i.tools : i.tools.slice(0, 6)

  useEffect(() => {
    if (!focus) return
    const t = window.setTimeout(() => root.current?.querySelector(`#section-${focus}`)?.scrollIntoView({ block: 'start' }), 260)
    return () => window.clearTimeout(t)
  }, [focus])

  return (
    <div ref={root} className="pb-8">
      {/* header */}
      <div className="relative overflow-hidden px-5 pb-5 pt-5">
        <div className="flex items-start gap-4">
          <BrandIcon id={i.id} icon={i.icon} size={56} />
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2">
              <h2 className="text-lg font-semibold tracking-tight text-fg">{i.display_name}</h2>
              <Badge tone={status.tone === 'neutral' ? 'neutral' : status.tone} icon={i.status === 'connecting' ? <Spinner size={10} /> : undefined}>
                {status.label}
              </Badge>
            </div>
            <p className="mt-1 text-sm leading-relaxed text-fg-muted">{i.description}</p>
          </div>
        </div>

        {i.connected && !builtin && (
          <div className="mt-4 flex items-center gap-2.5 rounded-xl border border-border bg-sunken/50 px-3.5 py-2.5">
            <IconUserCircle size={18} className="text-fg-subtle" />
            <div className="min-w-0 flex-1">
              <div className="text-2xs uppercase tracking-wide text-fg-subtle">Connected account</div>
              <div className="truncate text-sm font-medium text-fg">{i.account_label || 'Connected'}</div>
            </div>
            <StatusDot tone={i.status === 'error' ? 'danger' : 'success'} />
          </div>
        )}

        {i.status === 'error' && i.error && (
          <Alert className="mt-4" tone="danger" icon={<IconAlertTriangle />} title="Needs attention">
            {i.error}
          </Alert>
        )}

        {test && (
          <Alert className="mt-4" tone={test.ok ? 'success' : 'danger'} icon={test.ok ? <IconCircleCheck /> : <IconAlertTriangle />} title={test.ok ? 'Connection works' : 'Connection test failed'}>
            {test.detail}
          </Alert>
        )}

        <div className="mt-4 flex flex-wrap items-center gap-2">
          {!builtin && (!i.connected || i.status === 'error') && (
            <Button variant="primary" leftIcon={<IconPlugConnected size={15} />} onClick={() => onConnect(i.id)}>
              {i.status === 'error' ? 'Reconnect' : `Connect ${i.display_name}`}
            </Button>
          )}
          {(i.connected || builtin) && !i.alternative_for && (
            <Button
              variant="secondary"
              leftIcon={<IconStethoscope size={15} />}
              loading={actions.test.isPending}
              onClick={() =>
                actions.test.mutate(i.id, {
                  onSuccess: (r) => setTest(r),
                  onError: (e) => setTest({ ok: false, detail: errorMessage(e) })
                })
              }
            >
              Test connection
            </Button>
          )}
          {i.connected && !builtin && i.status !== 'error' && (
            <Button variant="ghost" onClick={() => onConnect(i.id)}>
              Reconnect
            </Button>
          )}
          {!builtin && (i.connected || i.status === 'error') && (
            <Button variant="ghost" className="ml-auto text-danger hover:bg-danger/10 hover:text-danger" leftIcon={<IconPlugConnectedX size={15} />} onClick={() => setConfirm(true)}>
              Disconnect
            </Button>
          )}
        </div>
      </div>

      {/* capabilities */}
      {i.tools.length > 0 ? (
        <Section
          id="section-tools"
          icon={<IconSparkles />}
          title="What Sentient can do"
          description={
            <span className="flex flex-wrap gap-x-3">
              {(['read', 'write', 'send', 'exec'] as const)
                .filter((r) => counts[r])
                .map((r) => (
                  <span key={r}>
                    {formatNumber(counts[r])} {RISK_LABEL[r].label.toLowerCase()}
                  </span>
                ))}
            </span>
          }
        >
          <ul className="divide-y divide-border overflow-hidden rounded-xl border border-border bg-surface">
            {tools.map((t) => {
              const risk = RISK_LABEL[riskOf(t.risk)]
              return (
                <li key={t.name} className="flex items-start gap-3 px-3.5 py-2.5">
                  <div className="min-w-0 flex-1">
                    <div className="text-sm font-medium text-fg">{toolLabel(t.name, i.id)}</div>
                    <div className="mt-0.5 text-xs leading-relaxed text-fg-subtle">{plainDescription(t.description)}</div>
                  </div>
                  <Tooltip content={risk.hint} side="left">
                    <span className="mt-0.5">
                      <Badge size="xs" tone={risk.tone}>
                        {risk.label}
                      </Badge>
                    </span>
                  </Tooltip>
                </li>
              )
            })}
          </ul>
          {i.tools.length > 6 && (
            <button
              type="button"
              onClick={() => setShowAllTools((s) => !s)}
              className="mt-2 flex items-center gap-1 text-xs font-medium text-fg-muted hover:text-fg"
            >
              <IconChevronDown size={13} className={cn('transition-transform', showAllTools && 'rotate-180')} />
              {showAllTools ? 'Show fewer' : `Show all ${i.tools.length}`}
            </button>
          )}
        </Section>
      ) : replaces ? (
        <Section id="section-tools" icon={<IconSparkles />} title="What it does">
          <button
            type="button"
            onClick={() => onOpenOther(replaces.id)}
            className="flex w-full items-center gap-3 rounded-xl border border-border bg-surface px-3.5 py-3 text-left transition-colors hover:bg-hover"
          >
            <BrandIcon id={replaces.id} icon={replaces.icon} size={32} />
            <span className="min-w-0 flex-1 text-sm text-fg-muted">
              An optional upgrade for the built-in <span className="font-medium text-fg">{replaces.display_name}</span>. Once connected,{' '}
              {replaces.display_name.toLowerCase()} requests use {i.display_name} instead. It adds no new abilities of its own.
            </span>
            <IconArrowUpRight size={15} className="text-fg-subtle" />
          </button>
        </Section>
      ) : null}

      {/* privacy */}
      {i.privacy_filters.supported && (
        <Section id="section-privacy" icon={<IconShieldLock />} title="Privacy filters" description={`Choose what Sentient is never allowed to see in ${i.display_name}.`}>
          <PrivacyFiltersEditor integration={i} />
        </Section>
      )}

      {/* triggers */}
      {i.triggers.length > 0 && (
        <Section id="section-triggers" icon={<IconBolt />} title="Triggers" description="Run a task automatically when something happens here.">
          <ul className="space-y-2">
            {i.triggers.map((t) => (
              <li key={t.event} className="flex items-center gap-3 rounded-xl border border-border bg-surface px-3.5 py-2.5">
                <span className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-accent/10 text-accent-text">
                  <IconBolt size={16} />
                </span>
                <div className="min-w-0 flex-1">
                  <div className="text-sm font-medium text-fg">{t.label}</div>
                  <div className="text-xs text-fg-subtle">{TRIGGER_HINT[t.event] ?? `Start a task on ${t.label.toLowerCase()}.`}</div>
                </div>
                <Button
                  size="sm"
                  variant="secondary"
                  leftIcon={<IconPlus size={14} />}
                  onClick={() =>
                    navigate(
                      `/tasks?compose=1&prompt=${encodeURIComponent(`When there's a ${(t.label || t.event.replace(/_/g, ' ')).toLowerCase()} in ${i.display_name}, `)}`
                    )
                  }
                >
                  Create task
                </Button>
              </li>
            ))}
          </ul>
          {!i.connected && <p className="mt-2 text-xs text-fg-subtle">Triggers start working once {i.display_name} is connected.</p>}
        </Section>
      )}

      {/* upgrades for builtins */}
      {upgrades.length > 0 && (
        <Section
          id="section-upgrades"
          icon={<IconPlugConnected />}
          title="Optional upgrades"
          description={`${i.display_name} works without any setup. Add a key from one of these services for richer or more reliable results.`}
        >
          <ul className="space-y-2">
            {upgrades.map((u) => (
              <li key={u.id} className="flex items-center gap-3 rounded-xl border border-border bg-surface px-3.5 py-2.5">
                <BrandIcon id={u.id} icon={u.icon} size={32} />
                <button type="button" className="min-w-0 flex-1 text-left" onClick={() => onOpenOther(u.id)}>
                  <div className="text-sm font-medium text-fg hover:underline">{u.display_name}</div>
                  <div className="line-clamp-1 text-xs text-fg-subtle">{u.connected ? `In use${u.account_label ? ` · ${u.account_label}` : ''}` : u.description}</div>
                </button>
                {u.connected ? (
                  <Badge tone="success">In use</Badge>
                ) : (
                  <Button size="sm" variant="secondary" onClick={() => onConnect(u.id)}>
                    Add key
                  </Button>
                )}
              </li>
            ))}
          </ul>
        </Section>
      )}

      <ConfirmDialog
        open={confirm}
        onOpenChange={setConfirm}
        title={`Disconnect ${i.display_name}?`}
        description={`Sentient will forget the saved sign-in for ${i.display_name} and stop using it. Tasks that depend on ${i.display_name} will be paused until you reconnect.`}
        confirmLabel="Disconnect"
        onConfirm={async () => {
          try {
            await actions.disconnect.mutateAsync(i.id)
            setTest(null)
            toast.success(`${i.display_name} disconnected`, { description: 'Tasks that used it are paused.' })
          } catch (e) {
            toast.error(`Couldn't disconnect ${i.display_name}`, { description: errorMessage(e) })
          }
        }}
      />
    </div>
  )
}
