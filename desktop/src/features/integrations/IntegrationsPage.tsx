import { IconLock, IconPlugConnected, IconSearch, IconServer2, IconWebhook, IconX } from '@tabler/icons-react'
import { LiveUpdatesRow } from '@/features/automations/LiveUpdates'
import { WebhooksSection } from '@/features/automations/Webhooks'
import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { useSearchParams } from 'react-router'
import { Button, EmptyState, Input, Skeleton } from '@/components/ui'
import { ConnectDialog } from '@/features/integrations/ConnectDialog'
import { IntegrationCard } from '@/features/integrations/IntegrationCard'
import { IntegrationDrawer } from '@/features/integrations/IntegrationDrawer'
import { McpServersSection } from '@/features/integrations/McpServers'
import { CATEGORIES, sortIntegrations, type CategoryFilter } from '@/features/integrations/meta'
import { useIntegrations } from '@/hooks/integrations'
import { useMcpServers } from '@/hooks/integrations'
import { errorMessage } from '@/lib/api'
import type { Integration } from '@/lib/types'
import { cn } from '@/lib/utils'

/**
 * Integrations: connect apps, manage privacy filters and custom MCP servers.
 *
 * Deep links (also used by screenshots): `?connect=<id>`, `?open=<id>[&focus=privacy|tools|triggers|upgrades]`,
 * `?section=mcp|webhooks`, `?q=<search>`, `?category=<id>`.
 */
export function IntegrationsPage() {
  const [params, setParams] = useSearchParams()
  const { data, isLoading, isError, error, refetch } = useIntegrations()
  const mcp = useMcpServers()
  const [query, setQuery] = useState(params.get('q') ?? '')
  const [category, setCategory] = useState<CategoryFilter>((params.get('category') as CategoryFilter) || 'all')
  const scroller = useRef<HTMLDivElement>(null)

  const openId = params.get('open')
  const connectId = params.get('connect')
  const focus = params.get('focus')

  const setParam = (key: string, value: string | null, extra?: Record<string, string | null>) => {
    setParams(
      (prev) => {
        const next = new URLSearchParams(prev)
        for (const [k, v] of Object.entries({ [key]: value, ...extra })) {
          if (v === null) next.delete(k)
          else next.set(k, v)
        }
        return next
      },
      { replace: true }
    )
  }

  const all = useMemo(() => sortIntegrations(data ?? []), [data])
  const byId = (id: string | null) => (id ? (all.find((i) => i.id === id) ?? null) : null)

  const q = query.trim().toLowerCase()
  const matches = (i: Integration) =>
    (category === 'all' || i.category === category) &&
    (!q || `${i.display_name} ${i.description} ${i.id}`.toLowerCase().includes(q))

  const visible = all.filter(matches)
  const connected = visible.filter((i) => i.auth_type !== 'builtin' && (i.connected || i.status === 'error' || i.status === 'connecting'))
  const builtins = visible.filter((i) => i.auth_type === 'builtin')
  const available = visible.filter(
    (i) => i.auth_type !== 'builtin' && !connected.includes(i) && (!i.alternative_for || !!q || category !== 'all')
  )

  const counts = useMemo(() => {
    const c: Record<string, number> = { all: 0 }
    for (const i of all) {
      if (i.alternative_for) continue
      c.all++
      c[i.category] = (c[i.category] ?? 0) + 1
    }
    return c
  }, [all])
  const connectedCount = all.filter((i) => i.auth_type !== 'builtin' && i.connected).length
  const builtinCount = all.filter((i) => i.auth_type === 'builtin').length
  const showMcp = !q && (category === 'all' || category === 'development')

  const showWebhooks = !q && category === 'all'

  useEffect(() => {
    const section = params.get('section')
    const target = section === 'mcp' ? '#mcp-servers' : section === 'webhooks' ? '#webhooks' : null
    if (!target || isLoading || mcp.isLoading) return
    const t = window.setTimeout(() => scroller.current?.querySelector(target)?.scrollIntoView({ block: 'start' }), 120)
    return () => window.clearTimeout(t)
  }, [params, isLoading, mcp.isLoading])

  const upgradesFor = (id: string) => all.filter((x) => x.alternative_for === id).length
  const card = (i: Integration) => (
    <IntegrationCard
      key={i.id}
      integration={i}
      upgrades={upgradesFor(i.id)}
      onOpen={() => setParam('open', i.id, { focus: null })}
      onConnect={() => setParam('connect', i.id)}
    />
  )

  return (
    <div ref={scroller} className="h-full overflow-y-auto">
      <div className="mx-auto max-w-[1180px] px-8 pb-16 pt-7">
        {/* hero */}
        <header className="relative overflow-hidden rounded-2xl border border-border bg-elevated/50 px-7 py-6">
          <div
            aria-hidden
            className="pointer-events-none absolute -right-20 -top-28 size-80 rounded-full opacity-[0.13] blur-3xl"
            style={{ background: 'radial-gradient(circle, var(--accent), transparent 70%)' }}
          />
          <div className="relative flex flex-wrap items-end gap-x-8 gap-y-4">
            <div className="min-w-0 flex-1 basis-80">
              <div className="flex items-center gap-2 text-xs font-medium text-accent-text">
                <IconPlugConnected size={15} /> Integrations
              </div>
              <h1 className="mt-1.5 text-2xl font-semibold tracking-tight text-fg">Bring your apps to Sentient</h1>
              <p className="mt-1.5 max-w-xl text-sm leading-relaxed text-fg-muted">
                Connect the apps you already use so Sentient can catch you up, suggest what to do next and act for you. Everything runs on this computer.
              </p>
              <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-fg-subtle">
                {data && (
                  <>
                    <span>
                      <span className="font-semibold text-fg">{connectedCount}</span> connected
                    </span>
                    <span>
                      <span className="font-semibold text-fg">{builtinCount}</span> ready without setup
                    </span>
                  </>
                )}
                <span className="flex items-center gap-1">
                  <IconLock size={12} /> Keys stay in your system keychain
                </span>
                <button
                  type="button"
                  onClick={() => {
                    setQuery('')
                    setCategory('all')
                    setParam('section', 'webhooks')
                  }}
                  className="flex items-center gap-1 text-fg-subtle underline-offset-4 hover:text-fg hover:underline"
                >
                  <IconWebhook size={12} /> Webhooks
                </button>
              </div>
            </div>
            <Input
              size="lg"
              wrapperClassName="w-full max-w-[320px]"
              leftIcon={<IconSearch />}
              placeholder="Search apps"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Escape' && setQuery('')}
              rightSlot={
                query ? (
                  <button type="button" aria-label="Clear search" onClick={() => setQuery('')} className="flex size-6 items-center justify-center rounded text-fg-subtle hover:text-fg">
                    <IconX size={14} />
                  </button>
                ) : undefined
              }
            />
          </div>
        </header>

        {/* category tabs */}
        <nav className="mt-5 flex flex-wrap items-center gap-1.5" aria-label="Categories">
          {[{ id: 'all', label: 'All' }, ...CATEGORIES].map((c) => {
            const active = category === c.id
            return (
              <button
                key={c.id}
                type="button"
                onClick={() => setCategory(c.id as CategoryFilter)}
                className={cn(
                  'flex h-8 items-center gap-1.5 rounded-full border px-3 text-sm transition-colors',
                  active ? 'border-accent/40 bg-accent/12 font-medium text-fg' : 'border-border text-fg-muted hover:border-border-strong hover:text-fg'
                )}
              >
                {c.label}
                {data && <span className={cn('text-xs tabular-nums', active ? 'text-accent-text' : 'text-fg-subtle')}>{counts[c.id] ?? 0}</span>}
              </button>
            )
          })}
        </nav>

        {isLoading ? (
          <div className="mt-8 grid gap-3 [grid-template-columns:repeat(auto-fill,minmax(250px,1fr))]">
            {Array.from({ length: 8 }, (_, i) => (
              <Skeleton key={i} className="h-[150px] rounded-xl" />
            ))}
          </div>
        ) : isError ? (
          <EmptyState
            icon={<IconPlugConnected />}
            title="Couldn't load integrations"
            description={errorMessage(error)}
            action={<Button onClick={() => void refetch()}>Try again</Button>}
          />
        ) : (
          <div className="mt-7 space-y-9">
            {!q && <LiveUpdatesRow />}
            {connected.length > 0 && (
              <Group title="Connected" description="Apps Sentient can use right now.">
                {connected.map(card)}
              </Group>
            )}
            {available.length > 0 && (
              <Group title={q || category !== 'all' ? 'Apps' : 'Available to connect'} description={q || category !== 'all' ? undefined : 'Connect in a minute. Each one comes with a step-by-step guide.'}>
                {available.map(card)}
              </Group>
            )}
            {builtins.length > 0 && (
              <Group title="Works without setup" description="Built into Sentient. Nothing to connect, no keys needed.">
                {builtins.map(card)}
              </Group>
            )}
            {!connected.length && !available.length && !builtins.length && (
              <EmptyState
                icon={<IconSearch />}
                title={q ? `Nothing matches “${query.trim()}”` : 'No apps in this category yet'}
                description="If the app has an MCP server, you can add it below as a custom server."
                action={
                  <Button
                    variant="secondary"
                    leftIcon={<IconServer2 size={15} />}
                    onClick={() => {
                      setQuery('')
                      setCategory('all')
                      setParam('section', 'mcp')
                    }}
                  >
                    Add an MCP server
                  </Button>
                }
              />
            )}
            {showWebhooks && (
              <div className="border-t border-border pt-8">
                <WebhooksSection id="webhooks" />
              </div>
            )}
            {showMcp && (
              <div className="border-t border-border pt-8">
                <McpServersSection id="mcp-servers" />
              </div>
            )}
          </div>
        )}
      </div>

      <IntegrationDrawer
        integration={byId(openId)}
        all={all}
        open={!!openId}
        focus={focus}
        onOpenChange={(o) => !o && setParam('open', null, { focus: null })}
        onConnect={(id) => setParam('connect', id)}
        onOpenOther={(id) => setParam('open', id, { focus: null })}
      />
      <ConnectDialog integration={byId(connectId)} open={!!connectId} onOpenChange={(o) => !o && setParam('connect', null)} />
    </div>
  )
}

function Group({ title, description, children }: { title: string; description?: string; children: ReactNode }) {
  return (
    <section>
      <div className="mb-3">
        <h2 className="text-md font-semibold text-fg">{title}</h2>
        {description && <p className="mt-0.5 text-sm text-fg-subtle">{description}</p>}
      </div>
      <div className="grid gap-3 [grid-template-columns:repeat(auto-fill,minmax(250px,1fr))]">{children}</div>
    </section>
  )
}
