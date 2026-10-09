/**
 * "Rules for apps and tools": lasting Allow / Ask / Never choices per app or per tool
 * (`tools.approvals.rules`, docs/API.md section 2, ADR 0016). "Default" means no rule.
 */
import { IconChevronDown, IconShoppingCart, IconTrash } from '@tabler/icons-react'
import { useMemo, useState, type ReactNode } from 'react'
import { Alert, Badge, Button, FormSection, SegmentedControl, Skeleton } from '@/components/ui'
import { useConfigEditor } from '@/hooks/config'
import { useTools } from '@/hooks/core'
import { errorMessage, isNotImplemented } from '@/lib/api'
import type { ApprovalRule, ToolInfo, ToolPlugin } from '@/lib/types'
import { cn, humanize } from '@/lib/utils'
import { ToolTile } from '../../tasks/tools'

type Choice = ApprovalRule | 'default'

const CHOICES: Array<{ value: Choice; label: string }> = [
  { value: 'default', label: 'Default' },
  { value: 'allow', label: 'Allow' },
  { value: 'ask', label: 'Ask' },
  { value: 'never', label: 'Never' }
]

const RULE_WORDS: Record<ApprovalRule, string> = {
  allow: 'goes ahead without asking',
  ask: 'always asks first',
  never: "can't be used"
}

function toolLabel(plugin: ToolPlugin, tool: ToolInfo): string {
  const prefix = `${plugin.id}_`
  return humanize(tool.name.startsWith(prefix) ? tool.name.slice(prefix.length) : tool.name)
}


export function ApprovalRulesSection({ query }: { query: string }) {
  const tools = useTools()
  const { config, patch } = useConfigEditor()
  // app id -> open; apps with rules on single tools start open
  const [openState, setOpenState] = useState<Record<string, boolean>>({})

  const rules = useMemo(() => {
    const out: Record<string, ApprovalRule> = {}
    for (const [k, v] of Object.entries(config?.tools.approvals.rules ?? {})) if (v) out[k] = v
    return out
  }, [config])

  const q = query.trim().toLowerCase()
  const isOpen = (p: ToolPlugin) => !!q || (openState[p.id] ?? p.tools.some((t) => rules[t.name]))
  const toggle = (p: ToolPlugin) => setOpenState((prev) => ({ ...prev, [p.id]: !isOpen(p) }))

  const setRule = (key: string, choice: Choice, appId?: string) => {
    if (appId) setOpenState((prev) => ({ ...prev, [appId]: true })) // keep the list open while editing its tools
    patch({ tools: { approvals: { rules: { [key]: choice === 'default' ? null : choice } } } }, { immediate: true })
  }

  const plugins = useMemo(() => {
    const list = (tools.data ?? []).filter((p) => p.tools.length)
    const matches = (p: ToolPlugin) =>
      !q || `${p.display_name} ${p.id} ${p.description}`.toLowerCase().includes(q) || p.tools.some((t) => `${t.name} ${t.description}`.toLowerCase().includes(q))
    const sorted = [...list.filter(matches)].sort((a, b) => a.display_name.localeCompare(b.display_name))
    return {
      // apps you connect (sign in or add a key) versus abilities that come with Sentient
      apps: sorted.filter((p) => p.auth !== 'none'),
      builtIn: sorted.filter((p) => p.auth === 'none')
    }
  }, [tools.data, q])

  // rules for apps or tools that are not listed right now (an app that was disconnected, say)
  const orphans = useMemo(() => {
    const known = new Set<string>()
    for (const p of tools.data ?? []) {
      known.add(p.id)
      for (const t of p.tools) known.add(t.name)
    }
    return Object.keys(rules).filter((k) => !known.has(k)).sort()
  }, [tools.data, rules])

  const description = 'Choose what Sentient may do without checking with you. Default follows the setting above.'

  if (tools.isLoading || !config) {
    return (
      <FormSection title="Rules for apps and tools" description={description}>
        <div className="space-y-2 p-4">
          {[0, 1, 2].map((i) => (
            <Skeleton key={i} className="h-9" />
          ))}
        </div>
      </FormSection>
    )
  }

  if (tools.isError) {
    return (
      <FormSection title="Rules for apps and tools" description={description}>
        <div className="p-4">
          <Alert tone={isNotImplemented(tools.error) ? 'info' : 'danger'} title="Couldn't load your apps and tools">
            {isNotImplemented(tools.error) ? 'This version of the engine does not list its tools yet.' : errorMessage(tools.error)}
          </Alert>
        </div>
      </FormSection>
    )
  }

  const nothing = !plugins.apps.length && !plugins.builtIn.length && !orphans.length

  return (
    <div className="space-y-3">
      <FormSection title="Rules for apps and tools" description={description}>
        <div className="flex items-start gap-2.5 px-4 py-3 text-xs text-fg-muted">
          <IconShoppingCart size={15} className="mt-px shrink-0 text-fg-subtle" />
          <span>Purchases always ask, whatever you choose here. Never hides that app or tool from Sentient completely.</span>
        </div>
        {nothing && <div className="px-4 py-6 text-center text-sm text-fg-subtle">{q ? 'No apps or tools match your search.' : 'No apps or tools to set rules for yet.'}</div>}
        {plugins.apps.length > 0 && <GroupHeading>Apps and services</GroupHeading>}
        {plugins.apps.map((p) => (
          <PluginRows key={p.id} plugin={p} rules={rules} open={isOpen(p)} onToggle={() => toggle(p)} onChange={setRule} />
        ))}
        {plugins.builtIn.length > 0 && <GroupHeading>Built in</GroupHeading>}
        {plugins.builtIn.map((p) => (
          <PluginRows key={p.id} plugin={p} rules={rules} open={isOpen(p)} onToggle={() => toggle(p)} onChange={setRule} />
        ))}
        {orphans.length > 0 && !q && (
          <>
            <GroupHeading>Other rules</GroupHeading>
            {orphans.map((key) => (
              <div key={key} className="flex items-center gap-3 px-4 py-2.5">
                <div className="min-w-0 flex-1">
                  <div className="truncate text-sm text-fg">{humanize(key)}</div>
                  <div className="text-xs text-fg-subtle">Not connected right now. This {RULE_WORDS[rules[key]]}.</div>
                </div>
                <Badge size="xs">{CHOICES.find((c) => c.value === rules[key])?.label}</Badge>
                <Button size="sm" variant="ghost" leftIcon={<IconTrash size={13} />} onClick={() => setRule(key, 'default')}>
                  Remove
                </Button>
              </div>
            ))}
          </>
        )}
      </FormSection>
    </div>
  )
}

function GroupHeading({ children }: { children: ReactNode }) {
  return <div className="bg-sunken/40 px-4 py-1.5 text-2xs font-semibold uppercase tracking-wide text-fg-subtle">{children}</div>
}

function PluginRows({
  plugin,
  rules,
  open,
  onToggle,
  onChange
}: {
  plugin: ToolPlugin
  rules: Record<string, ApprovalRule>
  open: boolean
  onToggle: () => void
  onChange: (key: string, choice: Choice, appId?: string) => void
}) {
  const appRule = rules[plugin.id]
  const ownRules = plugin.tools.filter((t) => rules[t.name]).length
  const count = plugin.tools.length

  return (
    <div>
      <div className="flex flex-col gap-2.5 px-4 py-3 sm:flex-row sm:items-center sm:gap-4">
        <button
          type="button"
          onClick={onToggle}
          aria-expanded={open}
          className="flex min-w-0 flex-1 items-center gap-3 text-left"
        >
          <ToolTile name={plugin.id} />
          <span className="min-w-0 flex-1">
            <span className="flex items-center gap-2 text-sm font-medium text-fg">
              <span className="truncate">{plugin.display_name}</span>
              {ownRules > 0 && (
                <Badge size="xs" tone="accent">
                  {ownRules} tool rule{ownRules === 1 ? '' : 's'}
                </Badge>
              )}
            </span>
            <span className="flex items-center gap-1 text-xs text-fg-subtle">
              {appRule ? `Every tool ${RULE_WORDS[appRule]}` : `${count} tool${count === 1 ? '' : 's'}`}
              <IconChevronDown size={13} className={cn('transition-transform', open && 'rotate-180')} />
            </span>
          </span>
        </button>
        <SegmentedControl
          size="sm"
          aria-label={`Rule for ${plugin.display_name}`}
          value={appRule ?? 'default'}
          onChange={(v) => onChange(plugin.id, v)}
          options={CHOICES}
        />
      </div>
      {open && (
        <div className="border-t border-border bg-sunken/30">
          {plugin.tools.map((t) => {
            const own = rules[t.name]
            const effective = own ?? appRule
            return (
              <div key={t.name} className="flex flex-col gap-2 py-2.5 pl-14 pr-4 sm:flex-row sm:items-center sm:gap-4">
                <div className="min-w-0 flex-1">
                  <div className="truncate text-sm text-fg">{toolLabel(plugin, t)}</div>
                  {/* tool descriptions are written for the model, so rows show only the plain name */}
                  {!own && appRule && (
                    <div className="truncate text-xs text-fg-subtle">{`Follows ${plugin.display_name}: ${RULE_WORDS[appRule]}.`}</div>
                  )}
                </div>
                <SegmentedControl
                  size="sm"
                  aria-label={`Rule for ${toolLabel(plugin, t)}`}
                  value={own ?? 'default'}
                  onChange={(v) => onChange(t.name, v, plugin.id)}
                  options={CHOICES}
                  className={cn(!own && effective && 'opacity-80')}
                />
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
