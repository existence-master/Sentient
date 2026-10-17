/**
 * Claude through your own Claude Code (experimental, issue #206, ADR 0022, docs/API.md §3).
 * Off by default. Sentient runs the claude program already on this computer, signed in by the user, and never reads
 * that login. It only answers chats; background work keeps using the other models.
 */
import { IconAlertTriangle, IconCheck, IconExternalLink, IconTerminal2 } from '@tabler/icons-react'
import { toast } from 'sonner'
import { Alert, Badge, Button, Card, Skeleton, StatusDot, Switch } from '@/components/ui'
import { ModelTest } from '@/features/models/ModelTest'
import { useConfigEditor } from '@/hooks/config'
import { useClaudeCodeStatus, useHardware, useLocalModels, useSetRoles } from '@/hooks/models'
import { errorMessage } from '@/lib/api'
import { getBridge } from '@/lib/bridge'
import { otherModel, refusingRoles } from '@/lib/claudeCode'
import { modelShortName, providerOf, recommendFastLocal, recommendLocal, ROLE_META } from '@/lib/models'
import type { RoleName } from '@/lib/types'

export const CLAUDE_CODE_WARNING = 'Experimental. Uses your own Claude Code install and login. Anthropic may change how this is counted or allowed.'
/** Claude Code starts fresh for every reply and sends text in bursts (issue #246). */
export const CLAUDE_CODE_STREAMING = 'Replies take a few seconds to start, and the text arrives in larger pieces than with your other models.'
const CLAUDE_CODE_URL = 'https://claude.com/claude-code'

export function ClaudeCodeSection() {
  const { config, patch } = useConfigEditor()
  const enabled = !!config?.models.experimental_claude_code
  const status = useClaudeCodeStatus(enabled)
  const setRoles = useSetRoles()
  const primary = config?.models.roles.primary ?? ''
  const models = status.data?.models ?? []
  const testModel = providerOf(primary) === 'claude-code' ? primary : models[0]

  const chooseForChat = (model: string) =>
    setRoles.mutate(
      { primary: model },
      {
        onSuccess: () => toast.success('Chat now uses Claude through Claude Code', { description: modelShortName(model) }),
        onError: (e) => toast.error("Couldn't change the model", { description: errorMessage(e) })
      }
    )

  return (
    <section className="space-y-3">
      <div className="px-1">
        <h3 className="text-sm font-semibold text-fg">Claude through your Claude Code</h3>
        <p className="mt-0.5 text-xs text-fg-subtle">Chat with Claude using the Claude Code program already on this computer.</p>
      </div>
      <Card className="space-y-3.5 p-4">
        <div className="flex items-start gap-3">
          <div className="flex size-9 shrink-0 items-center justify-center rounded-lg border border-border bg-elevated text-accent-text">
            <IconTerminal2 size={18} />
          </div>
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2 text-sm font-medium text-fg">
              Use my Claude Code for chats
              <Badge size="xs" tone="warning">
                Experimental
              </Badge>
            </div>
            <p className="mt-0.5 text-xs leading-relaxed text-fg-subtle">
              Sentient starts Claude Code with the plan you signed in to and never sees or keeps your Claude login. API keys and other sign-in settings in your environment are left out, so it always uses that plan. Claude Code's own tools stay off: everything it does goes through Sentient's tools and approvals.
            </p>
          </div>
          <Switch
            checked={enabled}
            onCheckedChange={(on) => patch({ models: { experimental_claude_code: on } }, { immediate: true })}
            aria-label="Use my Claude Code for chats"
          />
        </div>

        <Alert tone="warning">
          {CLAUDE_CODE_WARNING} It only answers your chats: tasks, suggestions and other background work keep using your other models, and it can't be the memory search model. {CLAUDE_CODE_STREAMING}
        </Alert>

        {enabled &&
          (status.isLoading ? (
            <Skeleton className="h-9 rounded-lg" />
          ) : status.isError ? (
            <Alert tone="danger" title="Couldn't check for Claude Code">
              {errorMessage(status.error)}
            </Alert>
          ) : !status.data?.installed ? (
            <div className="flex flex-wrap items-center gap-3">
              <span className="flex min-w-0 flex-1 items-center gap-2 text-xs text-fg-muted">
                <StatusDot tone="danger" />
                {status.data?.detail}
              </span>
              <Button size="sm" variant="secondary" leftIcon={<IconExternalLink size={13} />} onClick={() => void getBridge().openExternal(CLAUDE_CODE_URL)}>
                Get Claude Code
              </Button>
            </div>
          ) : (
            <div className="space-y-3">
              <div className="flex items-center gap-2 text-xs text-fg-muted">
                <StatusDot tone={status.data.version ? 'success' : 'warning'} />
                <span className="min-w-0 truncate">
                  {status.data.version ? `Found Claude Code ${status.data.version}. ` : ''}
                  {status.data.detail}
                </span>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                {models.map((m) =>
                  m === primary ? (
                    <Badge key={m} tone="success">
                      <IconCheck size={12} /> Chat uses {modelShortName(m)}
                    </Badge>
                  ) : (
                    <Button key={m} size="sm" variant="secondary" loading={setRoles.isPending && setRoles.variables?.primary === m} onClick={() => chooseForChat(m)}>
                      Use {modelShortName(m)} for chat
                    </Button>
                  )
                )}
                <div className="flex-1" />
                <ModelTest model={testModel} role="primary" className="shrink-0" />
              </div>
            </div>
          ))}
      </Card>
    </section>
  )
}

/**
 * Settings > Models (#442): roles that use Claude Code but refuse it, their own model or the main model they fall
 * back to. Claude Code only answers chats, so memory, summaries and tasks on those roles fail until they use another
 * model. Which roles accept it comes from the engine.
 */
export function ClaudeCodeRoleWarning() {
  const { config } = useConfigEditor()
  const status = useClaudeCodeStatus(!!config?.models.experimental_claude_code)
  const local = useLocalModels()
  const hardware = useHardware()
  const setRoles = useSetRoles()
  if (!config) return null
  const roles = config.models.roles as unknown as Record<string, string | null>
  const refusing = refusingRoles(roles, status.data?.roles)
  if (!refusing.length) return null
  const chat = refusing.filter((r) => r.role !== 'embedding')
  const rec = hardware.data?.recommendation
  const installed = (local.data?.ollama.models ?? []).map((m) => m.name)
  const recInstalled = rec && !rec.cloud_first && installed.some((n) => n === rec.name || n === `${rec.name}:latest`)
  const fix = otherModel([
    roles.fast,
    roles.executor,
    roles.planner,
    ...(config.models.fallbacks.fast ?? []),
    ...(config.models.fallbacks.primary ?? []),
    // tasks need a model that handles tools: the one sized for this computer first
    recInstalled ? rec.model : null,
    local.data ? recommendLocal(local.data) : null,
    local.data ? recommendFastLocal(local.data) : null
  ])
  const label = (r: { role: string }) => ROLE_META[r.role as RoleName].label
  const own = refusing.filter((r) => !r.inherited)
  const inherited = refusing.filter((r) => r.inherited)
  const parts = [
    own.length ? `${listText(own.map(label))} ${own.length === 1 ? 'is' : 'are'} set to it` : '',
    inherited.length ? `${listText(inherited.map(label))} ${inherited.length === 1 ? 'gets' : 'get'} it from your main model` : ''
  ].filter(Boolean)
  const useFix = () =>
    setRoles.mutate(Object.fromEntries(chat.map((r) => [r.role, fix])), {
      onSuccess: () => toast.success(`${listText(chat.map((r) => ROLE_META[r.role as RoleName].label))} now ${chat.length === 1 ? 'uses' : 'use'} ${modelShortName(fix)}`),
      onError: (e) => toast.error("Couldn't change the model", { description: errorMessage(e) })
    })

  return (
    <Alert tone="warning" icon={<IconAlertTriangle />} title="Some work can't run on Claude Code">
      <p>
        Claude Code only answers your chats, but {parts.join(', and ')}. Until {refusing.length === 1 ? 'it uses' : 'they use'} another model,{' '}
        {consequences(refusing.map((r) => r.role))} won&apos;t work. You can pick one under Roles.
      </p>
      {fix && chat.length > 0 && (
        <Button size="sm" variant="secondary" className="mt-2.5" loading={setRoles.isPending} onClick={useFix}>
          Use {modelShortName(fix)} for {chat.length === 1 ? ROLE_META[chat[0].role as RoleName].label.toLowerCase() : 'these'}
        </Button>
      )}
    </Alert>
  )
}

function consequences(roles: string[]): string {
  const out: string[] = []
  if (roles.includes('fast')) out.push('new memories', 'summaries')
  if (roles.includes('planner') || roles.includes('executor')) out.push('tasks')
  if (roles.includes('embedding')) out.push('memory search')
  return listText(out)
}

function listText(items: string[]): string {
  return items.length < 2 ? (items[0] ?? '') : `${items.slice(0, -1).join(', ')} and ${items[items.length - 1]}`
}
