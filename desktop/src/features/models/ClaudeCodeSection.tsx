/**
 * Claude through your own Claude Code (experimental, issue #206, ADR 0022, docs/API.md §3), for chats or, after a
 * separate warning, for everything (#444, ADR 0032). Off by default. Sentient runs the claude program already on this
 * computer, signed in by the user, and never reads that login.
 */
import { IconAlertTriangle, IconCheck, IconExternalLink, IconTerminal2 } from '@tabler/icons-react'
import { useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { toast } from 'sonner'
import { Alert, Badge, Button, Skeleton, StatusDot, Switch } from '@/components/ui'
import { openExternal } from '@/features/integrations/InstructionsGuide'
import { ANTHROPIC_PLAN_ARTICLE, CLAUDE_CODE_EVERYWHERE_WARNING, ClaudeCodeEverywhereDialog } from '@/features/models/ClaudeCodeEverywhere'
import { ModelTest } from '@/features/models/ModelTest'
import { SchemaForm } from '@/features/settings/SchemaForm'
import { useConfigEditor } from '@/hooks/config'
import { useClaudeCodeStatus, useHardware, useLocalModels, useSetRoles } from '@/hooks/models'
import { qk } from '@/hooks/queryKeys'
import { api, errorMessage } from '@/lib/api'
import { claudeCodeState, fastStandIns, isClaudeCode, otherModel, refusingRoles } from '@/lib/claudeCode'
import { modelShortName, recommendFastLocal, recommendLocal, ROLE_META } from '@/lib/models'
import type { RoleName } from '@/lib/types'

export const CLAUDE_CODE_WARNING = 'Experimental. Uses your own Claude Code install and login. Anthropic may change how this is counted or allowed.'
/** Claude Code starts fresh for every reply and sends text in bursts (issue #246). */
export const CLAUDE_CODE_STREAMING = 'Replies take a few seconds to start, and the text arrives in larger pieces than with your other models.'
const CLAUDE_CODE_URL = 'https://claude.com/claude-code'
export const CLAUDE_CODE_FAST_MODEL = 'claude-code/haiku'

/**
 * Claude Code for chats (`way="chats"`) or for everything (`way="everything"`). `onUseForChat` puts a Claude Code
 * model on chat; `onUseForEverything` puts Claude Code on every job (main and fast models; planner and executor follow
 * the main model).
 */
export function ClaudeCodePanel({
  way,
  primary,
  fast,
  others = {},
  onUseForChat,
  onUseForEverything
}: {
  way: 'chats' | 'everything'
  /** The main and fast models in use (saved, or picked so far in setup). */
  primary: string
  fast: string
  /** The other jobs that can use Claude Code (planner, executor). Empty follows the main model. */
  others?: Record<string, string | null | undefined>
  onUseForChat: (model: string) => void
  onUseForEverything: (primary: string, fast: string) => void
}) {
  const qc = useQueryClient()
  const { config } = useConfigEditor()
  const enabled = !!config?.models.experimental_claude_code
  const everywhere = enabled && !!config?.models.claude_code_everywhere
  const status = useClaudeCodeStatus(enabled)
  const [asking, setAsking] = useState(false)
  const models = status.data?.models ?? []
  const fastModel = status.data?.fast_model ?? CLAUDE_CODE_FAST_MODEL
  const testModel = isClaudeCode(primary) ? primary : models[0]
  const state = claudeCodeState(enabled, everywhere)
  const maxConcurrent = config?.models.claude_code_max_concurrent ?? 2

  // Saved at once, and only a save that worked says so. Then every Claude Code status is asked again: the roles that
  // may use it change with the switches.
  const save = async (models: Record<string, boolean>, done?: { title: string; description: string }) => {
    try {
      const res = await api.config.patch({ models })
      qc.setQueryData(qk.config, res.config)
      void qc.invalidateQueries({ queryKey: qk.bootstrap })
      if (done) toast.success(done.title, { description: done.description })
    } catch (e) {
      toast.error("Couldn't change Claude Code", { description: errorMessage(e) })
    } finally {
      void qc.invalidateQueries({ queryKey: qk.claudeCodeAll })
    }
  }
  const setEnabled = (on: boolean) => void save(on ? { experimental_claude_code: true } : { experimental_claude_code: false, claude_code_everywhere: false })
  const turnOnEverything = () =>
    void save({ experimental_claude_code: true, claude_code_everywhere: true }, { title: 'Claude Code can now do everything', description: 'Pick it for each job below, or under Roles.' })
  const turnOffEverything = () =>
    void save({ claude_code_everywhere: false }, { title: 'Claude Code answers only your chats again', description: 'Jobs that used it move to another model.' })
  // every job: main and fast on Claude Code, and the other jobs on it too or following the main model
  const allOnClaudeCode = isClaudeCode(primary) && isClaudeCode(fast) && Object.values(others).every((m) => !m || isClaudeCode(m))

  return (
    <div className="space-y-3.5">
      <div className="flex items-start gap-3">
        <div className="flex size-9 shrink-0 items-center justify-center rounded-lg border border-border bg-elevated text-accent-text">
          <IconTerminal2 size={18} />
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2 text-sm font-medium text-fg">
            {way === 'chats' ? 'Use my Claude Code for chats' : 'Use my Claude Code for everything'}
            <Badge size="xs" tone="warning">
              Experimental
            </Badge>
            <Badge size="xs" tone={state.tone}>
              Now: {state.label}
            </Badge>
          </div>
          <p className="mt-0.5 text-xs leading-relaxed text-fg-subtle">
            Sentient starts Claude Code with the plan you signed in to and never sees or keeps your Claude login. API keys and other sign-in settings in your environment are left out, so it always uses that plan. Claude Code's own tools stay off: everything it does goes through Sentient's tools and approvals.
          </p>
        </div>
        {way === 'chats' ? (
          <Switch checked={enabled} onCheckedChange={setEnabled} aria-label="Use my Claude Code for chats" />
        ) : (
          <Switch checked={everywhere} onCheckedChange={(on) => (on ? setAsking(true) : turnOffEverything())} aria-label="Use my Claude Code for everything" />
        )}
      </div>

      {way === 'chats' ? (
        <Alert tone="warning">
          {CLAUDE_CODE_WARNING} {everywhere ? 'It also does everything else now, because you turned that on.' : 'It only answers your chats: tasks, suggestions and other background work keep using your other models, and it can\'t be the memory search model.'} {CLAUDE_CODE_STREAMING}
        </Alert>
      ) : (
        <Alert tone="danger" title={everywhere ? 'On: Claude Code does everything' : 'Off until you accept the risk'}>
          <span className="block">{CLAUDE_CODE_EVERYWHERE_WARNING}</span>
          <button type="button" onClick={() => openExternal(ANTHROPIC_PLAN_ARTICLE)} className="mt-1.5 inline-flex items-center gap-1 font-medium text-accent-text hover:underline">
            Anthropic&apos;s article <IconExternalLink size={12} />
          </button>
        </Alert>
      )}

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
            <Button size="sm" variant="secondary" leftIcon={<IconExternalLink size={13} />} onClick={() => openExternal(CLAUDE_CODE_URL)}>
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
              {way === 'chats' &&
                models.map((m) =>
                  m === primary ? (
                    <Badge key={m} tone="success">
                      <IconCheck size={12} /> Chat uses {modelShortName(m)}
                    </Badge>
                  ) : (
                    <Button key={m} size="sm" variant="secondary" onClick={() => onUseForChat(m)}>
                      Use {modelShortName(m)} for chat
                    </Button>
                  )
                )}
              {way === 'everything' &&
                everywhere &&
                models[0] &&
                (allOnClaudeCode ? (
                  <Badge tone="success">
                    <IconCheck size={12} /> Every job uses Claude Code
                  </Badge>
                ) : (
                  <Button size="sm" variant="secondary" onClick={() => onUseForEverything(models[0], fastModel)}>
                    Use Claude Code for every job
                  </Button>
                ))}
              <div className="flex-1" />
              <ModelTest model={testModel} role="primary" className="shrink-0" />
            </div>
          </div>
        ))}

      {way === 'everything' && everywhere && (
        <SchemaForm section="models" title="How many at once" include={['models.claude_code_max_concurrent']} />
      )}

      <ClaudeCodeEverywhereDialog open={asking} onOpenChange={setAsking} onConfirm={turnOnEverything} maxConcurrent={maxConcurrent} />
    </div>
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
  // the engine does the fast role's quick jobs on another model when it can (#474): say which, not that they fail
  const [standIn] = fastStandIns(roles, config.models.fallbacks, status.data?.roles)
  const stuck = standIn ? refusing.filter((r) => r.role !== 'fast') : refusing
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
  const own = stuck.filter((r) => !r.inherited)
  const inherited = stuck.filter((r) => r.inherited)
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
      {stuck.length > 0 && (
        <p>
          Claude Code only answers your chats, but {parts.join(', and ')}. Until {stuck.length === 1 ? 'it uses' : 'they use'} another model,{' '}
          {consequences(stuck.map((r) => r.role))} won&apos;t work. You can pick one under Roles.
        </p>
      )}
      {standIn && (
        <p>
          Claude Code can&apos;t do quick background jobs, so {standIn} does them.
          {stuck.length === 0 && ' You can pick another model for them under Roles.'}
        </p>
      )}
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
