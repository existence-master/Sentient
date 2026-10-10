/**
 * Claude through your own Claude Code (experimental, issue #206, ADR 0022, docs/API.md §3).
 * Off by default. Sentient runs the claude program already on this computer, signed in by the user, and never reads
 * that login. It only answers chats; background work keeps using the other models.
 */
import { IconCheck, IconExternalLink, IconTerminal2 } from '@tabler/icons-react'
import { toast } from 'sonner'
import { Alert, Badge, Button, Card, Skeleton, StatusDot, Switch } from '@/components/ui'
import { ModelTest } from '@/features/models/ModelTest'
import { useConfigEditor } from '@/hooks/config'
import { useClaudeCodeStatus, useSetRoles } from '@/hooks/models'
import { errorMessage } from '@/lib/api'
import { getBridge } from '@/lib/bridge'
import { modelShortName, providerOf } from '@/lib/models'

export const CLAUDE_CODE_WARNING = 'Experimental. Uses your own Claude Code install and login. Anthropic may change how this is counted or allowed.'
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
              Sentient starts Claude Code with your own sign-in and never sees or keeps your Claude login. Claude Code's own tools stay off: everything it does goes through Sentient's tools and approvals.
            </p>
          </div>
          <Switch
            checked={enabled}
            onCheckedChange={(on) => patch({ models: { experimental_claude_code: on } }, { immediate: true })}
            aria-label="Use my Claude Code for chats"
          />
        </div>

        <Alert tone="warning">
          {CLAUDE_CODE_WARNING} It only answers your chats: tasks, suggestions and other background work keep using your other models, and it can't be the memory search model.
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
