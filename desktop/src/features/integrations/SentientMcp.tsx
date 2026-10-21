import { IconCheck, IconCopy, IconShieldCheck } from '@tabler/icons-react'
import { useState } from 'react'
import { toast } from 'sonner'
import { Alert, Button, FormRow, FormSection, IconButton, Skeleton, Switch } from '@/components/ui'
import { useConfigEditor } from '@/hooks/config'
import { useConfig } from '@/hooks/core'
import { errorMessage } from '@/lib/api'
import { copyText } from '@/lib/utils'

const CLAUDE_CODE_COMMAND = 'claude mcp add sentient -- sentient mcp'
const HTTP_START_COMMAND = 'sentient mcp --transport http'
const HTTP_SNIPPET = `{
  "mcpServers": {
    "sentient": {
      "url": "http://127.0.0.1:8765/mcp",
      "headers": {
        "Authorization": "Bearer <engine token>"
      }
    }
  }
}`

function CopyButton({ text, label }: { text: string; label: string }) {
  const [copied, setCopied] = useState(false)
  return (
    <IconButton
      size="xs"
      label={copied ? 'Copied' : label}
      icon={copied ? <IconCheck size={13} className="text-success" /> : <IconCopy size={13} />}
      onClick={() =>
        void copyText(text).then((ok) => {
          if (!ok) {
            toast.error("Couldn't copy to the clipboard")
            return
          }
          setCopied(true)
          window.setTimeout(() => setCopied(false), 1400)
        })
      }
    />
  )
}

function CodeBlock({ label, code, copyLabel }: { label: string; code: string; copyLabel: string }) {
  return (
    <div>
      <div className="mb-1.5 flex items-center justify-between gap-3">
        <span className="text-sm font-medium text-fg">{label}</span>
        <CopyButton text={code} label={copyLabel} />
      </div>
      <pre className="selectable overflow-x-auto rounded-xl border border-border bg-sunken px-3.5 py-3 font-mono text-[12px] leading-relaxed text-fg">{code}</pre>
    </div>
  )
}

/**
 * Let other apps use Sentient over MCP (issue #443).
 *
 * Off by default. When on, MCP apps on this computer can use a small set of
 * Sentient tools for memory, tasks and notifications. All of it still goes
 * through Sentient itself, so plans and risky steps ask the user as usual.
 * Settings live in `integrations.mcp_server_enabled` and
 * `integrations.mcp_server_memories_skip_review` (PATCH /api/config).
 */
export function SentientMcpSection({ id }: { id?: string }) {
  const { config, isLoading, error, setValue } = useConfigEditor()
  const configQuery = useConfig()

  const enabled = !!config?.integrations.mcp_server_enabled
  const skipReview = !!config?.integrations.mcp_server_memories_skip_review

  return (
    <section id={id} className="scroll-mt-6">
      <div className="mb-3">
        <h2 className="text-md font-semibold text-fg">Let other apps use Sentient</h2>
        <p className="mt-0.5 max-w-2xl text-sm text-fg-subtle">
          Apps like Claude Code and Cursor that speak MCP can use Sentient on this computer. They can search and save memories, make and check tasks, and send you a
          notification. This is off by default.
        </p>
      </div>

      {isLoading ? (
        <Skeleton className="h-40 rounded-xl" />
      ) : error || !config ? (
        <Alert
          tone="danger"
          title="Couldn't load settings"
          action={
            <Button size="sm" onClick={() => void configQuery.refetch()}>
              Try again
            </Button>
          }
        >
          {error ? errorMessage(error) : 'Sentient did not return its settings. Try again.'}
        </Alert>
      ) : (
        <div className="space-y-5">
          <FormSection>
            <FormRow
              label="Let other apps use Sentient"
              description="When this is on, MCP apps on this computer can reach Sentient. It only works on this computer and still needs the engine token."
              htmlFor="sentient-mcp-enabled"
            >
              <Switch
                id="sentient-mcp-enabled"
                checked={enabled}
                onCheckedChange={(v) => setValue('integrations.mcp_server_enabled', v, { immediate: true })}
              />
            </FormRow>
            <FormRow
              label="Let MCP apps save memories without review"
              description="By default, a memory added by another app waits in Review for your approval. Turn this on only if you want those memories saved right away."
              htmlFor="sentient-mcp-skip-review"
            >
              <Switch
                id="sentient-mcp-skip-review"
                checked={skipReview}
                disabled={!enabled}
                onCheckedChange={(v) => setValue('integrations.mcp_server_memories_skip_review', v, { immediate: true })}
              />
            </FormRow>
          </FormSection>

          <div className="space-y-4 rounded-xl border border-border bg-surface px-4 py-4">
            <div>
              <div className="text-sm font-semibold text-fg">Add Sentient to your app</div>
              <p className="mt-0.5 text-xs leading-relaxed text-fg-subtle">
                Turn the switch on first, then add Sentient to the app you use. The app starts Sentient for you each time it needs it.
              </p>
            </div>
            <CodeBlock label="Claude Code" code={CLAUDE_CODE_COMMAND} copyLabel="Copy Claude Code command" />
            <div className="space-y-2">
              <CodeBlock label="Other apps, with HTTP" code={HTTP_SNIPPET} copyLabel="Copy HTTP settings" />
              <p className="text-xs leading-relaxed text-fg-subtle">
                For the HTTP address above, start Sentient in HTTP mode first with <code className="font-mono text-fg-muted">{HTTP_START_COMMAND}</code>. Keep it on this
                computer only, and replace <code className="font-mono text-fg-muted">&lt;engine token&gt;</code> with the engine token: the HTTP address asks for it
                on every call.
              </p>
            </div>
          </div>

          <Alert tone="info" icon={<IconShieldCheck />} title="How Sentient keeps this safe">
            <ul className="list-disc space-y-1 pl-4">
              <li>There are no tools to send messages, delete things or run commands.</li>
              <li>A task made this way still needs you to approve its plan in Sentient before it runs.</li>
              <li>When an app asks Sentient a question, Sentient&apos;s own tools still follow Sentient&apos;s approval rules and ask you when they should.</li>
            </ul>
          </Alert>
        </div>
      )}
    </section>
  )
}
