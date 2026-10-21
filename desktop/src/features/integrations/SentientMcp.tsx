import { IconCheck, IconCopy, IconShieldCheck } from '@tabler/icons-react'
import { useState } from 'react'
import { toast } from 'sonner'
import { Alert, Button, FormRow, FormSection, IconButton, Skeleton, Switch } from '@/components/ui'
import { useConfigEditor } from '@/hooks/config'
import { useConfig } from '@/hooks/core'
import { api, errorMessage } from '@/lib/api'
import { copyText } from '@/lib/utils'
import { useConnection } from '@/stores/connection'

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
 * Sentient tools for memory, tasks and notifications, signing in with a key
 * made only for this (kept in the keychain, replaceable below) that opens
 * nothing else. All of it still goes through Sentient itself, so plans and
 * risky steps ask the user as usual. Settings live in
 * `integrations.mcp_server_enabled` and
 * `integrations.mcp_server_memories_skip_review` (PATCH /api/config).
 */
export function SentientMcpSection({ id }: { id?: string }) {
  const { config, isLoading, error, setValue } = useConfigEditor()
  const configQuery = useConfig()
  const backend = useConnection((s) => s.backend)
  const [replacing, setReplacing] = useState(false)

  const enabled = !!config?.integrations.mcp_server_enabled
  const skipReview = !!config?.integrations.mcp_server_memories_skip_review

  // The command an app should start: the bundled engine in an installed build, the checkout's
  // Python in a development one. The bridge reports the path of the engine it launched; with no
  // bridge (a browser tab), fall back to the plain command name.
  const enginePath = backend?.pythonPath
  const bundled = /sentient-engine(\.exe)?$/i.test(enginePath ?? '')
  const command = enginePath ? `"${enginePath}"` : 'sentient'
  const args = bundled || !enginePath ? ['mcp'] : ['-m', 'sentient', 'mcp']
  const claudeCodeCommand = `claude mcp add sentient -- ${command} ${args.join(' ')}`
  const snippet = `{
  "mcpServers": {
    "sentient": {
      "command": ${JSON.stringify(enginePath ?? 'sentient')},
      "args": ${JSON.stringify(args)}
    }
  }
}`

  const replaceKey = () => {
    setReplacing(true)
    void api.integrations
      .replaceMcpKey()
      .then(() => toast.success('Key replaced', { description: 'Restart your MCP apps so they pick up the new key.' }))
      .catch((e: unknown) => toast.error("Couldn't replace the key", { description: errorMessage(e) }))
      .finally(() => setReplacing(false))
  }

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
              description="When this is on, MCP apps on this computer can reach Sentient. It only works on this computer."
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
            <FormRow
              label="App access key"
              description="Apps prove themselves with a key made only for this, kept in your keychain. It opens only the few things the tools need, never your settings or approvals. Replacing it cuts off every app still holding the old key until the app restarts."
            >
              <Button size="sm" variant="secondary" loading={replacing} disabled={!enabled} onClick={replaceKey}>
                Replace key
              </Button>
            </FormRow>
          </FormSection>

          <div className="space-y-4 rounded-xl border border-border bg-surface px-4 py-4">
            <div>
              <div className="text-sm font-semibold text-fg">Add Sentient to your app</div>
              <p className="mt-0.5 text-xs leading-relaxed text-fg-subtle">
                Turn the switch on first, then add Sentient to the app you use. The app starts Sentient for you each time it needs it. The settings hold no address and no
                key: the command finds the running Sentient by itself.
              </p>
            </div>
            <CodeBlock label="Claude Code" code={claudeCodeCommand} copyLabel="Copy Claude Code command" />
            <CodeBlock label="Other apps" code={snippet} copyLabel="Copy settings" />
          </div>

          <Alert tone="info" icon={<IconShieldCheck />} title="How Sentient keeps this safe">
            <ul className="list-disc space-y-1 pl-4">
              <li>There are no tools to send messages, delete things or run commands.</li>
              <li>A task made this way is made in the app&apos;s name, and its plan always waits for your approval before it runs.</li>
              <li>
                When an app asks Sentient a question, Sentient treats the words as the app&apos;s, not yours: anything that sends, deletes or runs code asks you first,
                in the window and as a notification.
              </li>
              <li>A memory an app saves waits in Review for your approval, unless you allow saving right away above.</li>
            </ul>
          </Alert>
        </div>
      )}
    </section>
  )
}
