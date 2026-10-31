import { IconCheck, IconCopy, IconKey } from '@tabler/icons-react'
import { useState } from 'react'
import { toast } from 'sonner'
import { Button, FormRow, FormSection } from '@/components/ui'
import { api, getConnection } from '@/lib/api'
import { copyText } from '@/lib/utils'
import { SchemaForm } from '../SchemaForm'

/**
 * Other apps on this computer chatting with Sentient through the OpenAI-compatible API (#211, docs/API.md section 2).
 * The two switches come from the schema; the address and the key, shown once, are here.
 */
export function ConnectedAppsSection({ query }: { query: string }) {
  const [key, setKey] = useState('')
  const [making, setMaking] = useState(false)
  const [copied, setCopied] = useState('')
  const address = `${getConnection()?.baseUrl ?? 'http://127.0.0.1:7777'}/v1`

  async function copy(what: string, text: string) {
    if (await copyText(text)) {
      setCopied(what)
      setTimeout(() => setCopied(''), 1500)
    }
  }

  async function makeKey() {
    setMaking(true)
    try {
      setKey((await api.openaiApi.makeKey()).key)
    } catch (e) {
      toast.error("Couldn't make a key", { description: e instanceof Error ? e.message : undefined })
    } finally {
      setMaking(false)
    }
  }

  const copyButton = (what: string, text: string) => (
    <Button
      size="xs"
      variant="ghost"
      leftIcon={copied === what ? <IconCheck size={12} /> : <IconCopy size={12} />}
      onClick={() => void copy(what, text)}
    >
      {copied === what ? 'Copied' : 'Copy'}
    </Button>
  )

  return (
    <SchemaForm
      section="openai_api"
      title="Connected apps"
      description="Editors, scripts and shortcuts on this computer can chat with Sentient through an OpenAI-compatible API."
      filter={query}
      footer={
        <FormSection>
          <FormRow label="Address" description="Use it as the base URL in the app. It changes when Sentient restarts.">
            <span className="flex items-center gap-2">
              <code className="font-mono text-xs text-fg-muted">{address}</code>
              {copyButton('address', address)}
            </span>
          </FormRow>
          <FormRow
            label="API key"
            description={
              key ? "Copy it now. It won't be shown again." : 'Each app needs it. Making a new key stops the old one working.'
            }
            stack={!!key}
          >
            {key ? (
              <span className="flex items-center gap-2">
                <code className="break-all font-mono text-xs text-fg">{key}</code>
                {copyButton('key', key)}
              </span>
            ) : (
              <Button size="sm" leftIcon={<IconKey size={14} />} loading={making} onClick={() => void makeKey()}>
                Make a key
              </Button>
            )}
          </FormRow>
        </FormSection>
      }
    />
  )
}
