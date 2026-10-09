import { IconCheck, IconCopy, IconFolder, IconRefresh } from '@tabler/icons-react'
import { useEffect, useState } from 'react'
import { toast } from 'sonner'
import { Button, Card, FormRow, FormSection, JsonView, Skeleton, StatusDot } from '@/components/ui'
import { useBootstrap, useConfig } from '@/hooks/core'
import { getBridge } from '@/lib/bridge'
import { copyText } from '@/lib/utils'
import { useConnection } from '@/stores/connection'
import type { OpenPathTarget, VersionInfo } from '@/types/bridge'
import { HermesImportCard } from '@/features/onboarding/HermesImport'
import { SchemaForm } from '../SchemaForm'
import type { SectionProps } from '../SettingsPage'

const FOLDERS: Array<{ target: OpenPathTarget; label: string; description: string }> = [
  { target: 'home', label: 'Data folder', description: 'Everything Sentient stores: database, config and files.' },
  { target: 'logs', label: 'Logs', description: 'backend.log and diagnostics.' },
  { target: 'files', label: 'Files', description: 'Attachments and files Sentient creates for you.' },
  { target: 'workspace', label: 'Workspace', description: 'SOUL.md, USER.md, MEMORY.md and daily notes.' },
  { target: 'skills', label: 'Skills', description: 'SKILL.md folders.' }
]

export function AdvancedSection({ query }: SectionProps) {
  const bridge = getBridge()
  const bootstrap = useBootstrap()
  const config = useConfig()
  const backend = useConnection((s) => s.backend)
  const socket = useConnection((s) => s.socket)
  const restart = useConnection((s) => s.restart)
  const [version, setVersion] = useState<VersionInfo | null>(null)
  const [copied, setCopied] = useState(false)
  const [restarting, setRestarting] = useState(false)

  useEffect(() => {
    void bridge.getVersion().then(setVersion)
  }, [bridge])

  return (
    <div className="space-y-8">
      <FormSection title="Move from another assistant">
        <HermesImportCard />
      </FormSection>
      <FormSection title="Folders" description={bootstrap.data ? <span className="font-mono">{bootstrap.data.home}</span> : undefined}>
        {FOLDERS.map((f) => (
          <FormRow key={f.target} label={f.label} description={f.description}>
            <Button size="sm" leftIcon={<IconFolder size={14} />} disabled={!bridge.isDesktop} onClick={() => void bridge.openPath(f.target)}>
              Open
            </Button>
          </FormRow>
        ))}
      </FormSection>

      <FormSection title="Engine">
        <FormRow label="Status" description={backend.message}>
          <span className="flex items-center gap-2 text-sm text-fg-muted">
            <StatusDot tone={backend.state === 'ready' ? (socket === 'open' ? 'success' : 'warning') : 'danger'} />
            {backend.state === 'ready' ? (socket === 'open' ? 'Running · live connection open' : 'Running · reconnecting') : backend.state}
          </span>
        </FormRow>
        <FormRow label="Restart engine" description="Stops and starts Sentient's background engine. Chats in progress are interrupted.">
          <Button
            size="sm"
            leftIcon={<IconRefresh size={14} />}
            loading={restarting}
            disabled={!bridge.isDesktop}
            onClick={async () => {
              setRestarting(true)
              try {
                await restart()
                toast.success('Engine restarted')
              } finally {
                setRestarting(false)
              }
            }}
          >
            Restart
          </Button>
        </FormRow>
        <FormRow label="Version">
          <span className="text-right font-mono text-xs leading-relaxed text-fg-muted">
            engine {bootstrap.data?.version ?? '…'}
            <br />
            app {version?.app ?? '…'} · electron {version?.electron ?? '…'}
          </span>
        </FormRow>
      </FormSection>

      <SchemaForm section="integrations" title="Web & integrations" filter={query} />

      <section className="space-y-2.5">
        <div className="flex items-end justify-between px-1">
          <div>
            <h3 className="text-sm font-semibold text-fg">Raw configuration</h3>
            <p className="mt-0.5 text-xs text-fg-subtle">Read-only view of config.yaml. API keys are never stored here.</p>
          </div>
          <div className="flex gap-1.5">
            <Button size="xs" variant="ghost" leftIcon={<IconRefresh size={12} />} onClick={() => void config.refetch()}>
              Reload
            </Button>
            <Button
              size="xs"
              variant="ghost"
              leftIcon={copied ? <IconCheck size={12} /> : <IconCopy size={12} />}
              disabled={!config.data}
              onClick={async () => {
                if (await copyText(JSON.stringify(config.data, null, 2))) {
                  setCopied(true)
                  setTimeout(() => setCopied(false), 1400)
                }
              }}
            >
              {copied ? 'Copied' : 'Copy JSON'}
            </Button>
          </div>
        </div>
        <Card className="max-h-[480px] overflow-auto p-4">{config.data ? <JsonView value={config.data} collapsedDepth={1} /> : <Skeleton className="h-40" />}</Card>
      </section>
    </div>
  )
}
