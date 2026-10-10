import { IconArrowRight, IconBolt, IconDownload, IconMicrophone, IconPlayerPlay, IconRefresh, IconVolume } from '@tabler/icons-react'
import { useNavigate } from 'react-router'
import { toast } from 'sonner'
import { Alert, Badge, Button, Card, FormRow, FormSection, ProgressBar, Skeleton, StatusDot, Switch } from '@/components/ui'
import { useConfigEditor } from '@/hooks/config'
import { useBootstrap, useTools } from '@/hooks/core'
import { useProactivityActions, useProactivityStatus } from '@/hooks/notifications'
import { useSpeak, useVoicePrepare, useVoiceStatus } from '@/hooks/voice'
import { errorMessage, isNotImplemented } from '@/lib/api'
import { relativeTime } from '@/lib/utils'
import { SchemaForm } from '../SchemaForm'
import type { SectionProps } from '../SettingsPage'
import { WAKE_KEYS, WakeWordSection } from './AbilitySections'
import { ApprovalRulesSection } from './ApprovalRules'
import { DICTATION_KEYS, DictationSection } from './Dictation'

export function MemorySection({ query }: SectionProps) {
  const navigate = useNavigate()
  return (
    <SchemaForm
      section="memory"
      title="Learning and recall"
      filter={query}
      footer={
        <Card interactive className="flex cursor-pointer items-center gap-3 px-4 py-3.5" onClick={() => navigate('/memory')}>
          <div className="min-w-0 flex-1">
            <div className="text-sm font-medium text-fg">Browse what Sentient remembers</div>
            <div className="text-xs text-fg-subtle">See, edit and delete individual memories.</div>
          </div>
          <IconArrowRight size={16} className="text-fg-subtle" />
        </Card>
      }
    />
  )
}

export function TasksSection({ query }: SectionProps) {
  return <SchemaForm section="tasks" title="Running tasks" filter={query} />
}

export function ProactivitySection({ query }: SectionProps) {
  const status = useProactivityStatus()
  const { pollNow } = useProactivityActions()
  const missing = status.isError && isNotImplemented(status.error)

  return (
    <div className="space-y-8">
      <Card className="p-4">
        <div className="flex items-center gap-3">
          <div className="flex size-9 items-center justify-center rounded-lg bg-accent/10 text-accent-text">
            <IconBolt size={18} />
          </div>
          <div className="min-w-0 flex-1">
            {status.isLoading ? (
              <Skeleton className="h-4 w-48" />
            ) : missing ? (
              <>
                <div className="text-sm font-medium text-fg">Watching for suggestions</div>
                <div className="text-xs text-fg-subtle">Status appears once Gmail or Calendar is connected in Integrations.</div>
              </>
            ) : status.data ? (
              <>
                <div className="flex items-center gap-2 text-sm font-medium text-fg">
                  <StatusDot tone={status.data.enabled ? 'success' : 'neutral'} />
                  {status.data.enabled ? 'Active' : 'Paused'}
                  <Badge size="xs">{status.data.suggestions_today} today</Badge>
                </div>
                <div className="text-xs text-fg-subtle">
                  {status.data.sources.length
                    ? status.data.sources.map((s) => `${s.source}: ${s.last_error ? 'error' : s.last_poll_at ? `checked ${relativeTime(s.last_poll_at)}` : 'not checked yet'}`).join(' · ')
                    : 'No connected sources yet.'}
                </div>
              </>
            ) : (
              <div className="text-sm text-danger">{errorMessage(status.error)}</div>
            )}
          </div>
          <Button
            size="sm"
            leftIcon={<IconRefresh size={14} />}
            disabled={missing || status.isLoading}
            loading={pollNow.isPending}
            onClick={() =>
              pollNow.mutate(undefined, {
                onSuccess: (r) => toast.success('Checked your apps', { description: `${r.events} new event${r.events === 1 ? '' : 's'}` }),
                onError: (e) => toast.error("Couldn't check now", { description: errorMessage(e) })
              })
            }
          >
            Check now
          </Button>
        </div>
      </Card>
      <SchemaForm section="proactivity" title="Suggestions" filter={query} />
    </div>
  )
}

export function ApprovalsSection({ query }: SectionProps) {
  const tools = useTools()
  const { config, setValue } = useConfigEditor()
  const disabled = config?.tools.disabled ?? []

  return (
    <div className="space-y-8">
      <SchemaForm section="tools" title="Approvals" filter={query} exclude={['tools.disabled']} />
      <ApprovalRulesSection query={query} />
      <FormSection title="Tools" description="Turn off tool groups Sentient shouldn't use at all.">
        {tools.isLoading ? (
          <div className="p-4">
            <Skeleton className="h-24" />
          </div>
        ) : (
          (tools.data ?? []).map((p) => {
            const risks = Array.from(new Set(p.tools.map((t) => t.risk)))
            return (
              <FormRow
                key={p.id}
                label={
                  <span className="flex items-center gap-2">
                    {p.display_name}
                    {risks.map((r) => (
                      <Badge key={r} size="xs" tone={r === 'read' ? 'neutral' : r === 'write' ? 'warning' : 'danger'}>
                        {r}
                      </Badge>
                    ))}
                  </span>
                }
                description={`${p.description} ${p.tools.length} tool${p.tools.length === 1 ? '' : 's'}.`}
              >
                <Switch
                  checked={!disabled.includes(p.id)}
                  onCheckedChange={(on) => setValue('tools.disabled', on ? disabled.filter((d) => d !== p.id) : [...disabled, p.id], { immediate: true })}
                />
              </FormRow>
            )
          })
        )}
      </FormSection>
    </div>
  )
}

export function EvolutionSection({ query }: SectionProps) {
  return (
    <div className="space-y-8">
      <SchemaForm section="evolution" title="Learning new skills" filter={query} />
      <SchemaForm section="skills" title="Skill library" filter={query} />
    </div>
  )
}

export function VoiceSection({ query }: SectionProps) {
  const status = useVoiceStatus()
  const speak = useSpeak()
  const prepare = useVoicePrepare()
  const bootstrap = useBootstrap()
  const missing = status.isError && isNotImplemented(status.error)

  return (
    <div className="space-y-8">
      {missing ? (
        <Alert tone="info" icon={<IconMicrophone />} title="The voice engine isn't installed in this build yet">
          Your choices below are saved and take effect when voice arrives.
        </Alert>
      ) : (
        <Card className="divide-y divide-border">
          <div className="flex items-center gap-3 px-4 py-3">
            <IconMicrophone size={17} className="text-fg-subtle" />
            <div className="min-w-0 flex-1 text-sm">
              {status.isLoading ? (
                <Skeleton className="h-4 w-48" />
              ) : (
                <>
                  <span className="font-medium text-fg">Speech recognition</span>
                  <span className="ml-2 text-xs text-fg-subtle">
                    {[status.data?.stt.provider, status.data?.stt.model, status.data?.stt.device].filter(Boolean).join(' · ')}
                    {status.data?.stt.ready === false && ' · loads on first use'}
                  </span>
                  {status.data?.stt.note && <div className="text-xs text-fg-subtle">{status.data.stt.note}</div>}
                  {status.data?.stt.error && <div className="text-xs text-danger">{status.data.stt.error}</div>}
                </>
              )}
            </div>
            <StatusDot tone={status.data?.stt.ready ? 'success' : 'warning'} />
          </div>
          <div className="flex items-center gap-3 px-4 py-3">
            <IconVolume size={17} className="text-fg-subtle" />
            <div className="min-w-0 flex-1 text-sm">
              <span className="font-medium text-fg">Voice</span>
              <span className="ml-2 text-xs text-fg-subtle">
                {[status.data?.tts.provider, status.data?.tts.voice, status.data?.tts.backend].filter(Boolean).join(' · ')}
              </span>
              {status.data?.tts.error && <div className="text-xs text-danger">{status.data.tts.error}</div>}
            </div>
            <Button
              size="sm"
              leftIcon={<IconPlayerPlay size={13} />}
              loading={speak.isPending}
              onClick={() =>
                speak.mutate(
                  { text: `Hi ${bootstrap.data?.assistant.user_name || 'there'}, this is how I sound.` },
                  { onError: (e) => toast.error("Couldn't play the voice", { description: errorMessage(e) }) }
                )
              }
            >
              Test voice
            </Button>
          </div>
          <div className="px-4 py-3">
            <div className="flex items-center gap-3">
              <IconDownload size={17} className="text-fg-subtle" />
              <div className="min-w-0 flex-1 text-sm">
                <span className="font-medium text-fg">Local voice models</span>
                <div className="text-xs text-fg-subtle">{prepare.error ?? (prepare.running || prepare.done ? prepare.stage : 'Download speech models so voice works offline.')}</div>
              </div>
              <Button size="sm" loading={prepare.running} disabled={prepare.done} onClick={() => void prepare.run()}>
                {prepare.done ? 'Ready' : 'Prepare'}
              </Button>
            </div>
            {(prepare.running || prepare.done) && <ProgressBar className="mt-3" value={prepare.progress} tone={prepare.done ? 'success' : 'accent'} />}
          </div>
        </Card>
      )}
      <WakeWordSection query={query} />
      <DictationSection query={query} />
      <SchemaForm section="voice" title="Voice settings" filter={query} exclude={[...WAKE_KEYS, ...DICTATION_KEYS]} />
    </div>
  )
}
