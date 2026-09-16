/** Settings for the newer abilities: knowing you, code execution, browser, helpers and the wake word. */
import { IconArrowRight, IconCode, IconDownload, IconEar, IconMoonStars, IconShieldLock, IconUserHeart, IconUsersGroup, IconWorldWww } from '@tabler/icons-react'
import type { ReactNode } from 'react'
import { useNavigate } from 'react-router'
import { Alert, Badge, Button, Card, FormRow, FormSection, ProgressBar, Skeleton, StatusDot, Switch } from '@/components/ui'
import { setAlwaysListening, useWakeStore } from '@/features/voice/wake'
import { useConfigEditor } from '@/hooks/config'
import { useConfigSchema } from '@/hooks/core'
import { useVoicePrepare, useVoiceStatus } from '@/hooks/voice'
import { notReady, useSandboxStatus } from '@/lib/leap/hooks-b'
import type { VoiceStatusB } from '@/lib/leap/types-b'
import type { VoicePrepareTarget } from '@/lib/types'
import { SchemaForm } from '../SchemaForm'
import type { SectionProps } from '../SettingsPage'

export const WAKE_KEYS = ['voice.wake_word', 'voice.wake_engine', 'voice.wake_sensitivity', 'voice.wake_model', 'voice.wake_whisper_model', 'voice.wake_earcon', 'voice.follow_up_seconds']

function useHasSections(...sections: string[]): boolean | null {
  const schema = useConfigSchema()
  if (!schema.data) return null
  return sections.some((s) => !!schema.data?.properties?.[s])
}

function NotYet({ what }: { what: string }) {
  return (
    <Alert tone="info" title={`${what} settings arrive with the next engine update`}>
      Nothing to set up yet. They’ll appear here on their own.
    </Alert>
  )
}

function Explainer({ icon, title, children, action }: { icon: ReactNode; title: string; children: ReactNode; action?: ReactNode }) {
  return (
    <Card className="flex items-start gap-3 px-4 py-3.5">
      <span className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-accent/10 text-accent-text">{icon}</span>
      <div className="min-w-0 flex-1">
        <div className="text-sm font-medium text-fg">{title}</div>
        <div className="mt-0.5 text-xs leading-relaxed text-fg-subtle">{children}</div>
      </div>
      {action}
    </Card>
  )
}

// ---------------------------------------------------------------------------- knowing you
export function KnowingSection({ query }: SectionProps) {
  const navigate = useNavigate()
  const has = useHasSections('user_model', 'dreaming')
  return (
    <div className="space-y-8">
      <Card interactive className="flex cursor-pointer items-center gap-3 px-4 py-3.5" onClick={() => navigate('/about')}>
        <span className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-accent/10 text-accent-text">
          <IconUserHeart size={18} />
        </span>
        <div className="min-w-0 flex-1">
          <div className="text-sm font-medium text-fg">See how Sentient understands you</div>
          <div className="text-xs text-fg-subtle">Every preference and goal it has noticed, with the reasons. Correct anything that isn’t you.</div>
        </div>
        <IconArrowRight size={16} className="text-fg-subtle" />
      </Card>
      {has === false ? (
        <NotYet what="Knowing you" />
      ) : (
        <>
          <SchemaForm section="user_model" title="Your picture" filter={query} />
          <SchemaForm
            section="dreaming"
            title="Tidying up overnight"
            filter={query}
            footer={
              <Button size="sm" variant="ghost" leftIcon={<IconMoonStars size={14} />} onClick={() => navigate('/about/dreams')}>
                Read the dream journal
              </Button>
            }
          />
        </>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------- code execution
export function SandboxSection({ query }: SectionProps) {
  const status = useSandboxStatus()
  const has = useHasSections('sandbox')
  const s = status.data
  return (
    <div className="space-y-8">
      {status.isLoading ? (
        <Skeleton className="h-16 rounded-xl" />
      ) : s ? (
        <Card className="flex items-center gap-3 px-4 py-3.5">
          <span className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-accent/10 text-accent-text">
            <IconCode size={18} />
          </span>
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2 text-sm font-medium text-fg">
              <StatusDot tone={s.enabled ? 'success' : 'neutral'} />
              {s.enabled ? 'Ready to run scripts' : 'Turned off'}
            </div>
            <div className="text-xs text-fg-subtle">
              {s.backend === 'docker' ? 'Scripts run in a Docker container.' : 'Scripts run in a separate process on this computer.'} Python {s.python_version}.
            </div>
          </div>
          <Badge tone={s.docker_available ? 'success' : 'neutral'}>{s.docker_available ? 'Docker available' : 'No Docker'}</Badge>
        </Card>
      ) : status.isError && !notReady(status.error) ? (
        <Alert tone="warning" title="Couldn’t check code execution">
          It may still work. Try again in a moment.
        </Alert>
      ) : null}
      <Explainer icon={<IconShieldLock size={18} />} title="Scripts stay on a short leash">
        Scripts never see your API keys. They can look things up with Sentient’s tools, but anything that sends, deletes or spends still goes through your approvals.
      </Explainer>
      {has === false ? <NotYet what="Code execution" /> : <SchemaForm section="sandbox" title="Running scripts" filter={query} />}
    </div>
  )
}

// ---------------------------------------------------------------------------- browser
export function BrowserSection({ query }: SectionProps) {
  const has = useHasSections('browser')
  return (
    <div className="space-y-8">
      <Explainer icon={<IconWorldWww size={18} />} title="Signing in is always up to you">
        Sentient never logs in on its own. When a site needs you, it opens a window so you can sign in, and it remembers that for next time.
      </Explainer>
      {has === false ? <NotYet what="Browser" /> : <SchemaForm section="browser" title="Using the web" filter={query} />}
    </div>
  )
}

// ---------------------------------------------------------------------------- helpers
export function SubagentsSection({ query }: SectionProps) {
  const has = useHasSections('subagents')
  return (
    <div className="space-y-8">
      <Explainer icon={<IconUsersGroup size={18} />} title="Many hands for big jobs">
        For bigger requests, Sentient can split the work and let helpers research in parallel. Helpers never send, delete or run code, and they report back in your chat.
      </Explainer>
      {has === false ? <NotYet what="Helper" /> : <SchemaForm section="subagents" title="Helpers" filter={query} />}
    </div>
  )
}

// ---------------------------------------------------------------------------- wake word (inside Voice)
export function WakeWordSection({ query }: SectionProps) {
  const enabled = useWakeStore((s) => s.enabled)
  const status = useWakeStore((s) => s.status)
  const error = useWakeStore((s) => s.error)
  const voiceStatus = useVoiceStatus()
  const prepare = useVoicePrepare()
  const { config } = useConfigEditor()
  const wake = (voiceStatus.data as VoiceStatusB | undefined)?.wake
  const raw = config?.voice.wake_word || wake?.phrase || 'hey sentient'
  const phrase = raw.replace(/\b\p{L}/gu, (c) => c.toUpperCase())
  const q = query.trim().toLowerCase()
  const matches = !q || `wake word phrase hey sentient always listening hands-free follow-up chime ${phrase}`.toLowerCase().includes(q)

  const line =
    status === 'standby'
      ? `Listening for “${phrase}”`
      : status === 'starting'
        ? 'Getting the microphone ready…'
        : status === 'paused'
          ? 'Paused while Voice mode is open'
          : status === 'error'
            ? (error ?? 'Wake word listening stopped')
            : null

  return (
    <div className="space-y-6">
      {matches && (
        <FormSection title={`“${phrase}”`} description="Start talking to Sentient without touching your computer.">
          <FormRow
            label={`Always listening for “${phrase}”`}
            description="Sentient listens quietly for its name while the app is open, then opens Voice mode. Nothing is sent to any AI until you say it. Listening from the tray is coming soon."
          >
            <Switch checked={enabled} onCheckedChange={setAlwaysListening} aria-label="Always listening for the wake word" />
          </FormRow>
          {enabled && line && (
            <div className="flex items-center gap-2 px-4 py-2.5 text-xs text-fg-muted">
              <StatusDot tone={status === 'error' ? 'danger' : status === 'standby' ? 'success' : 'neutral'} pulse={status === 'standby'} />
              {line}
            </div>
          )}
          {wake && (
            <div className="flex flex-wrap items-center gap-3 px-4 py-3">
              <IconEar size={16} className="text-fg-subtle" />
              <div className="min-w-0 flex-1 text-sm">
                <span className="font-medium text-fg">Wake word detector</span>
                <span className="ml-2 text-xs text-fg-subtle">
                  {wake.engine === 'openwakeword' ? 'Lighter detector' : 'Small speech model'} · {wake.ready ? 'ready' : 'loads when first needed'}
                </span>
                {wake.error && <div className="text-xs text-danger">{wake.error}</div>}
                {prepare.error && <div className="text-xs text-danger">{prepare.error}</div>}
              </div>
              {!wake.ready && (
                <Button size="sm" leftIcon={<IconDownload size={13} />} loading={prepare.running} disabled={prepare.done} onClick={() => void prepare.run('wake' as VoicePrepareTarget)}>
                  {prepare.done ? 'Ready' : 'Prepare'}
                </Button>
              )}
              {(prepare.running || prepare.done) && <ProgressBar className="basis-full" value={prepare.progress} tone={prepare.done ? 'success' : 'accent'} />}
            </div>
          )}
        </FormSection>
      )}
      <SchemaForm section="voice" title="Wake word" include={WAKE_KEYS} filter={query} />
    </div>
  )
}
