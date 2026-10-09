/**
 * Settings generated from `GET /api/config/schema`. New backend fields show up
 * automatically; `LABELS` only makes them read nicely.
 */
import { IconX } from '@tabler/icons-react'
import { useEffect, useState, type ReactNode } from 'react'
import { FormRow, FormSection, Input, SegmentedControl, Select, Skeleton, Slider, Switch } from '@/components/ui'
import { getPath, useConfigEditor } from '@/hooks/config'
import { useConfigSchema } from '@/hooks/core'
import type { JsonSchema } from '@/lib/types'
import { cn, humanize } from '@/lib/utils'

interface FieldLabel {
  label?: string
  description?: string
  unit?: string
  percent?: boolean
  placeholder?: string
  options?: Record<string, string>
  hidden?: boolean
  /** Render a string field as a native time picker (HH:MM). */
  inputType?: 'time'
}

export const LABELS: Record<string, FieldLabel> = {
  // chat
  'chat.history_window': { label: 'Messages of context per reply', unit: 'messages' },
  'chat.compress_after_messages': { label: 'Summarize long chats after', unit: 'messages' },
  'chat.show_thinking': { label: 'Show thinking', description: "Show the model's reasoning in a collapsible block." },
  'chat.auto_title': { label: 'Name chats automatically' },
  // memory
  'memory.facts_top_k': { label: 'Facts recalled per message', unit: 'facts' },
  'memory.min_similarity': { label: 'Recall threshold', percent: true },
  'memory.duplicate_similarity': { label: 'Duplicate threshold', percent: true },
  'memory.extract_after_turn': { label: 'Learn from conversations' },
  'memory.workspace_budget_chars': { label: 'Workspace file budget', unit: 'chars' },
  'memory.graph_link_similarity': { label: 'Graph link threshold', percent: true },
  'memory.summarize_after_minutes': { label: 'Summarize conversations after', unit: 'min' },
  'memory.summary_chunk_messages': { label: 'Messages per summary', unit: 'messages' },
  // tasks
  'tasks.tick_seconds': { label: 'Scheduler check interval', unit: 's' },
  'tasks.max_concurrent_runs': { label: 'Runs at the same time' },
  'tasks.run_timeout_minutes': { label: 'Stop a run after', unit: 'min' },
  'tasks.max_tool_rounds': { label: 'Max tool steps per run', unit: 'steps' },
  'tasks.require_plan_approval': { label: 'Approve plans before they run' },
  'tasks.swarm_max_agents': { label: 'Parallel agents per swarm', unit: 'agents' },
  // proactivity
  'proactivity.enabled': { label: 'Proactive suggestions' },
  'proactivity.poll_interval_minutes': { label: 'Check connected apps every', unit: 'min' },
  'proactivity.base_confidence_threshold': { label: 'Minimum confidence', percent: true },
  'proactivity.quiet_hours': { label: 'Quiet hours', placeholder: '22:00-07:00' },
  'proactivity.heartbeat_minutes': { label: 'Periodic check-in', unit: 'min', description: 'A check-in with no trigger. 0 turns it off.' },
  'proactivity.followups': { label: 'Unanswered emails' },
  'proactivity.followups.enabled': { label: 'Notice unanswered emails' },
  'proactivity.followups.sources': { label: 'Email accounts to check' },
  'proactivity.followups.waiting_on_you_days': { label: 'Remind me to reply after', unit: 'days' },
  'proactivity.followups.waiting_on_them_days': { label: 'Offer a nudge after', unit: 'days' },
  'proactivity.followups.max_age_days': { label: 'Ignore emails quiet for over', unit: 'days' },
  'proactivity.followups.max_suggestions': { label: 'Most suggestions per check' },
  // approvals & tools
  'tools.approvals.mode': {
    label: 'Ask before acting',
    description: 'Sentient checks with you before it changes, sends, deletes or runs something. You can also require a yes for every tool.',
    options: { off: 'Never ask', ask: 'When it changes or sends', always: 'Before every tool' }
  },
  'tools.approvals.remember_session': {
    label: 'Remember “Allow for this chat”',
    description: "After you allow a tool for a chat, Sentient won't ask again for that tool in the same chat."
  },
  'tools.approvals.timeout_s': { label: 'Deny unanswered requests after', unit: 's' },
  'tools.disabled': { label: 'Disabled tools', hidden: true },
  'tools.approvals.rules': { label: 'Rules for apps and tools', hidden: true },
  // evolution & skills
  'evolution.review_enabled': { label: 'Learn skills from finished work' },
  'evolution.review_idle_minutes': { label: 'Review a chat once idle for', unit: 'min' },
  'evolution.min_tool_calls_for_review': { label: 'Only review work with at least', unit: 'tool calls' },
  'evolution.curator_enabled': { label: 'Tidy up skills automatically' },
  'evolution.curator_interval_hours': { label: 'Curator runs every', unit: 'h' },
  'evolution.stale_after_days': { label: 'Mark unused skills stale after', unit: 'days' },
  'evolution.archive_after_days': { label: 'Archive stale skills after', unit: 'days' },
  'evolution.user_profile_updates': { label: 'Keep USER.md and MEMORY.md up to date' },
  'skills.extra_dirs': { label: 'Extra skill folders', placeholder: 'Add a folder path and press Enter' },
  'skills.write_approval': { label: 'Review new skills before they activate' },
  // voice
  'voice.stt_provider': {
    label: 'Speech recognition',
    options: { faster_whisper: 'Local (faster-whisper)', openai: 'OpenAI', deepgram: 'Deepgram', elevenlabs: 'ElevenLabs' }
  },
  'voice.stt_model': { label: 'Recognition model', placeholder: 'base' },
  'voice.stt_device': { label: 'Run recognition on', options: { auto: 'Auto', cpu: 'CPU', cuda: 'GPU' } },
  'voice.stt_language': { label: 'Spoken language', placeholder: 'Auto-detect' },
  'voice.tts_provider': {
    label: 'Voice engine',
    options: { system: 'System voices', kokoro: 'Local (Kokoro)', openai: 'OpenAI', elevenlabs: 'ElevenLabs' }
  },
  'voice.tts_model': { label: 'Voice model', placeholder: 'Provider default' },
  'voice.tts_voice': { label: 'Voice', placeholder: 'Default voice' },
  'voice.tts_speed': { label: 'Speaking speed', unit: '×' },
  'voice.kokoro_variant': { label: 'Kokoro variant' },
  'voice.vad_silence_ms': { label: 'Pause that ends a sentence', unit: 'ms' },
  'voice.vad_min_speech_ms': { label: 'Ignore sounds shorter than', unit: 'ms' },
  'voice.vad_max_utterance_s': { label: 'Longest single utterance', unit: 's' },
  'voice.barge_in': { label: 'Let me interrupt by speaking', description: 'Talking while Sentient speaks stops it.' },
  'voice.wake_word': { label: 'Wake phrase', placeholder: 'hey sentient', description: 'Say this to start talking without touching anything.' },
  'voice.wake_engine': {
    label: 'How I listen for it',
    options: { whisper: 'Any phrase', openwakeword: 'Lighter detector' },
    description: 'The first understands any phrase you choose. The lighter one uses less power but only knows a few phrases like “hey jarvis”.'
  },
  'voice.wake_sensitivity': { label: 'Sensitivity', percent: true, description: 'Higher hears you from further away, but may wake up by mistake.' },
  'voice.wake_model': { label: 'Wake model', placeholder: 'Chosen from the phrase', description: 'Only for the lighter detector: a built-in model name or a custom model file.' },
  'voice.wake_whisper_model': { label: 'Listening model', options: { tiny: 'Tiny (fastest)', 'tiny.en': 'Tiny, English only', base: 'Base (more accurate)', 'base.en': 'Base, English only' } },
  'voice.wake_earcon': { label: 'Play a chime when I hear you' },
  'voice.follow_up_seconds': { label: 'Keep listening after a reply for', unit: 's', description: 'Ask a follow-up without saying the phrase again.' },
  // knowing you
  'user_model.enabled': { label: 'Build a picture of me', description: 'Sentient notices your preferences, goals and style over time. You can see and correct all of it on the About you page.' },
  'user_model.refresh_after_turns': { label: 'Look for new things about me every', unit: 'messages' },
  'user_model.min_refresh_hours': { label: 'But no more often than every', unit: ' h' },
  'user_model.max_open_questions': { label: 'Questions I can ask you at once', description: 'When Sentient isn’t sure about something, it asks instead of guessing.' },
  'user_model.max_active_insights': { label: 'Most things to keep in mind', unit: 'items' },
  // engine tuning, kept out of the way
  'user_model.role': { hidden: true },
  'user_model.max_operations': { hidden: true },
  'user_model.support_step': { hidden: true },
  'user_model.contradict_step': { hidden: true },
  'user_model.dispute_below': { hidden: true },
  'user_model.context_max_chars': { hidden: true },
  'user_model.context_min_similarity': { hidden: true },
  'user_model.recent_messages': { hidden: true },
  'user_model.recent_facts': { hidden: true },
  'user_model.recent_summaries': { hidden: true },
  'dreaming.enabled': { label: 'Tidy up memory overnight', description: 'Merge repeats, settle things that disagree and let old notes fade, then leave a short note about it.' },
  'dreaming.time': { label: 'Start at', inputType: 'time', description: 'Only when you haven’t used Sentient for a while.' },
  'dreaming.require_idle_minutes': { label: 'Wait until I’ve been away for', unit: 'min' },
  'dreaming.refresh_user_model': { label: 'Update my picture while tidying up' },
  'dreaming.notify': { label: 'Leave me a note when something changed' },
  'dreaming.max_model_calls': { label: 'Most AI calls per night', unit: 'calls', description: 'Keeps overnight tidying light on your computer.' },
  'dreaming.promote_min_recalls': { label: 'Keep a short-term note for good after it comes up', unit: 'times' },
  'dreaming.role': { hidden: true },
  'dreaming.max_facts': { hidden: true },
  'dreaming.merge_similarity': { hidden: true },
  'dreaming.merge_min_overlap': { hidden: true },
  'dreaming.contradiction_similarity': { hidden: true },
  // code execution
  'sandbox.enabled': { label: 'Let Sentient run small scripts', description: 'For crunching data or combining several tools at once. Anything that sends, deletes or spends still asks you first.' },
  'sandbox.backend': { label: 'Where scripts run', options: { auto: 'Automatic', process: 'Separate process', docker: 'Docker container' }, description: 'Automatic uses Docker when it is running, otherwise a separate process on this computer.' },
  'sandbox.timeout_s': { label: 'Stop a script after', unit: 's' },
  'sandbox.max_output_chars': { label: 'Longest output kept', unit: 'chars' },
  'sandbox.max_tool_calls': { label: 'Most tool uses per script', unit: 'uses' },
  'sandbox.max_concurrent_runs': { label: 'Scripts at the same time' },
  'sandbox.max_files': { label: 'Most files kept per run', unit: 'files' },
  'sandbox.max_file_mb': { label: 'Skip files larger than', unit: 'MB' },
  'sandbox.keep_workdirs': { label: 'Keep script folders for troubleshooting' },
  'sandbox.docker_image': { label: 'Docker image', placeholder: 'python:3.12-slim' },
  'sandbox.docker_memory_mb': { label: 'Memory limit in Docker', unit: 'MB' },
  'sandbox.docker_cpus': { label: 'Processor limit in Docker', unit: ' CPUs' },
  'sandbox.allow_network_in_docker': { label: 'Let Docker scripts use the internet' },
  // browser
  'browser.enabled': { label: 'Let Sentient use a web browser', description: 'For websites without an integration. It never types passwords or card numbers.' },
  'browser.engine': { label: 'Browser', options: { auto: 'Automatic', msedge: 'Microsoft Edge', chrome: 'Google Chrome', chromium: 'Chromium' } },
  'browser.headless': { label: 'Keep the browser hidden', description: 'You can still watch what it does in the live view.' },
  'browser.idle_minutes': { label: 'Close the hidden browser after', unit: ' min' },
  'browser.allow_domains': { label: 'Only allow these sites', placeholder: 'e.g. amazon.in, then press Enter' },
  'browser.block_domains': { label: 'Never open these sites', placeholder: 'Add a site and press Enter' },
  'browser.max_snapshot_chars': { label: 'Page description size', unit: 'chars' },
  'browser.max_extract_chars': { label: 'Page text size', unit: 'chars' },
  'browser.confirm_purchases': { label: 'Ask before buying, sending or deleting' },
  'browser.live_view': { label: 'Show a live view' },
  // helpers
  'subagents.enabled': { label: 'Let Sentient use helpers', description: 'Sentient can hand parts of a bigger job to helpers that work alongside the chat, then report back.' },
  'subagents.max_concurrent': { label: 'Helpers at the same time' },
  'subagents.max_rounds': { label: 'Most steps per helper', unit: 'steps' },
  'subagents.role': { label: 'Model helpers use', options: { executor: 'Task model', primary: 'Main model', fast: 'Fast model' } },
  'subagents.timeout_minutes': { label: 'Stop a helper after', unit: 'min' },
  // integrations
  'integrations.oauth_redirect_port': { label: 'OAuth callback port', description: '0 picks a free port each time.' },
  'integrations.search_provider': {
    label: 'Web search',
    options: { duckduckgo: 'DuckDuckGo', brave: 'Brave', google_cse: 'Google', searxng: 'SearXNG' }
  },
  'integrations.searxng_url': { label: 'SearXNG address', placeholder: 'https://search.example.com' },
  'integrations.weather_provider': { label: 'Weather', options: { open_meteo: 'Open-Meteo', accuweather: 'AccuWeather' } },
  'integrations.mcp_servers': { hidden: true }
}

export function deref(schema: JsonSchema | undefined, root: JsonSchema | undefined): JsonSchema | undefined {
  if (!schema) return schema
  if (schema.$ref && root?.$defs) {
    const name = schema.$ref.split('/').pop() as string
    return deref(root.$defs[name], root)
  }
  if (schema.allOf?.length === 1) return deref({ ...schema.allOf[0], ...schema, allOf: undefined }, root)
  if (schema.anyOf) {
    const nonNull = schema.anyOf.filter((s) => s.type !== 'null')
    if (nonNull.length === 1) return { ...deref(nonNull[0], root), description: schema.description, title: schema.title, default: schema.default, nullable: true }
  }
  return schema
}

interface FieldDef {
  path: string
  key: string
  schema: JsonSchema
  label: string
  description?: string
  meta: FieldLabel
}

interface Group {
  path: string
  title: string
  description?: string
  fields: FieldDef[]
}

function collectGroups(section: string, root: JsonSchema): Group[] {
  const top = deref(root.properties?.[section], root)
  if (!top?.properties) return []
  const groups: Group[] = []
  const walk = (schema: JsonSchema, prefix: string, title: string, description?: string) => {
    const group: Group = { path: prefix, title, description, fields: [] }
    groups.push(group)
    for (const [key, raw] of Object.entries(schema.properties ?? {})) {
      const s = deref(raw, root) as JsonSchema
      const path = `${prefix}.${key}`
      const meta = LABELS[path] ?? {}
      if (meta.hidden) continue
      if (s.type === 'object' && s.properties) {
        walk(s, path, meta.label ?? humanize(key), meta.description ?? s.description)
        continue
      }
      if (s.type === 'object') continue // free-form dicts are edited elsewhere
      group.fields.push({
        path,
        key,
        schema: s,
        label: meta.label ?? humanize(s.title ?? key),
        description: meta.description ?? s.description,
        meta
      })
    }
  }
  walk(top, section, humanize(section), top.description)
  return groups.filter((g) => g.fields.length)
}

export function schemaMatches(section: string, root: JsonSchema | undefined, query: string): boolean {
  if (!root || !query.trim()) return false
  const q = query.toLowerCase()
  return collectGroups(section, root).some((g) => g.fields.some((f) => `${f.label} ${f.description ?? ''} ${f.path}`.toLowerCase().includes(q)))
}

export function SchemaForm({
  section,
  title,
  description,
  filter = '',
  exclude = [],
  include,
  footer
}: {
  section: string
  title?: ReactNode
  description?: ReactNode
  filter?: string
  exclude?: string[]
  /** Only these field paths (in schema order); fields missing from the schema are skipped. */
  include?: string[]
  footer?: ReactNode
}) {
  const schema = useConfigSchema()
  const editor = useConfigEditor()

  if (schema.isLoading || editor.isLoading) {
    return (
      <div className="space-y-2 rounded-xl border border-border bg-surface p-4">
        {[0, 1, 2].map((i) => (
          <Skeleton key={i} className="h-9" />
        ))}
      </div>
    )
  }
  if (!schema.data || !editor.config) return null

  const q = filter.trim().toLowerCase()
  const groups = collectGroups(section, schema.data)
    .map((g) => ({
      ...g,
      fields: g.fields.filter(
        (f) => (!include || include.includes(f.path)) && !exclude.includes(f.path) && (!q || `${f.label} ${f.description ?? ''} ${f.path}`.toLowerCase().includes(q))
      )
    }))
    .filter((g) => g.fields.length)

  if (!groups.length) {
    return q ? <p className="px-1 text-sm text-fg-subtle">No matching settings here.</p> : null
  }

  return (
    <div className="space-y-6">
      {groups.map((g, i) => (
        <FormSection key={g.path} title={i === 0 ? (title ?? g.title) : g.title} description={i === 0 ? (description ?? undefined) : g.description}>
          {g.fields.map((f) => (
            <SchemaField key={f.path} field={f} value={getPath(editor.config, f.path)} error={editor.errors[f.path]} onChange={editor.setValue} />
          ))}
        </FormSection>
      ))}
      {footer}
    </div>
  )
}

function SchemaField({
  field,
  value,
  error,
  onChange
}: {
  field: FieldDef
  value: unknown
  error?: string
  onChange: (path: string, value: unknown, opts?: { immediate?: boolean }) => void
}) {
  const { schema: s, meta, path } = field
  const id = `cfg-${path.replace(/\./g, '-')}`
  let control: ReactNode = null
  let stack = false

  if (s.type === 'boolean') {
    control = <Switch id={id} checked={!!value} onCheckedChange={(v) => onChange(path, v, { immediate: true })} />
  } else if (s.enum) {
    const options = s.enum.map((v) => ({ value: String(v), label: meta.options?.[String(v)] ?? humanize(String(v)) }))
    const short = options.length <= 3 && options.every((o) => String(o.label).length <= 8)
    control = short ? (
      <SegmentedControl size="sm" value={String(value)} onChange={(v) => onChange(path, v, { immediate: true })} options={options} />
    ) : (
      <Select id={id} value={String(value)} onValueChange={(v) => onChange(path, v, { immediate: true })} options={options} className="w-60" />
    )
  } else if (s.type === 'number' && s.minimum !== undefined && s.maximum !== undefined) {
    const v = typeof value === 'number' ? value : Number(s.default ?? s.minimum)
    const range = s.maximum - s.minimum
    control = (
      <div className="flex w-64 items-center gap-3">
        <Slider value={v} min={s.minimum} max={s.maximum} step={range <= 1 ? 0.01 : 0.05} onValueChange={(n) => onChange(path, Math.round(n * 100) / 100)} aria-label={field.label} />
        <span className="w-12 shrink-0 text-right text-xs tabular-nums text-fg-muted">
          {meta.percent ? `${Math.round(v * 100)}%` : `${v.toFixed(2).replace(/\.?0+$/, '')}${meta.unit ?? ''}`}
        </span>
      </div>
    )
  } else if (s.type === 'integer' || s.type === 'number') {
    control = <NumberInput id={id} value={value as number} schema={s} unit={meta.unit} onCommit={(n) => onChange(path, n)} invalid={!!error} />
  } else if (s.type === 'array' && (s.items?.type === 'string' || !s.items)) {
    stack = true
    control = <TagInput value={Array.isArray(value) ? (value as string[]) : []} onChange={(v) => onChange(path, v, { immediate: true })} placeholder={meta.placeholder} />
  } else if (s.type === 'string' && meta.inputType === 'time') {
    control = <Input id={id} type="time" value={(value as string | null) ?? ''} invalid={!!error} onChange={(e) => e.target.value && onChange(path, e.target.value)} className="w-32" />
  } else if (s.type === 'string' || s.nullable) {
    control = (
      <Input
        id={id}
        value={(value as string | null) ?? ''}
        placeholder={meta.placeholder}
        invalid={!!error}
        onChange={(e) => onChange(path, s.nullable && !e.target.value ? null : e.target.value)}
        className="w-64"
      />
    )
  } else {
    return null
  }

  return (
    <FormRow id={`row-${path}`} label={field.label} description={field.description} error={error} htmlFor={id} stack={stack}>
      {control}
    </FormRow>
  )
}

function NumberInput({
  id,
  value,
  schema,
  unit,
  onCommit,
  invalid
}: {
  id: string
  value: number | undefined
  schema: JsonSchema
  unit?: string
  onCommit: (n: number) => void
  invalid?: boolean
}) {
  const [text, setText] = useState(value === undefined || value === null ? '' : String(value))
  useEffect(() => {
    setText(value === undefined || value === null ? '' : String(value))
  }, [value])
  const integer = schema.type === 'integer'
  const outOfRange = (n: number) => (schema.minimum !== undefined && n < schema.minimum) || (schema.maximum !== undefined && n > schema.maximum)
  const parsed = Number(text)
  const bad = text.trim() === '' || Number.isNaN(parsed) || outOfRange(parsed) || (integer && !Number.isInteger(parsed))

  return (
    <div className="flex items-center gap-2">
      <Input
        id={id}
        type="number"
        inputMode={integer ? 'numeric' : 'decimal'}
        value={text}
        min={schema.minimum}
        max={schema.maximum}
        step={integer ? 1 : 'any'}
        invalid={invalid || bad}
        onChange={(e) => {
          setText(e.target.value)
          const n = Number(e.target.value)
          if (e.target.value.trim() !== '' && !Number.isNaN(n) && !outOfRange(n) && (!integer || Number.isInteger(n))) onCommit(n)
        }}
        className="w-24 text-right tabular-nums"
      />
      {unit && <span className="min-w-6 text-xs text-fg-subtle">{unit}</span>}
      {bad && (schema.minimum !== undefined || schema.maximum !== undefined) && (
        <span className="sr-only">
          Between {schema.minimum ?? '-∞'} and {schema.maximum ?? '∞'}
        </span>
      )}
    </div>
  )
}

export function TagInput({ value, onChange, placeholder }: { value: string[]; onChange: (v: string[]) => void; placeholder?: string }) {
  const [draft, setDraft] = useState('')
  const add = () => {
    const t = draft.trim()
    if (t && !value.includes(t)) onChange([...value, t])
    setDraft('')
  }
  return (
    <div className={cn('flex w-full flex-wrap items-center gap-1.5 rounded-lg border border-border bg-field p-1.5 focus-within:border-accent/60 focus-within:ring-3 focus-within:ring-accent/15')}>
      {value.map((v) => (
        <span key={v} className="flex h-6 items-center gap-1 rounded-md bg-active pl-2 pr-1 font-mono text-xs text-fg">
          {v}
          <button type="button" aria-label={`Remove ${v}`} onClick={() => onChange(value.filter((x) => x !== v))} className="rounded p-0.5 text-fg-subtle hover:text-fg">
            <IconX size={11} />
          </button>
        </span>
      ))}
      <input
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ',') {
            e.preventDefault()
            add()
          } else if (e.key === 'Backspace' && !draft && value.length) onChange(value.slice(0, -1))
        }}
        onBlur={add}
        placeholder={value.length ? '' : placeholder}
        className="h-6 min-w-40 flex-1 bg-transparent px-1 text-sm text-fg outline-none placeholder:text-fg-subtle"
      />
    </div>
  )
}
