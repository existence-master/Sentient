import { IconAlertCircle, IconCheck, IconSearch, IconX } from '@tabler/icons-react'
import { AnimatePresence, motion } from 'motion/react'
import { useEffect, useMemo, useState, type ComponentType } from 'react'
import { NavLink, useParams } from 'react-router'
import { Input, ScrollArea, Spinner, Tooltip } from '@/components/ui'
import { useSaveStatus } from '@/hooks/config'
import { useConfigSchema } from '@/hooks/core'
import { cn } from '@/lib/utils'
import { schemaMatches } from './SchemaForm'
import { SETTINGS_SECTIONS, sectionById } from './sections'
import { AdvancedSection } from './sections/Advanced'
import { GeneralSection } from './sections/General'
import { ModelsSection } from './sections/Models'
import { PersonalitySection } from './sections/Personality'
import { ApprovalsSection, EvolutionSection, MemorySection, ProactivitySection, TasksSection, VoiceSection } from './sections/SchemaSections'
import { UsageSection } from './sections/Usage'
import { BrowserSection, KnowingSection, SandboxSection, SubagentsSection, TerminalSection } from './sections/AbilitySections'

export interface SectionProps {
  query: string
}

const SECTION_COMPONENTS: Record<string, ComponentType<SectionProps>> = {
  general: GeneralSection,
  models: ModelsSection,
  personality: PersonalitySection,
  memory: MemorySection,
  tasks: TasksSection,
  proactivity: ProactivitySection,
  voice: VoiceSection,
  approvals: ApprovalsSection,
  evolution: EvolutionSection,
  usage: UsageSection,
  knowing: KnowingSection,
  sandbox: SandboxSection,
  terminal: TerminalSection,
  browser: BrowserSection,
  subagents: SubagentsSection,
  advanced: AdvancedSection
}

export function SettingsPage() {
  const { section } = useParams()
  const active = sectionById(section) ?? SETTINGS_SECTIONS[0]
  const [query, setQuery] = useState('')
  const schema = useConfigSchema()
  const Body = SECTION_COMPONENTS[active.id]

  const visible = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return SETTINGS_SECTIONS
    return SETTINGS_SECTIONS.filter(
      (s) =>
        `${s.label} ${s.description} ${s.keywords.join(' ')}`.toLowerCase().includes(q) ||
        (s.schemaSections ?? []).some((sec) => schemaMatches(sec, schema.data, q))
    )
  }, [query, schema.data])

  return (
    <div className="flex h-full">
      <nav className="flex w-60 shrink-0 flex-col border-r border-border">
        <div className="px-3 pb-2 pt-4">
          <h1 className="px-1 pb-3 text-lg font-semibold tracking-tight text-fg">Settings</h1>
          <Input
            size="sm"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search settings"
            leftIcon={<IconSearch />}
            onKeyDown={(e) => e.key === 'Escape' && setQuery('')}
            rightSlot={
              query && (
                <button type="button" aria-label="Clear" onClick={() => setQuery('')} className="flex size-5 items-center justify-center rounded text-fg-subtle hover:text-fg">
                  <IconX size={12} />
                </button>
              )
            }
          />
        </div>
        <ScrollArea className="min-h-0 flex-1" viewportClassName="px-3 pb-4">
          <div className="space-y-0.5 pt-1">
            {visible.map((s) => (
              <NavLink
                key={s.id}
                to={`/settings/${s.id}`}
                className={cn(
                  'flex h-8.5 items-center gap-2.5 rounded-lg px-2.5 text-sm transition-colors',
                  s.id === active.id ? 'bg-active text-fg' : 'text-fg-muted hover:bg-hover hover:text-fg'
                )}
              >
                <s.icon size={16} stroke={1.75} className={s.id === active.id ? 'text-accent-text' : 'text-fg-subtle'} />
                {s.label}
              </NavLink>
            ))}
            {!visible.length && <p className="px-2.5 py-3 text-xs text-fg-subtle">No settings match “{query}”.</p>}
          </div>
        </ScrollArea>
      </nav>

      <div className="min-w-0 flex-1 overflow-y-auto" id="settings-scroll">
        <div className="mx-auto w-full max-w-[860px] px-10 pb-16 pt-7">
          <header className="mb-7 flex items-start gap-4">
            <div className="flex size-10 shrink-0 items-center justify-center rounded-xl border border-border bg-elevated text-accent-text shadow-soft">
              <active.icon size={20} />
            </div>
            <div className="min-w-0 flex-1">
              <h2 className="text-xl font-semibold tracking-tight text-fg">{active.label}</h2>
              <p className="mt-0.5 text-sm text-fg-muted">{active.description}</p>
            </div>
            <SaveIndicator />
          </header>
          <motion.div key={active.id} initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.18 }}>
            <Body query={query} />
          </motion.div>
        </div>
      </div>
    </div>
  )
}

export function SaveIndicator() {
  const status = useSaveStatus((s) => s.status)
  const error = useSaveStatus((s) => s.error)
  const savedAt = useSaveStatus((s) => s.savedAt)
  const [showSaved, setShowSaved] = useState(false)

  useEffect(() => {
    if (status !== 'saved') return
    setShowSaved(true)
    const t = setTimeout(() => setShowSaved(false), 2200)
    return () => clearTimeout(t)
  }, [status, savedAt])

  return (
    <div className="flex h-8 min-w-24 items-center justify-end">
      <AnimatePresence mode="wait">
        {status === 'saving' && (
          <motion.span key="saving" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex items-center gap-1.5 text-xs text-fg-subtle">
            <Spinner size={12} /> Saving…
          </motion.span>
        )}
        {status === 'saved' && showSaved && (
          <motion.span key="saved" initial={{ opacity: 0, y: 2 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="flex items-center gap-1.5 text-xs text-success">
            <IconCheck size={13} /> Saved
          </motion.span>
        )}
        {status === 'error' && (
          <motion.span key="error" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <Tooltip content={error}>
              <span className="flex cursor-default items-center gap-1.5 text-xs text-danger">
                <IconAlertCircle size={13} /> Couldn&apos;t save
              </span>
            </Tooltip>
          </motion.span>
        )}
      </AnimatePresence>
    </div>
  )
}
