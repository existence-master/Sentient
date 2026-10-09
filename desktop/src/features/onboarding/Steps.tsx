import {
  IconArrowRight,
  IconBolt,
  IconBrain,
  IconBrandGithub,
  IconBrandNotion,
  IconBrandSlack,
  IconBrandWhatsapp,
  IconBriefcase,
  IconCalendar,
  IconCheck,
  IconHeart,
  IconListCheck,
  IconLock,
  IconMail,
  IconMapPin,
  IconMoodSmile,
  IconPlugConnected,
  IconSunrise,
  IconTrophy,
  type Icon
} from '@tabler/icons-react'
import { motion } from 'motion/react'
import type { ReactNode } from 'react'
import { useNavigate } from 'react-router'
import { Logo } from '@/components/brand/Logo'
import { Button, Combobox, Field, Input, Skeleton, Switch, Textarea } from '@/components/ui'
import { SUGGESTIONS } from '@/features/chat/EmptyChat'
import { usePersonas } from '@/hooks/memory'
import { cn, listTimezones } from '@/lib/utils'
import { useOnboardingDraft } from './draft'

export function StepHeader({ title, subtitle, eyebrow }: { title: ReactNode; subtitle?: ReactNode; eyebrow?: ReactNode }) {
  return (
    <div className="mb-8">
      {eyebrow && <div className="mb-2 text-xs font-medium uppercase tracking-wider text-accent-text">{eyebrow}</div>}
      <h1 className="text-[26px] font-semibold leading-tight tracking-tight text-fg">{title}</h1>
      {subtitle && <p className="mt-2 text-md leading-relaxed text-fg-muted">{subtitle}</p>}
    </div>
  )
}

// ---------------------------------------------------------------------------- welcome
export function WelcomeStep({ onStart }: { onStart: () => void }) {
  const values: Array<{ icon: Icon; title: string; text: string }> = [
    { icon: IconBrain, title: 'Remembers', text: 'Learns what matters to you over time.' },
    { icon: IconListCheck, title: 'Works for you', text: 'Runs tasks in the background, on a schedule.' },
    { icon: IconLock, title: 'Yours', text: 'Runs on this computer. No account needed.' }
  ]
  return (
    <div className="flex flex-1 flex-col items-center justify-center py-8 text-center">
      <motion.div initial={{ scale: 0.85, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ duration: 0.5 }}>
        <Logo size={104} animated />
      </motion.div>
      <motion.h1 initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }} className="mt-9 text-[34px] font-semibold tracking-tight text-fg">
        Hi, I&apos;m Sentient.
      </motion.h1>
      <motion.p initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.22 }} className="mt-3 max-w-md text-md leading-relaxed text-fg-muted">
        A personal assistant that lives on your computer. Let&apos;s get to know each other so I can actually be useful.
      </motion.p>
      <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }} className="mt-10 grid w-full grid-cols-3 gap-3">
        {values.map((v) => (
          <div key={v.title} className="rounded-xl border border-border bg-surface/70 p-4 text-left">
            <v.icon size={18} className="text-accent-text" />
            <div className="mt-2.5 text-sm font-medium text-fg">{v.title}</div>
            <div className="mt-0.5 text-xs leading-relaxed text-fg-subtle">{v.text}</div>
          </div>
        ))}
      </motion.div>
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.4 }} className="mt-10 flex flex-col items-center gap-3">
        <Button autoFocus variant="primary" size="lg" rightIcon={<IconArrowRight size={16} />} onClick={onStart} className="px-7">
          Get started
        </Button>
        <span className="text-xs text-fg-subtle">Takes about two minutes</span>
      </motion.div>
    </div>
  )
}

// ---------------------------------------------------------------------------- about
export function AboutStep({ nameError, clearNameError }: { nameError: boolean; clearNameError: () => void }) {
  const d = useOnboardingDraft()
  const zones = listTimezones()
  return (
    <div>
      <StepHeader title="First, what should I call you?" subtitle="A few basics so replies, reminders and schedules feel right." />
      <div className="space-y-5">
        <Field label="Your name" htmlFor="ob-name" error={nameError ? 'Tell me what to call you' : undefined}>
          <Input
            id="ob-name"
            size="lg"
            autoFocus
            value={d.user_name}
            invalid={nameError}
            placeholder="e.g. Alex"
            onChange={(e) => {
              d.set({ user_name: e.target.value })
              if (e.target.value.trim()) clearNameError()
            }}
          />
        </Field>
        <Field label="What should I call myself?" htmlFor="ob-assistant" description="You can rename me any time.">
          <Input id="ob-assistant" value={d.assistant_name} placeholder="Sentient" onChange={(e) => d.set({ assistant_name: e.target.value })} />
        </Field>
        <div className="grid grid-cols-2 gap-4">
          <Field label="Timezone" description="Detected from this computer.">
            <Combobox
              value={d.timezone}
              onChange={(v) => d.set({ timezone: v })}
              groups={[{ id: 'tz', label: 'Timezones', options: zones.map((z) => ({ value: z, label: z.replace(/_/g, ' ') })) }]}
              searchPlaceholder="Search timezones…"
              allowCustom={false}
            />
          </Field>
          <Field label="Location" htmlFor="ob-location" optional description="For weather and local context.">
            <Input id="ob-location" leftIcon={<IconMapPin />} value={d.location} placeholder="City, Country" onChange={(e) => d.set({ location: e.target.value })} />
          </Field>
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------- context
export function ContextStep() {
  const d = useOnboardingDraft()
  const first = d.user_name.trim().split(/\s+/)[0]
  return (
    <div>
      <StepHeader
        title={first ? `Tell me a bit about your world, ${first}` : 'Tell me a bit about your world'}
        subtitle="Optional, but it makes a big difference. A couple of sentences is plenty."
      />
      <div className="space-y-5">
        <Field label="Work" htmlFor="ob-work" optional>
          <Textarea
            id="ob-work"
            autoFocus
            autoGrow
            minHeight={92}
            value={d.professional_context}
            onChange={(e) => d.set({ professional_context: e.target.value })}
            placeholder="I'm a product designer at a fintech startup. Most of my day is Figma, Slack and back-to-back meetings on Tuesdays."
          />
        </Field>
        <Field label="Life" htmlFor="ob-life" optional>
          <Textarea
            id="ob-life"
            autoGrow
            minHeight={92}
            value={d.personal_context}
            onChange={(e) => d.set({ personal_context: e.target.value })}
            placeholder="I'm training for a half marathon, learning Spanish, and my sister's birthday is in June."
          />
        </Field>
        <div className="flex gap-3 rounded-xl border border-border bg-surface/70 p-4">
          <IconBrain size={18} className="mt-0.5 shrink-0 text-accent-text" />
          <p className="text-sm leading-relaxed text-fg-muted">
            This becomes my memory. It&apos;s stored only on this computer, and you can see, edit or delete anything I remember from the Memory page.
          </p>
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------- personality
const PERSONA_ICON: Record<string, Icon> = { friendly: IconMoodSmile, professional: IconBriefcase, concise: IconBolt, coach: IconTrophy }

export function PersonalityStep() {
  const d = useOnboardingDraft()
  const personas = usePersonas()
  return (
    <div>
      <StepHeader title="How should I come across?" subtitle="Pick a starting personality. You can fine-tune every word of it later in Settings." />
      {personas.isLoading ? (
        <div className="grid grid-cols-2 gap-3">
          {[0, 1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-28 rounded-xl" />
          ))}
        </div>
      ) : (
        <div role="radiogroup" className="grid grid-cols-2 gap-3">
          {(personas.data ?? []).map((p) => {
            const active = d.persona === p.id
            const PIcon = PERSONA_ICON[p.id] ?? IconHeart
            return (
              <button
                key={p.id}
                type="button"
                role="radio"
                aria-checked={active}
                onClick={() => d.set({ persona: p.id })}
                className={cn(
                  'relative flex flex-col items-start rounded-xl border p-4 text-left transition-[border-color,background-color,box-shadow] duration-150',
                  active ? 'border-accent/60 bg-accent/[0.06] shadow-glow' : 'border-border bg-surface/70 hover:border-border-strong hover:bg-elevated'
                )}
              >
                <div className={cn('flex size-9 items-center justify-center rounded-lg', active ? 'bg-accent text-accent-fg' : 'bg-active text-fg-muted')}>
                  <PIcon size={18} />
                </div>
                <div className="mt-3 text-sm font-semibold text-fg">{p.name}</div>
                <div className="mt-1 text-xs leading-relaxed text-fg-subtle">{p.description}</div>
                {active && (
                  <span className="absolute right-3 top-3 flex size-5 items-center justify-center rounded-full bg-accent text-accent-fg">
                    <IconCheck size={12} stroke={3} />
                  </span>
                )}
              </button>
            )
          })}
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------- apps
const APPS: Array<{ icon: Icon; name: string }> = [
  { icon: IconMail, name: 'Gmail' },
  { icon: IconCalendar, name: 'Calendar' },
  { icon: IconBrandSlack, name: 'Slack' },
  { icon: IconBrandNotion, name: 'Notion' },
  { icon: IconBrandGithub, name: 'GitHub' },
  { icon: IconBrandWhatsapp, name: 'WhatsApp' }
]

export function AppsStep() {
  const d = useOnboardingDraft()
  return (
    <div>
      <StepHeader
        title="Bring your apps along"
        subtitle="With your inbox and calendar connected, I can draft replies, prepare you for meetings and suggest things before you ask."
      />
      <div className="grid grid-cols-3 gap-3">
        {APPS.map((a, i) => (
          <motion.div
            key={a.name}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.04 }}
            className="flex flex-col items-center gap-2.5 rounded-xl border border-border bg-surface/70 px-3 py-5"
          >
            <a.icon size={24} className="text-fg-muted" />
            <span className="text-sm text-fg">{a.name}</span>
          </motion.div>
        ))}
      </div>
      <div className="mt-5 flex gap-3 rounded-xl border border-border bg-surface/70 p-4">
        <IconPlugConnected size={18} className="mt-0.5 shrink-0 text-accent-text" />
        <p className="text-sm leading-relaxed text-fg-muted">
          You can connect apps any time from <span className="font-medium text-fg">Integrations</span> in the sidebar. Sign-in happens in your browser and keys stay in your system keychain.
        </p>
      </div>
      <label className="mt-3 flex cursor-pointer items-center gap-3 rounded-xl border border-border bg-surface/70 p-4">
        <IconSunrise size={18} className="shrink-0 text-accent-text" />
        <span className="min-w-0 flex-1">
          <span className="block text-sm font-medium text-fg">Daily Brief</span>
          <span className="block text-sm text-fg-muted">Weekday mornings at 07:30: today&apos;s meetings, emails that need you, tasks and the weather in a few lines.</span>
        </span>
        <Switch checked={d.daily_brief} onCheckedChange={(v) => d.set({ daily_brief: v })} aria-label="Send me a Daily Brief" />
      </label>
    </div>
  )
}

// ---------------------------------------------------------------------------- done
export function DoneStep() {
  const d = useOnboardingDraft()
  const navigate = useNavigate()
  const first = d.user_name.trim().split(/\s+/)[0]
  const start = (prompt?: { text: string; autoSend?: boolean }) => {
    d.reset()
    navigate('/chat', { replace: true, state: { fresh: Date.now(), prompt } })
  }
  return (
    <div className="flex flex-1 flex-col items-center justify-center py-8 text-center">
      <motion.div initial={{ scale: 0.6, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ type: 'spring', stiffness: 260, damping: 18 }} className="relative">
        <Logo size={96} animated />
        <span className="absolute -bottom-1 -right-1 flex size-8 items-center justify-center rounded-full border-4 border-bg bg-success text-white">
          <IconCheck size={16} stroke={3} />
        </span>
      </motion.div>
      <h1 className="mt-8 text-[30px] font-semibold tracking-tight text-fg">You&apos;re all set{first ? `, ${first}` : ''}.</h1>
      <p className="mt-2 max-w-md text-md text-fg-muted">I&apos;m ready when you are. Here are a few good ways to start:</p>
      <div className="mt-8 w-full max-w-md space-y-2">
        {SUGGESTIONS.map((s) => (
          <button
            key={s.title}
            type="button"
            onClick={() => start({ text: s.prompt, autoSend: s.autoSend })}
            className="group flex w-full items-center gap-3 rounded-xl border border-border bg-surface/70 px-4 py-3 text-left transition-colors hover:border-border-strong hover:bg-elevated"
          >
            <s.icon size={17} className="shrink-0 text-fg-subtle group-hover:text-accent-text" />
            <span className="flex-1 text-sm text-fg">{s.title}</span>
            <IconArrowRight size={15} className="text-fg-faint transition-transform group-hover:translate-x-0.5 group-hover:text-fg-muted" />
          </button>
        ))}
      </div>
      <Button autoFocus variant="primary" size="lg" className="mt-8 px-7" rightIcon={<IconArrowRight size={16} />} onClick={() => start()}>
        Start chatting
      </Button>
    </div>
  )
}
