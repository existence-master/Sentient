/**
 * "Connect Claude" (#444, ADR 0032): every way to use Claude in one card, in setup's Brain step and Settings > Models.
 * An API key, a Max or Team plan's API credits (also a key), or the user's own Claude Code for chats or for everything.
 * Sentient never offers a Claude sign-in and never reads a Claude login.
 */
import { IconCreditCard, IconKey, IconMessages, IconSparkles, IconTerminal2, type Icon } from '@tabler/icons-react'
import { useEffect, useRef, useState } from 'react'
import { toast } from 'sonner'
import { Badge, Card } from '@/components/ui'
import { ClaudeCodePanel } from '@/features/models/ClaudeCodeSection'
import { ClaudePlanGuide, useApplyPlan } from '@/features/models/ConnectPlans'
import { useConfig } from '@/hooks/core'
import { useSetRoles } from '@/hooks/models'
import { errorMessage } from '@/lib/api'
import { claudeWayFor, type ClaudeWay } from '@/lib/claudeCode'
import { modelShortName } from '@/lib/models'
import { cn } from '@/lib/utils'

/** The four choices, each with one plain line on what it covers and what it costs or risks. */
export const CLAUDE_WAYS: Array<{ value: ClaudeWay; title: string; line: string; icon: Icon; badge?: { label: string; tone: 'warning' | 'danger' } }> = [
  { value: 'key', title: 'API key', line: 'Claude for every job, billed per use to your Claude Console account.', icon: IconKey },
  {
    value: 'credits',
    title: "Use my Max or Team plan's credits",
    line: 'Claude for every job, paid from the API credits your plan includes each month. Your Claude app limits stay the same.',
    icon: IconCreditCard
  },
  {
    value: 'code_chats',
    title: 'Use my Claude Code for chats',
    line: "Only your chats, on your plan's limits. Anthropic may change how this is counted or allowed.",
    icon: IconMessages,
    badge: { label: 'Experimental', tone: 'warning' }
  },
  {
    value: 'code_everything',
    title: 'Use my Claude Code for everything',
    line: 'Chats, tasks, memory and background work on your plan. Uses your limits much faster and puts your Claude account at risk.',
    icon: IconTerminal2,
    badge: { label: 'Your risk', tone: 'danger' }
  }
]

export function ConnectClaude({
  primary,
  fast,
  others,
  onUseModels,
  onUseForChat,
  onUseForEverything
}: {
  primary: string
  fast: string
  /** The planner and executor models (empty follows the main model). */
  others?: Record<string, string | null | undefined>
  /** Claude through an API key (or the plan's credits) for the main and fast models. */
  onUseModels: (primary: string, fast: string) => void
  onUseForChat: (model: string) => void
  onUseForEverything: (primary: string, fast: string) => void
}) {
  const config = useConfig()
  const [way, setWay] = useState<ClaudeWay>(() => claudeWayFor(config.data?.models, primary))
  const picked = useRef(false)
  useEffect(() => {
    // once, when the saved setup is first known; later saves (a switch on this card) never move the choice
    if (picked.current || !config.data) return
    picked.current = true
    setWay(claudeWayFor(config.data.models, primary))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [config.data])
  const choose = (w: ClaudeWay) => {
    picked.current = true
    setWay(w)
  }

  return (
    <div className="space-y-4">
      <div role="radiogroup" aria-label="How to use Claude" className="grid grid-cols-2 gap-2">
        {CLAUDE_WAYS.map((w) => {
          const on = way === w.value
          return (
            <button
              key={w.value}
              type="button"
              role="radio"
              aria-checked={on}
              onClick={() => choose(w.value)}
              className={cn(
                'flex min-h-[84px] flex-col items-start gap-1 rounded-xl border px-3 py-2.5 text-left transition-colors',
                on ? 'border-accent/60 bg-accent/[0.06]' : 'border-border bg-surface/70 hover:border-border-strong'
              )}
            >
              <span className="flex w-full items-center gap-1.5 text-sm font-medium text-fg">
                <w.icon size={15} className={cn('shrink-0', on ? 'text-accent-text' : 'text-fg-subtle')} />
                <span className="min-w-0 flex-1">{w.title}</span>
                {w.badge && (
                  <Badge size="xs" tone={w.badge.tone}>
                    {w.badge.label}
                  </Badge>
                )}
              </span>
              <span className="text-xs leading-snug text-fg-subtle">{w.line}</span>
            </button>
          )
        })}
      </div>
      <div className="border-t border-border pt-4">
        {way === 'key' ? (
          <ClaudePlanGuide steps={false} onUseModels={onUseModels} />
        ) : way === 'credits' ? (
          <ClaudePlanGuide onUseModels={onUseModels} />
        ) : (
          <ClaudeCodePanel
            way={way === 'code_chats' ? 'chats' : 'everything'}
            primary={primary}
            fast={fast}
            others={others}
            onUseForChat={onUseForChat}
            onUseForEverything={onUseForEverything}
          />
        )}
      </div>
    </div>
  )
}

/** Settings > Models: the Connect Claude card, saving straight to the roles. */
export function ConnectClaudeSection() {
  const config = useConfig()
  const setRoles = useSetRoles()
  const applyPlan = useApplyPlan()
  const roles = config.data?.models.roles
  const done = (title: string, description?: string) => ({
    onSuccess: () => toast.success(title, { description }),
    onError: (e: unknown) => toast.error("Couldn't change the models", { description: errorMessage(e) })
  })
  return (
    <section className="space-y-3">
      <div className="px-1">
        <h3 className="flex items-center gap-1.5 text-sm font-semibold text-fg">
          <IconSparkles size={15} /> Connect Claude
        </h3>
        <p className="mt-0.5 text-xs text-fg-subtle">Every way to use Claude in Sentient. Keys stay in your system keychain, and Sentient never sees your Claude login.</p>
      </div>
      <Card className="p-4">
        <ConnectClaude
          primary={roles?.primary ?? ''}
          fast={roles?.fast ?? ''}
          others={{ planner: roles?.planner, executor: roles?.executor }}
          onUseModels={applyPlan('anthropic', 'Claude')}
          onUseForChat={(m) => setRoles.mutate({ primary: m }, done('Chat now uses Claude through Claude Code', modelShortName(m)))}
          onUseForEverything={(primary, fast) =>
            setRoles.mutate({ primary, fast, planner: null, executor: null }, done('Every job now uses your Claude Code', 'Memory search keeps its own model.'))
          }
        />
      </Card>
    </section>
  )
}
