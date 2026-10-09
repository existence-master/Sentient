import {
  IconAlertCircle,
  IconBulb,
  IconCheck,
  IconChevronRight,
  IconClockHour4,
  IconCopy,
  IconCornerDownRight,
  IconPlayerStop,
  IconRefresh
} from '@tabler/icons-react'
import { ApprovalCard } from './ApprovalCard'
import { BrowserCard, isBrowserTool } from './cards/BrowserCard'
import { AnimatePresence, motion } from 'motion/react'
import { memo, useEffect, useState } from 'react'
import { Logo } from '@/components/brand/Logo'
import { Button, IconButton, Markdown, Tooltip } from '@/components/ui'
import { turnText, type AssistantTurnView, type TurnSegment } from '@/lib/chatFold'
import { modelShortName } from '@/lib/models'
import type { ApprovalDecision } from '@/lib/types'
import { cn, copyText, formatNumber } from '@/lib/utils'
import { MemorySources } from './MemorySources'
import { ToolCallCard } from './ToolCallCard'

export interface AssistantTurnProps {
  turn: AssistantTurnView
  assistantName: string
  showThinking: boolean
  isLast: boolean
  onRetry?: () => void
  onApprove?: (approvalId: string, decision: ApprovalDecision) => void
  /** Queued `chat.steer` messages (live turn only). */
  steers?: Array<{ id: string; text: string }>
}

type Block =
  | { kind: 'thinking'; text: string; index: number }
  | { kind: 'text'; text: string; index: number }
  | { kind: 'tools'; callIds: string[]; index: number }
  | { kind: 'interjection'; text: string; index: number }

function toBlocks(segments: TurnSegment[]): Block[] {
  const out: Block[] = []
  segments.forEach((s, index) => {
    if (s.kind === 'tool') {
      const last = out[out.length - 1]
      if (last?.kind === 'tools') last.callIds.push(s.callId)
      else out.push({ kind: 'tools', callIds: [s.callId], index })
    } else if (s.kind === 'thinking') out.push({ kind: 'thinking', text: s.text, index })
    else if (s.kind === 'interjection') out.push({ kind: 'interjection', text: s.text, index })
    else out.push({ kind: 'text', text: s.text, index })
  })
  return out
}

/** Consecutive browser calls collapse into one browser card; everything else stays one card per call. */
function groupTools(callIds: string[], turn: AssistantTurnView): Array<{ browser: boolean; ids: string[] }> {
  const groups: Array<{ browser: boolean; ids: string[] }> = []
  for (const id of callIds) {
    const browser = isBrowserTool(turn.tools[id]?.name ?? '')
    const last = groups[groups.length - 1]
    if (browser && last?.browser) last.ids.push(id)
    else groups.push({ browser, ids: [id] })
  }
  return groups
}

/** Something the user added while the reply was running; `pending` until the model has seen it. */
export function Interjection({ text, pending }: { text: string; pending?: boolean }) {
  return (
    <motion.div initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} className="flex flex-col items-end gap-1">
      <div className="flex items-center gap-1 text-2xs text-fg-subtle">
        {pending ? <IconClockHour4 size={11} /> : <IconCornerDownRight size={11} />}
        {pending ? 'Sentient will see this next' : 'You added'}
      </div>
      <div
        className={cn(
          'selectable max-w-[85%] whitespace-pre-wrap break-words rounded-2xl rounded-br-md border px-3.5 py-2 text-sm leading-relaxed',
          pending ? 'border-dashed border-border-strong text-fg-muted' : 'border-border bg-elevated text-fg shadow-soft'
        )}
      >
        {text}
      </div>
    </motion.div>
  )
}

export const AssistantTurn = memo(function AssistantTurn({ turn, assistantName, showThinking, isLast, onRetry, onApprove, steers }: AssistantTurnProps) {
  const streaming = turn.status === 'streaming'
  const blocks = toBlocks(turn.segments)
  const lastBlock = blocks[blocks.length - 1]
  const textOut = turnText(turn)
  const [copied, setCopied] = useState(false)
  const visibleBlocks = showThinking ? blocks : blocks.filter((b) => b.kind !== 'thinking')
  const thinkingNow = streaming && lastBlock?.kind === 'thinking'
  const waiting = streaming && (!blocks.length || (!showThinking && thinkingNow))

  return (
    <div className="group/turn flex gap-3.5">
      <div className="pt-0.5">
        <Logo size={24} />
      </div>
      <div className="min-w-0 flex-1 space-y-3">
        <div className="flex h-6 items-center gap-2 text-sm font-medium text-fg">
          {assistantName}
          {turn.status === 'cancelled' && (
            <span className="flex items-center gap-1 text-xs font-normal text-fg-subtle">
              <IconPlayerStop size={11} /> Stopped
            </span>
          )}
        </div>

        {visibleBlocks.map((b) => {
          if (b.kind === 'thinking') {
            return <ThinkingBlock key={`t${b.index}`} text={b.text} active={streaming && b === lastBlock} />
          }
          if (b.kind === 'text') {
            return <Markdown key={`m${b.index}`} streaming={streaming && b === lastBlock}>{b.text}</Markdown>
          }
          if (b.kind === 'interjection') {
            return <Interjection key={`i${b.index}`} text={b.text} />
          }
          return (
            <div key={`c${b.index}`} className="space-y-1.5">
              {groupTools(b.callIds, turn).map((g) => {
                if (g.browser) {
                  const tools = g.ids.map((id) => turn.tools[id]).filter(Boolean)
                  const approvals = turn.approvals.filter((a) => g.ids.includes(a.callId))
                  return (
                    <div key={g.ids[0]} className="space-y-2">
                      <BrowserCard tools={tools} approvals={approvals} />
                      {approvals.map((a) => (
                        <ApprovalCard key={a.approvalId} approval={a} onRespond={onApprove} />
                      ))}
                    </div>
                  )
                }
                const tool = turn.tools[g.ids[0]]
                if (!tool) return null
                return <ToolCallCard key={tool.callId} tool={tool} approval={turn.approvals.find((a) => a.callId === tool.callId)} onApprove={onApprove} />
              })}
            </div>
          )
        })}

        {streaming && steers?.map((s) => <Interjection key={s.id} text={s.text} pending />)}

        {waiting && <TypingIndicator label={thinkingNow ? 'Thinking' : 'Working on it'} />}
        {streaming && lastBlock?.kind === 'tools' && Object.values(turn.tools).every((t) => t.status !== 'running' && t.status !== 'awaiting_approval') && (
          <TypingIndicator label="Reading the results" />
        )}

        {turn.error && turn.status !== 'cancelled' && (
          <div className="flex items-start gap-3 rounded-xl border border-danger/25 bg-danger/[0.07] px-3.5 py-3 text-sm">
            <IconAlertCircle size={17} className="mt-0.5 shrink-0 text-danger" />
            <div className="min-w-0 flex-1">
              <div className="font-medium text-fg">Something went wrong</div>
              <div className="selectable mt-0.5 break-words text-fg-muted">{turn.error.message}</div>
            </div>
            {isLast && onRetry && (
              <Button size="sm" variant="secondary" leftIcon={<IconRefresh size={14} />} onClick={onRetry}>
                Retry
              </Button>
            )}
          </div>
        )}

        {!streaming && !!turn.memorySources?.length && <MemorySources sources={turn.memorySources} assistantName={assistantName} />}

        {!streaming && (textOut || isLast) && (
          <div className={cn('-ml-1.5 flex h-7 items-center gap-0.5 transition-opacity', isLast ? 'opacity-100' : 'opacity-0 group-hover/turn:opacity-100')}>
            {textOut && (
              <IconButton
                size="sm"
                label={copied ? 'Copied' : 'Copy'}
                icon={copied ? <IconCheck size={14} /> : <IconCopy size={14} />}
                onClick={async () => {
                  if (await copyText(textOut)) {
                    setCopied(true)
                    setTimeout(() => setCopied(false), 1400)
                  }
                }}
              />
            )}
            {isLast && onRetry && <IconButton size="sm" label="Retry" icon={<IconRefresh size={14} />} onClick={onRetry} />}
            {turn.usage && (
              <Tooltip content={`${turn.usage.model} · ${turn.usage.prompt_tokens} in / ${turn.usage.completion_tokens} out`}>
                <span className="ml-1.5 cursor-default text-2xs text-fg-faint">
                  {modelShortName(turn.usage.model)} · {formatNumber(turn.usage.prompt_tokens + turn.usage.completion_tokens)} tokens
                </span>
              </Tooltip>
            )}
          </div>
        )}
      </div>
    </div>
  )
})

function ThinkingBlock({ text, active }: { text: string; active: boolean }) {
  const [open, setOpen] = useState(active)
  useEffect(() => {
    if (!active) setOpen(false)
  }, [active])
  const words = text.trim().split(/\s+/).filter(Boolean).length

  return (
    <div className="rounded-xl border border-border bg-sunken/35">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex h-8.5 w-full items-center gap-2 px-3 text-left text-sm text-fg-muted hover:text-fg"
      >
        <IconBulb size={15} className={cn(active ? 'text-accent-text' : 'text-fg-subtle')} />
        {active ? <ShimmerText>Thinking</ShimmerText> : <span>Thought process</span>}
        {!active && <span className="text-2xs text-fg-faint">{words} words</span>}
        <span className="flex-1" />
        <IconChevronRight size={14} className={cn('text-fg-faint transition-transform', open && 'rotate-90')} />
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
            <div className="selectable max-h-72 overflow-y-auto whitespace-pre-wrap border-t border-border px-3.5 py-2.5 text-[13px] leading-relaxed text-fg-subtle">
              {text.trim()}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

function ShimmerText({ children }: { children: string }) {
  return (
    <span
      className="animate-shimmer bg-clip-text font-medium text-transparent"
      style={{
        backgroundImage: 'linear-gradient(90deg, var(--fg-subtle) 30%, var(--fg) 50%, var(--fg-subtle) 70%)',
        backgroundSize: '200% 100%'
      }}
    >
      {children}
    </span>
  )
}

function TypingIndicator({ label }: { label: string }) {
  return (
    <div className="flex h-6 items-center gap-2 text-sm">
      <span className="flex gap-1">
        {[0, 1, 2].map((i) => (
          <motion.span
            key={i}
            className="size-1.5 rounded-full bg-accent"
            animate={{ opacity: [0.25, 1, 0.25], y: [0, -2, 0] }}
            transition={{ duration: 1.1, repeat: Infinity, delay: i * 0.15 }}
          />
        ))}
      </span>
      <ShimmerText>{label}</ShimmerText>
    </div>
  )
}
