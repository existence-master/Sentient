import { IconAlertTriangle, IconCircleCheck, IconClock, IconFileCheck, IconTerminal2 } from '@tabler/icons-react'
import type { Skill, SkillCheck, SkillCheckReport } from '@/lib/types'
import { cn } from '@/lib/utils'

/** True when approving runs the skill's code check (it waits for the user's yes, #386). */
export function checkWaits(skill: Skill): boolean {
  return skill.check?.status === 'waiting'
}

function headline(report: SkillCheckReport | null | undefined, hasCode: boolean): { tone: 'ok' | 'bad' | 'wait'; text: string } {
  if (!report) return { tone: 'wait', text: hasCode ? 'Sentient runs this check when you approve the skill.' : 'Sentient runs this check before it saves the skill.' }
  const failed = report.results.find((r) => r.ok === false)
  if (report.status === 'failed') return { tone: 'bad', text: `The check failed: ${failed?.detail ?? 'it did not pass'}. Approving runs it again.` }
  if (report.status === 'waiting') return { tone: 'wait', text: 'This check runs a little code, so Sentient runs it when you approve. It runs in the code sandbox, without your keys or passwords.' }
  if (report.status === 'not_run') return { tone: 'wait', text: report.results.find((r) => r.ok === null)?.detail ?? 'This check can’t run here.' }
  return { tone: 'ok', text: 'Checked: the skill works.' }
}

function CheckLine({ check }: { check: SkillCheck }) {
  if (check.file) {
    return (
      <li className="flex items-start gap-2 text-sm text-fg">
        <IconFileCheck size={15} className="mt-0.5 shrink-0 text-fg-subtle" />
        <span>
          The file <span className="font-mono text-[12px]">{check.file}</span> exists
          {check.expect ? <> and contains <span className="font-mono text-[12px]">“{check.expect}”</span></> : null}
        </span>
      </li>
    )
  }
  return (
    <li className="space-y-1.5 text-sm text-fg">
      <div className="flex items-start gap-2">
        <IconTerminal2 size={15} className="mt-0.5 shrink-0 text-fg-subtle" />
        <span>
          This code runs{check.expect ? <> and prints <span className="font-mono text-[12px]">“{check.expect}”</span></> : ' without an error'}
        </span>
      </div>
      <pre className="selectable ml-6 overflow-x-auto rounded-lg border border-border bg-sunken px-3 py-2 font-mono text-[12px] leading-relaxed text-fg">{check.run}</pre>
    </li>
  )
}

/** How Sentient checks a skill works (its `validate` header), and how the last check went. Renders nothing without checks. */
export function SkillChecks({ skill, className }: { skill: Skill; className?: string }) {
  const checks = skill.validate ?? []
  if (!checks.length) return null
  const { tone, text } = headline(skill.check, checks.some((c) => !!c.run))
  return (
    <div
      className={cn(
        'rounded-xl border px-4 py-3',
        tone === 'bad' ? 'border-danger/30 bg-danger/[0.06]' : tone === 'ok' ? 'border-success/30 bg-success/[0.06]' : 'border-border bg-sunken/50',
        className
      )}
    >
      <div className={cn('flex items-center gap-2 text-sm font-medium', tone === 'bad' ? 'text-danger' : tone === 'ok' ? 'text-success' : 'text-fg-muted')}>
        {tone === 'bad' ? <IconAlertTriangle size={16} /> : tone === 'ok' ? <IconCircleCheck size={16} /> : <IconClock size={16} />}
        How Sentient checks this skill
      </div>
      <p className="mt-1 text-sm text-fg">{text}</p>
      <ul className="mt-2 space-y-2">
        {checks.map((c, i) => (
          <CheckLine key={i} check={c} />
        ))}
      </ul>
    </div>
  )
}
