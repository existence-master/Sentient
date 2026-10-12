import { IconAlertTriangle, IconShieldExclamation } from '@tabler/icons-react'
import type { SkillLintFinding } from '@/lib/types'
import { cn } from '@/lib/utils'

/** Problems first, then suggestions. */
function sorted(findings: SkillLintFinding[]) {
  return [...findings].sort((a, b) => Number(b.level === 'error') - Number(a.level === 'error'))
}

export function lintSummary(findings: SkillLintFinding[] | undefined): { errors: number; warnings: number } {
  const list = findings ?? []
  return { errors: list.filter((f) => f.level === 'error').length, warnings: list.filter((f) => f.level !== 'error').length }
}

/** What the skill checker found, in plain words. Renders nothing when the skill is fine. */
export function LintList({ findings, title, compact, className }: { findings: SkillLintFinding[] | undefined; title?: string; compact?: boolean; className?: string }) {
  if (!findings?.length) return null
  const { errors } = lintSummary(findings)
  const heading = title ?? (errors ? 'Check this skill before Sentient relies on it' : 'Suggestions for this skill')
  if (compact) {
    return (
      <ul className={cn('space-y-0.5', className)}>
        {sorted(findings).map((f, i) => (
          <li key={i} className={cn('flex items-start gap-1.5 text-xs', f.level === 'error' ? 'text-danger' : 'text-warning')}>
            <IconAlertTriangle size={12} className="mt-0.5 shrink-0" />
            <span>{f.message}</span>
          </li>
        ))}
      </ul>
    )
  }
  return (
    <div className={cn('rounded-xl border px-4 py-3', errors ? 'border-danger/30 bg-danger/[0.06]' : 'border-warning/30 bg-warning/[0.06]', className)}>
      <div className={cn('flex items-center gap-2 text-sm font-medium', errors ? 'text-danger' : 'text-warning')}>
        {errors ? <IconShieldExclamation size={16} /> : <IconAlertTriangle size={16} />}
        {heading}
      </div>
      <ul className="mt-2 space-y-1.5">
        {sorted(findings).map((f, i) => (
          <li key={i} className="flex items-start gap-2 text-sm text-fg">
            <span className={cn('mt-1.5 size-1.5 shrink-0 rounded-full', f.level === 'error' ? 'bg-danger' : 'bg-warning')} />
            <span>{f.message}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}
