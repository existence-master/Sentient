import { IconAlertTriangle, IconFileCode, IconLink, IconShieldCheck, IconShieldExclamation, IconShieldSearch } from '@tabler/icons-react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { toast } from 'sonner'
import { Badge, Button, Dialog, Field, Input } from '@/components/ui'
import { qk } from '@/hooks/queryKeys'
import { api, errorMessage, isApiError } from '@/lib/api'
import type { SkillVetting } from '@/lib/types'
import { cn, relativeTime } from '@/lib/utils'

const SOURCE: Record<string, string> = { link: 'Added from a link', hermes: 'Brought over from Hermes' }

export function vetSourceLabel(v: SkillVetting): string {
  return SOURCE[v.source ?? ''] ?? 'Imported'
}

/** The safety check of a skill from outside (#146): the verdict in one sentence, where it came from, and what was
 * found in which file. */
export function SafetyReport({ vetting: v, className }: { vetting: SkillVetting; className?: string }) {
  const tone = v.verdict === 'high' ? 'danger' : v.verdict === 'check' ? 'warning' : 'success'
  const Icon = v.verdict === 'high' ? IconShieldExclamation : v.verdict === 'check' ? IconShieldSearch : IconShieldCheck
  const files = [...new Set(v.findings.map((f) => f.file).filter(Boolean))] as string[]
  return (
    <div
      className={cn(
        'rounded-xl border px-4 py-3',
        tone === 'danger' ? 'border-danger/30 bg-danger/[0.06]' : tone === 'warning' ? 'border-warning/30 bg-warning/[0.06]' : 'border-success/25 bg-success/[0.05]',
        className
      )}
    >
      <div className="flex items-start gap-2.5">
        <Icon size={18} className={cn('mt-0.5 shrink-0', tone === 'danger' ? 'text-danger' : tone === 'warning' ? 'text-warning' : 'text-success')} />
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-sm font-semibold text-fg">Safety check</span>
            <Badge size="xs" tone={tone}>
              {v.verdict === 'high' ? 'High risk' : v.verdict === 'check' ? 'Check this' : 'Nothing found'}
            </Badge>
          </div>
          <p className="mt-0.5 text-sm text-fg">{v.summary}</p>
          <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-0.5 text-xs text-fg-subtle">
            <span className="inline-flex min-w-0 items-center gap-1">
              <IconLink size={12} className="shrink-0" />
              {v.origin ? `${vetSourceLabel(v)}:` : vetSourceLabel(v)}
              {v.origin && (
                <span className="truncate font-mono" title={v.origin}>
                  {v.origin}
                </span>
              )}
            </span>
            {v.imported_at && <span>{relativeTime(v.imported_at)}</span>}
            <span>
              {v.files.length} file{v.files.length === 1 ? '' : 's'}
            </span>
          </div>
        </div>
      </div>
      {v.findings.length > 0 && (
        <div className="mt-2.5 space-y-2 pl-7">
          {(files.length ? files : ['']).map((file) => (
            <div key={file}>
              {files.length > 1 && (
                <div className="mb-0.5 inline-flex items-center gap-1 font-mono text-2xs text-fg-subtle">
                  <IconFileCode size={11} /> {file}
                </div>
              )}
              <ul className="space-y-1">
                {v.findings
                  .filter((f) => !file || f.file === file)
                  .map((f, i) => (
                    <li key={i} className="flex items-start gap-2 text-sm text-fg">
                      <span className={cn('mt-1.5 size-1.5 shrink-0 rounded-full', f.level === 'error' ? 'bg-danger' : 'bg-warning')} />
                      <span>{f.message}</span>
                    </li>
                  ))}
              </ul>
            </div>
          ))}
        </div>
      )}
      {v.changed_since_approval && (
        <p className="mt-2 flex items-center gap-1.5 pl-7 text-xs font-medium text-warning">
          <IconAlertTriangle size={13} /> Its files changed after you approved it.
        </p>
      )}
    </div>
  )
}

/** "Add from a link": a SKILL.md address, or a GitHub page, folder or repository that holds one. */
export function ImportSkillDialog({ open, onOpenChange, onImported }: { open: boolean; onOpenChange: (o: boolean) => void; onImported: (name: string) => void }) {
  const qc = useQueryClient()
  const [url, setUrl] = useState('')
  const [error, setError] = useState('')
  const run = useMutation({
    mutationFn: () => api.skills.importUrl(url.trim()),
    onSuccess: (s) => {
      void qc.invalidateQueries({ queryKey: qk.skills.all })
      const high = s.vetting?.verdict === 'high'
      ;(high ? toast.warning : toast.success)(high ? `${s.name} has high-risk problems` : `${s.name} is waiting for your review`, {
        description: high ? 'Read the safety check before you decide. We suggest you decline it.' : 'Nothing is used until you approve it.'
      })
      close(false)
      onImported(s.name)
    },
    onError: (e) => (isApiError(e) && e.status === 400 ? setError(e.detail) : toast.error("Couldn't add the skill", { description: errorMessage(e) }))
  })
  const close = (o: boolean) => {
    onOpenChange(o)
    if (!o) {
      setUrl('')
      setError('')
    }
  }
  return (
    <Dialog
      open={open}
      onOpenChange={close}
      title="Add a skill from a link"
      description="Sentient downloads the SKILL.md, runs a safety check and keeps it waiting for your review. Nothing is used until you approve it."
      footer={
        <>
          <Button variant="ghost" onClick={() => close(false)}>
            Cancel
          </Button>
          <Button variant="primary" leftIcon={<IconLink size={15} />} disabled={!/^https?:\/\/\S+$/.test(url.trim())} loading={run.isPending} onClick={() => run.mutate()}>
            Check and add
          </Button>
        </>
      }
    >
      <Field label="Web address" htmlFor="skill-url" error={error || undefined} description="A link to a SKILL.md, or to a GitHub folder or repository that has one.">
        <Input
          id="skill-url"
          autoFocus
          placeholder="https://github.com/…/SKILL.md"
          value={url}
          onChange={(e) => {
            setUrl(e.target.value)
            setError('')
          }}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && /^https?:\/\/\S+$/.test(url.trim())) run.mutate()
          }}
        />
      </Field>
    </Dialog>
  )
}
