import { IconPlus, IconSparkles } from '@tabler/icons-react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { toast } from 'sonner'
import { Alert, Button, Dialog, Tooltip } from '@/components/ui'
import { qk } from '@/hooks/queryKeys'
import { api, errorMessage, isApiError } from '@/lib/api'
import { SkillEditor, splitList, type SkillDraft } from './SkillEditor'
import { SKILL_TEMPLATE, slugify, validSkillName } from './meta'

/** POST /api/skills/review-now. Calls the model, so it reports progress honestly. */
export function ReviewNowButton({ onProposed }: { onProposed?: (names: string[]) => void }) {
  const qc = useQueryClient()
  const [elapsed, setElapsed] = useState(0)
  const review = useMutation({
    mutationFn: () => api.skills.reviewNow(),
    onSuccess: (r) => {
      void qc.invalidateQueries({ queryKey: qk.skills.all })
      const n = typeof r.reviewed === 'number' ? r.reviewed : r.reviewed ? 1 : 0
      if (r.proposed.length) {
        toast.success(`Proposed ${r.proposed.length} ${r.proposed.length === 1 ? 'skill' : 'skills'} for review`, { description: r.proposed.join(', ') })
        onProposed?.(r.proposed)
      } else {
        toast(n ? `Reviewed ${n} ${n === 1 ? 'conversation' : 'conversations'}` : 'Nothing new to review', {
          description: n ? "Nothing looked worth saving as a skill this time." : 'Chats and task runs with several tool calls are reviewed once they go quiet.'
        })
      }
    },
    onError: (e) => toast.error('Review failed', { description: errorMessage(e) })
  })

  useEffect(() => {
    if (!review.isPending) return setElapsed(0)
    const started = Date.now()
    const t = window.setInterval(() => setElapsed(Math.round((Date.now() - started) / 1000)), 1000)
    return () => window.clearInterval(t)
  }, [review.isPending])

  return (
    <Tooltip content={review.isPending ? 'Your model is reading recent chats and task runs' : 'Ask Sentient to look through recent chats and task runs for procedures worth saving. Uses your fast model.'}>
      <Button variant="secondary" leftIcon={<IconSparkles size={15} />} loading={review.isPending} onClick={() => review.mutate()}>
        {review.isPending ? `Reviewing recent work… ${elapsed}s` : 'Review recent work'}
      </Button>
    </Tooltip>
  )
}

const empty: SkillDraft = { name: '', description: '', body: SKILL_TEMPLATE, tags: '', requires_tools: '' }

export function CreateSkillDialog({ open, onOpenChange, onCreated }: { open: boolean; onOpenChange: (o: boolean) => void; onCreated: (name: string) => void }) {
  const qc = useQueryClient()
  const [draft, setDraft] = useState<SkillDraft>(empty)
  const [touched, setTouched] = useState(false)
  const [serverError, setServerError] = useState('')
  const slug = slugify(draft.name)
  const nameError = touched && draft.name && !validSkillName(slug) ? 'Use 2-64 lowercase letters, digits and dashes.' : serverError
  const valid = validSkillName(slug) && draft.description.trim() && draft.body.trim()

  const create = useMutation({
    mutationFn: () =>
      api.skills.create({ name: slug, description: draft.description.trim(), body: draft.body, tags: splitList(draft.tags), requires_tools: splitList(draft.requires_tools) }),
    onSuccess: (s) => {
      void qc.invalidateQueries({ queryKey: qk.skills.all })
      toast.success(`Created ${s.name}`, { description: 'It is active immediately.' })
      close(false)
      onCreated(s.name)
    },
    onError: (e) => {
      if (isApiError(e) && (e.status === 409 || e.status === 400)) setServerError(e.detail)
      else toast.error("Couldn't create skill", { description: errorMessage(e) })
    }
  })

  const close = (o: boolean) => {
    onOpenChange(o)
    if (!o) {
      setDraft(empty)
      setTouched(false)
      setServerError('')
    }
  }

  return (
    <Dialog
      open={open}
      onOpenChange={close}
      size="xl"
      modalLock
      title="Create a skill"
      description="Write down a procedure you want Sentient to follow. It lists skills by name and description and reads the full steps when one applies."
      footer={
        <>
          {slug && slug !== draft.name && <span className="mr-auto font-mono text-xs text-fg-subtle">saved as {slug}</span>}
          <Button variant="ghost" onClick={() => close(false)}>
            Cancel
          </Button>
          <Button variant="primary" leftIcon={<IconPlus size={15} />} disabled={!valid} loading={create.isPending} onClick={() => create.mutate()}>
            Create skill
          </Button>
        </>
      }
    >
      <div className="max-h-[70vh] overflow-y-auto pr-1">
        <SkillEditor
          nameEditable
          nameError={nameError || undefined}
          draft={draft}
          onChange={(d) => {
            setDraft(d)
            setTouched(true)
            if (d.name !== draft.name) setServerError('')
          }}
          minHeight={300}
        />
        {!draft.description.trim() && touched && (
          <Alert tone="info" className="mt-3">
            Add a one-line description so Sentient knows when to use this skill.
          </Alert>
        )}
      </div>
    </Dialog>
  )
}
