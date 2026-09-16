/** The warm summary at the top of "About you", plus the dialog to add something yourself. */
import { IconPlus, IconRefresh, IconUserHeart } from '@tabler/icons-react'
import { useState } from 'react'
import { toast } from 'sonner'
import { Button, Dialog, Field, Select, Skeleton, Textarea } from '@/components/ui'
import { useBootstrap } from '@/hooks/core'
import { errorMessage } from '@/lib/api'
import { useUserModelActions } from '@/lib/leap/hooks-b'
import type { InsightDimension, UserModel } from '@/lib/leap/types-b'
import { relativeTime } from '@/lib/utils'
import { DIMENSION_META, DIMENSION_ORDER, dimensionMeta } from './meta'

export function Portrait({ model, loading, onAdd }: { model?: UserModel; loading: boolean; onAdd: () => void }) {
  const { refresh } = useUserModelActions()
  const bootstrap = useBootstrap()
  const name = bootstrap.data?.assistant.user_name || 'you'
  const live = (model?.insights ?? []).filter((i) => i.status !== 'retired')
  const areas = new Set(live.map((i) => i.dimension)).size
  const confirmed = live.filter((i) => i.status === 'confirmed' || i.source === 'user').length

  const runRefresh = () =>
    refresh.mutate(undefined, {
      onSuccess: (r) => {
        const changes = r.added + r.updated
        toast.success(changes ? 'Your picture is up to date' : 'Nothing new to add yet', {
          description: changes
            ? `${r.added ? `${r.added} new` : ''}${r.added && r.updated ? ', ' : ''}${r.updated ? `${r.updated} refined` : ''}${r.questions ? `, and ${r.questions} question${r.questions === 1 ? '' : 's'} for you` : ''}.`
            : 'I’ll keep learning as we talk.'
        })
      },
      onError: (e) => toast.error('Couldn’t refresh right now', { description: errorMessage(e) })
    })

  return (
    <header className="relative overflow-hidden rounded-3xl border border-border bg-elevated/50 px-8 py-7">
      <div aria-hidden className="pointer-events-none absolute -right-24 -top-32 size-96 rounded-full opacity-[0.14] blur-3xl" style={{ background: 'radial-gradient(circle, var(--accent), transparent 70%)' }} />
      <div aria-hidden className="pointer-events-none absolute -bottom-40 left-10 size-80 rounded-full opacity-[0.07] blur-3xl" style={{ background: 'radial-gradient(circle, #a78bfa, transparent 70%)' }} />
      <div className="relative">
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2 text-xs font-medium text-accent-text">
            <IconUserHeart size={15} /> About you
          </div>
          <div className="flex-1" />
          <Button variant="ghost" size="sm" leftIcon={<IconRefresh size={14} />} loading={refresh.isPending} onClick={runRefresh}>
            Refresh my picture
          </Button>
          <Button variant="secondary" size="sm" leftIcon={<IconPlus size={14} />} onClick={onAdd}>
            Add something
          </Button>
        </div>
        <h1 className="mt-2 text-2xl font-semibold tracking-tight text-fg">How Sentient understands {name === 'you' ? 'you' : name}</h1>
        {loading ? (
          <div className="mt-4 max-w-3xl space-y-2">
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-11/12" />
            <Skeleton className="h-4 w-2/3" />
          </div>
        ) : model?.summary?.trim() ? (
          <p className="selectable mt-3 max-w-3xl text-[15px] leading-[1.75] text-fg-muted">{model.summary}</p>
        ) : (
          <p className="mt-3 max-w-2xl text-[15px] leading-relaxed text-fg-muted">
            I’m still getting to know you. As we talk, I’ll write down what I notice here, and you can correct anything that isn’t quite right.
          </p>
        )}
        {model && (
          <div className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-fg-subtle">
            <span>
              <b className="font-semibold text-fg">{live.length}</b> {live.length === 1 ? 'thing' : 'things'} I’ve noticed
            </span>
            <span>
              across <b className="font-semibold text-fg">{areas}</b> {areas === 1 ? 'part' : 'parts'} of your life
            </span>
            {confirmed > 0 && (
              <span>
                <b className="font-semibold text-fg">{confirmed}</b> confirmed by you
              </span>
            )}
            {model.updated_at && <span>Updated {relativeTime(model.updated_at)}</span>}
          </div>
        )}
      </div>
    </header>
  )
}

export function AddInsightDialog({ open, onOpenChange, initialDimension }: { open: boolean; onOpenChange: (o: boolean) => void; initialDimension?: InsightDimension }) {
  const { add } = useUserModelActions()
  const [dimension, setDimension] = useState<InsightDimension>(initialDimension ?? 'preferences')
  const [statement, setStatement] = useState('')

  const submit = () => {
    const s = statement.trim()
    if (!s) return
    add.mutate(
      { statement: s, dimension },
      {
        onSuccess: () => {
          toast.success('Got it', { description: 'I’ll keep this in mind from now on.' })
          setStatement('')
          onOpenChange(false)
        },
        onError: (e) => toast.error('Couldn’t save that', { description: errorMessage(e) })
      }
    )
  }

  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      title="Tell Sentient something about you"
      description="Anything that would help it help you. You can change or remove it any time."
      modalLock={!!statement.trim()}
      footer={
        <>
          <Button variant="ghost" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button variant="primary" loading={add.isPending} disabled={!statement.trim()} onClick={submit}>
            Save
          </Button>
        </>
      }
    >
      <div className="space-y-4">
        <Field label="What is it about?">
          <Select<InsightDimension>
            value={dimension}
            onValueChange={setDimension}
            options={DIMENSION_ORDER.map((d) => {
              const m = DIMENSION_META[d]
              return { value: d, label: m.label, description: m.blurb, icon: <m.icon size={15} style={{ color: m.color }} /> }
            })}
          />
        </Field>
        <Field label="In your words">
          <Textarea
            autoFocus
            autoGrow
            minHeight={88}
            value={statement}
            placeholder={dimensionMeta(dimension).example}
            onChange={(e) => setStatement(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && (e.metaKey || e.ctrlKey) && submit()}
          />
        </Field>
      </div>
    </Dialog>
  )
}
