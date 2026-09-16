import { IconAt, IconEyeOff, IconShieldLock, IconTag, IconTextCaption, IconX, type Icon } from '@tabler/icons-react'
import { AnimatePresence, motion } from 'motion/react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { toast } from 'sonner'
import { Alert, Button, Skeleton } from '@/components/ui'
import { useIntegrationActions, usePrivacyFilters } from '@/hooks/integrations'
import { errorMessage } from '@/lib/api'
import type { Integration, PrivacyFilters } from '@/lib/types'
import { cn } from '@/lib/utils'
import { PRIVACY_FIELD_META, splitValues } from './meta'

const FIELD_ICON: Record<string, Icon> = { keywords: IconTextCaption, emails: IconAt, labels: IconTag }
const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/

export function ChipInput({
  values,
  onChange,
  placeholder,
  validate,
  icon: IconCmp
}: {
  values: string[]
  onChange: (next: string[]) => void
  placeholder?: string
  validate?: (value: string) => string | null
  icon?: Icon
}) {
  const [draft, setDraft] = useState('')
  const [error, setError] = useState<string | null>(null)
  const input = useRef<HTMLInputElement>(null)

  const add = (raw: string) => {
    const incoming = splitValues(raw)
    if (!incoming.length) return true
    for (const v of incoming) {
      const problem = validate?.(v)
      if (problem) {
        setError(problem)
        return false
      }
    }
    const lower = new Set(values.map((v) => v.toLowerCase()))
    const next = [...values, ...incoming.filter((v) => !lower.has(v.toLowerCase()))]
    if (next.length !== values.length) onChange(next)
    setDraft('')
    setError(null)
    return true
  }

  return (
    <div>
      <div
        onClick={() => input.current?.focus()}
        className={cn(
          'flex min-h-9 w-full cursor-text flex-wrap items-center gap-1.5 rounded-lg border bg-field px-2 py-1.5 transition-[border-color,box-shadow] focus-within:border-accent/60 focus-within:ring-3 focus-within:ring-accent/15',
          error ? 'border-danger/60' : 'border-border hover:border-border-strong'
        )}
      >
        {IconCmp && <IconCmp size={15} className="ml-0.5 shrink-0 text-fg-subtle" />}
        <AnimatePresence initial={false}>
          {values.map((v) => (
            <motion.span
              key={v}
              layout
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              className="inline-flex h-6 max-w-full items-center gap-1 rounded-md border border-border-strong bg-elevated pl-2 pr-0.5 text-xs text-fg"
            >
              <span className="truncate">{v}</span>
              <button
                type="button"
                aria-label={`Remove ${v}`}
                onClick={(e) => {
                  e.stopPropagation()
                  onChange(values.filter((x) => x !== v))
                }}
                className="flex size-5 items-center justify-center rounded text-fg-subtle hover:bg-active hover:text-fg"
              >
                <IconX size={11} />
              </button>
            </motion.span>
          ))}
        </AnimatePresence>
        <input
          ref={input}
          value={draft}
          spellCheck={false}
          placeholder={values.length ? '' : placeholder}
          onChange={(e) => {
            setDraft(e.target.value)
            setError(null)
          }}
          onKeyDown={(e) => {
            if ((e.key === 'Enter' || e.key === ',' || e.key === 'Tab') && draft.trim()) {
              e.preventDefault()
              add(draft)
            } else if (e.key === 'Backspace' && !draft && values.length) {
              onChange(values.slice(0, -1))
            }
          }}
          onBlur={() => draft.trim() && add(draft)}
          onPaste={(e) => {
            const text = e.clipboardData.getData('text')
            if (/[,;\n]/.test(text)) {
              e.preventDefault()
              add(text)
            }
          }}
          className="h-6 min-w-24 flex-1 bg-transparent px-1 text-sm text-fg outline-none placeholder:text-fg-subtle"
        />
      </div>
      {error && <p className="mt-1 text-xs text-danger">{error}</p>}
    </div>
  )
}

function same(a: PrivacyFilters | undefined, b: PrivacyFilters | undefined, fields: string[]) {
  return fields.every((f) => JSON.stringify(a?.[f] ?? []) === JSON.stringify(b?.[f] ?? []))
}

export function PrivacyFiltersEditor({ integration }: { integration: Integration }) {
  const fields = integration.privacy_filters.fields
  const query = usePrivacyFilters(integration.id)
  const { setPrivacyFilters } = useIntegrationActions()
  const [draft, setDraft] = useState<PrivacyFilters | null>(null)

  useEffect(() => {
    if (query.data && draft === null) setDraft(query.data)
  }, [query.data, draft])

  const dirty = useMemo(() => !!draft && !same(draft, query.data, fields), [draft, query.data, fields])
  const total = fields.reduce((n, f) => n + (draft?.[f]?.length ?? 0), 0)

  if (query.isLoading || (!draft && !query.isError)) {
    return (
      <div className="space-y-3">
        {fields.map((f) => (
          <Skeleton key={f} className="h-14 rounded-lg" />
        ))}
      </div>
    )
  }
  if (query.isError || !draft) {
    return <Alert tone="danger" title="Couldn't load privacy filters">{errorMessage(query.error)}</Alert>
  }

  return (
    <div className="space-y-4">
      <div className="flex items-start gap-2.5 rounded-lg border border-border bg-sunken/50 px-3 py-2.5 text-xs leading-relaxed text-fg-muted">
        <IconShieldLock size={16} className="mt-px shrink-0 text-accent-text" />
        <span>
          Filtered items never reach Sentient: not in chat, suggestions or triggered tasks.{' '}
          {total ? `${total} filter${total === 1 ? '' : 's'} active.` : 'No filters yet.'}
        </span>
      </div>
      {fields.map((f) => {
        const meta = PRIVACY_FIELD_META[f] ?? { label: f, placeholder: 'Add a value', help: '' }
        return (
          <div key={f} className="space-y-1.5">
            <div className="flex items-baseline justify-between gap-3">
              <label className="text-sm font-medium text-fg">{meta.label}</label>
              <span className="text-2xs tabular-nums text-fg-subtle">{draft[f]?.length ?? 0}</span>
            </div>
            <ChipInput
              icon={FIELD_ICON[f] ?? IconEyeOff}
              values={draft[f] ?? []}
              placeholder={meta.placeholder}
              validate={f === 'emails' ? (v) => (EMAIL_RE.test(v) ? null : `"${v}" doesn't look like an email address.`) : undefined}
              onChange={(next) => setDraft({ ...draft, [f]: next })}
            />
            <p className="text-xs text-fg-subtle">{meta.help} Press Enter or comma to add.</p>
          </div>
        )
      })}
      <AnimatePresence initial={false}>
        {dirty && (
          <motion.div
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 4 }}
            className="flex items-center justify-end gap-2"
          >
            <span className="mr-auto text-xs text-fg-subtle">Unsaved changes</span>
            <Button size="sm" variant="ghost" onClick={() => setDraft(query.data ?? null)}>
              Discard
            </Button>
            <Button
              size="sm"
              variant="primary"
              loading={setPrivacyFilters.isPending}
              onClick={() =>
                setPrivacyFilters.mutate(
                  { id: integration.id, filters: { ...draft, keywords: draft.keywords ?? [], emails: draft.emails ?? [], labels: draft.labels ?? [] } },
                  {
                    onSuccess: () => toast.success(`Privacy filters saved for ${integration.display_name}`),
                    onError: (e) => toast.error("Couldn't save privacy filters", { description: errorMessage(e) })
                  }
                )
              }
            >
              Save filters
            </Button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
