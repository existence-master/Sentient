import { IconDeviceFloppy, IconRestore } from '@tabler/icons-react'
import { useEffect, useState } from 'react'
import { toast } from 'sonner'
import { Alert, Button, Card, ConfirmDialog, Skeleton, Textarea } from '@/components/ui'
import { useBootstrap } from '@/hooks/core'
import { useMemoryActions, usePersonas, useWorkspace } from '@/hooks/memory'
import { useHotkey } from '@/hooks/useHotkey'
import { errorMessage } from '@/lib/api'
import type { Persona, WorkspaceFileId } from '@/lib/types'
import { cn } from '@/lib/utils'
import type { SectionProps } from '../SettingsPage'

export function PersonalitySection(_props: SectionProps) {
  const ws = useWorkspace()
  const personas = usePersonas()
  const bootstrap = useBootstrap()
  const [soul, setSoul] = useState<string | null>(null)
  const [user, setUser] = useState<string | null>(null)
  const [pendingPreset, setPendingPreset] = useState<Persona | null>(null)

  useEffect(() => {
    if (ws.data) {
      setSoul((s) => s ?? ws.data.soul)
      setUser((u) => u ?? ws.data.user)
    }
  }, [ws.data])

  const names = { name: bootstrap.data?.assistant.name ?? 'Sentient', user: bootstrap.data?.assistant.user_name || 'the user' }
  const applyPreset = (p: Persona) => setSoul(p.soul_md.replace(/\{name\}/g, names.name).replace(/\{user\}/g, names.user))
  const soulDirty = ws.data && soul !== null && soul !== ws.data.soul

  if (ws.isLoading) return <Skeleton className="h-96 rounded-xl" />
  if (ws.isError) {
    return (
      <Alert tone="danger" title="Couldn't load the workspace files">
        {errorMessage(ws.error)}
      </Alert>
    )
  }

  return (
    <div className="space-y-8">
      <div className="space-y-3">
        <div className="px-1">
          <h3 className="text-sm font-semibold text-fg">Soul</h3>
          <p className="mt-0.5 text-xs text-fg-subtle">SOUL.md defines who {names.name} is: tone, values and habits. It&apos;s read before every reply.</p>
        </div>
        {!!personas.data?.some((p) => p.soul_md) && (
          <div className="flex flex-wrap gap-1.5 px-1">
            <span className="mr-1 self-center text-xs text-fg-subtle">Start from</span>
            {personas.data
              .filter((p) => p.soul_md)
              .map((p) => (
                <button
                  key={p.id}
                  type="button"
                  title={p.description}
                  onClick={() => (soulDirty ? setPendingPreset(p) : applyPreset(p))}
                  className="rounded-full border border-border bg-surface px-3 py-1 text-xs text-fg-muted transition-colors hover:border-border-strong hover:text-fg"
                >
                  {p.name}
                </button>
              ))}
          </div>
        )}
        <WorkspaceEditor which="soul" value={soul ?? ''} original={ws.data?.soul ?? ''} onChange={setSoul} minHeight={340} />
      </div>

      <div className="space-y-3">
        <div className="px-1">
          <h3 className="text-sm font-semibold text-fg">About you</h3>
          <p className="mt-0.5 text-xs text-fg-subtle">USER.md is what {names.name} knows about you in your own words. Sentient may add to it as it learns.</p>
        </div>
        <WorkspaceEditor which="user" value={user ?? ''} original={ws.data?.user ?? ''} onChange={setUser} minHeight={220} />
      </div>

      <ConfirmDialog
        open={!!pendingPreset}
        onOpenChange={(o) => !o && setPendingPreset(null)}
        title="Replace your edits?"
        description={`Loading “${pendingPreset?.name}” replaces the unsaved text in the editor.`}
        confirmLabel="Replace"
        tone="primary"
        onConfirm={() => {
          if (pendingPreset) applyPreset(pendingPreset)
        }}
      />
    </div>
  )
}

function WorkspaceEditor({
  which,
  value,
  original,
  onChange,
  minHeight
}: {
  which: WorkspaceFileId
  value: string
  original: string
  onChange: (v: string) => void
  minHeight: number
}) {
  const { writeWorkspace } = useMemoryActions()
  const dirty = value !== original
  const file = which === 'soul' ? 'SOUL.md' : which === 'user' ? 'USER.md' : 'MEMORY.md'

  const save = () =>
    writeWorkspace.mutate(
      { which, content: value },
      {
        onSuccess: () => toast.success(`${file} saved`),
        onError: (e) => toast.error(`Couldn't save ${file}`, { description: errorMessage(e) })
      }
    )

  useHotkey('mod+s', () => dirty && save(), { enabled: dirty })

  return (
    <Card className={cn('overflow-hidden transition-colors', dirty && 'border-accent/35')}>
      <div className="flex h-10 items-center gap-2 border-b border-border px-3.5">
        <span className="font-mono text-xs text-fg-muted">{file}</span>
        {dirty && <span className="size-1.5 rounded-full bg-accent" title="Unsaved changes" />}
        <div className="flex-1" />
        <Button size="xs" variant="ghost" leftIcon={<IconRestore size={13} />} disabled={!dirty} onClick={() => onChange(original)}>
          Revert
        </Button>
        <Button size="xs" variant={dirty ? 'primary' : 'secondary'} leftIcon={<IconDeviceFloppy size={13} />} disabled={!dirty} loading={writeWorkspace.isPending} onClick={save}>
          Save
        </Button>
      </div>
      <Textarea
        autoGrow
        minHeight={minHeight}
        maxHeight={720}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        spellCheck={false}
        className="rounded-none border-0 bg-transparent px-4 py-3.5 font-mono text-[12.5px] leading-relaxed shadow-none hover:border-0 focus:border-0 focus:ring-0"
      />
    </Card>
  )
}
