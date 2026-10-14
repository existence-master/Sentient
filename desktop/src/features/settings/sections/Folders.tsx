/**
 * Settings > Folders (#176): folders and Obsidian vaults Sentient searches when you ask about your notes. Add, remove
 * and check again, with what was found (docs/API.md §22). Sentient only reads these folders.
 */
import {
  IconAlertTriangle,
  IconFolderOpen,
  IconFolderPlus,
  IconFolders,
  IconNotebook,
  IconRefresh,
  IconTrash
} from '@tabler/icons-react'
import { useState } from 'react'
import { toast } from 'sonner'
import { Alert, Badge, Button, ConfirmDialog, EmptyState, FormSection, Input, ProgressBar, Skeleton } from '@/components/ui'
import { useAddFolder, useFolderIgnore, useFolders, useRemoveFolder, useRescanFolder } from '@/hooks/folders'
import { errorMessage } from '@/lib/api'
import { getBridge } from '@/lib/bridge'
import type { WatchedFolder } from '@/lib/types'
import { formatDuration, formatNumber, relativeTime } from '@/lib/utils'
import { SchemaForm } from '../SchemaForm'
import type { SectionProps } from '../SettingsPage'

const READ_ONLY = 'Sentient only reads these folders. It never changes, moves or deletes anything in them.'

function plural(n: number, one: string, many = `${one}s`): string {
  return `${formatNumber(n, false)} ${n === 1 ? one : many}`
}

export function FoldersSection({ query }: SectionProps) {
  const bridge = getBridge()
  const folders = useFolders()
  const add = useAddFolder()
  const data = folders.data

  const pick = async () => {
    const [dir] = await bridge.pickFiles({ directory: true, multiple: false })
    if (!dir) return
    add.mutate(dir, {
      onSuccess: () => toast.success('Folder added', { description: 'Sentient is reading it now. You can keep chatting.' }),
      onError: (err) => toast.error("That folder can't be added", { description: errorMessage(err) })
    })
  }

  if (folders.isLoading) {
    return (
      <div className="space-y-3">
        <Skeleton className="h-24" />
        <Skeleton className="h-24" />
      </div>
    )
  }
  if (folders.isError || !data) {
    return (
      <Alert tone="warning" icon={<IconAlertTriangle />} title="Folders aren't available" action={<Button size="sm" onClick={() => void folders.refetch()}>Try again</Button>}>
        {errorMessage(folders.error) || 'This version of the engine cannot search your folders.'}
      </Alert>
    )
  }

  return (
    <div className="space-y-8">
      <FormSection
        title="Your folders"
        description={`Add a folder of notes or documents, or an Obsidian vault, and Sentient searches it when you ask about your notes. Its answers show which notes they used. ${READ_ONLY}`}
        actions={
          <Button size="sm" variant="primary" leftIcon={<IconFolderPlus size={14} />} loading={add.isPending} disabled={!bridge.isDesktop} onClick={() => void pick()}>
            Add folder…
          </Button>
        }
      >
        {data.embedding_error && (
          <Alert tone="warning" icon={<IconAlertTriangle />} title="Searching by meaning is off">
            {data.embedding_error}
          </Alert>
        )}
        {data.folders.length === 0 ? (
          <EmptyState
            compact
            icon={<IconFolders />}
            title="No folders yet"
            description="Markdown, text, PDF and Word files are searched. Hidden folders like .obsidian are left out."
          />
        ) : (
          <div className="space-y-2.5">
            {data.folders.map((f) => (
              <FolderCard key={f.id} folder={f} />
            ))}
          </div>
        )}
      </FormSection>

      <SchemaForm section="folders" title="Reading and searching" filter={query} />
    </div>
  )
}

function statusBadge(f: WatchedFolder) {
  switch (f.status) {
    case 'scanning':
      return <Badge tone="accent">Reading</Badge>
    case 'waiting':
      return <Badge tone="neutral">Waiting</Badge>
    case 'missing':
      return <Badge tone="warning">Can't be found</Badge>
    default:
      return <Badge tone="success">Up to date</Badge>
  }
}

function FolderCard({ folder: f }: { folder: WatchedFolder }) {
  const bridge = getBridge()
  const remove = useRemoveFolder()
  const rescan = useRescanFolder()
  const setIgnore = useFolderIgnore()
  const [confirm, setConfirm] = useState(false)
  const [ignore, setIgnoreText] = useState(f.ignore.join(', '))
  const progress = f.progress && f.progress.total > 0 ? f.progress.done / f.progress.total : null
  const counts = [
    plural(f.files, 'file'),
    plural(f.chunks, 'piece'),
    f.chunks > 0 && f.embedded < f.chunks && `${formatNumber(f.chunks - f.embedded, false)} not yet searchable by meaning`,
    f.unreadable > 0 && `${formatNumber(f.unreadable, false)} couldn't be read`,
    f.skipped > 0 && `${formatNumber(f.skipped, false)} left out`,
    f.clippings > 0 && plural(f.clippings, 'clipped web page')
  ].filter(Boolean)
  const saveIgnore = () => {
    const list = ignore.split(/[,\n]/).map((s) => s.trim()).filter(Boolean)
    if (list.join(',') === f.ignore.join(',')) return
    setIgnore.mutate(
      { id: f.id, ignore: list },
      {
        onSuccess: () => toast.success('Saved', { description: 'Sentient reads the folder again with the new rules.' }),
        onError: (err) => toast.error("That didn't save", { description: errorMessage(err) })
      }
    )
  }

  return (
    <div className="rounded-xl border border-border bg-surface p-3.5" data-folder={f.name}>
      <div className="flex items-start gap-3">
        <div className="mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-lg border border-border bg-elevated text-fg-muted">
          {f.kind === 'obsidian' ? <IconNotebook size={16} /> : <IconFolders size={16} />}
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="truncate text-sm font-medium text-fg">{f.name}</span>
            {f.kind === 'obsidian' && <Badge size="xs" tone="info">Obsidian vault</Badge>}
            {statusBadge(f)}
          </div>
          <div className="selectable mt-0.5 break-all font-mono text-2xs text-fg-subtle">{f.path}</div>
          <div className="mt-1.5 text-xs text-fg-muted">{counts.join(' · ')}</div>
          <div className="mt-0.5 text-xs text-fg-subtle">
            {f.status === 'scanning' && f.progress
              ? `Reading ${formatNumber(f.progress.done, false)} of ${plural(f.progress.total, 'new or changed file')}`
              : f.last_scan_at
                ? `Last checked ${relativeTime(f.last_scan_at)}${f.last_scan_ms != null && f.last_scan_ms >= 1000 ? `, took ${formatDuration(f.last_scan_ms)}` : ''}`
                : 'Not read yet'}
          </div>
          {f.status === 'scanning' && <ProgressBar className="mt-2" value={progress} />}
          {f.error && <p className="mt-1.5 text-xs text-warning">{f.error}</p>}
          <div className="mt-2.5 flex items-center gap-2">
            <Input
              size="sm"
              value={ignore}
              onChange={(e) => setIgnoreText(e.target.value)}
              onBlur={saveIgnore}
              onKeyDown={(e) => e.key === 'Enter' && saveIgnore()}
              placeholder="Leave out, like Archive/, *.tmp"
              aria-label={`Leave out of ${f.name}`}
              wrapperClassName="flex-1"
            />
          </div>
        </div>
        <div className="flex shrink-0 flex-col items-end gap-1.5">
          <div className="flex gap-1">
            <Button size="sm" variant="ghost" leftIcon={<IconFolderOpen size={14} />} disabled={!bridge.isDesktop} onClick={() => void bridge.showInFolder(f.path)}>
              Open
            </Button>
            <Button
              size="sm"
              variant="ghost"
              leftIcon={<IconRefresh size={14} />}
              loading={rescan.isPending}
              disabled={f.status === 'scanning'}
              onClick={() => rescan.mutate(f.id)}
            >
              Check again
            </Button>
            <Button size="sm" variant="ghost" aria-label={`Remove ${f.name}`} leftIcon={<IconTrash size={14} />} onClick={() => setConfirm(true)}>
              Remove
            </Button>
          </div>
        </div>
      </div>
      <ConfirmDialog
        open={confirm}
        onOpenChange={setConfirm}
        title={`Stop searching ${f.name}?`}
        description="Sentient forgets what it found there. The folder and your files stay exactly as they are."
        confirmLabel="Remove"
        onConfirm={async () => {
          await remove.mutateAsync(f.id)
          toast.success(`${f.name} removed`)
        }}
      />
    </div>
  )
}
