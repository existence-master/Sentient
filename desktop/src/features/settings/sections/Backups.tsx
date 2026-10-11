/** Settings > Backups: back up now, restore with a clear confirmation, and the snapshot list (docs/API.md §20). */
import {
  IconAlertTriangle,
  IconArchive,
  IconCamera,
  IconCheck,
  IconDatabaseExport,
  IconFileZip,
  IconFolder,
  IconFolderOpen,
  IconHistory,
  IconKey,
  IconRestore
} from '@tabler/icons-react'
import { useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { toast } from 'sonner'
import { Alert, Badge, Button, Dialog, EmptyState, FormRow, FormSection, Skeleton } from '@/components/ui'
import { useBackUpNow, useBackups, useCancelRestore, useDismissRestore } from '@/hooks/backups'
import { useConfigEditor } from '@/hooks/config'
import { qk } from '@/hooks/queryKeys'
import { api, errorMessage } from '@/lib/api'
import { getBridge } from '@/lib/bridge'
import type { BackupEntry, BackupInspect, BackupSignIn, RestoreResult } from '@/lib/types'
import { formatBytes, formatDateTime, relativeTime } from '@/lib/utils'
import { useConnection } from '@/stores/connection'
import { SchemaForm } from '../SchemaForm'
import type { SectionProps } from '../SettingsPage'

const KEYCHAIN_NOTE = "Passwords, API keys and sign-ins stay in this computer's keychain and are never in a backup."
const DEVICES_NOTE = 'A backup also holds the certificate your devices trust, so keep backups somewhere private.'

function countsLine(e: BackupEntry): string {
  const c = e.counts
  const parts = [
    c.chats !== undefined && `${c.chats} ${c.chats === 1 ? 'chat' : 'chats'}`,
    c.memories !== undefined && `${c.memories} ${c.memories === 1 ? 'memory' : 'memories'}`,
    c.tasks !== undefined && `${c.tasks} ${c.tasks === 1 ? 'task' : 'tasks'}`,
    c.skills !== undefined && `${c.skills} ${c.skills === 1 ? 'skill' : 'skills'}`
  ].filter(Boolean)
  return parts.join(' · ')
}

function signInNames(items: BackupSignIn[]): string {
  return items.map((s) => s.label).join(', ')
}

export function BackupsSection({ query }: SectionProps) {
  const bridge = getBridge()
  const qc = useQueryClient()
  const backups = useBackups()
  const backUp = useBackUpNow()
  const cancel = useCancelRestore()
  const dismiss = useDismissRestore()
  const editor = useConfigEditor()
  const restartEngine = useConnection((s) => s.restart)
  const [confirm, setConfirm] = useState<BackupInspect | null>(null)
  const [opening, setOpening] = useState<string | null>(null)
  const [restoring, setRestoring] = useState(false)
  const data = backups.data

  const runBackup = (folder?: string) =>
    backUp.mutate(folder, {
      onSuccess: (made) => toast.success('Backed up', { description: made.path }),
      onError: (err) => toast.error("The backup didn't work", { description: errorMessage(err) })
    })

  const pickFolder = async (): Promise<string | undefined> => {
    const [dir] = await bridge.pickFiles({ directory: true, multiple: false })
    return dir
  }

  const openRestore = async (path: string) => {
    setOpening(path)
    try {
      setConfirm(await api.backups.inspect(path))
    } catch (err) {
      toast.error("That backup can't be restored", { description: errorMessage(err) })
    } finally {
      setOpening(null)
    }
  }

  const pickBackupFile = async () => {
    const [file] = await bridge.pickFiles({ multiple: false, filters: [{ name: 'Sentient backups', extensions: ['zip'] }] })
    if (file) await openRestore(file)
  }

  const restart = async () => {
    await restartEngine()
    window.location.reload() // every screen reads the restored data from scratch
  }

  const doRestore = async () => {
    if (!confirm) return
    setRestoring(true)
    try {
      await api.backups.restore(confirm.path)
      setConfirm(null)
      void qc.invalidateQueries({ queryKey: qk.backups })
      if (bridge.isDesktop) {
        toast.message('Restoring', { description: 'Sentient is restarting with your backup.' })
        await restart()
      } else {
        toast.success('Ready to restore', { description: 'Restart Sentient to finish.' })
      }
    } catch (err) {
      toast.error("That backup can't be restored", { description: errorMessage(err) })
    } finally {
      setRestoring(false)
    }
  }

  if (backups.isLoading) {
    return (
      <div className="space-y-3">
        <Skeleton className="h-28" />
        <Skeleton className="h-40" />
      </div>
    )
  }
  if (backups.isError || !data) {
    return (
      <Alert tone="warning" icon={<IconAlertTriangle />} title="Backups aren't available" action={<Button size="sm" onClick={() => void backups.refetch()}>Try again</Button>}>
        {errorMessage(backups.error) || 'This version of the engine has no backups.'}
      </Alert>
    )
  }

  const customFolder = data.folder !== data.default_folder

  return (
    <div className="space-y-8">
      {data.pending_restore && (
        <Alert
          tone="warning"
          icon={<IconRestore />}
          title="A restore is waiting"
          action={
            <div className="flex gap-1.5">
              <Button size="sm" variant="ghost" loading={cancel.isPending} onClick={() => cancel.mutate()}>
                Cancel restore
              </Button>
              {bridge.isDesktop && (
                <Button size="sm" variant="primary" onClick={() => void restart()}>
                  Restart now
                </Button>
              )}
            </div>
          }
        >
          The backup from {formatDateTime(data.pending_restore.created_at)} replaces your data when Sentient restarts.
        </Alert>
      )}

      {data.last_restore && <LastRestore result={data.last_restore} onDismiss={() => dismiss.mutate()} />}

      <FormSection title="Back up" description={`${KEYCHAIN_NOTE} ${DEVICES_NOTE}`}>
        <FormRow label="Back up now" description="Saves your chats, memory, tasks, settings, skills and files in one file.">
          <div className="flex flex-wrap justify-end gap-1.5">
            <Button
              size="sm"
              variant="ghost"
              disabled={!bridge.isDesktop || backUp.isPending}
              onClick={async () => {
                const dir = await pickFolder()
                if (dir) runBackup(dir)
              }}
            >
              To another folder…
            </Button>
            <Button size="sm" variant="primary" leftIcon={<IconDatabaseExport size={14} />} loading={backUp.isPending} onClick={() => runBackup()}>
              Back up now
            </Button>
          </div>
        </FormRow>
        <FormRow
          label="Backup folder"
          description={
            <>
              <span className="block break-all font-mono">{data.folder}</span>
              <span className="mt-1 block">
                {customFolder
                  ? 'The folder you picked.'
                  : data.folder.startsWith(data.home)
                    ? "Inside Sentient's data folder, because your Documents folder can't be used. Pick a folder elsewhere so a backup survives losing it."
                    : 'In your Documents folder, apart from your data, so a backup survives losing it.'}
              </span>
            </>
          }
        >
          <div className="flex flex-wrap justify-end gap-1.5">
            {customFolder && (
              <Button size="sm" variant="ghost" onClick={() => editor.setValue('backups.folder', null, { immediate: true })}>
                Use the default
              </Button>
            )}
            <Button
              size="sm"
              variant="ghost"
              leftIcon={<IconFolder size={14} />}
              disabled={!bridge.isDesktop}
              onClick={async () => {
                if (!(await bridge.showInFolder(data.folder))) toast.message('The folder appears after your first backup.')
              }}
            >
              Open
            </Button>
            <Button
              size="sm"
              leftIcon={<IconFolderOpen size={14} />}
              disabled={!bridge.isDesktop}
              onClick={async () => {
                const dir = await pickFolder()
                if (dir) {
                  editor.setValue('backups.folder', dir, { immediate: true })
                  toast.success('Backups go to the new folder')
                }
              }}
            >
              Choose…
            </Button>
          </div>
        </FormRow>
      </FormSection>

      <FormSection
        title="Restore"
        description="Pick a backup. You see what is in it and confirm before anything changes."
        actions={
          <Button size="xs" variant="ghost" leftIcon={<IconFileZip size={12} />} disabled={!bridge.isDesktop} loading={opening !== null && !data.backups.some((b) => b.path === opening)} onClick={() => void pickBackupFile()}>
            Restore from a file…
          </Button>
        }
      >
        {data.backups.length === 0 ? (
          <EmptyState compact icon={<IconArchive />} title="No backups here yet" description="Back up now, or restore a backup file from somewhere else." />
        ) : (
          data.backups.map((b) => <BackupRow key={b.path} entry={b} busy={opening === b.path} onRestore={() => void openRestore(b.path)} />)
        )}
      </FormSection>

      <FormSection
        title="Snapshots"
        description={`Sentient takes one before it updates its database, before an import and before a restore. The last ${data.snapshots_keep} are kept. They leave out your files folder.`}
      >
        {data.snapshots.length === 0 ? (
          <EmptyState compact icon={<IconCamera />} title="No snapshots yet" description="One appears the next time Sentient updates its database or you import something." />
        ) : (
          data.snapshots.map((s) => <BackupRow key={s.path} entry={s} busy={opening === s.path} onRestore={() => void openRestore(s.path)} />)
        )}
      </FormSection>

      <SchemaForm section="backups" title="Automatic backups" filter={query} exclude={['backups.folder']} />

      <RestoreDialog info={confirm} busy={restoring} desktop={bridge.isDesktop} onCancel={() => setConfirm(null)} onConfirm={() => void doRestore()} />
    </div>
  )
}

function BackupRow({ entry, busy, onRestore }: { entry: BackupEntry; busy: boolean; onRestore: () => void }) {
  const title = entry.kind === 'snapshot' ? entry.reason ? entry.reason.charAt(0).toUpperCase() + entry.reason.slice(1) : 'Snapshot' : formatDateTime(entry.created_at)
  const sub = [entry.kind === 'snapshot' ? formatDateTime(entry.created_at) : relativeTime(entry.created_at), countsLine(entry), formatBytes(entry.size)].filter(Boolean).join(' · ')
  return (
    <FormRow
      label={
        <span className="flex items-center gap-2">
          {title}
          {entry.kind === 'scheduled' && <Badge size="xs">Automatic</Badge>}
        </span>
      }
      description={sub}
    >
      <Button size="sm" leftIcon={<IconHistory size={14} />} loading={busy} onClick={onRestore}>
        Restore
      </Button>
    </FormRow>
  )
}

function RestoreDialog({
  info,
  busy,
  desktop,
  onCancel,
  onConfirm
}: {
  info: BackupInspect | null
  busy: boolean
  desktop: boolean
  onCancel: () => void
  onConfirm: () => void
}) {
  const snapshot = info?.kind === 'snapshot'
  return (
    <Dialog
      open={info !== null}
      onOpenChange={(o) => !o && !busy && onCancel()}
      title={snapshot ? 'Restore this snapshot?' : 'Restore this backup?'}
      description={info ? `Made ${formatDateTime(info.created_at)}${info.version ? ` by Sentient ${info.version}` : ''}.` : undefined}
      size="md"
      footer={
        <>
          <Button variant="ghost" onClick={onCancel} disabled={busy}>
            Cancel
          </Button>
          <Button variant="danger" leftIcon={<IconRestore size={14} />} loading={busy} onClick={onConfirm}>
            {desktop ? 'Restore and restart' : 'Restore'}
          </Button>
        </>
      }
    >
      {info && (
        <div className="space-y-3 text-sm text-fg-muted">
          {countsLine(info) && <p className="font-medium text-fg">{countsLine(info)}</p>}
          <p>
            {snapshot
              ? 'Your chats, memory, tasks, settings and skills go back to how they were then. Your files folder stays as it is.'
              : 'Your chats, memory, tasks, settings, skills and files are replaced with the ones in this backup. Anything added since then goes away.'}{' '}
            Sentient saves a snapshot of what you have now first, so you can undo this.
          </p>
          {info.newer && (
            <Alert tone="warning" icon={<IconAlertTriangle />}>
              A newer version of Sentient made this backup. Some things in it may not work in this version.
            </Alert>
          )}
          <Alert tone="info" icon={<IconKey />} title="Sign-ins are not in backups">
            {info.sign_ins.length
              ? `${KEYCHAIN_NOTE} On this computer they keep working. On another computer, sign in again to: ${signInNames(info.sign_ins)}.`
              : `${KEYCHAIN_NOTE} If you restore on another computer, add your API keys and connect your apps again.`}
          </Alert>
          <p className="text-xs text-fg-subtle">{desktop ? 'Sentient restarts to finish. A reply in progress stops.' : 'Restart Sentient to finish.'}</p>
        </div>
      )}
    </Dialog>
  )
}

function LastRestore({ result, onDismiss }: { result: RestoreResult; onDismiss: () => void }) {
  if (result.error) {
    return (
      <Alert tone="danger" icon={<IconAlertTriangle />} title="The restore didn't happen" action={<Button size="sm" variant="ghost" onClick={onDismiss}>Dismiss</Button>}>
        {result.error}
      </Alert>
    )
  }
  const redo = result.sign_ins.filter((s) => s.signed_in !== true)
  return (
    <Alert
      tone="success"
      icon={<IconCheck />}
      title={`Restored the ${result.kind === 'snapshot' ? 'snapshot' : 'backup'} from ${formatDateTime(result.backup_created_at)}`}
      action={
        <Button size="sm" variant="ghost" onClick={onDismiss}>
          Dismiss
        </Button>
      }
    >
      {redo.length ? `Sign in again to: ${signInNames(redo)}. ${KEYCHAIN_NOTE}` : 'Everything that was signed in still works.'}
      {result.snapshot && <span className="mt-1 block text-xs">Your data from before is kept as a snapshot below, if you want it back.</span>}
    </Alert>
  )
}
