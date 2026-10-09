/** Settings > Browser: named browser profiles and attaching to a browser you started (docs/API.md §12). */
import { IconAlertTriangle, IconLogin2, IconPencil, IconPlugConnected, IconPlus, IconTrash, IconWorldWww } from '@tabler/icons-react'
import { useState } from 'react'
import { toast } from 'sonner'
import { Alert, Badge, Button, ConfirmDialog, Dialog, Field, FormSection, IconButton, Input, SegmentedControl, Select, Skeleton, StatusDot } from '@/components/ui'
import { useBrowserActions, useBrowserProfileActions, useBrowserProfiles } from '@/features/browser/state'
import { errorMessage, isNotImplemented } from '@/lib/api'
import type { BrowserProfile, BrowserProfileKind } from '@/lib/types'

const ENGINES = [
  { value: 'same', label: 'Same as the Browser setting' },
  { value: 'msedge', label: 'Microsoft Edge' },
  { value: 'chrome', label: 'Google Chrome' },
  { value: 'chromium', label: 'Chromium' }
]

export function BrowserProfilesSection() {
  const profiles = useBrowserProfiles()
  const { remove } = useBrowserProfileActions()
  const { open } = useBrowserActions()
  const [creating, setCreating] = useState(false)
  const [editing, setEditing] = useState<BrowserProfile | null>(null)
  const [deleting, setDeleting] = useState<BrowserProfile | null>(null)

  if (profiles.isError && isNotImplemented(profiles.error)) return null

  const openProfile = (p: BrowserProfile) =>
    open.mutate(
      { profile: p.name },
      {
        onSuccess: () =>
          toast.success(p.kind === 'attach' ? 'Connected to your browser' : 'The browser is open', {
            description: p.kind === 'attach' ? 'Sentient opened a tab of its own in it.' : 'Sign in yourself in the window that opened, then close it when you are done.'
          }),
        onError: (e) => toast.error("Couldn't open that profile", { description: errorMessage(e) })
      }
    )

  return (
    <FormSection
      title="Profiles"
      description="Each profile keeps its own sign-ins, for example one for posting on a social account and one for shopping. A task can use a profile, and a skill can name one."
      actions={
        <Button size="sm" variant="secondary" leftIcon={<IconPlus size={14} />} disabled={profiles.isLoading} onClick={() => setCreating(true)}>
          Add profile
        </Button>
      }
    >
      {profiles.isLoading ? (
        <div className="p-4">
          <Skeleton className="h-12 rounded-lg" />
        </div>
      ) : profiles.isError ? (
        <div className="p-4">
          <Alert tone="danger" title="Couldn’t load your browser profiles" action={<Button size="sm" onClick={() => void profiles.refetch()}>Retry</Button>}>
            {errorMessage(profiles.error)}
          </Alert>
        </div>
      ) : (
        profiles.data?.profiles.map((p) => (
          <div key={p.name} className="flex flex-wrap items-center gap-x-3 gap-y-2 px-4 py-3">
            <span className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-accent/10 text-accent-text">
              {p.kind === 'attach' ? <IconPlugConnected size={16} /> : <IconWorldWww size={16} />}
            </span>
            <div className="min-w-0 flex-1 basis-48">
              <div className="flex items-center gap-2">
                <span className="truncate text-sm font-medium text-fg">{p.name}</span>
                {p.kind === 'attach' && <Badge size="xs">Your browser</Badge>}
                {p.running && (
                  <span className="flex items-center gap-1 text-2xs text-fg-subtle">
                    <StatusDot tone="success" /> In use
                  </span>
                )}
              </div>
              <div className="truncate text-xs text-fg-subtle">
                {p.notes || (p.kind === 'attach' ? p.endpoint : p.name === 'default' ? 'Used when nothing else is picked' : 'No notes')}
              </div>
            </div>
            <Button size="xs" variant="ghost" leftIcon={p.kind === 'attach' ? <IconPlugConnected size={13} /> : <IconLogin2 size={13} />} loading={open.isPending && typeof open.variables === 'object' && open.variables.profile === p.name} onClick={() => openProfile(p)}>
              {p.kind === 'attach' ? 'Connect' : 'Open to sign in'}
            </Button>
            <IconButton size="sm" label={`Edit ${p.name}`} icon={<IconPencil size={15} />} onClick={() => setEditing(p)} />
            {p.name !== 'default' && <IconButton size="sm" label={`Delete ${p.name}`} icon={<IconTrash size={15} />} onClick={() => setDeleting(p)} />}
          </div>
        ))
      )}

      <ProfileDialog open={creating} onOpenChange={setCreating} />
      <ProfileDialog open={!!editing} onOpenChange={(o) => !o && setEditing(null)} profile={editing ?? undefined} />
      <ConfirmDialog
        open={!!deleting}
        onOpenChange={(o) => !o && setDeleting(null)}
        title={`Delete “${deleting?.name ?? ''}”?`}
        description={
          deleting?.kind === 'attach'
            ? 'Sentient stops connecting to that browser. Your browser and everything in it stay as they are.'
            : 'Its sign-ins, cookies and history are deleted from this computer. Tasks that use it will ask you to pick another. This can’t be undone.'
        }
        confirmLabel="Delete profile"
        onConfirm={async () => {
          if (!deleting) return
          try {
            await remove.mutateAsync(deleting.name)
            toast.success('Profile deleted')
          } catch (e) {
            toast.error('Couldn’t delete it', { description: errorMessage(e) })
          }
        }}
      />
    </FormSection>
  )
}

function ProfileDialog({ open, onOpenChange, profile }: { open: boolean; onOpenChange: (o: boolean) => void; profile?: BrowserProfile }) {
  const { create, update } = useBrowserProfileActions()
  const editing = !!profile
  const [name, setName] = useState('')
  const [kind, setKind] = useState<BrowserProfileKind>('launch')
  const [engine, setEngine] = useState('same')
  const [endpoint, setEndpoint] = useState('')
  const [notes, setNotes] = useState('')
  const [seen, setSeen] = useState<string | null>(null)

  // load the profile being edited (or reset for a new one) each time the dialog opens
  const key = open ? (profile?.name ?? '+new') : null
  if (key !== seen) {
    setSeen(key)
    if (open) {
      setName(profile?.name ?? '')
      setKind(profile?.kind ?? 'launch')
      setEngine(profile?.engine || 'same')
      setEndpoint(profile?.endpoint ?? '')
      setNotes(profile?.notes ?? '')
    }
  }

  const pending = create.isPending || update.isPending
  const canSave = !!name.trim() && (kind === 'launch' || !!endpoint.trim())

  const save = () => {
    if (!canSave) return
    const eng = kind === 'launch' && engine !== 'same' ? engine : ''
    const done = { onSuccess: () => (onOpenChange(false), toast.success(editing ? 'Profile saved' : 'Profile added')), onError: (e: unknown) => toast.error('Couldn’t save the profile', { description: errorMessage(e) }) }
    if (profile) {
      update.mutate(
        {
          name: profile.name,
          patch: {
            ...(name.trim() !== profile.name ? { name: name.trim() } : {}),
            notes,
            ...(kind === 'attach' ? { endpoint } : { engine: eng })
          }
        },
        done
      )
    } else {
      create.mutate({ name: name.trim(), kind, notes, ...(kind === 'attach' ? { endpoint } : { engine: eng }) }, done)
    }
  }

  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      size="lg"
      title={editing ? `Edit “${profile?.name}”` : 'Add a browser profile'}
      description={editing ? undefined : 'A profile keeps its own sign-ins. Name it after what it’s for.'}
      footer={
        <>
          <Button variant="ghost" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button variant="primary" loading={pending} disabled={!canSave} onClick={save}>
            {editing ? 'Save' : 'Add profile'}
          </Button>
        </>
      }
    >
      <div className="space-y-4">
        <Field label="Name" description={profile?.name === 'default' ? 'The default profile keeps its name.' : 'Letters, numbers and dashes, for example x-posting.'}>
          <Input autoFocus value={name} disabled={profile?.name === 'default'} placeholder="e.g. x-posting" onChange={(e) => setName(e.target.value)} />
        </Field>
        {!editing && (
          <SegmentedControl
            fullWidth
            aria-label="Kind of profile"
            value={kind}
            onChange={setKind}
            options={[
              { value: 'launch', label: 'Sentient opens it', icon: <IconWorldWww size={14} /> },
              { value: 'attach', label: 'A browser I started', icon: <IconPlugConnected size={14} /> }
            ]}
          />
        )}
        {kind === 'launch' ? (
          <Field label="Browser">
            <Select value={engine} onValueChange={setEngine} options={ENGINES} aria-label="Browser for this profile" />
          </Field>
        ) : (
          <>
            <Field label="DevTools address" description="Only an address on this computer works, like http://127.0.0.1:9333.">
              <Input value={endpoint} placeholder="http://127.0.0.1:9333" onChange={(e) => setEndpoint(e.target.value)} />
            </Field>
            <div className="rounded-xl border border-border bg-sunken/60 p-3.5 text-xs leading-relaxed text-fg-muted">
              <div className="mb-1.5 text-sm font-medium text-fg">How to start your browser for this</div>
              Close it first, then start Brave, Chrome or Edge with a port and a separate profile folder, for example:
              <pre className="selectable mt-2 overflow-x-auto rounded-lg border border-border bg-field px-3 py-2 font-mono text-[11.5px] text-fg">
                brave.exe --remote-debugging-port=9333 --user-data-dir="C:\BrowserForSentient"
              </pre>
              Use chrome.exe or msedge.exe the same way. Sign in to your sites in that window. Sentient works in a tab of its own and never closes your browser.
            </div>
            <Alert tone="warning" icon={<IconAlertTriangle />} title="Any program on this computer can control that browser">
              While the browser runs with a DevTools port, other programs on this computer can use it too, including your signed-in sites. Only start it this way when you need it, and keep it to a separate profile.
            </Alert>
          </>
        )}
        <Field label="What it’s for" optional>
          <Input value={notes} placeholder="e.g. Signed in to my X account" onChange={(e) => setNotes(e.target.value)} />
        </Field>
      </div>
    </Dialog>
  )
}
