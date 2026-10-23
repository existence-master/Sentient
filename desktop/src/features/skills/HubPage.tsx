import {
  IconAlertTriangle,
  IconBuildingStore,
  IconCheck,
  IconCloudDownload,
  IconCode,
  IconDeviceDesktop,
  IconExternalLink,
  IconEyeCheck,
  IconFileDescription,
  IconFiles,
  IconGitCommit,
  IconKey,
  IconLicense,
  IconListCheck,
  IconPlus,
  IconRefresh,
  IconScript,
  IconSearch,
  IconShieldCheck,
  IconTool,
  IconTrash,
  IconUserPlus
} from '@tabler/icons-react'
import { useMemo, useState, type ReactNode } from 'react'
import { toast } from 'sonner'
import { Alert, Badge, Button, ConfirmDialog, Dialog, EmptyState, Field, Input, SegmentedControl, Sheet, Skeleton, Tooltip } from '@/components/ui'
import { openExternal } from '@/features/integrations/InstructionsGuide'
import { useHubActions, useHubBrowse, useHubInstalled, useHubPreview, useHubSources } from '@/hooks/skillsHub'
import { errorMessage, hubConfirmCode, isApiError } from '@/lib/api'
import type { HubAsk, HubFileChanges, HubInstalled, HubSkill, HubSource, HubTrust, SkillVetting } from '@/lib/types'
import { cn, relativeTime } from '@/lib/utils'
import { SafetyReport } from './SafetyReport'
import { SkillBody } from './SkillDetail'
import { splitFrontmatter } from './meta'

const ASK_ICON: Record<string, ReactNode> = {
  tools: <IconTool size={12} />,
  apps: <IconDeviceDesktop size={12} />,
  keys: <IconKey size={12} />,
  platform: <IconDeviceDesktop size={12} />,
  needs: <IconListCheck size={12} />,
  programs: <IconScript size={12} />
}
const ASK_SHORT: Record<string, string> = { tools: 'Tools', apps: 'Programs', keys: 'Keys', platform: 'Platforms', needs: 'Needs', programs: 'Scripts' }

export const shortCommit = (c: string | null | undefined) => (c ? c.slice(0, 7) : '')

/** Who stands behind a source: one Sentient lists, or one the user added. */
export function TrustBadge({ trust, size = 'xs' }: { trust: HubTrust | string; size?: 'xs' | 'sm' }) {
  return trust === 'trusted' ? (
    <Tooltip content="A source Sentient lists. Its skills still get the safety check and wait for your review.">
      <span>
        <Badge size={size} tone="success" icon={<IconShieldCheck />}>
          Listed by Sentient
        </Badge>
      </span>
    </Tooltip>
  ) : (
    <Tooltip content="A source you added. Sentient doesn't know who runs it, so installing from it asks you once more.">
      <span>
        <Badge size={size} tone="warning" icon={<IconUserPlus />}>
          Added by you
        </Badge>
      </span>
    </Tooltip>
  )
}

/** The Skills hub (#210): skills other people wrote, from a few sources. Installing puts a skill in pending review. */
export function HubPage({ onReview }: { onReview: (name: string) => void }) {
  const sources = useHubSources()
  const [picked, setPicked] = useState<string | null>(null)
  const [query, setQuery] = useState('')
  const [open, setOpen] = useState<HubSkill | null>(null)
  const [adding, setAdding] = useState(false)
  const [removing, setRemoving] = useState(false)
  const actions = useHubActions()
  const source = sources.data?.find((s) => s.id === picked) ?? sources.data?.[0] ?? null
  const listing = useHubBrowse(source?.id ?? null)

  const visible = useMemo(() => {
    const q = query.trim().toLowerCase()
    const all = listing.data?.skills ?? []
    return q ? all.filter((s) => s.name.includes(q) || s.description.toLowerCase().includes(q) || s.author.toLowerCase().includes(q)) : all
  }, [listing.data, query])

  if (sources.isError) {
    return (
      <Alert tone="danger" title="The Skills hub isn't available right now" action={<Button size="sm" onClick={() => void sources.refetch()}>Retry</Button>}>
        {errorMessage(sources.error)}
      </Alert>
    )
  }

  return (
    <div className="space-y-4">
      <InstalledPanel onReview={onReview} />

      <div className="flex flex-wrap items-center gap-1.5" role="group" aria-label="Sources">
        {sources.isLoading
          ? [0, 1].map((i) => <Skeleton key={i} className="h-8 w-28 rounded-full" />)
          : sources.data?.map((s) => {
              const active = s.id === source?.id
              return (
                <button
                  key={s.id}
                  type="button"
                  aria-pressed={active}
                  onClick={() => {
                    setPicked(s.id)
                    setQuery('')
                  }}
                  className={cn(
                    'inline-flex h-8 items-center gap-1.5 rounded-full border px-3 text-sm font-medium transition-colors',
                    active ? 'border-border-strong bg-active text-fg' : 'border-border bg-surface text-fg-muted hover:border-border-strong hover:text-fg'
                  )}
                >
                  {s.trust === 'trusted' ? <IconShieldCheck size={14} className="text-success" /> : <IconUserPlus size={14} className="text-warning" />}
                  {s.name}
                </button>
              )
            })}
        <Button size="sm" variant="ghost" leftIcon={<IconPlus size={14} />} onClick={() => setAdding(true)}>
          Add a source
        </Button>
      </div>

      {source && (
        <div className="rounded-xl border border-border bg-surface px-4 py-3">
          <div className="flex flex-wrap items-start gap-3">
            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-md font-semibold text-fg">{source.name}</span>
                <TrustBadge trust={source.trust} />
              </div>
              {source.description && <p className="mt-0.5 text-sm text-fg-muted">{source.description}</p>}
              <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-0.5 text-xs text-fg-subtle">
                <button type="button" className="inline-flex min-w-0 items-center gap-1 font-mono hover:text-accent-text" onClick={() => openExternal(source.url)}>
                  <IconExternalLink size={12} className="shrink-0" />
                  <span className="truncate">{source.url.replace(/^https?:\/\//, '')}</span>
                </button>
                {listing.data?.commit && (
                  <span className="inline-flex items-center gap-1" title={listing.data.commit}>
                    <IconGitCommit size={12} /> listed at {shortCommit(listing.data.commit)}
                  </span>
                )}
                {listing.data && <span>{listing.data.skills.length} skills</span>}
                {listing.data?.fetched_at && <span>checked {relativeTime(listing.data.fetched_at)}</span>}
              </div>
            </div>
            <Button size="sm" variant="ghost" leftIcon={<IconRefresh size={14} />} loading={actions.refresh.isPending} onClick={() => actions.refresh.mutate(source.id, { onError: (e) => toast.error("Couldn't refresh", { description: errorMessage(e) }) })}>
              Refresh
            </Button>
            {!source.default && (
              <Button size="sm" variant="ghost" className="text-danger hover:text-danger" leftIcon={<IconTrash size={14} />} onClick={() => setRemoving(true)}>
                Remove
              </Button>
            )}
          </div>
          {source.trust !== 'trusted' && (
            <p className="mt-2 flex items-start gap-1.5 text-xs text-warning">
              <IconAlertTriangle size={13} className="mt-px shrink-0" />
              You added this source, so Sentient can't vouch for who runs it. Every skill still gets the safety check, and installing asks you once more.
            </p>
          )}
        </div>
      )}

      {source && (
        <Input wrapperClassName="max-w-xs" leftIcon={<IconSearch />} placeholder={`Search ${source.name}`} value={query} onChange={(e) => setQuery(e.target.value)} />
      )}

      {listing.isLoading ? (
        <Grid>
          {[0, 1, 2, 3, 4, 5].map((i) => (
            <Skeleton key={i} className="h-40 rounded-xl" />
          ))}
        </Grid>
      ) : listing.isError ? (
        <Alert tone="danger" title={`Couldn't list ${source?.name ?? 'this source'}`} action={<Button size="sm" onClick={() => void listing.refetch()}>Try again</Button>}>
          {errorMessage(listing.error)}
        </Alert>
      ) : listing.data && !listing.data.skills.length ? (
        <EmptyState icon={<IconBuildingStore />} title="No skills here" description="This source has no folders with a SKILL.md that Sentient can read." />
      ) : listing.data && !visible.length ? (
        <EmptyState compact icon={<IconSearch />} title="No skills match" description="Try a different search." />
      ) : listing.data ? (
        <Grid>
          {visible.map((s) => (
            <HubCard key={s.path} skill={s} onOpen={() => setOpen(s)} />
          ))}
        </Grid>
      ) : null}

      {source && <HubPreviewSheet source={source} skill={open} onOpenChange={(o) => !o && setOpen(null)} onReview={onReview} />}
      <AddSourceDialog open={adding} onOpenChange={setAdding} onAdded={(s) => setPicked(s.id)} />
      {source && (
        <ConfirmDialog
          open={removing}
          onOpenChange={setRemoving}
          title={`Remove ${source.name}?`}
          description="It stops showing here. Skills you installed from it stay as they are."
          confirmLabel="Remove source"
          onConfirm={async () => {
            try {
              await actions.removeSource.mutateAsync(source.id)
              setPicked(null)
            } catch (e) {
              toast.error("Couldn't remove it", { description: errorMessage(e) })
            }
          }}
        />
      )}
    </div>
  )
}

function Grid({ children }: { children: ReactNode }) {
  return <div className="grid grid-cols-1 gap-3 md:grid-cols-2 2xl:grid-cols-3">{children}</div>
}

function HubCard({ skill: s, onOpen }: { skill: HubSkill; onOpen: () => void }) {
  return (
    <button
      type="button"
      onClick={onOpen}
      className="group flex h-full flex-col gap-2.5 rounded-xl border border-border bg-surface p-4 text-left transition-colors hover:border-border-strong hover:bg-elevated"
    >
      <div className="flex items-start gap-2">
        <div className="min-w-0 flex-1">
          <div className="truncate font-mono text-sm font-semibold text-fg">{s.name}</div>
          <div className="mt-0.5 truncate text-2xs text-fg-subtle">
            by {s.author}
            {s.version ? ` · v${s.version}` : ''}
          </div>
        </div>
        {s.installed ? (
          s.installed.update_available ? (
            <Badge size="xs" tone="info" icon={<IconCloudDownload />}>
              Update
            </Badge>
          ) : s.installed.status === 'pending' ? (
            <Badge size="xs" tone="accent" icon={<IconEyeCheck />}>
              In review
            </Badge>
          ) : (
            <Badge size="xs" tone="success" icon={<IconCheck />}>
              Installed
            </Badge>
          )
        ) : s.problem ? (
          <Badge size="xs" tone="neutral">
            Can't install
          </Badge>
        ) : null}
      </div>
      <p className="line-clamp-3 flex-1 text-sm leading-relaxed text-fg-muted">{s.description || 'No description.'}</p>
      <div className="flex flex-wrap items-center gap-1 border-t border-border pt-2.5">
        {s.asks.length ? (
          s.asks.map((a) => (
            <Tooltip key={a.kind} content={`${a.label}: ${a.items.join(', ')}${a.more ? ` and ${a.more} more` : ''}`}>
              <span className="inline-flex items-center gap-1 rounded-md border border-border px-1.5 py-0.5 text-2xs text-fg-subtle">
                {ASK_ICON[a.kind] ?? <IconListCheck size={12} />}
                {ASK_SHORT[a.kind] ?? a.label} {a.kind !== 'needs' && (a.items.length + (a.more ?? 0))}
              </span>
            </Tooltip>
          ))
        ) : (
          <span className="text-2xs text-fg-subtle">Only instructions, no tools or scripts</span>
        )}
      </div>
    </button>
  )
}

/** What a skill asks for, item by item. */
export function AsksList({ asks }: { asks: HubAsk[] }) {
  if (!asks.length) return <p className="text-sm text-fg-muted">Only written instructions. It doesn't name any tools, programs, keys or scripts.</p>
  return (
    <ul className="space-y-2">
      {asks.map((a) => (
        <li key={a.kind} className="flex items-start gap-2.5">
          <span className="mt-0.5 flex size-6 shrink-0 items-center justify-center rounded-md bg-active text-fg-muted">{ASK_ICON[a.kind] ?? <IconListCheck size={12} />}</span>
          <div className="min-w-0 flex-1">
            <div className="text-xs font-medium text-fg-muted">{a.label}</div>
            <div className="mt-0.5 flex flex-wrap gap-1">
              {a.items.map((i) => (
                <span key={i} className={cn('rounded-md border border-border bg-surface px-1.5 py-0.5 text-2xs text-fg', a.kind !== 'needs' && 'font-mono')}>
                  {i}
                </span>
              ))}
              {!!a.more && <span className="text-2xs text-fg-subtle">and {a.more} more</span>}
            </div>
          </div>
        </li>
      ))}
    </ul>
  )
}

function HubPreviewSheet({ source, skill, onOpenChange, onReview }: { source: HubSource; skill: HubSkill | null; onOpenChange: (o: boolean) => void; onReview: (name: string) => void }) {
  const preview = useHubPreview(skill ? source.id : null, skill?.path ?? null)
  const actions = useHubActions()
  const [mode, setMode] = useState<'rendered' | 'raw'>('rendered')
  const [ask, setAsk] = useState<'confirm_source' | 'high_risk' | null>(null)
  const [blocked, setBlocked] = useState<SkillVetting | null>(null)
  const p = preview.data
  const vetting = blocked ?? p?.vetting ?? null
  const high = vetting?.verdict === 'high'
  const installed = p?.installed ?? skill?.installed ?? null
  const body = useMemo(() => (p?.skill_md ? splitFrontmatter(p.skill_md).body : ''), [p?.skill_md])

  const install = async (confirmSource: boolean, acceptRisk: boolean) => {
    if (!skill) return
    try {
      const s = await actions.install.mutateAsync({ source: source.id, path: skill.path, confirm_source: confirmSource || undefined, accept_risk: acceptRisk || undefined })
      ;(s.vetting?.verdict === 'high' ? toast.warning : toast.success)(`${s.name} is waiting for your review`, {
        description: s.vetting?.verdict === 'high' ? 'Its safety check found high-risk problems. We suggest you decline it.' : 'Nothing is used until you approve it.'
      })
      onOpenChange(false)
      onReview(s.name)
    } catch (e) {
      const code = hubConfirmCode(e)
      if (code) {
        if (code === 'high_risk' && isApiError(e)) setBlocked(((e.body as { vetting?: SkillVetting }).vetting ?? null) as SkillVetting | null)
        setAsk(code)
      } else toast.error("Couldn't install it", { description: errorMessage(e) })
    }
  }
  const start = () => {
    if (!source.default) setAsk('confirm_source')
    else if (high) setAsk('high_risk')
    else void install(false, false)
  }

  return (
    <Sheet
      open={!!skill}
      onOpenChange={(o) => {
        onOpenChange(o)
        if (!o) {
          setAsk(null)
          setBlocked(null)
        }
      }}
      width={680}
      title={<span className="font-mono">{skill?.name}</span>}
      description={skill ? `by ${skill.author} · from ${source.name}` : undefined}
    >
      {skill && (
        <div className="flex min-h-full flex-col">
          <div className="flex-1 space-y-5 p-5">
            <div className="space-y-2">
              <p className="text-sm leading-relaxed text-fg">{skill.description || 'No description.'}</p>
              <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-fg-subtle">
                <TrustBadge trust={source.trust} />
                <span>Made by {skill.author}</span>
                {skill.version && <span>Version {skill.version}</span>}
                {skill.license && (
                  <span className="inline-flex items-center gap-1">
                    <IconLicense size={12} /> {skill.license}
                  </span>
                )}
                {p?.commit && (
                  <button type="button" className="inline-flex items-center gap-1 font-mono hover:text-accent-text" onClick={() => openExternal(skill.url)} title={p.commit}>
                    <IconGitCommit size={12} /> {shortCommit(p.commit)}
                  </button>
                )}
                {!p?.commit && (
                  <button type="button" className="inline-flex items-center gap-1 hover:text-accent-text" onClick={() => openExternal(skill.url)}>
                    <IconExternalLink size={12} /> Open its page
                  </button>
                )}
              </div>
            </div>

            <section className="rounded-xl border border-border bg-sunken/50 px-4 py-3">
              <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-fg-subtle">What it asks for</h3>
              <AsksList asks={p?.asks ?? skill.asks} />
            </section>

            {skill.problem ? (
              <Alert tone="warning" title="This skill can't be installed">
                {skill.problem}
              </Alert>
            ) : preview.isLoading ? (
              <div className="space-y-2">
                <div className="flex items-center gap-2 text-sm text-fg-muted">
                  <IconShieldCheck size={15} /> Downloading it and running the safety check…
                </div>
                <Skeleton className="h-20 rounded-xl" />
              </div>
            ) : preview.isError ? (
              <Alert tone="danger" title="Couldn't check this skill">
                {errorMessage(preview.error)}
              </Alert>
            ) : vetting ? (
              <SafetyReport vetting={vetting} />
            ) : null}

            {p && p.files.length > 0 && (
              <section>
                <h3 className="mb-1.5 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-fg-subtle">
                  <IconFiles size={13} /> {p.files.length} {p.files.length === 1 ? 'file' : 'files'}
                </h3>
                <ul className="flex flex-wrap gap-1.5">
                  {p.files.slice(0, 40).map((f) => (
                    <li key={f} className={cn('rounded-md border border-border bg-surface px-1.5 py-0.5 font-mono text-2xs', vetting?.programs.includes(f) ? 'text-warning' : 'text-fg-subtle')}>
                      {f}
                    </li>
                  ))}
                  {p.files.length > 40 && <li className="text-2xs text-fg-subtle">and {p.files.length - 40} more</li>}
                </ul>
              </section>
            )}

            {p?.skill_md && (
              <section className="space-y-2">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-semibold uppercase tracking-wide text-fg-subtle">SKILL.md as its maker wrote it</span>
                  <div className="flex-1" />
                  <SegmentedControl
                    size="sm"
                    aria-label="SKILL.md view"
                    value={mode}
                    onChange={setMode}
                    options={[
                      { value: 'rendered', label: 'Readable', icon: <IconFileDescription size={12} /> },
                      { value: 'raw', label: 'Raw', icon: <IconCode size={12} /> }
                    ]}
                  />
                </div>
                {mode === 'rendered' ? (
                  <SkillBody body={body} />
                ) : (
                  <pre className="selectable max-h-[480px] overflow-auto rounded-xl border border-border bg-sunken px-4 py-3 font-mono text-[12px] leading-relaxed text-fg">{p.skill_md}</pre>
                )}
              </section>
            )}
          </div>

          <div className="sticky bottom-0 flex flex-wrap items-center gap-2 border-t border-border bg-elevated/95 px-5 py-3 backdrop-blur">
            <p className={cn('mr-auto text-xs', high ? 'font-medium text-danger' : 'text-fg-subtle')}>
              {installed
                ? installed.status === 'pending'
                  ? 'Waiting for your review in Pending review.'
                  : 'Already installed. Check it for updates above.'
                : high
                  ? 'We suggest you don’t install it. It could act against you.'
                  : 'It goes to Pending review first. Nothing is used until you approve it.'}
            </p>
            {installed ? (
              installed.status === 'pending' ? (
                <Button variant="secondary" leftIcon={<IconEyeCheck size={15} />} onClick={() => onReview(installed.name)}>
                  Review it
                </Button>
              ) : null
            ) : high ? (
              <Button variant="ghost" className="text-danger hover:text-danger" leftIcon={<IconAlertTriangle size={15} />} disabled={!p || !!skill.problem} loading={actions.install.isPending} onClick={start}>
                Install anyway
              </Button>
            ) : (
              <Button variant="primary" leftIcon={<IconCloudDownload size={15} />} disabled={!p || !!skill.problem} loading={actions.install.isPending} onClick={start}>
                Install for review
              </Button>
            )}
          </div>
        </div>
      )}
      <ConfirmDialog
        open={ask === 'confirm_source'}
        onOpenChange={(o) => !o && setAsk(null)}
        tone="primary"
        title={`Install from ${source.name}?`}
        description="You added this source yourself, so Sentient can't vouch for who runs it. The skill still waits in Pending review with its safety check before Sentient uses it."
        confirmLabel="Yes, install for review"
        onConfirm={() => {
          setAsk(null)
          if (high) setAsk('high_risk')
          else void install(true, false)
        }}
      />
      <ConfirmDialog
        open={ask === 'high_risk'}
        onOpenChange={(o) => !o && setAsk(null)}
        title={`Install ${skill?.name ?? 'this skill'} anyway?`}
        description="The safety check found high-risk problems: it could tell Sentient to ignore its rules or send your data somewhere. It will still wait for review, and approving it will ask again."
        confirmLabel="Install anyway"
        onConfirm={() => {
          setAsk(null)
          void install(!source.default, true)
        }}
      />
    </Sheet>
  )
}

function AddSourceDialog({ open, onOpenChange, onAdded }: { open: boolean; onOpenChange: (o: boolean) => void; onAdded: (s: HubSource) => void }) {
  const actions = useHubActions()
  const [url, setUrl] = useState('')
  const [name, setName] = useState('')
  const [error, setError] = useState('')
  const valid = /^https?:\/\/\S+$/.test(url.trim())
  const close = (o: boolean) => {
    onOpenChange(o)
    if (!o) {
      setUrl('')
      setName('')
      setError('')
    }
  }
  const add = () =>
    actions.addSource.mutate(
      { url: url.trim(), name: name.trim() || undefined },
      {
        onSuccess: (s) => {
          toast.success(`Added ${s.name}`, { description: 'Its skills get the safety check, and installing asks you once more.' })
          close(false)
          onAdded(s)
        },
        onError: (e) => (isApiError(e) && (e.status === 400 || e.status === 409) ? setError(e.detail) : toast.error("Couldn't add the source", { description: errorMessage(e) }))
      }
    )
  return (
    <Dialog
      open={open}
      onOpenChange={close}
      title="Add a source of skills"
      description="A public GitHub repository with skill folders (each with a SKILL.md), or a site that publishes a skills index. Only add sources you trust."
      footer={
        <>
          <Button variant="ghost" onClick={() => close(false)}>
            Cancel
          </Button>
          <Button variant="primary" leftIcon={<IconPlus size={15} />} disabled={!valid} loading={actions.addSource.isPending} onClick={add}>
            Add source
          </Button>
        </>
      }
    >
      <div className="space-y-3">
        <Field label="Web address" htmlFor="hub-url" error={error || undefined} description="For example https://github.com/owner/repo, or a folder inside it.">
          <Input
            id="hub-url"
            autoFocus
            placeholder="https://github.com/…"
            value={url}
            onChange={(e) => {
              setUrl(e.target.value)
              setError('')
            }}
            onKeyDown={(e) => e.key === 'Enter' && valid && add()}
          />
        </Field>
        <Field label="Name (optional)" htmlFor="hub-name">
          <Input id="hub-name" placeholder="My team's skills" value={name} onChange={(e) => setName(e.target.value)} />
        </Field>
      </div>
    </Dialog>
  )
}

/** Skills installed from the hub, the version each is pinned to, and updates. */
function InstalledPanel({ onReview }: { onReview: (name: string) => void }) {
  const installed = useHubInstalled()
  const actions = useHubActions()
  const [risky, setRisky] = useState<string | null>(null)
  const list = installed.data?.skills ?? []
  if (!list.length) return null
  const getUpdate = async (name: string, acceptRisk = false) => {
    try {
      const r = await actions.update.mutateAsync({ name, acceptRisk })
      if (!r.changed) toast(`${name} is up to date`)
      else {
        toast.success(`The new version of ${name} is waiting for your review`, { description: changeSummary(r.changes) })
        onReview(name)
      }
    } catch (e) {
      if (hubConfirmCode(e) === 'high_risk') setRisky(name)
      else toast.error("Couldn't get the update", { description: errorMessage(e) })
    }
  }
  return (
    <div className="rounded-xl border border-border bg-surface">
      <div className="flex items-center gap-2 border-b border-border px-4 py-2.5">
        <span className="text-sm font-semibold text-fg">Installed from the hub</span>
        <span className="text-xs text-fg-subtle">Each one is pinned to the exact version you reviewed.</span>
        <div className="flex-1" />
        <Button size="sm" variant="secondary" leftIcon={<IconRefresh size={14} />} loading={actions.checkUpdates.isPending} onClick={() => actions.checkUpdates.mutate(undefined, { onError: (e) => toast.error("Couldn't check for updates", { description: errorMessage(e) }) })}>
          Check for updates
        </Button>
      </div>
      <ul className="divide-y divide-border">
        {list.map((h) => (
          <li key={h.name} className="flex flex-wrap items-center gap-x-3 gap-y-1 px-4 py-2 text-sm">
            <span className="font-mono font-medium text-fg">{h.name}</span>
            <span className="text-xs text-fg-subtle">
              {h.source_name} · {h.approved_commit ? `pinned at ${shortCommit(h.approved_commit)}` : h.commit ? `fetched at ${shortCommit(h.commit)}` : h.version ? `version ${h.version}` : 'pinned by its files'}
            </span>
            <div className="flex-1" />
            {h.status === 'pending' ? (
              <Button size="sm" variant="ghost" leftIcon={<IconEyeCheck size={14} />} onClick={() => onReview(h.name)}>
                {h.approved_commit ? 'Update in review' : 'In review'}
              </Button>
            ) : h.update_available ? (
              <Button size="sm" variant="primary" leftIcon={<IconCloudDownload size={14} />} loading={actions.update.isPending && actions.update.variables?.name === h.name} onClick={() => void getUpdate(h.name)}>
                Get update
              </Button>
            ) : h.problem ? (
              <span className="text-xs text-warning">{h.problem}</span>
            ) : h.update_available === false ? (
              <span className="inline-flex items-center gap-1 text-xs text-success">
                <IconCheck size={13} /> Up to date
              </span>
            ) : null}
          </li>
        ))}
      </ul>
      <ConfirmDialog
        open={!!risky}
        onOpenChange={(o) => !o && setRisky(null)}
        title={`Get the new ${risky ?? ''} anyway?`}
        description="The safety check found high-risk problems in the new version. It will wait for review, and the version you approved stays in use until then."
        confirmLabel="Get it anyway"
        onConfirm={() => {
          const n = risky
          setRisky(null)
          if (n) void getUpdate(n, true)
        }}
      />
    </div>
  )
}

export function changeSummary(c: HubFileChanges | null | undefined): string {
  if (!c) return ''
  const parts = [c.changed.length && `${c.changed.length} changed`, c.added.length && `${c.added.length} added`, c.removed.length && `${c.removed.length} removed`].filter(Boolean)
  return parts.length ? `Files: ${parts.join(', ')}.` : 'No files changed.'
}

/** On a pending card or a skill's page: where a hub skill came from, the pinned version and what an update changes. */
export function HubOrigin({ hub, className }: { hub: HubInstalled; className?: string }) {
  const c = hub.changes
  const update = hub.status === 'pending' && !!hub.approved_commit
  return (
    <div className={cn('rounded-xl border border-border bg-sunken/50 px-4 py-3', className)}>
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <IconBuildingStore size={15} className="text-fg-muted" />
        <span className="font-medium text-fg">From {hub.source_name}</span>
        <TrustBadge trust={hub.trust} />
        <span className="text-xs text-fg-subtle">made by {hub.made_by}</span>
      </div>
      <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-0.5 pl-6 text-xs text-fg-subtle">
        {update ? (
          <span className="inline-flex items-center gap-1 font-mono">
            <IconGitCommit size={12} /> {shortCommit(hub.approved_commit)} → {shortCommit(hub.commit) || 'new version'}
          </span>
        ) : hub.commit ? (
          <span className="inline-flex items-center gap-1 font-mono" title={hub.commit}>
            <IconGitCommit size={12} /> pinned at {shortCommit(hub.commit)}
          </span>
        ) : (
          <span>pinned by the hash of its files{hub.version ? `, version ${hub.version}` : ''}</span>
        )}
        <button type="button" className="inline-flex items-center gap-1 hover:text-accent-text" onClick={() => openExternal(hub.url)}>
          <IconExternalLink size={12} /> Open at this version
        </button>
      </div>
      {update && c && (
        <div className="mt-2 space-y-1 pl-6">
          <div className="text-xs font-medium text-fg-muted">What this update changes · {changeSummary(c)}</div>
          <ul className="flex flex-wrap gap-1.5">
            {c.changed.map((f) => (
              <FileChip key={`c${f}`} file={f} kind="changed" />
            ))}
            {c.added.map((f) => (
              <FileChip key={`a${f}`} file={f} kind="added" />
            ))}
            {c.removed.map((f) => (
              <FileChip key={`r${f}`} file={f} kind="removed" />
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

function FileChip({ file, kind }: { file: string; kind: 'added' | 'removed' | 'changed' }) {
  return (
    <li
      className={cn(
        'rounded-md border px-1.5 py-0.5 font-mono text-2xs',
        kind === 'added' ? 'border-success/30 bg-success/[0.06] text-success' : kind === 'removed' ? 'border-danger/30 bg-danger/[0.06] text-danger line-through' : 'border-info/30 bg-info/[0.06] text-info'
      )}
      title={kind}
    >
      {kind === 'added' ? '+ ' : kind === 'removed' ? '− ' : '~ '}
      {file}
    </li>
  )
}
