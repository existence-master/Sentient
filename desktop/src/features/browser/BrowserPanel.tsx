import {
  IconAppWindow,
  IconLayoutSidebarRightCollapse,
  IconLock,
  IconLogin2,
  IconWorldOff,
  IconWorldWww
} from '@tabler/icons-react'
import { AnimatePresence, motion } from 'motion/react'
import { useEffect, useRef, useState } from 'react'
import { useLocation } from 'react-router'
import { toast } from 'sonner'
import { Alert, Button, EmptyState, IconButton, Input, Select, Skeleton, StatusDot } from '@/components/ui'
import { api, errorMessage, isNotImplemented } from '@/lib/api'
import { cn, relativeTime } from '@/lib/utils'
import { previewMode } from '@/features/devices/preview'
import { hostOf, prettyUrl } from '@/features/chat/cards/bits'
import { useBrowserActions, useBrowserProfiles, useBrowserStatus, useBrowserView } from './state'

export const BROWSER_PANEL_WIDTH = 420

/** Right-side live view of Sentient's browser (§12). Docked next to the page so chatting continues. */
export function BrowserPanel() {
  const open = useBrowserView((s) => s.open)
  const setOpen = useBrowserView((s) => s.setOpen)
  const location = useLocation()
  const opened = useRef(false)

  // `?browser=1` opens the panel (deep links and screenshots).
  useEffect(() => {
    if (opened.current || !/(^|[?&])browser=1\b/.test(location.search)) return
    opened.current = true
    setOpen(true)
  }, [location.search, setOpen])

  return (
    <AnimatePresence initial={false}>
      {open && (
        <motion.aside
          key="browser-panel"
          initial={{ width: 0, opacity: 0 }}
          animate={{ width: BROWSER_PANEL_WIDTH, opacity: 1 }}
          exit={{ width: 0, opacity: 0 }}
          transition={{ duration: 0.22, ease: [0.2, 0.8, 0.2, 1] }}
          className="h-full shrink-0 overflow-hidden border-l border-t border-border bg-surface"
          aria-label="Browser live view"
        >
          <div style={{ width: BROWSER_PANEL_WIDTH }} className="flex h-full flex-col">
            <PanelBody onClose={() => setOpen(false)} />
          </div>
        </motion.aside>
      )}
    </AnimatePresence>
  )
}

function useNow(ms = 5000) {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const t = window.setInterval(() => setNow(Date.now()), ms)
    return () => window.clearInterval(t)
  }, [ms])
  return now
}

function PanelBody({ onClose }: { onClose: () => void }) {
  const status = useBrowserStatus()
  const frame = useBrowserView((s) => s.frame)
  const actions = useBrowserActions()
  const profiles = useBrowserProfiles()
  const [url, setUrl] = useState('')
  const [picked, setPicked] = useState<string | null>(null)
  const [shotFailed, setShotFailed] = useState(false)
  const now = useNow()
  const s = status.data
  const missing = isNotImplemented(status.error) && !previewMode()
  const running = !!s?.running
  // LIVE only while the browser runs and a frame arrived in the last few seconds.
  const live = running && !!frame && now - frame.at < 6000
  const activeTab = s?.tabs.find((t) => t.active)
  const shownUrl = frame?.url ?? activeTab?.url
  const imageSrc = frame?.image ?? (running && !shotFailed ? api.browser.screenshotUrl(Math.floor(now / 15000)) : undefined)

  const profileList = profiles.data?.profiles ?? []
  // a picked profile that was deleted or renamed meanwhile falls back to the open one
  const profile = (picked && profileList.some((p) => p.name === picked) ? picked : null) ?? s?.profile ?? 'default'
  const pickedProfile = profileList.find((p) => p.name === profile)

  const openForSignIn = () => {
    const raw = url.trim()
    const target = raw ? (/^https?:\/\//i.test(raw) ? raw : `https://${raw}`) : undefined
    actions.open.mutate({ url: target, profile: profileList.length > 1 ? profile : undefined }, {
      onSuccess: () => toast.success('The browser is open', { description: 'Sign in yourself in the window that opened, then close it when you are done.' }),
      onError: (e) => toast.error("Couldn't open the browser", { description: errorMessage(e) })
    })
  }

  const dot = missing || s?.available === false ? 'neutral' : running ? 'success' : 'neutral'
  const label = missing ? 'Not available yet' : s?.available === false ? 'Not set up' : running ? (live ? 'Working now' : 'Open') : 'Closed'

  return (
    <>
      <div className="flex h-12 shrink-0 items-center gap-2.5 border-b border-border pl-4 pr-2">
        <IconWorldWww size={17} className="text-accent-text" />
        <span className="text-sm font-semibold text-fg">Browser</span>
        <span className="flex items-center gap-1.5 text-xs text-fg-subtle">
          <StatusDot tone={dot} pulse={live} />
          {label}
        </span>
        <span className="flex-1" />
        <IconButton size="sm" label="Close panel" icon={<IconLayoutSidebarRightCollapse size={16} />} onClick={onClose} />
      </div>

      <div className="min-h-0 flex-1 space-y-5 overflow-y-auto p-4">
        {missing ? (
          <EmptyState
            compact
            icon={<IconWorldOff />}
            title="Browsing isn't available yet"
            description="This version of Sentient's engine can't use a browser yet. It will show up here once it can."
          />
        ) : (
          <>
            {s?.available === false && (
              <Alert tone="warning" title="Sentient needs a browser">
                {s.error || 'Install Microsoft Edge or Google Chrome so Sentient can browse websites for you.'}
              </Alert>
            )}

            <div className="overflow-hidden rounded-xl border border-border bg-sunken">
              <div className="flex h-8 items-center gap-1.5 border-b border-border bg-elevated/60 px-2.5 text-xs text-fg-muted">
                {shownUrl?.startsWith('https:') ? <IconLock size={12} className="shrink-0 text-fg-subtle" /> : <IconWorldWww size={12} className="shrink-0 text-fg-subtle" />}
                <span className="min-w-0 flex-1 truncate" title={shownUrl}>
                  {shownUrl ? prettyUrl(shownUrl, 56) : 'No page open'}
                </span>
                {live ? (
                  <span className="flex items-center gap-1 rounded-full bg-danger/12 px-1.5 text-[10px] font-semibold uppercase tracking-wide text-danger">
                    <span className="size-1.5 animate-pulse rounded-full bg-danger" /> Live
                  </span>
                ) : (
                  frame && (
                    <span className="shrink-0 rounded-full bg-active px-1.5 text-[10px] text-fg-subtle">
                      Last seen {relativeTime(new Date(frame.at).toISOString(), now)}
                    </span>
                  )
                )}
              </div>
              <div className="relative aspect-[16/10] w-full">
                {status.isLoading ? (
                  <Skeleton className="absolute inset-0 rounded-none" />
                ) : imageSrc ? (
                  <img
                    src={imageSrc}
                    alt={frame?.title ?? activeTab?.title ?? 'Browser'}
                    onError={() => !frame && setShotFailed(true)}
                    className={cn('absolute inset-0 size-full object-cover object-top transition-opacity', !running && 'opacity-45 grayscale-[35%]')}
                  />
                ) : (
                  <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 px-8 text-center">
                    <IconAppWindow size={28} stroke={1.4} className="text-fg-faint" />
                    <p className="text-sm text-fg-subtle">Nothing to show yet. When Sentient browses for you, you can watch it here.</p>
                  </div>
                )}
              </div>
            </div>
            {frame && (
              <p className="-mt-3 text-2xs text-fg-faint">
                {frame.title ? `${frame.title} · ` : ''}updated {relativeTime(new Date(frame.at).toISOString(), now)}
              </p>
            )}

            {!!s?.tabs.length && (
              <section>
                <div className="mb-1.5 text-2xs font-medium uppercase tracking-wide text-fg-subtle">Open tabs</div>
                <div className="divide-y divide-border rounded-xl border border-border">
                  {s.tabs.map((t) => (
                    <div key={t.index} className="flex items-center gap-2.5 px-3 py-2">
                      <IconWorldWww size={14} className={cn('shrink-0', t.active ? 'text-accent-text' : 'text-fg-faint')} />
                      <div className="min-w-0 flex-1">
                        <div className={cn('truncate text-sm', t.active ? 'text-fg' : 'text-fg-muted')}>{t.title || hostOf(t.url)}</div>
                        <div className="truncate text-2xs text-fg-subtle">{hostOf(t.url)}</div>
                      </div>
                      {t.active && <span className="text-2xs text-fg-subtle">Showing</span>}
                    </div>
                  ))}
                </div>
              </section>
            )}

            <section className="rounded-xl border border-border bg-elevated/40 p-4">
              <div className="flex items-start gap-3">
                <div className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-accent/10 text-accent-text">
                  <IconLogin2 size={17} />
                </div>
                <div className="min-w-0">
                  <div className="text-sm font-medium text-fg">Sign in to a website</div>
                  <p className="mt-1 text-xs leading-relaxed text-fg-muted">
                    Some sites, like an online store or your bank, need you to be signed in first. Open the browser, sign in yourself, then close the
                    window. Sentient uses the same browser afterwards. It never sees or types your passwords.
                  </p>
                </div>
              </div>
              <div className="mt-3 space-y-2">
                {profileList.length > 1 && (
                  <Select
                    size="sm"
                    aria-label="Browser profile"
                    value={profile}
                    onValueChange={setPicked}
                    options={profileList.map((p) => ({ value: p.name, label: p.notes ? `${p.name} (${p.notes})` : p.name }))}
                    className="w-full"
                  />
                )}
                <Input
                  size="sm"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && openForSignIn()}
                  placeholder="Website to open, for example example.com"
                  aria-label="Website to open"
                />
                <Button
                  size="sm"
                  variant="primary"
                  className="w-full"
                  loading={actions.open.isPending}
                  disabled={s?.available === false}
                  leftIcon={<IconLogin2 size={14} />}
                  onClick={openForSignIn}
                >
                  {pickedProfile?.kind === 'attach' ? 'Open in your browser' : 'Open browser to sign in'}
                </Button>
              </div>
            </section>
          </>
        )}
      </div>

      {!missing && (
        <div className="flex h-12 shrink-0 items-center gap-2 border-t border-border px-4">
          <span className="min-w-0 flex-1 truncate text-2xs text-fg-subtle">
            {s?.attached ? `Connected to your browser (${s.profile})` : s?.engine ? `Using ${s.engine}${s.profile && s.profile !== 'default' ? `, profile ${s.profile}` : ''}` : ''}
          </span>
          <Button
            size="sm"
            variant="ghost"
            disabled={!running}
            loading={actions.close.isPending}
            onClick={() => actions.close.mutate(undefined, { onError: (e) => toast.error("Couldn't close the browser", { description: errorMessage(e) }) })}
          >
            {s?.attached ? 'Disconnect' : 'Close browser'}
          </Button>
        </div>
      )}
    </>
  )
}
