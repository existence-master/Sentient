import { IconLayoutSidebarRightExpand, IconLock, IconWorldWww } from '@tabler/icons-react'
import { Button } from '@/components/ui'
import { openBrowserView, useBrowserView } from '@/features/browser/state'
import { api } from '@/lib/api'
import type { ApprovalView, ToolCallView } from '@/lib/chatFold'
import { cn, truncate } from '@/lib/utils'
import { outputPath } from '../files'
import { toolMeta } from '../toolMeta'
import { asRecord, DeclinedText, hostOf, prettyUrl, refLabel, Shimmer, StatusGlyph } from './bits'

export const isBrowserTool = (name: string) => name.startsWith('browser_')

interface BrowserResult {
  url?: string
  title?: string
  file?: string
  clicked?: string
  element?: string
}

function stepLabel(t: ToolCallView): string {
  const a = t.arguments as Record<string, unknown>
  const r = asRecord<BrowserResult>(t.result)
  const running = t.status === 'running'
  const meta = toolMeta(t.name)
  switch (t.name) {
    case 'browser_open': {
      const host = hostOf((a.url as string) || r?.url)
      return `${running ? 'Opening' : 'Opened'} ${host || 'a page'}`
    }
    case 'browser_click': {
      const target = refLabel(r?.clicked) ?? refLabel(r?.element) ?? refLabel(a.ref)
      if (target) return `${running ? 'Clicking' : 'Clicked'} “${truncate(target, 48)}”`
      return !running && r?.title ? `Clicked through to “${truncate(r.title, 44)}”` : meta[running ? 'running' : 'done']
    }
    case 'browser_type':
      return `${running ? 'Typing' : 'Typed'} "${truncate(String(a.text ?? ''), 40)}"`
    case 'browser_press':
      return `${running ? 'Pressing' : 'Pressed'} ${String(a.key ?? 'a key')}`
    case 'browser_select':
      return `${running ? 'Choosing' : 'Chose'} "${truncate(String(a.option ?? ''), 40)}"`
    case 'browser_switch_tab':
      return `${running ? 'Switching' : 'Switched'} to tab ${Number(a.index ?? 0) + 1}`
    default:
      return meta[running ? 'running' : 'done']
  }
}

/** A run of consecutive browser tool calls shown as one card: latest picture, address and the steps taken. */
export function BrowserCard({ tools, approvals = [] }: { tools: ToolCallView[]; approvals?: ApprovalView[] }) {
  const liveFrame = useBrowserView((s) => s.frame)
  const running = tools.some((t) => t.status === 'running' || t.status === 'awaiting_approval')
  const allDeclined = tools.length > 0 && tools.every((t) => t.status === 'denied')
  const results = tools.map((t) => asRecord<BrowserResult>(t.result))
  const progressFrame = [...tools].reverse().find((t) => t.progress?.frame)?.progress?.frame
  const shot = [...results].reverse().find((r) => r?.file)?.file
  const url =
    [...results].reverse().find((r) => r?.url)?.url ??
    progressFrame?.url ??
    ([...tools].reverse().find((t) => typeof (t.arguments as { url?: string }).url === 'string')?.arguments as { url?: string } | undefined)?.url
  const title = [...results].reverse().find((r) => r?.title)?.title ?? progressFrame?.title
  // While the browser is working, the domain event stream has the freshest picture.
  const frameImage =
    running && liveFrame && Date.now() - liveFrame.at < 60_000 ? liveFrame.image : (progressFrame?.image ?? (shot ? api.files.contentUrl(outputPath(shot)) : undefined))
  const steps = tools.slice(-4)
  const hidden = tools.length - steps.length

  return (
    <div className="flex gap-3.5 rounded-xl border border-border bg-surface p-2.5">
      <button
        type="button"
        onClick={openBrowserView}
        aria-label="Open live view"
        className={cn(
          'group relative w-40 shrink-0 self-start overflow-hidden rounded-lg border border-border',
          frameImage ? 'aspect-[16/10] bg-sunken' : 'h-[88px] bg-elevated/60'
        )}
      >
        {frameImage ? (
          <img src={frameImage} alt="" className="size-full object-cover object-top transition-transform duration-300 group-hover:scale-[1.03]" />
        ) : (
          <span className="flex size-full flex-col items-center justify-center gap-1 text-fg-subtle transition-colors group-hover:text-fg-muted">
            <span className="flex size-8 items-center justify-center rounded-full border border-border bg-surface">
              <IconWorldWww size={16} stroke={1.6} />
            </span>
            <span className="max-w-[90%] truncate text-2xs">{hostOf(url) || 'Browser'}</span>
          </span>
        )}
        {running && (
          <span className="absolute left-1.5 top-1.5 flex items-center gap-1 rounded-full bg-black/60 px-1.5 py-0.5 text-[10px] font-medium text-white backdrop-blur">
            <span className="size-1.5 animate-pulse rounded-full bg-danger" /> Live
          </span>
        )}
      </button>

      <div className="flex min-w-0 flex-1 flex-col py-0.5">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-fg">
            {running ? <Shimmer>Using the browser</Shimmer> : allDeclined ? 'Browser step not done' : 'Used the browser'}
          </span>
          <span className="text-2xs text-fg-faint">
            {tools.length} {tools.length === 1 ? 'step' : 'steps'}
          </span>
        </div>
        {url && (
          <div className="mt-0.5 flex min-w-0 items-center gap-1 text-xs text-fg-subtle" title={title ? `${title}\n${url}` : url}>
            {url.startsWith('https:') && <IconLock size={11} className="shrink-0" />}
            <span className="truncate">{prettyUrl(url)}</span>
          </div>
        )}
        <ul className="mt-2 space-y-1">
          {hidden > 0 && <li className="pl-5 text-2xs text-fg-faint">{hidden} earlier steps</li>}
          {steps.map((t) => {
            const meta = toolMeta(t.name)
            return (
              <li key={t.callId} className="flex items-center gap-2 text-xs">
                <meta.icon size={13} className={cn('shrink-0', t.status === 'error' ? 'text-danger' : t.status === 'denied' ? 'text-warning' : 'text-fg-subtle')} />
                {t.status === 'denied' ? (
                  <DeclinedText tool={t} approval={approvals.find((a) => a.callId === t.callId)} className="min-w-0 flex-1" />
                ) : (
                  <span className={cn('min-w-0 flex-1 truncate', t.status === 'running' ? 'text-fg' : 'text-fg-muted')}>{stepLabel(t)}</span>
                )}
                <StatusGlyph status={t.status} size={12} />
              </li>
            )
          })}
        </ul>
        <div className="mt-auto flex pt-2">
          <Button size="xs" variant="ghost" className="-ml-2 text-fg-muted" leftIcon={<IconLayoutSidebarRightExpand size={13} />} onClick={openBrowserView}>
            Open live view
          </Button>
        </div>
      </div>
    </div>
  )
}
