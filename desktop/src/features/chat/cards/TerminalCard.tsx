import { IconAlertCircle, IconExternalLink, IconFolder, IconPlayerStopFilled, IconTerminal2 } from '@tabler/icons-react'
import { useEffect, useRef } from 'react'
import { toast } from 'sonner'
import { Badge, Button } from '@/components/ui'
import { useStopCommand } from '@/hooks/terminal'
import { wasDenied, type ToolCallView } from '@/lib/chatFold'
import type { TerminalResult } from '@/lib/types'
import { cn, formatDuration } from '@/lib/utils'
import { openOutputFile } from '../files'
import { asRecord, DeclinedText, Shimmer, StatusGlyph } from './bits'

/** `terminal_run` (§18): the command, its folder, live output, exit code and a Stop button while it runs. */
export function TerminalCard({ tool }: { tool: ToolCallView }) {
  const args = tool.arguments as { command?: string; cwd?: string }
  const denied = tool.status === 'denied' || wasDenied(tool.result)
  const res = denied ? null : asRecord<TerminalResult>(tool.result)
  const running = tool.status === 'running'
  const waiting = tool.status === 'awaiting_approval'
  const refused = !!res && res.exit_code === null && !!res.error && !res.stopped && !res.timed_out
  const failed = !denied && (tool.status === 'error' || (res ? !res.ok : false))
  const stdout = res?.stdout ?? tool.progress?.stdout ?? ''
  const stderr = res?.stderr ?? tool.progress?.stderr ?? ''
  const duration = res?.duration_ms ?? tool.durationMs ?? null
  const folder = res?.cwd ?? args.cwd ?? null
  const stop = useStopCommand()
  const output = useRef<HTMLPreElement>(null)
  const onStop = () =>
    stop.mutate(tool.callId, {
      onSuccess: (r) => {
        if (!r.stopped) toast.info('That command already finished', { description: 'There was nothing left to stop.' })
      },
      onError: () => toast.error("Couldn't stop the command", { description: 'Try Stop everything in the title bar.' })
    })

  useEffect(() => {
    if (running && output.current) output.current.scrollTop = output.current.scrollHeight
  }, [running, stdout, stderr])

  const subline = waiting
    ? 'Waiting for your OK before running'
    : running
      ? null
      : denied
        ? 'Not run, so nothing changed'
        : refused
          ? 'Not run'
          : res?.stopped
            ? 'Stopped'
            : res?.timed_out
              ? 'Stopped because it took too long'
              : res && res.exit_code !== null
                ? `${res.exit_code === 0 ? 'Finished' : `Finished with exit code ${res.exit_code}`}${duration ? ` in ${formatDuration(duration)}` : ''}`
                : null

  return (
    <div className={cn('overflow-hidden rounded-xl border bg-surface', failed ? 'border-danger/30' : 'border-border')}>
      <div className="flex items-center gap-3 px-3.5 py-2.5">
        <div
          className={cn(
            'flex size-8 shrink-0 items-center justify-center rounded-lg',
            failed ? 'bg-danger/10 text-danger' : denied ? 'bg-warning/12 text-warning' : 'bg-accent/10 text-accent-text'
          )}
        >
          <IconTerminal2 size={17} />
        </div>
        <div className="min-w-0 flex-1">
          <div className="truncate font-mono text-sm text-fg">{denied ? <DeclinedText tool={tool} /> : args.command || 'A command'}</div>
          <div className="flex min-w-0 items-center gap-1 text-xs text-fg-subtle">
            {running ? (
              <Shimmer>Running on this computer</Shimmer>
            ) : (
              <span className="truncate">{subline}</span>
            )}
            {folder && !denied && (
              <span className="flex min-w-0 items-center gap-1 truncate">
                <IconFolder size={11} className="ml-1 shrink-0" />
                <span className="truncate font-mono">{folder}</span>
              </span>
            )}
          </div>
        </div>
        {running && (
          <Button size="sm" variant="ghost" leftIcon={<IconPlayerStopFilled size={12} />} loading={stop.isPending} onClick={onStop}>
            Stop
          </Button>
        )}
        {res?.shell && !running && <Badge size="xs">{res.shell}</Badge>}
        <span className="flex size-4 shrink-0 items-center justify-center">
          <StatusGlyph status={denied ? 'denied' : failed && tool.status === 'done' ? 'error' : tool.status} />
        </span>
      </div>

      {!waiting && !denied && (stdout || stderr || res?.error || res?.output_file) && (
        <div className="space-y-2.5 px-3.5 pb-3">
          {(stdout || stderr) && (
            <pre
              ref={output}
              className="selectable max-h-56 overflow-auto whitespace-pre-wrap break-words rounded-lg border border-border bg-sunken px-3 py-2 font-mono text-[12px] leading-relaxed text-fg-muted"
            >
              {stdout}
              {stderr && <span className="text-danger/90">{stdout && !stdout.endsWith('\n') ? '\n' : ''}{stderr}</span>}
              {running && <span className="ml-0.5 inline-block h-3.5 w-1.5 translate-y-0.5 animate-caret bg-fg-subtle" />}
            </pre>
          )}
          {running && tool.progress?.status && <div className="text-xs text-fg-subtle">{tool.progress.status}</div>}
          {res?.error && (
            <div className="flex items-start gap-2 rounded-lg border border-danger/25 bg-danger/[0.06] px-3 py-2 text-sm text-fg-muted">
              <IconAlertCircle size={15} className="mt-0.5 shrink-0 text-danger" />
              <span className="selectable min-w-0 break-words">{res.error}</span>
            </div>
          )}
          {res?.output_file && (
            <button
              type="button"
              onClick={() => void openOutputFile(res.output_file as string)}
              className="flex items-center gap-1.5 text-xs text-fg-subtle transition-colors hover:text-accent-text"
            >
              <IconExternalLink size={12} /> Open the full output
            </button>
          )}
        </div>
      )}
    </div>
  )
}
