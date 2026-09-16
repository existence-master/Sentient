import { IconAlertCircle, IconBox, IconChevronRight, IconCode, IconCpu, IconExternalLink, IconFile, IconFileSpreadsheet, IconTerminal2 } from '@tabler/icons-react'
import { AnimatePresence, motion } from 'motion/react'
import { useEffect, useRef, useState } from 'react'
import { Badge, JsonView, Markdown, Tooltip } from '@/components/ui'
import { api } from '@/lib/api'
import { wasDenied, type ToolCallView } from '@/lib/chatFold'
import type { SandboxResult } from '@/lib/types'
import { cn, formatDuration } from '@/lib/utils'
import { fileBase, IMAGE_FILE_RE, openOutputFile, outputPath } from '../files'
import { asRecord, DeclinedText, Shimmer, StatTiles, StatusGlyph } from './bits'

/** `execute_code` (§11): purpose, live output, result, created files, and the code folded away. */
export function CodeRunCard({ tool }: { tool: ToolCallView }) {
  const args = tool.arguments as { code?: string; purpose?: string }
  const denied = tool.status === 'denied' || wasDenied(tool.result)
  const res = denied ? null : asRecord<SandboxResult>(tool.result)
  const plainResult = res || denied ? null : tool.result
  const running = tool.status === 'running'
  const waiting = tool.status === 'awaiting_approval'
  const failed = !denied && (tool.status === 'error' || (res ? res.ok === false : false))
  const stdout = res?.stdout ?? tool.progress?.stdout ?? ''
  const stderr = res?.stderr ?? tool.progress?.stderr ?? ''
  const duration = res?.duration_ms ?? tool.durationMs ?? null
  const files = res?.files_created ?? []
  const code = (args.code ?? '').replace(/\s+$/, '')
  const lines = code ? code.split('\n').length : 0
  const [showCode, setShowCode] = useState(false)
  const output = useRef<HTMLPreElement>(null)

  useEffect(() => {
    if (running && output.current) output.current.scrollTop = output.current.scrollHeight
  }, [running, stdout, stderr])

  const title = args.purpose?.trim() || 'Ran some code'
  const subline = waiting
    ? 'Waiting for your OK before running'
    : running
      ? null
      : denied
        ? 'Not run, so nothing changed'
        : failed
          ? `Stopped with a problem${duration ? ` after ${formatDuration(duration)}` : ''}`
          : `Finished${duration ? ` in ${formatDuration(duration)}` : ''}`
  const backend = res?.backend
  const hasResult = res && res.result !== null && res.result !== undefined && res.result !== ''
  const showOutput = !!(stdout || stderr)

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
          <div className="truncate text-sm font-medium text-fg">{denied ? <DeclinedText tool={tool} /> : title}</div>
          <div className="truncate text-xs text-fg-subtle">{running ? <Shimmer>Running on this computer</Shimmer> : subline}</div>
        </div>
        {backend && (
          <Tooltip content={backend === 'docker' ? 'Ran inside an isolated container with no access to your keys' : 'Ran in a separate, locked-down folder with no access to your keys'}>
            <span>
              <Badge size="xs" icon={backend === 'docker' ? <IconBox /> : <IconCpu />}>
                {backend === 'docker' ? 'In a container' : 'On this computer'}
              </Badge>
            </span>
          </Tooltip>
        )}
        <span className="flex size-4 shrink-0 items-center justify-center">
          <StatusGlyph status={denied ? 'denied' : failed && tool.status === 'done' ? 'error' : tool.status} />
        </span>
      </div>

      {(showOutput || hasResult || res?.error || files.length > 0 || plainResult !== undefined) && !waiting && !denied && (
        <div className="space-y-2.5 px-3.5 pb-3">
          {showOutput && (
            <pre
              ref={output}
              className="selectable max-h-44 overflow-auto whitespace-pre-wrap break-words rounded-lg border border-border bg-sunken px-3 py-2 font-mono text-[12px] leading-relaxed text-fg-muted"
            >
              {stdout}
              {stderr && <span className="text-danger/90">{stdout && !stdout.endsWith('\n') ? '\n' : ''}{stderr}</span>}
              {running && <span className="ml-0.5 inline-block h-3.5 w-1.5 translate-y-0.5 animate-caret bg-fg-subtle" />}
            </pre>
          )}

          {hasResult && (
            <div>
              <div className="mb-1.5 text-2xs font-medium uppercase tracking-wide text-fg-subtle">Result</div>
              {typeof res.result === 'string' || typeof res.result === 'number' ? (
                <div className="selectable text-sm text-fg">{String(res.result)}</div>
              ) : (
                (StatTiles({ value: res.result }) ?? (
                  <div className="max-h-56 overflow-auto rounded-lg border border-border bg-sunken px-3 py-2">
                    <JsonView value={res.result} collapsedDepth={2} />
                  </div>
                ))
              )}
            </div>
          )}

          {!res && plainResult !== undefined && plainResult !== null && plainResult !== '' && (
            <pre className="selectable max-h-44 overflow-auto whitespace-pre-wrap rounded-lg border border-border bg-sunken px-3 py-2 font-mono text-[12px] text-fg-muted">
              {typeof plainResult === 'string' ? plainResult : JSON.stringify(plainResult, null, 2)}
            </pre>
          )}

          {res?.error && (
            <div className="flex items-start gap-2 rounded-lg border border-danger/25 bg-danger/[0.06] px-3 py-2 text-sm text-fg-muted">
              <IconAlertCircle size={15} className="mt-0.5 shrink-0 text-danger" />
              <span className="selectable min-w-0 break-words">{res.error}</span>
            </div>
          )}

          {files.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {files.map((f) => (
                <FileChip key={f} name={f} />
              ))}
            </div>
          )}
        </div>
      )}

      {code && (
        <>
          <button
            type="button"
            onClick={() => setShowCode((s) => !s)}
            aria-expanded={showCode}
            className="flex h-8 w-full items-center gap-2 border-t border-border px-3.5 text-left text-xs text-fg-subtle transition-colors hover:bg-hover hover:text-fg"
          >
            <IconCode size={14} />
            <span>{showCode ? 'Hide code' : 'Show code'}</span>
            <span className="text-fg-faint">
              {lines} {lines === 1 ? 'line' : 'lines'} of Python
            </span>
            <span className="flex-1" />
            <IconChevronRight size={13} className={cn('text-fg-faint transition-transform', showCode && 'rotate-90')} />
          </button>
          <AnimatePresence initial={false}>
            {showCode && (
              <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
                <div className="px-3.5 pb-3 [&_.md-code]:max-h-80 [&_.md-code]:overflow-auto">
                  <Markdown>{'```python\n' + code + '\n```'}</Markdown>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </>
      )}
    </div>
  )
}

function FileChip({ name }: { name: string }) {
  const base = fileBase(name)
  const image = IMAGE_FILE_RE.test(base)
  const sheet = /\.(csv|xlsx?|tsv)$/i.test(base)
  return (
    <button
      type="button"
      onClick={() => void openOutputFile(name)}
      title={`Open ${base}`}
      className="group flex h-11 max-w-64 items-center gap-2.5 rounded-lg border border-border bg-elevated/50 pl-1.5 pr-3 text-left transition-colors hover:border-border-strong hover:bg-elevated"
    >
      {image ? (
        <img src={api.files.contentUrl(outputPath(name))} alt="" className="size-8 rounded-md border border-border object-cover" />
      ) : (
        <span className="flex size-8 items-center justify-center rounded-md bg-accent/10 text-accent-text">
          {sheet ? <IconFileSpreadsheet size={16} /> : <IconFile size={16} />}
        </span>
      )}
      <span className="min-w-0">
        <span className="block truncate text-xs font-medium text-fg">{base}</span>
        <span className="flex items-center gap-1 text-2xs text-fg-subtle group-hover:text-accent-text">
          Open <IconExternalLink size={10} />
        </span>
      </span>
    </button>
  )
}
