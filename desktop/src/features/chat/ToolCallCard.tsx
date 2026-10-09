import { IconCheck, IconChevronRight, IconShieldQuestion, IconX } from '@tabler/icons-react'
import { AnimatePresence, motion } from 'motion/react'
import { useState } from 'react'
import { JsonView, Spinner } from '@/components/ui'
import type { ApprovalView, ToolCallView } from '@/lib/chatFold'
import { cn, formatDuration } from '@/lib/utils'
import { api } from '@/lib/api'
import { ApprovalCard } from './ApprovalCard'
import { asRecord, DeclinedText, StatusGlyph } from './cards/bits'
import { CodeRunCard } from './cards/CodeRunCard'
import { isSubagentTool, SubagentCard } from './cards/SubagentCard'
import { TerminalCard } from './cards/TerminalCard'
import { IMAGE_FILE_RE, openOutputFile, outputPath } from './files'
import { argsPreview, toolMeta } from './toolMeta'

function GenericToolCard({
  tool,
  approval,
  onApprove
}: {
  tool: ToolCallView
  approval?: ApprovalView
  onApprove?: (approvalId: string, decision: 'allow' | 'allow_session' | 'deny') => void
}) {
  const [open, setOpen] = useState(false)
  const meta = toolMeta(tool.name)
  const running = tool.status === 'running'
  const failed = tool.status === 'error'
  const denied = tool.status === 'denied'
  const preview = argsPreview(tool.arguments)
  const hasResult = tool.result !== undefined && tool.result !== null && tool.result !== ''

  return (
    <div className="space-y-2">
      <div
        className={cn(
          'overflow-hidden rounded-xl border bg-surface transition-colors',
          failed ? 'border-danger/30' : denied ? 'border-border' : 'border-border',
          open && 'border-border-strong'
        )}
      >
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          aria-expanded={open}
          className="flex h-9 w-full items-center gap-2.5 px-3 text-left text-sm transition-colors hover:bg-hover"
        >
          <span className={cn('flex size-5 shrink-0 items-center justify-center', failed ? 'text-danger' : denied ? 'text-warning' : 'text-fg-subtle')}>
            <meta.icon size={15} stroke={1.75} />
          </span>
          {denied ? (
            <DeclinedText tool={tool} approval={approval} className="min-w-0 flex-1 font-medium" />
          ) : (
            <>
              <span className={cn('shrink-0 font-medium', running ? 'text-fg-muted' : 'text-fg')}>{running ? meta.running : meta.done}</span>
              {preview && <span className="min-w-0 flex-1 truncate font-mono text-xs text-fg-subtle">{preview}</span>}
              {!preview && <span className="flex-1" />}
            </>
          )}
          {tool.durationMs !== undefined && tool.durationMs !== null && !running && (
            <span className="shrink-0 text-2xs tabular-nums text-fg-faint">{formatDuration(tool.durationMs)}</span>
          )}
          <span className="flex size-4 shrink-0 items-center justify-center">
            {running ? (
              <Spinner size={13} className="text-fg-subtle" />
            ) : tool.status === 'awaiting_approval' ? (
              <IconShieldQuestion size={14} className="text-warning" />
            ) : failed ? (
              <IconX size={14} className="text-danger" />
            ) : denied ? (
              <StatusGlyph status="denied" />
            ) : (
              <IconCheck size={14} className="text-success" />
            )}
          </span>
          <IconChevronRight size={14} className={cn('shrink-0 text-fg-faint transition-transform duration-150', open && 'rotate-90')} />
        </button>
        <AnimatePresence initial={false}>
          {open && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.16 }}
              className="overflow-hidden"
            >
              <div className="space-y-3 border-t border-border bg-sunken/40 px-3.5 py-3">
                <div>
                  <div className="mb-1 text-2xs font-medium uppercase tracking-wide text-fg-subtle">Tool</div>
                  <div className="font-mono text-xs text-fg-muted">{tool.name}</div>
                </div>
                {Object.keys(tool.arguments ?? {}).length > 0 && (
                  <div>
                    <div className="mb-1 text-2xs font-medium uppercase tracking-wide text-fg-subtle">Arguments</div>
                    <JsonView value={tool.arguments} collapsedDepth={3} />
                  </div>
                )}
                <div>
                  <div className={cn('mb-1 text-2xs font-medium uppercase tracking-wide', failed ? 'text-danger' : 'text-fg-subtle')}>
                    {failed ? 'Error' : 'Result'}
                  </div>
                  {running ? (
                    <div className="text-xs text-fg-subtle">Waiting for result…</div>
                  ) : !hasResult ? (
                    <div className="text-xs text-fg-subtle">No result</div>
                  ) : typeof tool.result === 'string' ? (
                    <pre className={cn('selectable max-h-72 overflow-auto whitespace-pre-wrap break-words font-mono text-xs', failed ? 'text-danger' : 'text-fg-muted')}>
                      {tool.result}
                    </pre>
                  ) : (
                    <div className="max-h-80 overflow-auto">
                      <JsonView value={tool.result} collapsedDepth={2} />
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
      {approval && <ApprovalCard approval={approval} onRespond={onApprove} />}
    </div>
  )
}

type ApproveFn = (approvalId: string, decision: 'allow' | 'allow_session' | 'deny') => void

/** Picks a dedicated card for code runs, commands, helpers and device photos; everything else uses the generic row. */
export function ToolCallCard({ tool, approval, onApprove }: { tool: ToolCallView; approval?: ApprovalView; onApprove?: ApproveFn }) {
  const special =
    tool.name === 'execute_code' ? (
      <CodeRunCard tool={tool} />
    ) : tool.name === 'terminal_run' ? (
      <TerminalCard tool={tool} />
    ) : isSubagentTool(tool.name) ? (
      <SubagentCard tool={tool} />
    ) : null
  if (special) {
    return (
      <div className="space-y-2">
        {special}
        {approval && <ApprovalCard approval={approval} onRespond={onApprove} />}
      </div>
    )
  }
  return (
    <div className="space-y-2">
      <GenericToolCard tool={tool} approval={approval} onApprove={onApprove} />
      <DeviceMedia tool={tool} />
    </div>
  )
}

/** `device_take_photo` / `device_capture_screen` results: show the picture and what Sentient saw. */
function DeviceMedia({ tool }: { tool: ToolCallView }) {
  if (tool.name !== 'device_take_photo' && tool.name !== 'device_capture_screen' && tool.name !== 'browser_screenshot') return null
  const res = asRecord<{ file?: string; description?: string }>(tool.result)
  if (!res?.file || !IMAGE_FILE_RE.test(res.file)) return null
  return (
    <div className="flex items-start gap-3 pl-1">
      <button type="button" onClick={() => void openOutputFile(res.file as string)} className="shrink-0 overflow-hidden rounded-xl border border-border bg-sunken" title="Open picture">
        <img src={api.files.contentUrl(outputPath(res.file))} alt={res.description ?? ''} className="h-28 max-w-52 object-cover" />
      </button>
      {res.description && <p className="pt-1 text-sm leading-relaxed text-fg-muted">{res.description}</p>}
    </div>
  )
}
