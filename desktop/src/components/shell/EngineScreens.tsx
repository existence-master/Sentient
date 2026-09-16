import { IconCopy, IconFolderOpen, IconRefresh } from '@tabler/icons-react'
import { motion } from 'motion/react'
import { useEffect, useState } from 'react'
import { Logo } from '@/components/brand/Logo'
import { Button, ProgressBar, Tooltip } from '@/components/ui'
import { errorMessage } from '@/lib/api'
import { getBridge } from '@/lib/bridge'
import { copyText } from '@/lib/utils'
import { useConnection } from '@/stores/connection'
import type { BackendStatus } from '@/types/bridge'
import { TitleBar } from './TitleBar'

/** Branded splash while the engine starts. */
export function Splash({ status }: { status: BackendStatus }) {
  const [slow, setSlow] = useState(false)
  useEffect(() => {
    const t = setTimeout(() => setSlow(true), 12_000)
    return () => clearTimeout(t)
  }, [])

  const message =
    status.state === 'restarting'
      ? `Restarting the engine${status.retryInMs ? ` in ${Math.ceil(status.retryInMs / 1000)}s` : ''}…`
      : status.state === 'crashed'
        ? 'The engine stopped. Trying again…'
        : (status.message ?? 'Waking up your assistant…')

  return (
    <div className="flex h-full flex-col bg-bg">
      <TitleBar minimal />
      <div className="relative flex flex-1 flex-col items-center justify-center overflow-hidden">
        <div
          aria-hidden
          className="pointer-events-none absolute left-1/2 top-1/2 size-[520px] -translate-x-1/2 -translate-y-1/2 rounded-full opacity-[0.07] blur-3xl"
          style={{ background: 'radial-gradient(circle, var(--accent), transparent 65%)' }}
        />
        <motion.div initial={{ opacity: 0, scale: 0.94 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.5 }} className="flex flex-col items-center">
          <Logo size={88} animated />
          <h1 className="mt-7 text-2xl font-semibold tracking-tight text-fg">Sentient</h1>
          <p className="mt-2 text-sm text-fg-muted">{message}</p>
          <div className="mt-6 w-44">
            <ProgressBar size="xs" />
          </div>
          <p className="mt-6 h-5 text-xs text-fg-subtle transition-opacity" style={{ opacity: slow ? 1 : 0 }}>
            The first launch can take a little longer while everything loads.
          </p>
        </motion.div>
      </div>
    </div>
  )
}

/** Shown when the engine couldn't start (or the bootstrap request failed). */
export function EngineError({ status, error, onRetry }: { status?: BackendStatus; error?: unknown; onRetry?: () => void }) {
  const restart = useConnection((s) => s.restart)
  const bridge = getBridge()
  const [copied, setCopied] = useState(false)
  const logTail = status?.logTail?.trim()
  // Packaged builds ship a frozen engine; only a development checkout runs Python itself.
  const bundledEngine = /sentient-engine(\.exe)?$/i.test(status?.pythonPath ?? '')

  useEffect(() => {
    const t = setTimeout(() => bridge.readyForScreenshot(), 600)
    return () => clearTimeout(t)
  }, [bridge])

  const detail = status?.message ?? (error ? errorMessage(error) : 'Something went wrong.')

  return (
    <div className="flex h-full flex-col bg-bg">
      <TitleBar minimal />
      <div className="flex flex-1 items-center justify-center overflow-y-auto p-8">
        <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} className="w-full max-w-2xl">
          <div className="flex items-center gap-4">
            <Logo size={48} glow="white" />
            <div>
              <h1 className="text-xl font-semibold tracking-tight text-fg">Sentient&apos;s engine didn&apos;t start</h1>
              <p className="mt-1 text-sm text-fg-muted">{detail}</p>
            </div>
          </div>

          {logTail && (
            <div className="mt-6 overflow-hidden rounded-xl border border-border bg-sunken">
              <div className="flex h-9 items-center justify-between border-b border-border px-3.5 text-xs text-fg-subtle">
                <span>Last lines of backend.log</span>
                <button
                  type="button"
                  className="flex items-center gap-1 rounded px-1.5 py-0.5 hover:bg-hover hover:text-fg"
                  onClick={async () => {
                    if (await copyText(logTail)) {
                      setCopied(true)
                      setTimeout(() => setCopied(false), 1500)
                    }
                  }}
                >
                  <IconCopy size={12} /> {copied ? 'Copied' : 'Copy'}
                </button>
              </div>
              <pre className="selectable max-h-72 overflow-y-auto whitespace-pre-wrap break-all px-3.5 py-3 font-mono text-[11.5px] leading-relaxed text-fg-muted">
                {logTail.replace(/\n{2,}/g, '\n')}
              </pre>
            </div>
          )}

          <div className="mt-6 flex flex-wrap items-center gap-2">
            <Button variant="primary" leftIcon={<IconRefresh size={16} />} onClick={() => (onRetry ? onRetry() : void restart())}>
              Retry
            </Button>
            {bridge.isDesktop && (
              <Button leftIcon={<IconFolderOpen size={16} />} onClick={() => void bridge.openPath('logs')}>
                Open logs folder
              </Button>
            )}
          </div>

          <div className="mt-8 space-y-2 text-sm text-fg-subtle">
            <p className="font-medium text-fg-muted">Things to check</p>
            <ul className="list-disc space-y-1 pl-5">
              {bundledEngine ? (
                <li>Your antivirus isn&apos;t blocking Sentient&apos;s engine. Reinstalling Sentient usually puts it back.</li>
              ) : (
                <>
                  <li>Python 3.12+ is installed, or set SENTIENT_PYTHON to the Python that has Sentient installed.</li>
                  <li>In a development checkout, the repository&apos;s .venv exists and `pip install -e .` has been run.</li>
                </>
              )}
              <li>Another program isn&apos;t blocking local connections to 127.0.0.1.</li>
            </ul>
            {status?.pythonPath &&
              (bundledEngine ? (
                <Tooltip content={status.pythonPath}>
                  <p className="w-fit cursor-default pt-2 text-xs">Sentient engine</p>
                </Tooltip>
              ) : (
                <p className="pt-2 font-mono text-xs">Engine: {status.pythonPath}</p>
              ))}
            {status?.logPath && <p className="font-mono text-xs">Log: {status.logPath}</p>}
          </div>
        </motion.div>
      </div>
    </div>
  )
}
