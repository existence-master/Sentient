/**
 * Visual smoke hook.
 *
 *   SENTIENT_SMOKE_SCREENSHOT=<png path>   enable; where to write the capture
 *   SENTIENT_SMOKE_ROUTE=/settings/models  hash route to open (default "/")
 *   SENTIENT_SMOKE_SIZE=1280x820           content size
 *   SENTIENT_SMOKE_DELAY_MS=700            settle time after the renderer reports ready
 *   SENTIENT_SMOKE_TIMEOUT_MS=180000       capture anyway (exit code 2) if never ready
 *
 * The renderer calls `window.sentient.readyForScreenshot()` once the route has
 * rendered and no queries are fetching; we capture the page, write the PNG and quit.
 */
import type { BrowserWindow } from 'electron'
import { mkdirSync, writeFileSync } from 'node:fs'
import { dirname } from 'node:path'

export interface SmokeConfig {
  out: string
  route: string
  size: { width: number; height: number }
  delayMs: number
  timeoutMs: number
}

export function smokeConfig(): SmokeConfig | null {
  const out = process.env.SENTIENT_SMOKE_SCREENSHOT?.trim()
  if (!out) return null
  const m = /^(\d+)x(\d+)$/.exec(process.env.SENTIENT_SMOKE_SIZE?.trim() ?? '')
  return {
    out,
    route: process.env.SENTIENT_SMOKE_ROUTE?.trim() || '/',
    size: m ? { width: Number(m[1]), height: Number(m[2]) } : { width: 1280, height: 820 },
    delayMs: Number(process.env.SENTIENT_SMOKE_DELAY_MS) || 700,
    timeoutMs: Number(process.env.SENTIENT_SMOKE_TIMEOUT_MS) || 180_000
  }
}

export class SmokeRunner {
  private done = false

  constructor(
    private cfg: SmokeConfig,
    private getWindow: () => BrowserWindow | null,
    private finish: (exitCode: number) => void
  ) {
    setTimeout(() => void this.capture('timeout'), cfg.timeoutMs).unref()
  }

  ready(): void {
    void this.capture('ready')
  }

  private async capture(reason: 'ready' | 'timeout'): Promise<void> {
    if (this.done) return
    this.done = true
    await new Promise((r) => setTimeout(r, reason === 'ready' ? this.cfg.delayMs : 0))
    const win = this.getWindow()
    try {
      if (!win || win.isDestroyed()) throw new Error('no window')
      const image = await win.webContents.capturePage()
      mkdirSync(dirname(this.cfg.out), { recursive: true })
      writeFileSync(this.cfg.out, image.toPNG())
      const { width, height } = image.getSize()
      console.log(`[smoke] ${reason}: wrote ${this.cfg.out} (${width}x${height}) route ${this.cfg.route}`)
      this.finish(reason === 'ready' ? 0 : 2)
    } catch (err) {
      console.error('[smoke] capture failed', err)
      this.finish(3)
    }
  }
}
