/**
 * Spawns and supervises the engine.
 *
 * - development: `.venv/Scripts/python.exe -m sentient serve` from the repo
 * - packaged:    the frozen `resources/engine/sentient-engine.exe serve`, so the
 *                user never installs Python (see electron/main/paths.ts)
 *
 * Everything else is identical in both modes, including `~/.sentient` as the data folder:
 * - picks a free loopback port and a random 32-byte token per launch
 * - pipes stdout/stderr to <home>/logs/backend.log (token redacted) and keeps a tail
 * - polls GET /api/health (then verifies the token with /api/bootstrap)
 * - restarts on crash with exponential backoff, gives up after repeated failures
 * - kills the whole process tree on stop (Windows: taskkill /T /F)
 */
import { type ChildProcess, spawn, spawnSync } from 'node:child_process'
import { randomBytes } from 'node:crypto'
import { EventEmitter } from 'node:events'
import { createWriteStream, mkdirSync, type WriteStream } from 'node:fs'
import { createServer } from 'node:net'
import { join } from 'node:path'
import type { BackendStatus } from '../../src/types/bridge'
import { homePaths, resolveEngine } from './paths'

const START_TIMEOUT_MS = Number(process.env.SENTIENT_BACKEND_START_TIMEOUT_MS) || 120_000
const MAX_CONSECUTIVE_CRASHES = 5
const TAIL_LINES = 300

function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const srv = createServer()
    srv.unref()
    srv.on('error', reject)
    srv.listen(0, '127.0.0.1', () => {
      const addr = srv.address()
      const port = typeof addr === 'object' && addr ? addr.port : 0
      srv.close(() => resolve(port))
    })
  })
}

function portIsFree(port: number): Promise<boolean> {
  return new Promise((resolve) => {
    const srv = createServer()
    srv.once('error', () => resolve(false))
    srv.listen(port, '127.0.0.1', () => srv.close(() => resolve(true)))
  })
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

export class BackendManager extends EventEmitter {
  readonly token = randomBytes(32).toString('base64url')
  port = 0
  status: BackendStatus = { state: 'starting' }

  private proc: ChildProcess | null = null
  private exited = true
  private log: WriteStream | null = null
  private tail: string[] = []
  private partial = ''
  private crashes = 0
  private readyAt = 0
  private generation = 0
  private restartTimer: NodeJS.Timeout | null = null
  private stopping = false
  private enginePath = ''
  private frozen = false

  get baseUrl(): string {
    return `http://127.0.0.1:${this.port}`
  }

  get logPath(): string {
    return join(homePaths.logs(), 'backend.log')
  }

  logTail(lines = 80): string {
    return this.tail.slice(-lines).join('\n')
  }

  /** Start (or restart from scratch, e.g. the user pressed Retry). */
  async start(): Promise<void> {
    this.stopping = false
    this.crashes = 0
    await this.killCurrent()
    await this.spawnOnce()
  }

  async restart(): Promise<void> {
    await this.start()
  }

  /** Synchronous so it can run inside before-quit. */
  stop(): void {
    this.stopping = true
    this.generation++
    if (this.restartTimer) clearTimeout(this.restartTimer)
    this.killTreeSync()
    this.setStatus({ state: 'stopped' })
    this.log?.end()
    this.log = null
  }

  // ------------------------------------------------------------------ internals
  private setStatus(s: BackendStatus): void {
    // `pythonPath` is what the error screen shows; in a packaged build it is the frozen engine.
    this.status = { pythonPath: this.enginePath, logPath: this.logPath, ...s }
    this.emit('status', this.status)
  }

  private openLog(): void {
    if (this.log) return
    try {
      mkdirSync(homePaths.logs(), { recursive: true })
      this.log = createWriteStream(this.logPath, { flags: 'a' })
    } catch {
      this.log = null
    }
  }

  private record(chunk: string): void {
    const text = chunk.split(this.token).join('<token>')
    this.log?.write(text)
    const lines = (this.partial + text).split(/\r?\n/)
    this.partial = lines.pop() ?? ''
    this.tail.push(...lines)
    if (this.tail.length > TAIL_LINES) this.tail.splice(0, this.tail.length - TAIL_LINES)
  }

  private async spawnOnce(): Promise<void> {
    const gen = ++this.generation
    if (!this.port || !(await portIsFree(this.port))) this.port = await freePort()
    const engine = resolveEngine()
    this.enginePath = engine.command
    this.frozen = engine.frozen
    this.openLog()
    this.record(
      `\n===== ${new Date().toISOString()} starting engine: ${engine.command} (${engine.source}) port ${this.port} =====\n`
    )
    this.setStatus({ state: 'starting', attempt: this.crashes + 1 })

    const env: NodeJS.ProcessEnv = {
      ...process.env,
      SENTIENT_GATEWAY_TOKEN: this.token,
      PYTHONUNBUFFERED: '1',
      PYTHONIOENCODING: 'utf-8'
    }
    // SENTIENT_HOME (if set) is inherited from process.env; otherwise the engine uses ~/.sentient.

    // The frozen engine runs from the data folder, which must exist before we chdir into it.
    try {
      mkdirSync(engine.cwd, { recursive: true })
    } catch {
      /* the engine creates it too */
    }

    let child: ChildProcess
    try {
      child = spawn(engine.command, [...engine.args, '--host', '127.0.0.1', '--port', String(this.port)], {
        cwd: engine.cwd,
        env,
        windowsHide: true,
        stdio: ['ignore', 'pipe', 'pipe'],
        detached: process.platform !== 'win32'
      })
    } catch (err) {
      this.fail(`${this.missingEngineMessage()} (${String(err)})`)
      return
    }
    this.proc = child
    this.exited = false
    child.stdout?.setEncoding('utf8').on('data', (d: string) => this.record(d))
    child.stderr?.setEncoding('utf8').on('data', (d: string) => this.record(d))
    child.on('error', (err: NodeJS.ErrnoException) => {
      if (gen !== this.generation) return
      this.exited = true
      if (err.code === 'ENOENT') {
        this.fail(this.missingEngineMessage())
      } else {
        this.record(`\n[shell] spawn error: ${err.message}\n`)
      }
    })
    child.on('exit', (code, signal) => {
      this.exited = true
      if (this.proc === child) this.proc = null
      this.record(`\n[shell] engine exited (code ${code ?? 'null'}${signal ? `, signal ${signal}` : ''})\n`)
      if (gen === this.generation && !this.stopping) this.onUnexpectedExit(code, signal)
    })

    const result = await this.waitReady(gen)
    if (gen !== this.generation || this.stopping) return
    if (result === 'ready') {
      this.readyAt = Date.now()
      this.setStatus({ state: 'ready', baseUrl: this.baseUrl })
    } else if (result === 'timeout') {
      this.generation++ // ignore the exit we are about to cause
      this.killTreeSync()
      this.fail(`The engine didn't answer within ${Math.round(START_TIMEOUT_MS / 1000)} seconds.`)
    }
    // 'exited' is handled by onUnexpectedExit
  }

  private async waitReady(gen: number): Promise<'ready' | 'timeout' | 'exited' | 'superseded'> {
    const deadline = Date.now() + START_TIMEOUT_MS
    while (Date.now() < deadline) {
      if (gen !== this.generation) return 'superseded'
      if (this.exited) return 'exited'
      try {
        const r = await fetch(`${this.baseUrl}/api/health`, { signal: AbortSignal.timeout(1500) })
        if (r.ok) {
          const auth = await fetch(`${this.baseUrl}/api/bootstrap`, {
            headers: { Authorization: `Bearer ${this.token}` },
            signal: AbortSignal.timeout(5000)
          })
          if (auth.ok) return 'ready'
        }
      } catch {
        /* not up yet */
      }
      await sleep(300)
    }
    return 'timeout'
  }

  private onUnexpectedExit(code: number | null, signal: NodeJS.Signals | null): void {
    if (this.status.state === 'failed') return
    if (this.readyAt && Date.now() - this.readyAt > 120_000) this.crashes = 0
    this.readyAt = 0
    this.crashes++
    const message = `The engine stopped unexpectedly (${signal ? `signal ${signal}` : `exit code ${code}`}).`
    if (this.crashes >= MAX_CONSECUTIVE_CRASHES) {
      this.fail(`${message} It failed ${this.crashes} times in a row.`)
      return
    }
    const retryInMs = Math.min(30_000, 1000 * 2 ** (this.crashes - 1))
    this.setStatus({ state: 'crashed', message, logTail: this.logTail(), willRestart: true, attempt: this.crashes })
    this.setStatus({ state: 'restarting', message, retryInMs, attempt: this.crashes })
    const gen = this.generation
    this.restartTimer = setTimeout(() => {
      if (gen === this.generation && !this.stopping) void this.spawnOnce()
    }, retryInMs)
  }

  /** Plain words for the two very different ways the engine can be missing. */
  private missingEngineMessage(): string {
    if (this.frozen) {
      return (
        "Sentient couldn't start its own engine. The installation looks incomplete - " +
        'reinstalling Sentient should fix it.'
      )
    }
    return (
      `Python wasn't found (${this.enginePath}). Install Python 3.12+ and create the project ` +
      'virtualenv, or set SENTIENT_PYTHON to your Python executable.'
    )
  }

  private fail(message: string): void {
    this.record(`\n[shell] ${message}\n`)
    this.setStatus({ state: 'failed', message, logTail: this.logTail(), willRestart: false })
  }

  private async killCurrent(): Promise<void> {
    this.generation++
    if (this.restartTimer) clearTimeout(this.restartTimer)
    const child = this.proc
    if (!child || this.exited) return
    const done = new Promise<void>((r) => child.once('exit', () => r()))
    this.killTreeSync()
    await Promise.race([done, sleep(5000)])
  }

  private killTreeSync(): void {
    const child = this.proc
    if (!child || child.pid === undefined || this.exited) return
    try {
      if (process.platform === 'win32') {
        spawnSync('taskkill', ['/pid', String(child.pid), '/T', '/F'], { windowsHide: true, timeout: 10_000 })
      } else {
        try {
          process.kill(-child.pid, 'SIGTERM')
        } catch {
          child.kill('SIGTERM')
        }
        setTimeout(() => {
          try {
            process.kill(-(child.pid as number), 'SIGKILL')
          } catch {
            /* already gone */
          }
        }, 3000).unref()
      }
    } catch {
      /* best effort */
    }
  }
}
