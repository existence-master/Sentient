/**
 * Reconnecting WebSocket to `WS /ws` (docs/API.md §1).
 *
 *   import { live } from '@/lib/ws'
 *   const off = live.onDomain('task.updated', (e) => upsert(e.data))
 *   live.onChat((e) => ...)            // chat/agent events (no dot in type)
 *   live.send({ type: 'chat.cancel', session_id })
 *
 * Exponential backoff (0.5s -> 15s), keepalive ping every 25s, stale detection.
 */
import type {
  ChatEvent,
  ClientMessage,
  DomainEvent,
  DomainEventType,
  ServerMessage
} from './types'
import type { Connection } from '@/types/bridge'

export type SocketState = 'idle' | 'connecting' | 'open' | 'reconnecting' | 'closed'

type Listener<T> = (value: T) => void

const PING_MS = 25_000
const STALE_MS = 70_000
const MAX_BACKOFF_MS = 15_000

export function isDomainEvent(msg: { type: string }): msg is DomainEvent {
  return msg.type.includes('.') && msg.type !== 'approval.ack'
}

export class LiveSocket {
  state: SocketState = 'idle'
  private ws: WebSocket | null = null
  private conn: Connection | null = null
  private attempt = 0
  private reconnectTimer: number | undefined
  private pingTimer: number | undefined
  private lastMessageAt = 0
  private manual = false
  private readonly all = new Set<Listener<ServerMessage>>()
  private readonly stateListeners = new Set<Listener<SocketState>>()

  connect(conn: Connection): void {
    const same = this.conn && this.conn.baseUrl === conn.baseUrl && this.conn.token === conn.token
    this.conn = conn
    this.manual = false
    if (same && (this.state === 'open' || this.state === 'connecting')) return
    this.teardown()
    this.open()
  }

  disconnect(): void {
    this.manual = true
    this.teardown()
    this.setState('closed')
  }

  /** Force a reconnect now (e.g. after the engine restarted). */
  reconnect(): void {
    if (!this.conn) return
    this.attempt = 0
    this.teardown()
    this.open()
  }

  /** Returns false when the socket isn't open (callers may fall back to REST). */
  send(msg: ClientMessage): boolean {
    if (this.ws?.readyState !== WebSocket.OPEN) return false
    this.ws.send(JSON.stringify(msg))
    return true
  }

  get isOpen(): boolean {
    return this.ws?.readyState === WebSocket.OPEN
  }

  /** Resolves true once open, false after `timeoutMs`. */
  waitOpen(timeoutMs = 4000): Promise<boolean> {
    if (this.isOpen) return Promise.resolve(true)
    return new Promise((resolve) => {
      const t = window.setTimeout(() => {
        off()
        resolve(false)
      }, timeoutMs)
      const off = this.onState((s) => {
        if (s === 'open') {
          clearTimeout(t)
          off()
          resolve(true)
        }
      })
    })
  }

  subscribe(fn: Listener<ServerMessage>): () => void {
    this.all.add(fn)
    return () => this.all.delete(fn)
  }

  onChat(fn: Listener<ChatEvent>): () => void {
    return this.subscribe((msg) => {
      if (!isDomainEvent(msg)) fn(msg as ChatEvent)
    })
  }

  onDomain<K extends DomainEventType>(type: K | '*', fn: Listener<DomainEvent<K>>): () => void {
    return this.subscribe((msg) => {
      if (isDomainEvent(msg) && (type === '*' || msg.type === type)) fn(msg as DomainEvent<K>)
    })
  }

  onState(fn: Listener<SocketState>): () => void {
    this.stateListeners.add(fn)
    return () => this.stateListeners.delete(fn)
  }

  // ------------------------------------------------------------------ internals
  private setState(s: SocketState) {
    if (this.state === s) return
    this.state = s
    this.stateListeners.forEach((l) => l(s))
  }

  private open(): void {
    if (!this.conn) return
    const url = `${this.conn.baseUrl.replace(/^http/, 'ws')}/ws?token=${encodeURIComponent(this.conn.token)}`
    this.setState(this.attempt === 0 ? 'connecting' : 'reconnecting')
    let ws: WebSocket
    try {
      ws = new WebSocket(url)
    } catch {
      this.scheduleReconnect()
      return
    }
    this.ws = ws
    ws.onopen = () => {
      this.attempt = 0
      this.lastMessageAt = Date.now()
      this.setState('open')
      this.startPing()
    }
    ws.onmessage = (ev) => {
      this.lastMessageAt = Date.now()
      if (typeof ev.data !== 'string') return
      let msg: ServerMessage
      try {
        msg = JSON.parse(ev.data) as ServerMessage
      } catch {
        return
      }
      if (!msg || typeof msg.type !== 'string') return
      this.all.forEach((l) => {
        try {
          l(msg)
        } catch (err) {
          console.error('[ws] listener failed', err)
        }
      })
    }
    ws.onclose = () => {
      if (this.ws !== ws) return
      this.stopPing()
      this.ws = null
      if (!this.manual) this.scheduleReconnect()
    }
    ws.onerror = () => {
      /* onclose follows */
    }
  }

  private scheduleReconnect(): void {
    this.setState('reconnecting')
    const delay = Math.min(MAX_BACKOFF_MS, 500 * 2 ** this.attempt) * (0.8 + Math.random() * 0.4)
    this.attempt++
    clearTimeout(this.reconnectTimer)
    this.reconnectTimer = window.setTimeout(() => this.open(), delay)
  }

  private startPing(): void {
    this.stopPing()
    this.pingTimer = window.setInterval(() => {
      if (Date.now() - this.lastMessageAt > STALE_MS) {
        this.ws?.close()
        return
      }
      this.send({ type: 'ping' })
    }, PING_MS)
  }

  private stopPing(): void {
    clearInterval(this.pingTimer)
    this.pingTimer = undefined
  }

  private teardown(): void {
    clearTimeout(this.reconnectTimer)
    this.stopPing()
    const ws = this.ws
    this.ws = null
    if (ws) {
      ws.onclose = null
      ws.onmessage = null
      try {
        ws.close()
      } catch {
        /* ignore */
      }
    }
  }
}

/** The app-wide live socket. */
export const live = new LiveSocket()
