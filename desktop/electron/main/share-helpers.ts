/**
 * Pure helpers for sharing a window or a screen region (issue #172). No Electron imports, so
 * `npm test` can run them under plain Node.
 */

export interface Size {
  width: number
  height: number
}

export interface Rect extends Size {
  x: number
  y: number
}

/** Smallest region worth sharing, in screen points; a smaller drag counts as a click. */
export const MIN_REGION = 8
/** Longest side of a shared image. Vision models shrink bigger images anyway, and pay for every pixel. */
export const MAX_SIDE = 1920

/** The rectangle between two corners of a drag, whichever way it went. */
export function rectFromDrag(a: { x: number; y: number }, b: { x: number; y: number }): Rect {
  return { x: Math.min(a.x, b.x), y: Math.min(a.y, b.y), width: Math.abs(b.x - a.x), height: Math.abs(b.y - a.y) }
}

/**
 * A region reported by the overlay page, kept inside the display and rounded, or null when it is
 * missing, malformed or smaller than MIN_REGION. The page is ours, but its answer is checked anyway.
 */
export function cleanRegion(raw: unknown, display: Size): Rect | null {
  if (!raw || typeof raw !== 'object') return null
  const r = raw as Record<string, unknown>
  const nums = [r.x, r.y, r.width, r.height]
  if (!nums.every((n) => typeof n === 'number' && Number.isFinite(n))) return null
  const [x, y, w, h] = nums as number[]
  const clamp = (v: number, max: number) => Math.round(Math.max(0, Math.min(max, v)))
  const left = clamp(x, display.width)
  const top = clamp(y, display.height)
  const out = { x: left, y: top, width: clamp(x + w, display.width) - left, height: clamp(y + h, display.height) - top }
  return out.width >= MIN_REGION && out.height >= MIN_REGION ? out : null
}

/**
 * Map a region in display points onto a screenshot of that display, which has its own pixel size
 * (the display's scale factor, or a smaller thumbnail). Always inside the image and at least 1x1.
 */
export function regionInImage(region: Rect, display: Size, image: Size): Rect {
  const sx = image.width / display.width
  const sy = image.height / display.height
  const x = Math.max(0, Math.min(image.width - 1, Math.floor(region.x * sx)))
  const y = Math.max(0, Math.min(image.height - 1, Math.floor(region.y * sy)))
  const right = Math.max(x + 1, Math.min(image.width, Math.ceil((region.x + region.width) * sx)))
  const bottom = Math.max(y + 1, Math.min(image.height, Math.ceil((region.y + region.height) * sy)))
  return { x, y, width: right - x, height: bottom - y }
}

/** `size` scaled down (never up) so its longest side is at most `max`. */
export function fitWithin(size: Size, max = MAX_SIDE): Size {
  const longest = Math.max(size.width, size.height)
  if (longest <= max || longest <= 0) return { width: Math.round(size.width), height: Math.round(size.height) }
  const k = max / longest
  return { width: Math.max(1, Math.round(size.width * k)), height: Math.max(1, Math.round(size.height * k)) }
}

export interface WindowSource {
  /** desktopCapturer id, "window:<native id>:0" */
  id: string
  name: string
  /** The thumbnail is empty (a minimized or cloaked window). */
  empty?: boolean
}

/** Desktop and system surfaces that are listed as windows on some systems but are never what the user means. */
const NOT_A_WINDOW = new Set(['program manager', 'windows input experience', 'desktop', 'dock', 'menubar', 'window server', 'notification center', 'control center'])

/** The native part of a source id, so "window:123:0" and "window:123:1" compare equal. */
function nativeId(id: string): string {
  const m = /^window:([^:]+)/.exec(id)
  return m ? m[1] : id
}

/**
 * The window to share: the front-most one that isn't Sentient's own. The capturer lists windows
 * front to back (z-order on Windows, CGWindowList order on macOS), so the first other window is the
 * one the user was looking at when they pressed the shortcut, also when Sentient itself had focus.
 */
export function pickActiveWindow<T extends WindowSource>(sources: T[], ownIds: Iterable<string>): T | null {
  const own = new Set([...ownIds].filter(Boolean).map(nativeId))
  return (
    sources.find((s) => !own.has(nativeId(s.id)) && !s.empty && s.name.trim() !== '' && !NOT_A_WINDOW.has(s.name.trim().toLowerCase())) ??
    null
  )
}

/**
 * A file name for the capture: the window's title (so the model and the files folder show what it
 * was), or "Screen region". Characters Windows or macOS refuse in file names are dropped.
 */
export function captureFileName(kind: 'window' | 'region', title: string): string {
  const clean = title
    .replace(/[<>:"/\\|?*\u0000-\u001f]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/^\.+/, '')
    .slice(0, 80)
    .trim()
  const base = kind === 'region' ? 'Screen region' : clean || 'Shared window'
  return `${base}.png`
}
