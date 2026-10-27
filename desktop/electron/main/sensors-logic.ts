/**
 * #170 The tray's microphone and camera indicator: what is live, the tooltip and menu text, and the red dot drawn
 * on the tray icon. Pure functions (no Electron), so `npm test` covers them.
 *
 * The engine sends the whole list to the desktop app's device connection (`sensors`, docs/API.md section 24). Push to
 * talk and dictation are known here first, so they show even before the engine's list arrives.
 */

/** One live microphone or camera (docs/API.md section 24, SensorUse). */
export interface LiveSensor {
  id: string
  sensor: 'mic' | 'camera'
  device_id: string
  device: string
  device_kind: string
  purpose: string
  since: string
  can_stop: boolean
}

export interface LiveSensors {
  mic: boolean
  camera: boolean
  active: LiveSensor[]
  camera_off: Array<{ device_id: string; device: string }>
}

/** The id the engine gives push to talk and dictation on this computer (reported as the desktop device's `mic`). */
export const LOCAL_MIC_ID = 'device:desktop:mic'

/** Read the engine's `sensors` message; anything malformed reads as nothing live. */
export function parseSensors(msg: unknown): LiveSensors {
  const m = (msg ?? {}) as Partial<LiveSensors>
  const active = Array.isArray(m.active)
    ? m.active.filter((u): u is LiveSensor => !!u && typeof u.id === 'string' && (u.sensor === 'mic' || u.sensor === 'camera'))
    : []
  const cameraOff = Array.isArray(m.camera_off) ? m.camera_off.filter((d) => d && typeof d.device === 'string') : []
  return { mic: active.some((u) => u.sensor === 'mic'), camera: active.some((u) => u.sensor === 'camera'), active, camera_off: cameraOff }
}

/** The engine's list plus push to talk or dictation on this computer, which the shell knows about first. */
export function liveSensors(engine: LiveSensors | null, localMic: string | null, now = new Date()): LiveSensor[] {
  const uses = [...(engine?.active ?? [])]
  if (localMic && !uses.some((u) => u.id === LOCAL_MIC_ID)) {
    uses.push({
      id: LOCAL_MIC_ID, sensor: 'mic', device_id: 'desktop', device: 'This computer', device_kind: 'desktop',
      purpose: localMic, since: now.toISOString(), can_stop: true
    }) // prettier-ignore
  }
  return uses
}

function what(uses: LiveSensor[]): string {
  const mic = uses.some((u) => u.sensor === 'mic')
  const camera = uses.some((u) => u.sensor === 'camera')
  return mic && camera ? 'microphone and camera' : mic ? 'microphone' : 'camera'
}

function where(device: string): string {
  return device === 'This computer' ? 'this computer' : device
}

/** Tray tooltip while something is live, e.g. "Sentient: microphone on (this computer, Push to talk)". */
export function sensorTooltip(uses: LiveSensor[]): string | null {
  if (!uses.length) return null
  if (uses.length === 1) return `Sentient: ${what(uses)} on (${where(uses[0].device)}, ${uses[0].purpose})`
  const devices = new Set(uses.map((u) => u.device))
  return `Sentient: ${what(uses)} on (${devices.size === 1 ? where(uses[0].device) : `${devices.size} devices`})`
}

/** Tray menu item that turns one off, e.g. "Turn off microphone: Maya's glasses (Voice chat)". */
export function sensorMenuLabel(use: LiveSensor): string {
  return `Turn off ${use.sensor === 'mic' ? 'microphone' : 'camera'}: ${use.device === 'This computer' ? 'this computer' : use.device} (${use.purpose})`
}

/**
 * Draw the "live" dot into a tray icon bitmap (BGRA, as `nativeImage.toBitmap()` returns it on Windows and Linux):
 * a red dot with a white ring in the bottom right corner. Returns a new buffer.
 */
export function drawLiveDot(bitmap: Uint8Array, width: number, height: number): Uint8Array {
  const out = new Uint8Array(bitmap)
  const r = Math.max(3, Math.round(Math.min(width, height) * 0.22))
  const cx = width - r - 1
  const cy = height - r - 1
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const d = Math.hypot(x + 0.5 - (cx + 0.5), y + 0.5 - (cy + 0.5))
      if (d > r + 0.5) continue
      const i = (y * width + x) * 4
      const ring = d > r - 1
      out[i] = ring ? 255 : 38 // B
      out[i + 1] = ring ? 255 : 38 // G
      out[i + 2] = ring ? 255 : 220 // R
      out[i + 3] = 255
    }
  }
  return out
}
