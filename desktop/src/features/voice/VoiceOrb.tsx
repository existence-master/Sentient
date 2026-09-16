import { useEffect, useRef } from 'react'
import type { VoiceStateB } from '@/lib/leap/types-b'
import { cn } from '@/lib/utils'
import { levels } from './session'

type RGB = [number, number, number]

function cssColor(value: string, fallback: RGB): RGB {
  try {
    const c = document.createElement('canvas')
    c.width = c.height = 1
    const x = c.getContext('2d', { willReadFrequently: true })
    if (!x) return fallback
    x.fillStyle = `rgb(${fallback.join(',')})`
    x.fillStyle = value
    x.fillRect(0, 0, 1, 1)
    const d = x.getImageData(0, 0, 1, 1).data
    return [d[0], d[1], d[2]]
  } catch {
    return fallback
  }
}

function rotateHue([r, g, b]: RGB, deg: number): RGB {
  const rn = r / 255
  const gn = g / 255
  const bn = b / 255
  const max = Math.max(rn, gn, bn)
  const min = Math.min(rn, gn, bn)
  const l = (max + min) / 2
  let h = 0
  let s = 0
  if (max !== min) {
    const d = max - min
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min)
    h = max === rn ? (gn - bn) / d + (gn < bn ? 6 : 0) : max === gn ? (bn - rn) / d + 2 : (rn - gn) / d + 4
    h *= 60
  }
  h = (h + deg + 360) % 360
  const c = (1 - Math.abs(2 * l - 1)) * s
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1))
  const m = l - c / 2
  const [r1, g1, b1] = h < 60 ? [c, x, 0] : h < 120 ? [x, c, 0] : h < 180 ? [0, c, x] : h < 240 ? [0, x, c] : h < 300 ? [x, 0, c] : [c, 0, x]
  return [Math.round((r1 + m) * 255), Math.round((g1 + m) * 255), Math.round((b1 + m) * 255)]
}

const rgba = ([r, g, b]: RGB, a: number) => `rgba(${r},${g},${b},${Math.max(0, Math.min(1, a))})`

const MOOD: Record<VoiceStateB | 'connecting', number> = {
  standby: 0.2,
  idle: 0.35,
  connecting: 0.45,
  listening: 0.75,
  transcribing: 0.85,
  thinking: 0.9,
  speaking: 1
}

/**
 * The voice visualizer: Sentient's dark core ringed by living light. Mic level drives it while
 * listening, output level while speaking; orbiting arcs mean transcribing/thinking.
 */
export function VoiceOrb({
  state,
  connecting = false,
  muted = false,
  size = 360,
  onClick,
  label,
  className
}: {
  state: VoiceStateB
  connecting?: boolean
  muted?: boolean
  size?: number
  onClick?: () => void
  label?: string
  className?: string
}) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const props = useRef({ state, connecting, muted })
  props.current = { state, connecting, muted }

  useEffect(() => {
    const el = canvas.current
    const ctx = el?.getContext('2d')
    if (!el || !ctx) return
    const dpr = Math.min(2, window.devicePixelRatio || 1)
    el.width = Math.round(size * dpr)
    el.height = Math.round(size * dpr)

    let accent: RGB = [124, 92, 255]
    let light = false
    const readTheme = () => {
      const root = document.documentElement
      accent = cssColor(getComputedStyle(root).getPropertyValue('--accent').trim() || '#7c5cff', accent)
      light = root.classList.contains('light')
    }
    readTheme()
    const themeTimer = window.setInterval(readTheme, 1000)

    let energy = 0
    let mood = 0.3
    let spin = 0
    let raf = 0
    let last = performance.now()
    const ripples: number[] = []
    let lastRipple = 0

    const S = size
    const cx = S / 2
    const cy = S / 2
    const R = S * 0.25

    const frame = (now: number) => {
      const dt = Math.min(0.05, (now - last) / 1000)
      last = now
      const t = now / 1000
      const { state: st, connecting: conn, muted: mute } = props.current
      const busy = st === 'thinking' || st === 'transcribing' || conn

      const raw = mute ? 0 : st === 'listening' ? levels.mic * 5.5 : st === 'speaking' ? levels.out * 4.2 : 0
      const target = Math.min(1, raw)
      energy += (target - energy) * (target > energy ? 0.32 : 0.07)
      mood += ((conn ? MOOD.connecting : MOOD[st]) * (mute ? 0.6 : 1) - mood) * 0.05
      spin += dt * (busy ? 2.4 : 0.35)

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, S, S)
      // standby breathes slowly, like something resting
      const breathe = Math.sin(t * (st === 'standby' ? 0.55 : 1.1)) * 0.5 + 0.5

      // halo
      const halo = ctx.createRadialGradient(cx, cy, R * 0.5, cx, cy, R * 2)
      halo.addColorStop(0, rgba(accent, (light ? 0.22 : 0.3) * mood * (0.7 + energy * 0.6)))
      halo.addColorStop(0.55, rgba(accent, (light ? 0.08 : 0.1) * mood))
      halo.addColorStop(1, rgba(accent, 0))
      ctx.fillStyle = halo
      ctx.fillRect(0, 0, S, S)

      // ripples while speaking
      if (st === 'speaking' && energy > 0.28 && t - lastRipple > 0.42) {
        ripples.push(t)
        lastRipple = t
      }
      for (let i = ripples.length - 1; i >= 0; i--) {
        const age = t - ripples[i]
        if (age > 1.6) {
          ripples.splice(i, 1)
          continue
        }
        ctx.beginPath()
        ctx.arc(cx, cy, R * (1.08 + age * 0.62), 0, Math.PI * 2)
        ctx.strokeStyle = rgba(accent, (1 - age / 1.6) * 0.32)
        ctx.lineWidth = 1.5
        ctx.stroke()
      }

      // living light: three noisy layers
      ctx.globalCompositeOperation = light ? 'source-over' : 'lighter'
      const layers: Array<{ hue: number; phase: number; speed: number; scale: number }> = [
        { hue: 0, phase: 0, speed: 1, scale: 1 },
        { hue: 14, phase: 2.1, speed: -0.8, scale: 0.94 },
        { hue: -12, phase: 4.2, speed: 0.6, scale: 0.9 }
      ]
      const N = 96
      ctx.filter = `blur(${Math.round(S * 0.012)}px)`
      for (const L of layers) {
        const amp = R * (0.03 + energy * 0.26 + (busy ? 0.035 : 0)) * L.scale
        const color = rotateHue(accent, L.hue)
        ctx.beginPath()
        for (let i = 0; i <= N; i++) {
          const a = (i / N) * Math.PI * 2
          const ta = t * L.speed
          const n =
            Math.sin(3 * a + ta * 1.3 + L.phase + spin * 0.3) * 0.5 +
            Math.sin(5 * a - ta * 1.9 + L.phase * 1.7) * 0.3 * (0.5 + energy) +
            Math.sin(2 * a + ta * 0.8 - L.phase) * 0.2
          const r = R * (1.02 + (1 - L.scale) * 0.6) + amp * n + breathe * R * 0.025
          const x = cx + Math.cos(a) * r
          const y = cy + Math.sin(a) * r
          if (i === 0) ctx.moveTo(x, y)
          else ctx.lineTo(x, y)
        }
        ctx.closePath()
        const g = ctx.createRadialGradient(cx, cy, R * 0.55, cx, cy, R * 1.35)
        g.addColorStop(0, rgba(color, (light ? 0.5 : 0.62) * mood))
        g.addColorStop(0.7, rgba(color, (light ? 0.3 : 0.34) * mood))
        g.addColorStop(1, rgba(color, 0.02))
        ctx.fillStyle = g
        ctx.fill()
      }
      ctx.globalCompositeOperation = 'source-over'
      ctx.filter = 'none'

      // orbit arcs while working
      if (busy) {
        for (let k = 0; k < 2; k++) {
          const rr = R * (1.34 + k * 0.13)
          const start = spin * (k ? -1.15 : 1) + k * 2
          const arc = ctx.createLinearGradient(cx + Math.cos(start) * rr, cy + Math.sin(start) * rr, cx + Math.cos(start + 1.8) * rr, cy + Math.sin(start + 1.8) * rr)
          arc.addColorStop(0, rgba(accent, 0))
          arc.addColorStop(1, rgba(rotateHue(accent, k ? 30 : -20), 0.75))
          ctx.beginPath()
          ctx.arc(cx, cy, rr, start, start + 1.8)
          ctx.strokeStyle = arc
          ctx.lineWidth = 2.2
          ctx.lineCap = 'round'
          ctx.stroke()
        }
      }

      // dark core
      const rc = R * (0.8 - energy * 0.04)
      const core = ctx.createRadialGradient(cx - rc * 0.32, cy - rc * 0.38, rc * 0.05, cx, cy, rc)
      core.addColorStop(0, '#2b2b35')
      core.addColorStop(0.65, '#0d0d12')
      core.addColorStop(1, '#040406')
      ctx.beginPath()
      ctx.arc(cx, cy, rc, 0, Math.PI * 2)
      ctx.fillStyle = core
      ctx.fill()
      ctx.lineWidth = 1.25
      ctx.strokeStyle = rgba(accent, 0.35 + energy * 0.4)
      ctx.stroke()
      // glint
      const glint = ctx.createRadialGradient(cx - rc * 0.35, cy - rc * 0.45, 0, cx - rc * 0.35, cy - rc * 0.45, rc * 0.55)
      glint.addColorStop(0, 'rgba(255,255,255,0.10)')
      glint.addColorStop(1, 'rgba(255,255,255,0)')
      ctx.fillStyle = glint
      ctx.beginPath()
      ctx.arc(cx, cy, rc, 0, Math.PI * 2)
      ctx.fill()
      // inner pulse
      const pulse = ctx.createRadialGradient(cx, cy, 0, cx, cy, rc * (0.35 + energy * 0.45))
      pulse.addColorStop(0, rgba(accent, 0.12 + energy * 0.35))
      pulse.addColorStop(1, rgba(accent, 0))
      ctx.fillStyle = pulse
      ctx.beginPath()
      ctx.arc(cx, cy, rc, 0, Math.PI * 2)
      ctx.fill()

      raf = requestAnimationFrame(frame)
    }
    raf = requestAnimationFrame(frame)
    return () => {
      cancelAnimationFrame(raf)
      window.clearInterval(themeTimer)
    }
  }, [size])

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={!onClick}
      aria-label={label ?? 'Voice visualizer'}
      className={cn('no-drag relative rounded-full outline-none focus-visible:ring-4 focus-visible:ring-accent/25 disabled:cursor-default', className)}
      style={{ width: size, height: size }}
    >
      <canvas ref={canvas} style={{ width: size, height: size }} className="pointer-events-none" />
    </button>
  )
}
