import { useId } from 'react'
import { cn } from '@/lib/utils'

/**
 * The Sentient orb (v2 mark: a dark core ringed by soft light), redrawn as a
 * theme-aware SVG. `glow="accent"` tints the halo with the current accent.
 */
export function Logo({
  size = 28,
  glow = 'accent',
  animated = false,
  className
}: {
  size?: number
  glow?: 'accent' | 'white'
  animated?: boolean
  className?: string
}) {
  const id = useId().replace(/:/g, '')
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 64 64"
      aria-label="Sentient"
      role="img"
      className={cn('shrink-0 overflow-visible', className)}
    >
      <defs>
        <radialGradient id={`halo-${id}`} cx="50%" cy="50%" r="50%">
          <stop offset="52%" stopColor={glow === 'accent' ? 'var(--accent)' : '#ffffff'} stopOpacity="0.95" />
          <stop offset="70%" stopColor={glow === 'accent' ? 'var(--accent)' : '#ffffff'} stopOpacity="0.35" />
          <stop offset="100%" stopColor={glow === 'accent' ? 'var(--accent)' : '#ffffff'} stopOpacity="0" />
        </radialGradient>
        <radialGradient id={`core-${id}`} cx="42%" cy="38%" r="70%">
          <stop offset="0%" stopColor="#26262e" />
          <stop offset="100%" stopColor="#050507" />
        </radialGradient>
      </defs>
      <circle cx="32" cy="32" r="31" fill={`url(#halo-${id})`} className={animated ? 'origin-center animate-breathe' : undefined} />
      <circle cx="32" cy="32" r="18.5" fill={`url(#core-${id})`} />
      <circle cx="32" cy="32" r="18.5" fill="none" stroke="#ffffff" strokeOpacity="0.14" strokeWidth="0.8" />
    </svg>
  )
}

export function Wordmark({ className, name = 'Sentient' }: { className?: string; name?: string }) {
  return <span className={cn('font-semibold tracking-tight text-fg', className)}>{name}</span>
}
