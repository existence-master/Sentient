import {
  IconAlertTriangle,
  IconArrowRight,
  IconCircleCheckFilled,
  IconDownload,
  IconEar,
  IconHandStop,
  IconMessageCircle,
  IconMicrophone,
  IconMicrophoneOff,
  IconPhoneOff,
  IconPlayerStopFilled,
  IconSend,
  IconSettings,
  IconVolume,
  IconX
} from '@tabler/icons-react'
import { AnimatePresence, motion } from 'motion/react'
import { useEffect, useRef, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router'
import { TitleBar } from '@/components/shell/TitleBar'
import { Alert, Button, IconButton, Input, Kbd, ProgressBar, SegmentedControl, Skeleton, StatusDot, Tooltip, type Tone } from '@/components/ui'
import { useHotkey } from '@/hooks/useHotkey'
import { useVoicePrepare, useVoiceStatus } from '@/hooks/voice'
import { errorMessage, isNotImplemented } from '@/lib/api'
import { useConfig } from '@/hooks/core'
import type { VoiceStateB, VoiceStatusB } from '@/lib/leap/types-b'
import type { VoiceStatus } from '@/lib/types'
import { useWakeStore } from './wake'
import { cn, isEditableTarget } from '@/lib/utils'
import { PREVIEW_KINDS, previewAllowed, previewState, type PreviewKind } from './preview'
import { useVoiceStore, voice, type VoicePhase } from './session'
import { VoiceOrb } from './VoiceOrb'
import { VoiceTranscript } from './VoiceTranscript'

/** Mounted VoiceMode views; the session ends only when this drops to zero. */
let mountedVoiceViews = 0

/**
 * Full-screen voice conversation over `WS /ws/voice` (docs/API.md §9).
 * Entered from the sidebar, the composer, the tray and the global shortcut (`voice-mode` command).
 * Dev-only: `#/voice?voicePreview=listening|transcribing|thinking|speaking|setup|ended`.
 */
export function VoiceMode() {
  const navigate = useNavigate()
  const [params] = useSearchParams()
  const status = useVoiceStatus()
  const prepare = useVoicePrepare()
  const phase = useVoiceStore((s) => s.phase)
  const state = useVoiceStore((s) => s.state)
  const turns = useVoiceStore((s) => s.turns)
  const sessionId = useVoiceStore((s) => s.sessionId)
  const mode = useVoiceStore((s) => s.mode)
  const muted = useVoiceStore((s) => s.muted)
  const talking = useVoiceStore((s) => s.talking)
  const micError = useVoiceStore((s) => s.micError)
  const error = useVoiceStore((s) => s.error)
  const preview = useVoiceStore((s) => s.preview)
  const wokeAt = useVoiceStore((s) => s.wokeAt)
  const wakePhrase = useVoiceStore((s) => s.wakePhrase)
  const followUpUntil = useVoiceStore((s) => s.followUpUntil)
  const config = useConfig()
  const phrase = titleCase(wakePhrase || (status.data as VoiceStatusB | undefined)?.wake?.phrase || config.data?.voice.wake_word || 'Hey Sentient')
  const autoStarted = useRef(false)

  const previewKind = params.get('voicePreview') as PreviewKind | null
  const previewing = !!previewKind && PREVIEW_KINDS.includes(previewKind) && previewAllowed()

  // preview (dev only)
  useEffect(() => {
    if (!previewing || !previewKind) return
    return voice.loadPreview(previewState(previewKind))
  }, [previewing, previewKind])

  // Leave the route -> end the session. Deferred so StrictMode / Fast Refresh remounts don't end it.
  useEffect(() => {
    mountedVoiceViews++
    return () => {
      mountedVoiceViews--
      window.setTimeout(() => mountedVoiceViews === 0 && voice.dispose(), 0)
    }
  }, [])

  const sttReady = !!status.data?.stt.ready
  const ttsReady = !!status.data?.tts.ready
  const engineError = status.data?.stt.error || status.data?.tts.error

  // Woken by background "Hey Sentient" listening: start at once and show the wake animation.
  const wokeFromBackground = params.get('wake') === '1'
  useEffect(() => {
    if (previewing || autoStarted.current || phase !== 'setup' || !wokeFromBackground) return
    autoStarted.current = true
    void voice.start({ woke: true })
  }, [previewing, phase, wokeFromBackground])

  // Go straight to listening when the models are already warm.
  useEffect(() => {
    if (previewing || autoStarted.current || phase !== 'setup' || !status.data) return
    if (sttReady && ttsReady && !engineError) {
      autoStarted.current = true
      void voice.start()
    }
  }, [previewing, phase, status.data, sttReady, ttsReady, engineError])

  const close = () => {
    voice.stop()
    if (window.history.length > 1) navigate(-1)
    else navigate('/chat')
  }
  useHotkey('escape', close)

  // Push-to-talk on Space; M toggles mute.
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (isEditableTarget(e.target) || e.ctrlKey || e.metaKey || e.altKey) return
      if (e.code === 'Space') {
        e.preventDefault()
        if (!e.repeat) voice.pttDown()
      } else if (e.key.toLowerCase() === 'm' && !e.repeat) voice.toggleMute()
    }
    const up = (e: KeyboardEvent) => {
      if (e.code === 'Space') voice.pttUp()
    }
    window.addEventListener('keydown', down)
    window.addEventListener('keyup', up)
    return () => {
      window.removeEventListener('keydown', down)
      window.removeEventListener('keyup', up)
    }
  }, [])

  const live = phase === 'live'
  const connecting = phase === 'connecting'
  const speaking = live && state === 'speaking'

  const orbClick = () => {
    if (!live) return
    if (state === 'standby') voice.wakeNow()
    else if (state === 'speaking' || state === 'thinking') voice.interrupt()
    else if (muted) voice.toggleMute()
  }

  return (
    <div className="relative flex h-full flex-col overflow-hidden bg-bg">
      <TitleBar minimal />
      <Backdrop state={live ? state : 'idle'} />

      <header className="relative z-10 flex items-center gap-3 px-6 pb-2 pt-1">
        <div className="flex items-center gap-2 text-sm font-semibold text-fg">
          <IconEar size={17} className="text-accent-text" />
          Voice mode
        </div>
        <EngineChips status={status.data} loading={status.isLoading} unavailable={status.isError} />
        <WakeChip status={status.data as VoiceStatusB | undefined} phrase={phrase} />
        <div className="flex-1" />
        {sessionId && !preview && (live || phase === 'ended') && (
          <Button size="sm" variant="ghost" leftIcon={<IconMessageCircle size={15} />} onClick={() => navigate(`/chat/${sessionId}`)}>
            Open in chat
          </Button>
        )}
        <IconButton label="Voice settings" icon={<IconSettings size={17} />} onClick={() => navigate('/settings/voice')} />
        <IconButton label="Close voice mode" shortcut="Esc" icon={<IconX size={18} />} onClick={close} />
      </header>

      <div className="relative z-10 flex min-h-0 flex-1">
        <main className="flex min-w-0 flex-1 flex-col items-center justify-center px-6 pb-6">
          {phase === 'setup' ? (
            <SetupCard status={status.data} statusError={status.error} loading={status.isLoading} prepare={prepare} />
          ) : phase === 'ended' ? (
            <EndedCard sessionId={sessionId} turns={turns.length} preview={preview} />
          ) : phase === 'error' ? (
            <div className="w-full max-w-md space-y-4 text-center">
              <VoiceOrb state="idle" size={220} className="mx-auto" />
              <Alert tone="danger" icon={<IconAlertTriangle />} title="Voice mode stopped">
                {error ?? 'Something went wrong.'}
              </Alert>
              <Button variant="primary" onClick={() => void voice.start({ sessionId, keepTranscript: true })}>
                Reconnect
              </Button>
            </div>
          ) : (
            <>
              <div className="relative -my-6">
                <VoiceOrb
                  state={state}
                  connecting={connecting}
                  muted={muted}
                  size={360}
                  onClick={orbClick}
                  label={speaking ? 'Interrupt' : state === 'standby' ? 'Wake without the phrase' : 'Voice visualizer'}
                />
                <WakeRipple wokeAt={wokeAt} />
              </div>
              <StateLabel phase={phase} state={state} muted={muted} mode={mode} talking={talking} phrase={phrase} followUp={!!followUpUntil && followUpUntil > Date.now()} />
              <FollowUpIndicator />
              <LiveCaption />
              {micError && (
                <Alert tone="warning" icon={<IconMicrophoneOff />} className="mt-4 max-w-md">
                  {micError}
                </Alert>
              )}
              <ControlDock />
            </>
          )}
        </main>

        <aside className="hidden w-[400px] shrink-0 flex-col border-l border-border bg-surface/50 backdrop-blur-sm md:flex">
          <div className="flex h-11 shrink-0 items-center gap-2 border-b border-border px-5">
            <span className="text-sm font-semibold text-fg">Transcript</span>
            {sessionId && <span className="text-2xs text-fg-subtle">Saved to your chats</span>}
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto">
            <VoiceTranscript turns={turns} live={live} speaking={speaking} className="min-h-full" />
          </div>
          <TypeInstead disabled={!live} />
        </aside>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------- pieces
function Backdrop({ state }: { state: VoiceStateB }) {
  const strong = state === 'speaking' || state === 'thinking'
  return (
    <motion.div
      aria-hidden
      className="pointer-events-none absolute inset-0"
      animate={{ opacity: strong ? 1 : 0.7 }}
      transition={{ duration: 1.2 }}
      style={{
        background:
          'radial-gradient(60% 55% at 38% 52%, color-mix(in oklab, var(--accent) 11%, transparent), transparent 70%), radial-gradient(40% 40% at 75% 20%, color-mix(in oklab, var(--accent) 6%, transparent), transparent 70%)'
      }}
    />
  )
}

function readiness(ready: boolean | undefined, error: string | undefined): { tone: Tone; label: string } {
  if (error) return { tone: 'danger', label: 'Error' }
  if (ready) return { tone: 'success', label: 'Ready' }
  return { tone: 'warning', label: 'Loads on first use' }
}

function EngineChips({ status, loading, unavailable }: { status?: VoiceStatus; loading: boolean; unavailable: boolean }) {
  if (loading) return <Skeleton className="h-6 w-56 rounded-full" />
  if (unavailable || !status) {
    return (
      <span className="flex items-center gap-1.5 rounded-full border border-border px-2.5 py-0.5 text-xs text-fg-subtle">
        <StatusDot tone="neutral" /> Voice engine unavailable
      </span>
    )
  }
  const stt = readiness(status.stt.ready, status.stt.error)
  const tts = readiness(status.tts.ready, status.tts.error)
  return (
    <div className="hidden items-center gap-1.5 lg:flex">
      <Tooltip content={status.stt.error ?? status.stt.note ?? `Speech recognition · ${stt.label}${status.stt.device ? ` · ${status.stt.device}` : ''}`}>
        <span className="flex items-center gap-1.5 rounded-full border border-border bg-surface/60 px-2.5 py-0.5 text-xs text-fg-muted">
          <IconMicrophone size={12} />
          <StatusDot tone={stt.tone} />
          {status.stt.model || status.stt.provider}
        </span>
      </Tooltip>
      <Tooltip content={status.tts.error ?? `Voice · ${tts.label}`}>
        <span className="flex items-center gap-1.5 rounded-full border border-border bg-surface/60 px-2.5 py-0.5 text-xs text-fg-muted">
          <IconVolume size={12} />
          <StatusDot tone={tts.tone} />
          {status.tts.voice || status.tts.provider}
        </span>
      </Tooltip>
    </div>
  )
}

const LABEL: Record<VoiceStateB, string> = {
  standby: 'Resting',
  idle: 'Ready',
  listening: 'Listening',
  transcribing: 'Got it',
  thinking: 'Thinking',
  speaking: 'Speaking'
}

function StateLabel({
  phase,
  state,
  muted,
  mode,
  talking,
  phrase,
  followUp
}: {
  phase: VoicePhase
  state: VoiceStateB
  muted: boolean
  mode: 'handsfree' | 'ptt'
  talking: boolean
  phrase: string
  followUp: boolean
}) {
  const connecting = phase === 'connecting'
  const title = connecting
    ? 'Connecting'
    : state === 'standby'
      ? `Say “${phrase}”`
      : muted && state === 'listening'
        ? 'Muted'
        : mode === 'ptt' && state === 'listening' && talking
          ? 'Listening'
          : LABEL[state]
  const hint = connecting
    ? 'Warming up the microphone and voice'
    : state === 'standby'
      ? 'I’m resting until I hear it. Nothing is sent to any AI until then.'
      : state === 'listening' && followUp && !muted
        ? `Go ahead, no need to say “${phrase}” again`
        : state === 'speaking'
      ? 'Tap the orb or just start talking to interrupt'
      : state === 'thinking'
        ? 'Working on it. Start talking to change your request'
        : state === 'transcribing'
          ? 'Turning your words into text'
          : muted
            ? 'Your microphone is off. Press M or tap the orb to unmute'
            : mode === 'ptt'
              ? talking
                ? 'Release to send'
                : 'Hold Space or the talk button while you speak'
              : "Go ahead, I'm listening"

  return (
    <div className="flex h-[68px] flex-col items-center text-center">
      <AnimatePresence mode="wait">
        <motion.div
          key={title}
          initial={{ opacity: 0, y: 6, filter: 'blur(4px)' }}
          animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
          exit={{ opacity: 0, y: -6, filter: 'blur(4px)' }}
          transition={{ duration: 0.25 }}
          className="text-2xl font-semibold tracking-tight text-fg"
        >
          {title}
          {(connecting || state === 'thinking' || state === 'transcribing') && <Dots />}
        </motion.div>
      </AnimatePresence>
      <p className="mt-1.5 text-sm text-fg-subtle">{hint}</p>
    </div>
  )
}

function Dots() {
  return (
    <span className="ml-2 inline-flex gap-[3px] align-middle">
      {[0, 1, 2].map((i) => (
        <motion.span
          key={i}
          className="inline-block size-1 rounded-full bg-fg-muted"
          animate={{ opacity: [0.25, 1, 0.25] }}
          transition={{ duration: 1.2, repeat: Infinity, delay: i * 0.18 }}
        />
      ))}
    </span>
  )
}

/** Last line of the conversation under the orb on narrow windows (the transcript panel is hidden). */
function LiveCaption() {
  const last = useVoiceStore((s) => s.turns[s.turns.length - 1])
  if (!last) return null
  const text = last.role === 'assistant' ? last.text || last.spoken.join(' ') : last.text
  if (!text) return null
  return <p className="mt-2 line-clamp-2 max-w-lg text-center text-sm text-fg-muted md:hidden">{text}</p>
}

function ControlDock() {
  const phase = useVoiceStore((s) => s.phase)
  const state = useVoiceStore((s) => s.state)
  const mode = useVoiceStore((s) => s.mode)
  const muted = useVoiceStore((s) => s.muted)
  const talking = useVoiceStore((s) => s.talking)
  const wakeMode = useVoiceStore((s) => s.wakeMode)
  const live = phase === 'live'
  const canInterrupt = live && (state === 'speaking' || state === 'thinking')

  return (
    <div className="mt-6 flex flex-col items-center gap-3">
      <div className="flex items-center gap-2 rounded-full border border-border-strong bg-elevated/80 p-1.5 shadow-pop backdrop-blur">
        <SegmentedControl
          size="sm"
          value={mode}
          onChange={(m) => voice.setMode(m)}
          aria-label="Input mode"
          options={[
            { value: 'handsfree', label: 'Hands-free' },
            { value: 'ptt', label: 'Push to talk' }
          ]}
        />
        <span className="mx-0.5 h-6 w-px bg-border" />
        <Tooltip content={wakeMode ? 'Listen all the time instead' : 'Rest until I hear “Hey Sentient”'}>
          <button
            type="button"
            aria-pressed={wakeMode}
            disabled={!live}
            onClick={() => voice.setWakeMode(!wakeMode)}
            className={cn(
              'no-drag flex h-10 items-center gap-1.5 rounded-full px-3.5 text-sm font-medium transition-colors disabled:opacity-45',
              wakeMode ? 'bg-accent/15 text-accent-text ring-1 ring-accent/30' : 'bg-active text-fg hover:bg-hover'
            )}
          >
            <IconEar size={17} />
            Wake word
          </button>
        </Tooltip>
        {mode === 'ptt' && (
          <button
            type="button"
            disabled={!live}
            onPointerDown={(e) => {
              e.currentTarget.setPointerCapture(e.pointerId)
              voice.pttDown()
            }}
            onPointerUp={() => voice.pttUp()}
            onPointerCancel={() => voice.pttUp()}
            className={cn(
              'no-drag flex h-10 items-center gap-2 rounded-full px-4 text-sm font-medium transition-colors disabled:opacity-45',
              talking ? 'bg-accent text-accent-fg shadow-glow' : 'bg-active text-fg hover:bg-hover'
            )}
          >
            <IconMicrophone size={16} />
            {talking ? 'Release to send' : 'Hold to talk'}
          </button>
        )}
        <Tooltip content={muted ? 'Unmute (M)' : 'Mute (M)'}>
          <button
            type="button"
            aria-pressed={muted}
            aria-label={muted ? 'Unmute microphone' : 'Mute microphone'}
            onClick={() => voice.toggleMute()}
            className={cn(
              'no-drag flex size-10 items-center justify-center rounded-full transition-colors',
              muted ? 'bg-danger/15 text-danger hover:bg-danger/25' : 'bg-active text-fg hover:bg-hover'
            )}
          >
            {muted ? <IconMicrophoneOff size={18} /> : <IconMicrophone size={18} />}
          </button>
        </Tooltip>
        <Tooltip content="Interrupt">
          <button
            type="button"
            aria-label="Interrupt"
            disabled={!canInterrupt}
            onClick={() => voice.interrupt()}
            className="no-drag flex size-10 items-center justify-center rounded-full bg-active text-fg transition-colors hover:bg-hover disabled:opacity-35"
          >
            {state === 'speaking' ? <IconPlayerStopFilled size={16} /> : <IconHandStop size={18} />}
          </button>
        </Tooltip>
        <Tooltip content="End conversation">
          <button
            type="button"
            aria-label="End conversation"
            onClick={() => voice.stop()}
            className="no-drag flex h-10 items-center gap-2 rounded-full bg-danger px-4 text-sm font-medium text-white transition-opacity hover:opacity-90"
          >
            <IconPhoneOff size={17} />
            End
          </button>
        </Tooltip>
      </div>
      <div className="flex items-center gap-3 text-2xs text-fg-faint">
        <span className="flex items-center gap-1">
          <Kbd>Space</Kbd> push to talk
        </span>
        <span className="flex items-center gap-1">
          <Kbd>M</Kbd> mute
        </span>
        <span className="flex items-center gap-1">
          <Kbd>Esc</Kbd> close
        </span>
      </div>
    </div>
  )
}

const titleCase = (s: string) => s.trim().replace(/\b\p{L}/gu, (c) => c.toUpperCase())

function WakeChip({ status, phrase }: { status?: VoiceStatusB; phrase: string }) {
  const always = useWakeStore((s) => s.enabled)
  const wakeMode = useVoiceStore((s) => s.wakeMode)
  const wake = status?.wake
  if (!always && !wakeMode && !wake?.error) return null
  const tone: Tone = wake?.error ? 'danger' : wake && !wake.ready ? 'warning' : 'success'
  return (
    <Tooltip content={wake?.error ?? (always ? `Always listening for “${phrase}” while Sentient is open` : `Resting until I hear “${phrase}”`)}>
      <span className="hidden items-center gap-1.5 rounded-full border border-border bg-surface/60 px-2.5 py-0.5 text-xs text-fg-muted lg:flex">
        <IconEar size={12} />
        <StatusDot tone={tone} />
        {always ? 'Always listening' : 'Wake word'}
      </span>
    </Tooltip>
  )
}

/** A gentle ripple and "I'm listening" when the wake word is heard. */
function WakeRipple({ wokeAt }: { wokeAt: number | null }) {
  const [visible, setVisible] = useState(false)
  useEffect(() => {
    if (!wokeAt) return
    setVisible(true)
    const t = window.setTimeout(() => setVisible(false), 4200)
    return () => window.clearTimeout(t)
  }, [wokeAt])
  return (
    <AnimatePresence>
      {visible && wokeAt && (
        <motion.div key={wokeAt} className="pointer-events-none absolute inset-0 flex items-center justify-center" exit={{ opacity: 0 }}>
          {[0, 1, 2].map((i) => (
            <motion.span
              key={i}
              className="absolute rounded-full border-2 border-accent/60"
              style={{ width: '48%', height: '48%' }}
              initial={{ scale: 0.85, opacity: 0.8 }}
              animate={{ scale: 2.1, opacity: 0 }}
              transition={{ duration: 1.8, delay: i * 0.3, ease: 'easeOut', repeat: 1, repeatDelay: 0.3 }}
            />
          ))}
          <motion.span
            initial={{ opacity: 0, y: 8, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            className="absolute bottom-[13%] rounded-full border border-accent/30 bg-accent/15 px-3 py-1 text-xs font-medium text-accent-text backdrop-blur"
          >
            I’m listening
          </motion.span>
        </motion.div>
      )}
    </AnimatePresence>
  )
}

/** Countdown while follow-ups are accepted without the wake word. */
function FollowUpIndicator() {
  const until = useVoiceStore((s) => s.followUpUntil)
  const total = useVoiceStore((s) => s.followUpMs) ?? 8000
  const state = useVoiceStore((s) => s.state)
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    if (!until) return
    const t = window.setInterval(() => setNow(Date.now()), 200)
    return () => window.clearInterval(t)
  }, [until])
  const left = until ? until - now : 0
  const show = !!until && left > 0 && state !== 'standby' && state !== 'speaking'
  const frac = Math.max(0, Math.min(1, left / total))
  const R = 7
  const C = 2 * Math.PI * R
  return (
    <div className="flex h-9 items-center justify-center">
      <AnimatePresence>
        {show && (
          <motion.div
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="flex items-center gap-2 rounded-full border border-accent/25 bg-accent/10 py-1 pl-1.5 pr-3 text-xs text-fg"
          >
            <svg width="18" height="18" viewBox="0 0 18 18" className="-rotate-90" aria-hidden>
              <circle cx="9" cy="9" r={R} fill="none" stroke="var(--active)" strokeWidth="2" />
              <circle cx="9" cy="9" r={R} fill="none" stroke="var(--accent)" strokeWidth="2" strokeDasharray={C} strokeDashoffset={C * (1 - frac)} strokeLinecap="round" />
            </svg>
            Still listening for a follow-up · {Math.ceil(left / 1000)}s
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

function TypeInstead({ disabled }: { disabled: boolean }) {
  const [text, setText] = useState('')
  const send = () => {
    if (!text.trim()) return
    voice.sendText(text)
    setText('')
  }
  return (
    <form
      className="flex shrink-0 items-center gap-2 border-t border-border p-3"
      onSubmit={(e) => {
        e.preventDefault()
        send()
      }}
    >
      <Input value={text} disabled={disabled} placeholder={disabled ? 'Start a conversation to type' : 'Type instead…'} onChange={(e) => setText(e.target.value)} />
      <IconButton type="submit" variant="primary" label="Send" icon={<IconSend size={15} />} disabled={disabled || !text.trim()} />
    </form>
  )
}

function SetupCard({
  status,
  statusError,
  loading,
  prepare
}: {
  status?: VoiceStatus
  statusError: unknown
  loading: boolean
  prepare: ReturnType<typeof useVoicePrepare>
}) {
  const connected = !!status
  const stt = status ? readiness(status.stt.ready, status.stt.error) : null
  const tts = status ? readiness(status.tts.ready, status.tts.error) : null
  const needsPrep = !!status && (!status.stt.ready || !status.tts.ready)
  const errors = [status?.stt.error, status?.tts.error].filter(Boolean) as string[]
  const navigate = useNavigate()

  return (
    <div className="flex w-full max-w-lg flex-col items-center text-center">
      <VoiceOrb state="idle" size={260} className="-my-4" />
      <h1 className="mt-2 text-2xl font-semibold tracking-tight text-fg">Talk to Sentient</h1>
      <p className="mt-1.5 max-w-sm text-sm text-fg-muted">Have a hands-free conversation. Interrupt any time, approve actions by saying yes or no.</p>

      {loading ? (
        <Skeleton className="mt-6 h-24 w-full rounded-xl" />
      ) : !connected ? (
        <Alert className="mt-6 w-full text-left" tone="warning" icon={<IconAlertTriangle />} title="The voice engine isn't available">
          {isNotImplemented(statusError) ? 'This version of the engine has no voice support.' : errorMessage(statusError)}
        </Alert>
      ) : (
        <div className="mt-6 w-full divide-y divide-border overflow-hidden rounded-xl border border-border bg-surface/70 text-left backdrop-blur">
          <SetupRow icon={<IconMicrophone size={16} />} title="Speech recognition" detail={[status.stt.provider, status.stt.model, status.stt.device].filter(Boolean).join(' · ')} state={stt} note={status.stt.note} />
          <SetupRow icon={<IconVolume size={16} />} title="Voice" detail={[status.tts.provider, status.tts.voice].filter(Boolean).join(' · ')} state={tts} />
          {(prepare.running || prepare.done || prepare.error) && (
            <div className="space-y-2 px-4 py-3">
              <div className="flex items-center justify-between text-xs">
                <span className={prepare.error ? 'text-danger' : 'text-fg-muted'}>{prepare.error ?? prepare.stage}</span>
                {typeof prepare.progress === 'number' && <span className="tabular-nums text-fg-subtle">{Math.round(prepare.progress * 100)}%</span>}
              </div>
              {!prepare.error && <ProgressBar value={prepare.done ? 1 : prepare.progress} tone={prepare.done ? 'success' : 'accent'} />}
            </div>
          )}
        </div>
      )}

      {errors.length > 0 && (
        <Alert className="mt-3 w-full text-left" tone="danger" icon={<IconAlertTriangle />} action={<Button size="xs" variant="ghost" onClick={() => navigate('/settings/voice')}>Settings</Button>}>
          {errors.join(' ')}
        </Alert>
      )}

      <div className="mt-6 flex flex-wrap items-center justify-center gap-2">
        <Button size="lg" variant="primary" leftIcon={<IconMicrophone size={17} />} disabled={!connected} onClick={() => void voice.start()}>
          Start talking
        </Button>
        {needsPrep && !prepare.done && (
          <Button size="lg" variant="secondary" leftIcon={<IconDownload size={16} />} loading={prepare.running} onClick={() => void prepare.run('all')}>
            Prepare voice models
          </Button>
        )}
      </div>
      {needsPrep && !prepare.running && !prepare.done && (
        <p className="mt-3 max-w-sm text-xs text-fg-subtle">
          Local models load the first time you talk, which can take a minute. Prepare them now to watch the progress.
        </p>
      )}
    </div>
  )
}

function SetupRow({ icon, title, detail, state, note }: { icon: React.ReactNode; title: string; detail: string; state: { tone: Tone; label: string } | null; note?: string }) {
  return (
    <div className="flex items-center gap-3 px-4 py-3">
      <span className="flex size-8 shrink-0 items-center justify-center rounded-lg border border-border bg-elevated text-fg-muted">{icon}</span>
      <div className="min-w-0 flex-1">
        <div className="text-sm font-medium text-fg">{title}</div>
        <div className="truncate text-xs text-fg-subtle">{note ?? detail}</div>
      </div>
      {state && (
        <span className="flex items-center gap-1.5 text-xs text-fg-muted">
          <StatusDot tone={state.tone} />
          {state.label}
        </span>
      )}
    </div>
  )
}

function EndedCard({ sessionId, turns, preview }: { sessionId: string | null; turns: number; preview: boolean }) {
  const navigate = useNavigate()
  return (
    <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="flex max-w-sm flex-col items-center text-center">
      <span className="flex size-14 items-center justify-center rounded-full bg-success/12 text-success">
        <IconCircleCheckFilled size={30} />
      </span>
      <h2 className="mt-4 text-xl font-semibold tracking-tight text-fg">Conversation saved</h2>
      <p className="mt-1.5 text-sm text-fg-muted">
        {turns ? `${Math.ceil(turns / 2)} exchange${turns > 2 ? 's' : ''} saved to your chats, so you can pick it up in text any time.` : 'Nothing was said, so there is nothing to keep.'}
      </p>
      <div className="mt-6 flex items-center gap-2">
        {sessionId && turns > 0 && (
          <Button variant="secondary" rightIcon={<IconArrowRight size={15} />} disabled={preview} onClick={() => navigate(`/chat/${sessionId}`)}>
            Open in chat
          </Button>
        )}
        <Button variant="primary" leftIcon={<IconMicrophone size={16} />} disabled={preview} onClick={() => void voice.start()}>
          Talk again
        </Button>
      </div>
    </motion.div>
  )
}
