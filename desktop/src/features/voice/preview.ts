/**
 * Dev-only mock data for `#/voice?voicePreview=listening|transcribing|thinking|speaking|setup|ended`.
 * Read only when the app isn't packaged (screenshots, design work without a mic).
 */
import type { VoiceStateB } from '@/lib/leap/types-b'
import type { VoiceTurn } from './session'

export const PREVIEW_KINDS = ['listening', 'transcribing', 'thinking', 'speaking', 'setup', 'ended', 'standby', 'woke', 'followup'] as const
export type PreviewKind = (typeof PREVIEW_KINDS)[number]

export function previewAllowed(): boolean {
  return !window.location.href.includes('app.asar')
}

let n = 0
const turn = (t: Partial<VoiceTurn> & Pick<VoiceTurn, 'role' | 'text'>): VoiceTurn => ({
  id: `preview-${++n}`,
  spoken: [],
  tools: [],
  approvals: [],
  done: true,
  ...t
})

const base = (): VoiceTurn[] => [
  turn({ role: 'user', text: "What's on my calendar tomorrow?" }),
  turn({
    role: 'assistant',
    text: 'You have three things tomorrow: the design review at 11, lunch with Priya at 1, and the Lumen Health check-in at 4.',
    tools: [{ callId: 'c1', name: 'gcalendar_list_events', status: 'done' }]
  }),
  turn({ role: 'user', text: 'Can you move the design review to Thursday afternoon?' }),
  turn({
    role: 'assistant',
    text: "Done. It's now Thursday at 3 PM, and everyone's calendars are free then.",
    tools: [
      { callId: 'c2', name: 'gcalendar_find_free_time', status: 'done' },
      { callId: 'c3', name: 'gcalendar_update_event', status: 'done' }
    ]
  })
]

export function previewState(kind: PreviewKind): {
  phase: 'live' | 'setup' | 'ended'
  state: VoiceStateB
  turns: VoiceTurn[]
  sessionId: string | null
  engine: { stt: string; tts: string } | null
  wakeMode?: boolean
  wokeAt?: number | null
  followUpUntil?: number | null
  followUpMs?: number | null
} {
  const engine = { stt: 'faster_whisper', tts: 'kokoro' }
  switch (kind) {
    case 'setup':
      return { phase: 'setup', state: 'idle', turns: [], sessionId: null, engine: null }
    case 'standby':
      return { phase: 'live', state: 'standby', turns: base(), sessionId: 'preview-session', engine, wakeMode: true }
    case 'woke':
      return { phase: 'live', state: 'listening', turns: base(), sessionId: 'preview-session', engine, wakeMode: true, wokeAt: Date.now() }
    case 'followup':
      return { phase: 'live', state: 'listening', turns: base(), sessionId: 'preview-session', engine, wakeMode: true, followUpUntil: Date.now() + 14_000, followUpMs: 16_000 }
    case 'ended':
      return { phase: 'ended', state: 'idle', turns: base(), sessionId: 'preview-session', engine }
    case 'listening':
      return { phase: 'live', state: 'listening', turns: base(), sessionId: 'preview-session', engine }
    case 'transcribing':
      return { phase: 'live', state: 'transcribing', turns: base(), sessionId: 'preview-session', engine }
    case 'thinking':
      return {
        phase: 'live',
        state: 'thinking',
        sessionId: 'preview-session',
        engine,
        turns: [
          ...base(),
          turn({ role: 'user', text: "Reply to Priya and tell her Thursday at 3 works, and I'll share the Figma link before." }),
          turn({ role: 'assistant', text: '', done: false, tools: [{ callId: 'c4', name: 'gmail_search', status: 'running' }] })
        ]
      }
    case 'speaking':
      return {
        phase: 'live',
        state: 'speaking',
        sessionId: 'preview-session',
        engine,
        turns: [
          ...base(),
          turn({ role: 'user', text: "Reply to Priya and tell her Thursday at 3 works, and I'll share the Figma link before." }),
          turn({
            role: 'assistant',
            done: false,
            text: "I found Priya's email and drafted a short reply saying Thursday at 3 works and that you'll send the Figma link beforehand. I need your approval to send it.",
            spoken: ["I found Priya's email and drafted a short reply."],
            tools: [
              { callId: 'c4', name: 'gmail_search', status: 'done' },
              { callId: 'c5', name: 'gmail_reply', status: 'awaiting' }
            ],
            approvals: [
              {
                approvalId: 'a1',
                callId: 'c5',
                name: 'gmail_reply',
                risk: 'send',
                reason: 'Reply to Priya Sharma: “Thursday at 3 works, I’ll share the Figma link before.”',
                arguments: { to: 'priya@northwind.example', subject: 'Re: Design review moved to Thursday?' },
                status: 'pending'
              }
            ]
          })
        ]
      }
  }
}
