/**
 * Push to talk and dictation into any app (#169), the window's side. The shell owns the global shortcuts and the
 * listening pill (electron/main/dictation.ts); this file
 * - keeps the shell's shortcuts in step with `voice.dictation` and remembers what the shell could register,
 * - sends what was said with push to talk to the open chat (or a new one) and reads the answer aloud.
 */
import { useEffect } from 'react'
import { create } from 'zustand'
import { useConfig } from '@/hooks/core'
import { qk } from '@/hooks/queryKeys'
import { api } from '@/lib/api'
import { getBridge } from '@/lib/bridge'
import { queryClient } from '@/lib/queryClient'
import type { DictationConfig, SentientConfig } from '@/lib/types'
import { useChat } from '@/stores/chat'
import type { DictationShellSettings, DictationStatus } from '@/types/bridge'

export const useDictationStatus = create<{ status: DictationStatus | null }>(() => ({ status: null }))

export function shellSettings(d: DictationConfig): DictationShellSettings {
  return {
    pushToTalk: d.push_to_talk,
    pushToTalkShortcut: d.push_to_talk_shortcut,
    dictate: d.dictate,
    dictateShortcut: d.dictate_shortcut,
    stopAfterSilenceS: d.stop_after_silence_s
  }
}

/** Mounted once inside the app: pushes `voice.dictation` to the shell whenever it changes. */
export function DictationShellSync() {
  const config = useConfig()
  const dictation = config.data?.voice?.dictation
  const settings = dictation ? JSON.stringify(shellSettings(dictation)) : ''
  useEffect(() => {
    const bridge = getBridge()
    if (!settings || !bridge.isDesktop) return
    let live = true
    void bridge.dictation.apply(JSON.parse(settings) as DictationShellSettings).then((status) => {
      if (live) useDictationStatus.setState({ status })
    })
    return () => {
      live = false
    }
  }, [settings])
  return null
}

/** The chat the window shows (`#/chat/<id>`), if any. */
export function openChatId(hash: string): string | null {
  const m = /^#?\/chat\/([^/?#]+)/.exec(hash)
  return m ? decodeURIComponent(m[1]) : null
}

let playing: HTMLAudioElement | null = null

async function speak(text: string): Promise<void> {
  try {
    const blob = await api.voice.speak(text)
    const url = URL.createObjectURL(blob)
    playing?.pause()
    const audio = new Audio(url)
    playing = audio
    audio.onended = () => URL.revokeObjectURL(url)
    await audio.play()
  } catch {
    /* the answer is on screen; a missing voice is not worth an error */
  }
}

/** Read the answer to the message sent with `clientId` aloud once it has finished. */
function speakWhenDone(clientId: string): void {
  let unsubscribe = () => undefined as void
  const giveUp = window.setTimeout(() => unsubscribe(), 5 * 60_000)
  unsubscribe = useChat.subscribe((s) => {
    const turn = Object.values(s.live).find((l) => l.clientId === clientId)
    if (!turn || turn.streaming) return
    unsubscribe()
    window.clearTimeout(giveUp)
    if (turn.turn.status !== 'done') return
    const text = turn.turn.segments
      .map((seg) => (seg.kind === 'text' ? seg.text : ''))
      .join('')
      .trim()
    if (text) void speak(text)
  })
}

/**
 * Send what was said with push to talk: into the chat the window shows, or a new chat. Returns where to go so the
 * user sees the answer.
 */
export async function sendPushToTalk(text: string, hash: string): Promise<{ route: string; state?: Record<string, unknown> }> {
  const sessionId = openChatId(hash)
  const key = await useChat.getState().send(sessionId, { text })
  const clientId = useChat.getState().live[key]?.clientId
  const config = queryClient.getQueryData<SentientConfig>(qk.config)
  if (clientId && config?.voice?.dictation?.speak_replies !== false) speakWhenDone(clientId)
  return sessionId ? { route: `/chat/${sessionId}` } : { route: '/chat', state: { fresh: Date.now(), pendingKey: key } }
}
