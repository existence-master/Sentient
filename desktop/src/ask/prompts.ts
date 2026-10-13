/**
 * The quick actions of the ask window (#173). Each sends one short message with the selected text attached; the
 * engine adds the text marked as outside content. No imports, so `npm test` runs them in Node.
 */

export type AskAction = 'explain' | 'summarize' | 'translate' | 'reply'

export const ACTIONS: Array<{ id: AskAction; label: string }> = [
  { id: 'explain', label: 'Explain' },
  { id: 'summarize', label: 'Summarize' },
  { id: 'translate', label: 'Translate' },
  { id: 'reply', label: 'Reply' }
]

/** "English" for "en", "German" for "de-AT"; the code itself when the name isn't known. */
export function languageName(code: string | undefined): string {
  const tag = String(code ?? '').trim() || 'en'
  try {
    return new Intl.DisplayNames(['en'], { type: 'language' }).of(tag.split('-')[0]) ?? tag
  } catch {
    return tag
  }
}

/** The message an action sends. Short and plain, so small local models follow it. */
export function actionPrompt(action: AskAction, language?: string): string {
  switch (action) {
    case 'explain':
      return 'Explain the selected text in plain words.'
    case 'summarize':
      return 'Summarize the selected text in a few short points.'
    case 'translate': {
      const target = languageName(language)
      return `Translate the selected text into ${target}. If it is already in ${target}, translate it into English. Give only the translation.`
    }
    case 'reply':
      return 'Write a short reply to the selected text that I could send. Give only the reply text.'
  }
}

/** What the window says about the empty state (`AskState.empty`). */
export function emptyMessage(empty: string | undefined, keys: string, platform: string): string {
  switch (empty) {
    case 'password':
      return 'A password box had focus, so Sentient didn’t copy anything.'
    case 'failed':
      return platform === 'linux'
        ? 'Sentient couldn’t copy from that app. Installing xdotool lets it copy for you. You can still ask a question below.'
        : 'Sentient couldn’t copy from that app. You can still ask a question below.'
    case 'own-window':
      return `Select text in another app, then press ${keys}. You can also ask a question below.`
    default:
      return `No text was selected. Select some text in any app, then press ${keys}. You can also ask a question below.`
  }
}

/** The file the selection is uploaded as (`source: "selection"`): named after the app. */
export function selectionFileName(app: string): string {
  const safe = String(app ?? '').replace(/[\\/:*?"<>|\x00-\x1f]/g, ' ').replace(/\s+/g, ' ').trim().slice(0, 50)
  return `${safe && safe !== 'another app' ? safe : 'Selected text'}.txt`
}
