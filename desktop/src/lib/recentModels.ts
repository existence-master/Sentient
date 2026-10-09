/** Chat models used recently on this computer, newest first, for the title bar model menu. */
const KEY = 'sentient.recentChatModels'
const MAX = 5

export function recentChatModels(): string[] {
  try {
    const list = JSON.parse(localStorage.getItem(KEY) ?? '[]') as unknown
    return Array.isArray(list) ? list.filter((m): m is string => typeof m === 'string' && !!m).slice(0, MAX) : []
  } catch {
    return []
  }
}

export function rememberChatModel(model: string | null | undefined): void {
  if (!model) return
  try {
    const next = [model, ...recentChatModels().filter((m) => m !== model)].slice(0, MAX)
    localStorage.setItem(KEY, JSON.stringify(next))
  } catch {
    // storage unavailable: the menu just shows no recent models
  }
}
