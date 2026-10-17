/**
 * Claude through your own Claude Code (ADR 0022): asking for its status right after the switch changes.
 *
 * The switch updates the screen at once, but the engine only knows once the setting is saved. A status asked for
 * in between still says "off" and has no version, so it is asked again until the engine agrees (#246).
 */

/** The engine hasn't saved the switch yet. */
export class SwitchNotSavedYet extends Error {
  constructor() {
    super('Sentient is still saving this setting. Try again in a moment.')
    this.name = 'SwitchNotSavedYet'
  }
}

export const SAVE_RETRIES = 12
export const SAVE_RETRY_MS = 400

/** Ask for the status, and fail with `SwitchNotSavedYet` while the engine still has the switch the other way. */
export async function statusOnceSaved<T extends { enabled: boolean }>(get: () => Promise<T>, enabled: boolean): Promise<T> {
  const status = await get()
  if (status.enabled !== enabled) throw new SwitchNotSavedYet()
  return status
}

/** React Query `retry`: only waiting for the save is worth another try. */
export function retryWhileSaving(failures: number, error: unknown): boolean {
  return error instanceof SwitchNotSavedYet && failures < SAVE_RETRIES
}

// ---------------------------------------------------------------------------- which roles may use it (#442)

/** A `claude-code/<model>` model: answered by the user's own Claude Code. */
export function isClaudeCode(model: string | null | undefined): boolean {
  return !!model && model.startsWith('claude-code/')
}

/**
 * Whether `role` may use `model`. Claude Code only answers chats, so only the roles the engine names (`roles` from
 * `GET /api/models/claude-code`) may use it. Unknown roles (the engine hasn't said yet) never get it.
 */
export function allowedForRole(model: string | null | undefined, role: string | undefined, accepting: readonly string[] | undefined): boolean {
  if (!isClaudeCode(model)) return true
  return !!role && !!accepting?.includes(role)
}

export interface RefusingRole {
  role: string
  model: string
  /** No model of its own: it uses the main model. */
  inherited: boolean
}

/**
 * Roles that use Claude Code but refuse it: their own model, or the main model an empty role falls back to (the
 * engine's rule for every role). In `roles` order.
 */
export function refusingRoles(roles: Record<string, string | null | undefined>, accepting: readonly string[] | undefined): RefusingRole[] {
  if (!accepting) return []
  const out: RefusingRole[] = []
  for (const [role, own] of Object.entries(roles)) {
    const model = own || roles.primary
    if (model && isClaudeCode(model) && !accepting.includes(role)) out.push({ role, model, inherited: !own })
  }
  return out
}

/** The first model that isn't Claude Code, for the work Claude Code refuses. '' when there is none. */
export function otherModel(candidates: readonly (string | null | undefined)[]): string {
  return candidates.find((m): m is string => !!m && !isClaudeCode(m)) ?? ''
}
