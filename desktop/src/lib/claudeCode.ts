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

/**
 * The roles the "use another model" button changes: the ones still stuck, never the embedding role (it needs an
 * embedding model, not a chat one). A fast role the engine covers with a stand-in is not stuck, so it keeps its setting.
 */
export function fixableRoles(stuck: readonly RefusingRole[]): RefusingRole[] {
  return stuck.filter((r) => r.role !== 'embedding')
}

/** The first model that isn't Claude Code, for the work Claude Code refuses. '' when there is none. */
export function otherModel(candidates: readonly (string | null | undefined)[]): string {
  return candidates.find((m): m is string => !!m && !isClaudeCode(m)) ?? ''
}

/**
 * The models that do the fast role's quick jobs (memory decisions and extraction, summaries, chat titles) while every
 * model it has is Claude Code, in the engine's order: the main model's fallbacks, the executor, the planner, the main
 * model, never Claude Code. [] when the engine keeps the refusal: Claude Code does everything (`accepting` has `fast`),
 * the fast role has another model of its own, or no other model is set up.
 *
 * This mirrors `stand_ins` in `sentient/llm/claude_code.py` (#474); change them together so Settings names the model
 * the engine uses.
 */
export function fastStandIns(
  roles: Record<string, string | null | undefined>,
  fallbacks: Record<string, readonly string[] | undefined>,
  accepting: readonly string[] | undefined
): string[] {
  if (!accepting || accepting.includes('fast')) return []
  const chain = [roles.fast || roles.primary, ...(fallbacks.fast ?? [])]
  if (!chain[0] || !chain.every(isClaudeCode)) return []
  const candidates = [...(fallbacks.primary ?? []), roles.executor, roles.planner, roles.primary]
  return [...new Set(candidates.filter((m): m is string => !!m && !isClaudeCode(m)))]
}

/** What Settings shows next to Claude Code: off, chats only, or everything. */
export function claudeCodeState(enabled: boolean, everywhere: boolean): { label: string; tone: 'neutral' | 'warning' | 'danger' } {
  if (!enabled) return { label: 'Off', tone: 'neutral' }
  return everywhere ? { label: 'Does everything', tone: 'danger' } : { label: 'Chats only', tone: 'warning' }
}

export type ClaudeWay = 'key' | 'credits' | 'code_chats' | 'code_everything'

/** The Connect Claude choice that matches what is set up now (`primary` is the main model in use or picked). */
export function claudeWayFor(models: { experimental_claude_code?: boolean; claude_code_everywhere?: boolean } | undefined, primary: string): ClaudeWay {
  if (models?.experimental_claude_code && models.claude_code_everywhere) return 'code_everything'
  if (isClaudeCode(primary)) return 'code_chats'
  return 'key'
}
