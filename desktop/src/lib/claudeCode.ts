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
