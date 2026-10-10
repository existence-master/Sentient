import type { ScreenShare } from '@/types/bridge'

/** How long a picture from the share shortcuts waits for a new chat to pick it up (the app may still be starting). */
const MAX_WAIT_MS = 60_000

let pending: { share: ScreenShare; at: number } | null = null

/** Hold a picture from Share this window / Share a region until the new chat opens. */
export function holdScreenShare(share: ScreenShare): void {
  pending = { share, at: Date.now() }
}

/** The held picture as a file for the composer, once; null when there is none or it is stale. */
export function takeScreenShare(): File | null {
  const held = pending
  pending = null
  if (!held || Date.now() - held.at > MAX_WAIT_MS) return null
  return new File([held.share.data as Uint8Array<ArrayBuffer>], held.share.fileName, { type: held.share.mime })
}
