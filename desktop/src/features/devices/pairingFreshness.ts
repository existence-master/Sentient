/** Does the pairing code on screen still point where a device can reach Sentient? Pure, for `npm test` (#477).
 *
 *  A code's links are fixed when it is made: the Wi-Fi listener's `https://<LAN IP>:<port>/node/` while it runs,
 *  else this computer's own address. When the listener starts, stops or moves, the code on screen is stale and
 *  the Add a device dialog makes a new one (replacing the old code, so it stops working). */
import type { DeviceLanInfo, DevicePairing } from '../../lib/types'

export interface PairingFreshness {
  /** The listener is starting: wait for it before showing a code. */
  starting: boolean
  /** The code's links point somewhere the listener no longer is: make a new code. */
  stale: boolean
  /** Where the code should point: the listener's web address, or `local` for this computer. */
  target: string
}

const withoutCode = (url: string) => url.replace(/#.*$/, '')

export function pairingFreshness(info: DevicePairing | null | undefined, lan: DeviceLanInfo | null | undefined): PairingFreshness {
  const starting = !!lan?.enabled && !lan.running && !lan.error
  const lanBase = lan?.running ? lan.web_urls?.[0] : undefined
  const target = lanBase ?? 'local'
  if (!info || !lan || starting) return { starting, stale: false, target }
  // Engines that send `web_url` say exactly where the code points; otherwise the fingerprint (only set while the
  // listener runs) tells a Wi-Fi code from a local one.
  const madeOnLan = info.web_url ? /^https:/i.test(info.web_url) : !!info.fingerprint
  const stale = lanBase ? (info.web_url ? withoutCode(info.web_url) !== lanBase : !madeOnLan) : madeOnLan
  return { starting, stale, target }
}

/** How often to ask the engine about the Wi-Fi listener while the dialog is open: quickly while it starts, so the
 *  new code appears as soon as it is ready; slowly otherwise, to notice it stopping or the network changing. */
export function lanPollMs(lan: DeviceLanInfo | null | undefined): number {
  return lan?.enabled && !lan.running && !lan.error ? 500 : 5000
}
