/**
 * #544 Which loopback port the engine starts on. Pure (no Electron, no sockets), so `npm test` covers it.
 *
 * The desktop remembers the engine's port in its own settings file and tries it first on the next launch, so an
 * app that saved Sentient's address (an OpenAI-style client, a script) keeps working after a restart. If that port
 * is taken, by another program or a second Sentient, it picks a free one exactly as before. A missing or broken
 * stored value never stops the engine from starting.
 */

/** The remembered port, or undefined when the stored value is missing or isn't a usable port number. */
export function rememberedPort(value: unknown): number | undefined {
  // Never a privileged port: the engine never picked one, so such a value was edited in by hand or is corrupt.
  return typeof value === 'number' && Number.isInteger(value) && value >= 1024 && value <= 65535 ? value : undefined
}

export interface PortProbe {
  /** Whether nothing else listens on this loopback port right now. */
  isFree: (port: number) => Promise<boolean>
  /** A port the OS says is free (listen on port 0). */
  pickFree: () => Promise<number>
}

/**
 * The port to start the engine on: `preferred` (the port it already used this session, else the remembered one)
 * when it is a usable port and free, otherwise a fresh free port. Probe errors count as "taken".
 */
export async function choosePort(preferred: unknown, probe: PortProbe): Promise<number> {
  const port = rememberedPort(preferred)
  if (port !== undefined) {
    const free = await probe.isFree(port).catch(() => false)
    if (free) return port
  }
  return probe.pickFree()
}

/** Where the engine's last port is kept between launches (the desktop's own settings file, prefs.ts). */
export interface PortMemory {
  load: () => unknown
  save: (port: number) => void
}

/** The stored value, or undefined when there is no memory or reading it fails. */
export function loadRememberedPort(memory: PortMemory | undefined): unknown {
  try {
    return memory?.load()
  } catch {
    return undefined
  }
}

/** Save the port the engine answered on, unless already stored. Never throws: next launch just picks a free one. */
export function rememberPort(memory: PortMemory | undefined, port: number): void {
  try {
    if (loadRememberedPort(memory) !== port) memory?.save(port)
  } catch {
    /* non-fatal */
  }
}
