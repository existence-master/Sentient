import { Menu, nativeImage, Tray } from 'electron'
import trayIconPng from '../../resources/icon.png?asset'
import trayIconIco from '../../resources/icon.ico?asset'

export interface TrayHandlers {
  open(): void
  newChat(): void
  voiceMode(): void
  toggleProactivity(): void
  quit(): void
  proactivityEnabled(): boolean | undefined
  backendReady(): boolean
  alwaysListening(): boolean
  toggleAlwaysListening(): void
  /** §17 Stop everything: undefined until the engine said. */
  stopped(): boolean | undefined
  toggleStopped(): void
  /** Push to talk or dictation has the microphone on (#169). */
  micOn(): boolean
}

export const STOP_ACCELERATOR = 'CommandOrControl+Alt+Shift+S'

const WAKE_LABEL = "Listen for 'Hey Sentient'"

export class AppTray {
  private tray: Tray

  constructor(private h: TrayHandlers) {
    const img =
      process.platform === 'win32'
        ? nativeImage.createFromPath(trayIconIco)
        : nativeImage.createFromPath(trayIconPng).resize({ width: 18, height: 18 })
    this.tray = new Tray(img.isEmpty() ? nativeImage.createFromPath(trayIconPng).resize({ width: 16, height: 16 }) : img)
    this.tray.setToolTip('Sentient')
    this.tray.on('click', () => h.open())
    this.refresh()
  }

  /** Privacy: the tooltip always says when the microphone is listening for the wake word. */
  private baseTooltip(): string {
    if (this.h.micOn()) return 'Sentient (microphone on)'
    return this.h.alwaysListening() ? "Sentient (listening for 'Hey Sentient')" : 'Sentient'
  }

  refresh(): void {
    const pro = this.h.proactivityEnabled()
    const ready = this.h.backendReady()
    const listening = this.h.alwaysListening()
    const stopped = this.h.stopped() === true
    if (!this.flashTimer) this.tray.setToolTip(this.baseTooltip())
    if (process.platform === 'darwin') this.tray.setTitle(listening || this.h.micOn() ? ' ●' : '')
    this.tray.setContextMenu(
      Menu.buildFromTemplate([
        { label: 'Open Sentient', click: () => this.h.open() },
        { label: 'New chat', accelerator: 'CommandOrControl+Shift+Space', click: () => this.h.newChat() },
        { label: 'Voice mode', click: () => this.h.voiceMode() },
        { label: WAKE_LABEL, type: 'checkbox', checked: listening, click: () => this.h.toggleAlwaysListening() },
        { type: 'separator' },
        {
          label: stopped ? 'Resume Sentient' : 'Stop everything',
          accelerator: stopped ? undefined : STOP_ACCELERATOR,
          enabled: ready,
          click: () => this.h.toggleStopped()
        },
        {
          label: pro === false ? 'Resume proactivity' : 'Pause proactivity',
          enabled: ready && pro !== undefined,
          click: () => this.h.toggleProactivity()
        },
        { type: 'separator' },
        { label: 'Quit Sentient', click: () => this.h.quit() }
      ])
    )
  }

  private flashTimer: NodeJS.Timeout | null = null

  /** Briefly change the tray tooltip so a capture is visible even when the window is hidden. */
  flash(text: string): void {
    this.tray.setToolTip(`Sentient: ${text}`)
    if (this.flashTimer) clearTimeout(this.flashTimer)
    this.flashTimer = setTimeout(() => {
      this.flashTimer = null
      this.tray.setToolTip(this.baseTooltip())
    }, 6000)
  }

  destroy(): void {
    this.tray.destroy()
  }
}
