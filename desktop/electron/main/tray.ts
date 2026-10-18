import { Menu, nativeImage, Tray, type MenuItemConstructorOptions, type NativeImage } from 'electron'
import trayIconPng from '../../resources/icon.png?asset'
import trayIconIco from '../../resources/icon.ico?asset'
import { drawLiveDot, sensorMenuLabel, sensorTooltip, type LiveSensor } from './sensors-logic'
import { NEW_CHAT_ACCELERATOR, STOP_ACCELERATOR } from './shortcuts'

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
  /** #170 Every live microphone and camera, here and on paired devices. */
  sensors(): LiveSensor[]
  /** Turn one off (`id`), or all of them. */
  stopSensor(id?: string): void
  /** Show the window with the title bar's microphone and camera details open. */
  showSensors(): void
  shareWindow(): void
  shareRegion(): void
  /** Current shortcut for a share item, "" when turned off. */
  shareAccelerators(): { window: string; region: string }
}

const WAKE_LABEL = "Listen for 'Hey Sentient'"

/** #170 The tray icon with a red "live" dot, drawn once from the app icon. */
function liveIcon(base: NativeImage): NativeImage {
  const size = process.platform === 'win32' ? 32 : 36 // drawn at 2x, so it stays sharp on high-density screens
  const img = base.resize({ width: size, height: size, quality: 'best' })
  const dotted = drawLiveDot(img.toBitmap(), size, size)
  return nativeImage.createFromBitmap(Buffer.from(dotted), { width: size, height: size, scaleFactor: 2 })
}

export class AppTray {
  private tray: Tray
  private normalIcon: NativeImage
  private dotIcon: NativeImage | null = null
  private live = false

  constructor(private h: TrayHandlers) {
    const img =
      process.platform === 'win32'
        ? nativeImage.createFromPath(trayIconIco)
        : nativeImage.createFromPath(trayIconPng).resize({ width: 18, height: 18 })
    this.normalIcon = img.isEmpty() ? nativeImage.createFromPath(trayIconPng).resize({ width: 16, height: 16 }) : img
    this.tray = new Tray(this.normalIcon)
    this.tray.setToolTip('Sentient')
    // while a microphone or camera is live, a click shows which one and what for, with Turn off
    this.tray.on('click', () => (this.h.sensors().length ? h.showSensors() : h.open()))
    this.refresh()
  }

  /** Privacy: the tooltip always says when a microphone or camera is live, or listening for the wake word. */
  private baseTooltip(): string {
    const live = sensorTooltip(this.h.sensors())
    if (live) return live
    if (this.h.micOn()) return 'Sentient (microphone on)'
    return this.h.alwaysListening() ? "Sentient (listening for 'Hey Sentient')" : 'Sentient'
  }

  /** The red dot on the tray icon whenever a microphone or camera is live (#170). */
  private showLive(live: boolean): void {
    if (live === this.live) return
    this.live = live
    if (process.platform === 'darwin') return // the menu bar shows a dot as the title instead
    try {
      this.dotIcon ??= liveIcon(nativeImage.createFromPath(trayIconPng))
      this.tray.setImage(live ? this.dotIcon : this.normalIcon)
    } catch {
      /* the tooltip and menu still say it */
    }
  }

  private sensorItems(uses: LiveSensor[]): MenuItemConstructorOptions[] {
    if (!uses.length) return []
    const items: MenuItemConstructorOptions[] = uses.map((u) => ({
      label: sensorMenuLabel(u),
      enabled: u.can_stop,
      click: () => this.h.stopSensor(u.id)
    }))
    if (uses.filter((u) => u.can_stop).length > 1) items.push({ label: 'Turn off all', click: () => this.h.stopSensor() })
    items.push({ label: 'Show details', click: () => this.h.showSensors() }, { type: 'separator' })
    return items
  }

  refresh(): void {
    const pro = this.h.proactivityEnabled()
    const ready = this.h.backendReady()
    const listening = this.h.alwaysListening()
    const stopped = this.h.stopped() === true
    const share = this.h.shareAccelerators()
    const sensors = this.h.sensors()
    this.showLive(sensors.length > 0)
    if (!this.flashTimer) this.tray.setToolTip(this.baseTooltip())
    if (process.platform === 'darwin') this.tray.setTitle(listening || this.h.micOn() || sensors.length ? ' ●' : '')
    this.tray.setContextMenu(
      Menu.buildFromTemplate([
        ...this.sensorItems(sensors),
        { label: 'Open Sentient', click: () => this.h.open() },
        { label: 'New chat', accelerator: NEW_CHAT_ACCELERATOR, click: () => this.h.newChat() },
        { label: 'Voice mode', click: () => this.h.voiceMode() },
        { label: 'Share this window', accelerator: share.window || undefined, click: () => this.h.shareWindow() },
        { label: 'Share a region', accelerator: share.region || undefined, click: () => this.h.shareRegion() },
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
