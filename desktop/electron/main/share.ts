/**
 * Share this window / Share a region (issue #172).
 *
 * Only ever runs because the user pressed a shortcut or picked it from the tray menu. It takes one
 * picture, shows a short flash where it was taken, and hands the picture to a new chat in the main
 * window. Nothing is sent to a model until the user writes a question and presses send; the chat
 * then counts as having read outside content (ADR 0018), because a screen can show anyone's text.
 *
 * Uses Electron's desktopCapturer and screen APIs only. On macOS the system asks for Screen
 * Recording permission the first time; until it is given, Sentient explains where to turn it on.
 */
import { BrowserWindow, desktopCapturer, dialog, type Display, type NativeImage, type Rectangle, screen, shell, systemPreferences } from 'electron'
import type { ScreenShare } from '../../src/types/bridge'
import { captureFileName, cleanRegion, fitWithin, pickActiveWindow, type Rect, regionInImage } from './share-helpers'

const PICK_TIMEOUT_MS = 120_000
const FLASH_MS = 900
const ACCENT = '#f5a524'

export interface ShareHost {
  /** Open a new chat with the picture attached. */
  deliver(share: ScreenShare): void
  /** Say what happened in the tray tooltip (the window may be hidden). */
  notice(text: string): void
}

// The overlay is a page of its own (no preload, no Node): it shows a still picture of the screen and
// resolves `pickRegion()` with the rectangle the user dragged, in screen points, or null.
const REGION_HTML = `<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'">
<title>Share a region</title><style>
html,body{margin:0;height:100%;overflow:hidden;cursor:crosshair;user-select:none;background:#000;font:13px system-ui,-apple-system,"Segoe UI",sans-serif}
#shot{position:fixed;left:0;top:0;width:100%;height:100%;pointer-events:none}
#dim{position:fixed;inset:0;background:rgba(0,0,0,.45)}
#sel{position:fixed;display:none;border:2px solid ${ACCENT};box-shadow:0 0 0 9999px rgba(0,0,0,.45);box-sizing:border-box}
#size{position:absolute;right:0;top:100%;margin-top:6px;padding:2px 6px;border-radius:6px;background:rgba(0,0,0,.75);color:#fff;font-size:12px;white-space:nowrap}
#hint{position:fixed;left:50%;top:24px;transform:translateX(-50%);padding:8px 14px;border-radius:999px;background:rgba(20,20,24,.88);color:#fff;box-shadow:0 4px 18px rgba(0,0,0,.35)}
</style></head><body><img id="shot" alt=""><div id="dim"></div><div id="sel"><span id="size"></span></div>
<div id="hint">Drag over what you want to share. Press Esc to cancel.</div>
<script>
const shot = document.getElementById('shot'), dim = document.getElementById('dim'), sel = document.getElementById('sel'), size = document.getElementById('size'), hint = document.getElementById('hint')
window.setShot = (src) => { shot.src = src; return shot.decode().then(() => true, () => false) }
// the window may sit a little off the display (the macOS menu bar): line the picture up with the real screen
window.place = (dx, dy, w, h) => Object.assign(shot.style, { left: -dx + 'px', top: -dy + 'px', width: w + 'px', height: h + 'px' })
window.pickRegion = () => new Promise((resolve) => {
  let start = null
  const rect = (e) => ({ x: Math.min(start.x, e.clientX), y: Math.min(start.y, e.clientY), width: Math.abs(e.clientX - start.x), height: Math.abs(e.clientY - start.y) })
  const finish = (r) => { removeEventListener('keydown', onKey, true); resolve(r) }
  const onKey = (e) => { if (e.key === 'Escape') { e.preventDefault(); finish(null) } }
  addEventListener('keydown', onKey, true)
  addEventListener('contextmenu', (e) => { e.preventDefault(); finish(null) })
  addEventListener('pointerdown', (e) => {
    if (e.button !== 0) return
    start = { x: e.clientX, y: e.clientY }
    try { document.body.setPointerCapture(e.pointerId) } catch {}
    hint.style.display = 'none'
  })
  addEventListener('pointermove', (e) => {
    if (!start) return
    const r = rect(e)
    dim.style.display = 'none'
    Object.assign(sel.style, { display: 'block', left: r.x + 'px', top: r.y + 'px', width: r.width + 'px', height: r.height + 'px' })
    size.textContent = Math.round(r.width) + ' x ' + Math.round(r.height)
  })
  addEventListener('pointerup', (e) => {
    if (!start) return
    const r = rect(e)
    start = null
    if (r.width >= 8 && r.height >= 8) { finish(r); return }
    sel.style.display = 'none'; dim.style.display = 'block'; hint.style.display = 'block'
  })
})
</script></body></html>`

// A short flash and outline where the picture was taken, so a capture is always visible.
const FLASH_HTML = `<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
<style>html,body{margin:0;height:100%;overflow:hidden;background:transparent}
div{position:fixed;inset:0;border:4px solid ${ACCENT};box-sizing:border-box;background:rgba(255,255,255,.28);animation:f ${FLASH_MS}ms ease-out forwards}
@keyframes f{0%{opacity:1}25%{background:rgba(255,255,255,0)}100%{opacity:0;background:rgba(255,255,255,0)}}</style>
</head><body><div></div></body></html>`

const html = (page: string) => `data:text/html;charset=utf-8,${encodeURIComponent(page)}`

function cursorDisplay(): Display {
  return screen.getDisplayNearestPoint(screen.getCursorScreenPoint())
}

function lockDown(win: BrowserWindow): void {
  win.webContents.setWindowOpenHandler(() => ({ action: 'deny' }))
  win.webContents.on('will-navigate', (event) => event.preventDefault())
}

/** Show a brief outline and flash over `bounds` (screen points). Never focusable, never in the way of clicks. */
function flash(bounds: Rectangle): void {
  const win = new BrowserWindow({
    ...bounds,
    show: false,
    frame: false,
    transparent: true,
    focusable: false,
    skipTaskbar: true,
    resizable: false,
    movable: false,
    hasShadow: false,
    alwaysOnTop: true,
    enableLargerThanScreen: true,
    webPreferences: { sandbox: true, contextIsolation: true, nodeIntegration: false }
  })
  lockDown(win)
  win.setIgnoreMouseEvents(true)
  win.setAlwaysOnTop(true, 'screen-saver')
  win.webContents
    .loadURL(html(FLASH_HTML))
    .then(() => {
      if (win.isDestroyed()) return
      win.setBounds(bounds)
      win.showInactive()
    })
    .catch(() => undefined)
  setTimeout(() => {
    if (!win.isDestroyed()) win.destroy()
  }, FLASH_MS + 150)
}

function toPng(image: NativeImage): Buffer {
  const size = image.getSize()
  const fit = fitWithin(size)
  const scaled = fit.width < size.width ? image.resize({ width: fit.width, height: fit.height, quality: 'best' }) : image
  return scaled.toPNG()
}

export class ScreenSharer {
  private busy = false

  constructor(private host: ShareHost) {}

  /** The front-most window that isn't Sentient's. */
  async shareWindow(): Promise<void> {
    await this.once(async () => {
      if (!(await this.mayCapture())) return
      const display = cursorDisplay()
      const own = BrowserWindow.getAllWindows().map((w) => {
        try {
          return w.getMediaSourceId()
        } catch {
          return ''
        }
      })
      const sources = await desktopCapturer.getSources({
        types: ['window'],
        thumbnailSize: fitWithin({ width: display.size.width * display.scaleFactor, height: display.size.height * display.scaleFactor }),
        fetchWindowIcons: false
      })
      if (!(await this.captured())) return
      const source = pickActiveWindow(
        sources.map((s) => ({ id: s.id, name: s.name, empty: s.thumbnail.isEmpty(), source: s })),
        own
      )
      if (!source) {
        await dialog.showMessageBox({
          type: 'info',
          title: 'Sentient',
          message: "There's no window to share",
          detail: 'Click the window you want to ask about, then press the shortcut again. Windows that are minimized can\'t be shared.',
          noLink: true
        })
        return
      }
      flash(display.bounds)
      this.host.notice(`shared the window "${source.name}"`)
      this.host.deliver({
        kind: 'window',
        title: source.name,
        fileName: captureFileName('window', source.name),
        mime: 'image/png',
        data: toPng(source.source.thumbnail)
      })
    })
  }

  /** A rectangle the user drags on the screen under the mouse pointer. */
  async shareRegion(): Promise<void> {
    await this.once(async () => {
      if (!(await this.mayCapture())) return
      const display = cursorDisplay()
      const sources = await desktopCapturer.getSources({
        types: ['screen'],
        thumbnailSize: {
          width: Math.round(display.size.width * display.scaleFactor),
          height: Math.round(display.size.height * display.scaleFactor)
        }
      })
      if (!(await this.captured())) return
      const source = sources.find((s) => s.display_id === String(display.id)) ?? (sources.length === 1 ? sources[0] : undefined)
      if (!source || source.thumbnail.isEmpty()) {
        dialog.showErrorBox('Sentient', "Couldn't take a picture of this screen. Please try again.")
        return
      }
      const region = await this.pickRegion(display, source.thumbnail)
      if (!region) return
      const crop = regionInImage(region, display.size, source.thumbnail.getSize())
      flash({ x: display.bounds.x + region.x, y: display.bounds.y + region.y, width: region.width, height: region.height })
      this.host.notice('shared part of your screen')
      this.host.deliver({
        kind: 'region',
        title: 'Screen region',
        fileName: captureFileName('region', ''),
        mime: 'image/png',
        data: toPng(source.thumbnail.crop(crop))
      })
    })
  }

  /** One capture at a time; a second press while the overlay is open is ignored. */
  private async once(run: () => Promise<void>): Promise<void> {
    if (this.busy) return
    this.busy = true
    try {
      await run()
    } catch (err) {
      console.warn('[share] capture failed', err)
      dialog.showErrorBox('Sentient', `Couldn't share your screen: ${err instanceof Error ? err.message : String(err)}`)
    } finally {
      this.busy = false
    }
  }

  private async pickRegion(display: Display, image: NativeImage): Promise<Rect | null> {
    const { bounds } = display
    const win = new BrowserWindow({
      ...bounds,
      show: false,
      frame: false,
      resizable: false,
      movable: false,
      minimizable: false,
      maximizable: false,
      fullscreenable: false,
      skipTaskbar: true,
      hasShadow: false,
      alwaysOnTop: true,
      enableLargerThanScreen: true,
      backgroundColor: '#000000',
      title: 'Share a region',
      webPreferences: { sandbox: true, contextIsolation: true, nodeIntegration: false }
    })
    lockDown(win)
    win.setAlwaysOnTop(true, 'screen-saver')
    if (process.platform === 'darwin') win.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true })
    const closed = new Promise<null>((resolve) => win.once('closed', () => resolve(null)))
    let timer: NodeJS.Timeout | undefined
    const timeout = new Promise<null>((resolve) => {
      timer = setTimeout(() => resolve(null), PICK_TIMEOUT_MS)
    })
    try {
      await win.loadURL(html(REGION_HTML))
      const still = `data:image/jpeg;base64,${image.toJPEG(90).toString('base64')}`
      await win.webContents.executeJavaScript(`window.setShot(${JSON.stringify(still)})`)
      win.setBounds(bounds) // macOS keeps new windows below the menu bar until they are placed again
      win.show()
      win.focus()
      // where the page's (0, 0) really is on the display; regions are measured from the display's corner
      const content = win.getContentBounds()
      const dx = content.x - bounds.x
      const dy = content.y - bounds.y
      await win.webContents.executeJavaScript(`window.place(${dx}, ${dy}, ${bounds.width}, ${bounds.height})`)
      const picked = (win.webContents.executeJavaScript('window.pickRegion()', true) as Promise<unknown>).catch(() => null)
      const raw = (await Promise.race([picked, closed, timeout])) as Partial<Rect> | null
      if (!raw || typeof raw !== 'object') return null
      return cleanRegion({ ...raw, x: Number(raw.x) + dx, y: Number(raw.y) + dy }, bounds)
    } finally {
      if (timer) clearTimeout(timer)
      if (!win.isDestroyed()) win.destroy()
    }
  }

  // ------------------------------------------------------------------ macOS Screen Recording permission
  private screenAccess(): 'granted' | 'not-determined' | 'denied' {
    if (process.platform !== 'darwin') return 'granted'
    const status = systemPreferences.getMediaAccessStatus('screen')
    return status === 'granted' || status === 'not-determined' ? status : 'denied'
  }

  /** Before capturing: false (after explaining) when macOS already said no. */
  private async mayCapture(): Promise<boolean> {
    if (this.screenAccess() !== 'denied') return true
    await this.explainPermission()
    return false
  }

  /** After capturing: the first capture on macOS triggers the system's question; without a yes the picture only shows the desktop. */
  private async captured(): Promise<boolean> {
    if (this.screenAccess() === 'granted') return true
    await this.explainPermission()
    return false
  }

  private async explainPermission(): Promise<void> {
    const { response } = await dialog.showMessageBox({
      type: 'info',
      buttons: ['Open Screen Recording settings', 'Not now'],
      defaultId: 0,
      cancelId: 1,
      noLink: true,
      title: 'Sentient',
      message: 'Sentient needs your permission to see the screen',
      detail:
        'To share a window or part of your screen, open System Settings > Privacy & Security > Screen & System Audio Recording and turn on Sentient. Then quit Sentient and open it again.\n\nNothing was captured.'
    })
    if (response === 0) void shell.openExternal('x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture')
  }
}
