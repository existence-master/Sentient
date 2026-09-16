import { BrowserWindow, nativeTheme, screen, shell } from 'electron'
import { join } from 'node:path'
import windowIcon from '../../resources/icon.png?asset'
import { resolvedTheme, shellState, THEME_BG, THEME_TITLEBAR } from './prefs'

export const TITLEBAR_HEIGHT = 40

export interface CreateWindowOptions {
  route: string
  smokeSize?: { width: number; height: number }
  startHidden?: boolean
  smoke?: boolean
}

function visibleOnSomeDisplay(x: number, y: number, w: number, h: number): boolean {
  return screen.getAllDisplays().some(({ workArea: a }) => {
    const ix = Math.max(0, Math.min(x + w, a.x + a.width) - Math.max(x, a.x))
    const iy = Math.max(0, Math.min(y + h, a.y + a.height) - Math.max(y, a.y))
    return ix * iy > 100 * 100
  })
}

export function isInternalUrl(url: string): boolean {
  const dev = process.env.ELECTRON_RENDERER_URL
  if (dev && url.startsWith(dev)) return true
  return url.startsWith('file://')
}

export function createMainWindow(opts: CreateWindowOptions): BrowserWindow {
  const prefs = shellState.prefs()
  const theme = resolvedTheme(prefs, nativeTheme.shouldUseDarkColors)
  const saved = opts.smokeSize ? undefined : shellState.window()
  const width = opts.smokeSize?.width ?? saved?.width ?? 1280
  const height = opts.smokeSize?.height ?? saved?.height ?? 820
  const pos =
    saved?.x !== undefined && saved?.y !== undefined && visibleOnSomeDisplay(saved.x, saved.y, width, height)
      ? { x: saved.x, y: saved.y }
      : {}
  const titleBar = prefs.titleBar ?? THEME_TITLEBAR[theme]

  const win = new BrowserWindow({
    width,
    height,
    ...pos,
    minWidth: 960,
    minHeight: 640,
    show: false,
    title: 'Sentient',
    icon: windowIcon,
    backgroundColor: THEME_BG[theme],
    autoHideMenuBar: true,
    ...(process.platform === 'darwin'
      ? { titleBarStyle: 'hiddenInset' as const, trafficLightPosition: { x: 14, y: 13 } }
      : {
          titleBarStyle: 'hidden' as const,
          titleBarOverlay: { color: titleBar.color, symbolColor: titleBar.symbolColor, height: TITLEBAR_HEIGHT }
        }),
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      contextIsolation: true,
      sandbox: true,
      nodeIntegration: false,
      webviewTag: false,
      spellcheck: true,
      backgroundThrottling: !opts.smoke
    }
  })
  if (opts.smokeSize) win.setContentSize(opts.smokeSize.width, opts.smokeSize.height)
  if (saved?.maximized && !opts.smoke) win.maximize()

  win.once('ready-to-show', () => {
    if (opts.smoke) win.showInactive()
    else if (!opts.startHidden) win.show()
  })

  // Remember size and position.
  let saveTimer: NodeJS.Timeout | null = null
  const persist = () => {
    if (opts.smoke) return
    if (saveTimer) clearTimeout(saveTimer)
    saveTimer = setTimeout(() => {
      if (win.isDestroyed() || win.isMinimized()) return
      const maximized = win.isMaximized()
      const b = maximized ? (shellState.window() ?? win.getNormalBounds()) : win.getBounds()
      shellState.setWindow({ x: b.x, y: b.y, width: b.width, height: b.height, maximized })
    }, 400)
  }
  win.on('resize', persist)
  win.on('move', persist)
  win.on('maximize', persist)
  win.on('unmaximize', persist)

  // Links open in the system browser; the app never navigates away from itself.
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (/^(https?:|mailto:)/i.test(url)) void shell.openExternal(url)
    return { action: 'deny' }
  })
  win.webContents.on('will-navigate', (event, url) => {
    if (isInternalUrl(url)) return
    event.preventDefault()
    if (/^(https?:|mailto:)/i.test(url)) void shell.openExternal(url)
  })
  win.webContents.on('will-attach-webview', (event) => event.preventDefault())

  const hash = opts.route.startsWith('/') ? opts.route : `/${opts.route}`
  if (process.env.ELECTRON_RENDERER_URL) {
    void win.loadURL(`${process.env.ELECTRON_RENDERER_URL}#${hash}`)
  } else {
    void win.loadFile(join(__dirname, '../renderer/index.html'), { hash })
  }
  return win
}
