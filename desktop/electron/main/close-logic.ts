/**
 * #439 What closing the main window does. Pure (no Electron), so `npm test` covers it.
 *
 * The app keeps hidden helper windows (the voice pill, the ask window) alive for the next press, so Electron's
 * `window-all-closed` never fires once one of them exists. Closing the main window therefore has to quit on purpose
 * when it isn't hiding to the tray.
 */

/** `hide`: keep running in the tray. `quit`: quit the app (helpers and engine too). `close`: just let it close. */
export type CloseAction = 'hide' | 'quit' | 'close'

export interface CloseState {
  /** The app is already quitting (tray Quit, the OS shutting down, `app.quit()`). */
  quitting: boolean
  /** A smoke run: it ends itself. */
  smoke: boolean
  /** The tray icon exists, so a hidden window can be found again. */
  hasTray: boolean
  /** Settings, minimize to tray (`ui.minimize_to_tray`); unset means on, as in the config default. */
  minimizeToTray: boolean | undefined
  platform: NodeJS.Platform | string
}

export function closeAction(s: CloseState): CloseAction {
  if (s.quitting || s.smoke) return 'close'
  if (s.hasTray && s.minimizeToTray !== false) return 'hide'
  // macOS apps stay in the Dock with no windows open; clicking the Dock icon brings the window back.
  if (s.platform === 'darwin') return 'close'
  return 'quit'
}
