/**
 * Keeping the user's clipboard: dictation (#169) and Ask about selected text (#173) borrow the clipboard to move text
 * in and out of other apps, then put back what was there.
 */
import { clipboard, ClipboardItem } from 'electron'

export type SavedClipboard = ClipboardItem[] | null

/** Everything on the clipboard (every format of every item), or null when it can't be read. */
export async function saveClipboard(): Promise<SavedClipboard> {
  try {
    const saved: ClipboardItem[] = []
    for (const item of await clipboard.read()) {
      const entries: Record<string, Blob | string | Electron.ClipboardBookmark> = {}
      for (const type of item.types) {
        try {
          entries[type] = await item.getType(type)
        } catch {
          /* a format that can't be read back */
        }
      }
      if (Object.keys(entries).length) saved.push(new ClipboardItem(entries))
    }
    return saved
  } catch {
    return null
  }
}

/** Put `saved` back. With `onlyIf`, only while the clipboard still holds that text, so a newer copy is kept. */
export async function restoreClipboard(saved: SavedClipboard, onlyIf?: string): Promise<boolean> {
  try {
    if (saved === null) return false
    if (onlyIf !== undefined && (await clipboard.readText()) !== onlyIf) return false
    if (saved.length) await clipboard.write(saved)
    else clipboard.clear()
    return true
  } catch (err) {
    console.warn(`[shell] could not restore the clipboard: ${String(err)}`)
    return false
  }
}
