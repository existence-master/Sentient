import { toast } from 'sonner'
import { api } from '@/lib/api'
import { getBridge } from '@/lib/bridge'

/** Tool results name output files either relative to `files/` (`outputs/x.csv`) or bare (`x.csv` in `files/outputs/`). */
export function outputPath(name: string): string {
  const clean = name.replace(/\\/g, '/').replace(/^\/+/, '')
  return clean.includes('/') ? clean : `outputs/${clean}`
}

export function fileBase(name: string): string {
  return name.split(/[\\/]/).pop() ?? name
}

export const IMAGE_FILE_RE = /\.(png|jpe?g|gif|webp|bmp|svg)$/i

export async function openOutputFile(name: string): Promise<void> {
  const bridge = getBridge()
  const path = outputPath(name)
  if (bridge.isDesktop) {
    const ok = await bridge.openFile(path)
    if (!ok) toast.error("Couldn't open that file", { description: 'It may have been moved or deleted.' })
    return
  }
  window.open(api.files.contentUrl(path), '_blank', 'noopener,noreferrer')
}
