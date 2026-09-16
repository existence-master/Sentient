import { useCallback, useRef, useState } from 'react'
import { toast } from 'sonner'
import { api, errorMessage } from '@/lib/api'
import type { AttachmentView } from '@/lib/chatFold'
import { uid } from '@/lib/utils'

export interface PendingAttachment {
  id: string
  file: File
  status: 'uploading' | 'done' | 'error'
  progress: number
  /** Server-side name once uploaded (e.g. `uploads/report.pdf`). */
  name?: string
  size: number
  mime: string
  previewUrl?: string
  error?: string
  abort?: AbortController
}

const MAX_BYTES = 50 * 1024 * 1024

/** Upload files to `POST /api/files` with progress; returns chips state + helpers. */
export function useAttachments() {
  const [items, setItems] = useState<PendingAttachment[]>([])
  const itemsRef = useRef(items)
  itemsRef.current = items

  const update = (id: string, patch: Partial<PendingAttachment>) =>
    setItems((list) => list.map((x) => (x.id === id ? { ...x, ...patch } : x)))

  const start = useCallback((item: PendingAttachment) => {
    const abort = new AbortController()
    update(item.id, { status: 'uploading', progress: 0, error: undefined, abort })
    api.files
      .upload(item.file, item.file.name || `pasted-${Date.now()}.png`, {
        signal: abort.signal,
        onProgress: (p) => update(item.id, { progress: p })
      })
      .then((res) => update(item.id, { status: 'done', progress: 1, name: res.name, size: res.size, mime: res.mime }))
      .catch((err) => {
        if ((err as Error)?.name === 'AbortError') return
        update(item.id, { status: 'error', error: errorMessage(err) })
      })
  }, [])

  const add = useCallback(
    (files: FileList | File[]) => {
      const next: PendingAttachment[] = []
      for (const file of Array.from(files)) {
        if (file.size > MAX_BYTES) {
          toast.error(`${file.name} is too large`, { description: 'Attachments can be up to 50 MB.' })
          continue
        }
        next.push({
          id: uid(),
          file,
          status: 'uploading',
          progress: 0,
          size: file.size,
          mime: file.type,
          previewUrl: file.type.startsWith('image/') ? URL.createObjectURL(file) : undefined
        })
      }
      if (!next.length) return
      setItems((list) => [...list, ...next])
      next.forEach(start)
    },
    [start]
  )

  const remove = useCallback((id: string) => {
    const item = itemsRef.current.find((x) => x.id === id)
    item?.abort?.abort()
    if (item?.previewUrl) URL.revokeObjectURL(item.previewUrl)
    setItems((list) => list.filter((x) => x.id !== id))
  }, [])

  const retry = useCallback((id: string) => {
    const item = itemsRef.current.find((x) => x.id === id)
    if (item) start(item)
  }, [start])

  /** Hand the uploaded files over to a message (preview URLs now belong to the message). */
  const take = useCallback((): AttachmentView[] => {
    const done = itemsRef.current.filter((x) => x.status === 'done' && x.name)
    setItems([])
    return done.map((x) => ({ name: x.name as string, size: x.size, mime: x.mime, previewUrl: x.previewUrl }))
  }, [])

  return {
    items,
    add,
    remove,
    retry,
    take,
    uploading: items.some((x) => x.status === 'uploading'),
    ready: items.filter((x) => x.status === 'done').length
  }
}
