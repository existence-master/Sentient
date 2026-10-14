/**
 * "Remembered: ... Undo" (#234): each time Sentient saves a memory from this chat, a small toast says what it saved,
 * with Undo. Undo deletes a new memory, or puts an updated one back the way it was (`memory.updated.previous`). A
 * memory that waits for review says so instead, with a link to the review inbox.
 */
import { useEffect } from 'react'
import { useNavigate } from 'react-router'
import { toast } from 'sonner'
import { qk } from '@/hooks/queryKeys'
import { api, errorMessage } from '@/lib/api'
import { queryClient } from '@/lib/queryClient'
import type { MemoryUpdatedData } from '@/lib/types'
import { truncate } from '@/lib/utils'
import { live } from '@/lib/ws'

async function undo(d: MemoryUpdatedData): Promise<void> {
  if (d.id === null) return
  try {
    if (d.action === 'UPDATE' && typeof d.previous === 'string') {
      await api.memories.update(d.id, d.previous)
      toast.success('Put back the way it was')
    } else {
      await api.memories.delete(d.id)
      toast.success('Forgotten')
    }
    void queryClient.invalidateQueries({ queryKey: qk.memories.all })
  } catch (err) {
    toast.error("Couldn't undo it", { description: errorMessage(err) })
  }
}

export function useMemorySaveToasts(sessionId: string | undefined) {
  const navigate = useNavigate()
  useEffect(() => {
    if (!sessionId) return
    return live.onDomain('memory.updated', (e) => {
      const d = e.data as MemoryUpdatedData
      if (d.session_id !== sessionId || d.id === null || !d.content || d.action === 'DELETE') return
      const text = truncate(d.content, 140)
      if (d.status === 'pending') {
        toast(`Waiting for your review: ${text}`, {
          description:
            d.reason === 'instruction'
              ? 'It tells Sentient how to act, so it waits for your OK.'
              : 'This chat read an email or web page, so it waits for your OK.',
          action: { label: 'Review', onClick: () => navigate('/memory?tab=review') }
        })
        return
      }
      toast(`${d.action === 'UPDATE' ? 'Updated' : 'Remembered'}: ${text}`, {
        action: { label: 'Undo', onClick: () => void undo(d) },
        duration: 8000
      })
    })
  }, [sessionId, navigate])
}
