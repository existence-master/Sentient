/**
 * Domain events for desktop agent B screens. Installed once by App.tsx.
 *
 *   user_model.updated   -> refetch ['user-model']
 *   dream.updated        -> upsert into ['memories', 'dreams'] (and refetch the user model when one completes)
 *   source.items         -> webhook calls refresh ['hooks'] (last called / call count)
 *   task.run_finished    -> nothing extra (task.updated already carries the task)
 *
 * These event types are not in lib/types DomainEventMap (owned by desktop agent A), so they are read through
 * the untyped `live.subscribe` stream.
 */
import type { QueryClient } from '@tanstack/react-query'
import { installWakeListener } from '@/features/voice/wake'
import { live } from '@/lib/ws'
import { qkB } from './api-b'
import { installLeapDemo } from './demo-b'
import { upsertDream } from './hooks-b'
import type { Dream, SourceItemsData } from './types-b'

export function installLeapBEvents(queryClient: QueryClient): () => void {
  const offs: Array<() => void> = []

  offs.push(
    live.subscribe((raw) => {
      const msg = raw as { type: string; data?: unknown }
      switch (msg.type) {
        case 'integration.updated':
          // feed state changes with connections; the builtin `webhook` integration changes when hooks do
          void queryClient.invalidateQueries({ queryKey: qkB.feeds })
          if ((msg.data as { id?: string } | undefined)?.id === 'webhook') void queryClient.invalidateQueries({ queryKey: qkB.hooks })
          break
        case 'user_model.updated':
          void queryClient.invalidateQueries({ queryKey: qkB.userModel })
          break
        case 'dream.updated': {
          const d = msg.data as Dream | undefined
          if (!d?.id) break
          queryClient.setQueryData<Dream[]>(qkB.dreams, (old) => upsertDream(old, d))
          if (d.status === 'completed') void queryClient.invalidateQueries({ queryKey: qkB.userModel })
          break
        }
        case 'source.items': {
          const data = msg.data as SourceItemsData | undefined
          if (data?.origin === 'webhook' || data?.source === 'webhook') void queryClient.invalidateQueries({ queryKey: qkB.hooks })
          break
        }
      }
    })
  )

  offs.push(installLeapDemo(queryClient))
  offs.push(installWakeListener())
  return () => offs.forEach((off) => off())
}
