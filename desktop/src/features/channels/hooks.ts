/** React Query hooks for §14 messaging channels. Live updates arrive via `channel.updated` (lib/events.ts). */
import { useMutation, useQuery, useQueryClient, type QueryClient } from '@tanstack/react-query'
import { api } from '@/lib/api'
import type { Channel, ChannelPairing } from '@/lib/types'
import { previewChannels, previewOr } from '@/features/devices/preview'

export const channelKeys = {
  all: ['channels'] as const
}

export function upsertChannel(qc: QueryClient, channel: Channel) {
  qc.setQueryData<Channel[]>(channelKeys.all, (old) => {
    if (!old) return old
    const i = old.findIndex((c) => c.id === channel.id)
    if (i === -1) return [...old, channel]
    const next = old.slice()
    next[i] = channel
    return next
  })
}

export function useChannels() {
  return useQuery({ queryKey: channelKeys.all, queryFn: () => previewOr(api.channels.list, previewChannels) })
}

function current(qc: QueryClient, id: string): Channel {
  return (qc.getQueryData<Channel[]>(channelKeys.all) ?? previewChannels()).find((c) => c.id === id) as Channel
}

export function useChannelActions() {
  const qc = useQueryClient()
  const onSuccess = (c: Channel) => upsertChannel(qc, c)
  return {
    connect: useMutation({
      mutationFn: ({ id, fields }: { id: string; fields: Record<string, string> }) =>
        previewOr(
          () => api.channels.connect(id, fields),
          () => ({ ...current(qc, id), status: 'connected' as const, account_label: id === 'whatsapp' ? '+15550100123' : '@sentient_demo_bot', error: null })
        ),
      onSuccess
    }),
    disconnect: useMutation({
      mutationFn: (id: string) =>
        previewOr(() => api.channels.disconnect(id), () => ({ ...current(qc, id), status: 'disconnected' as const, account_label: null })),
      onSuccess
    }),
    pairing: useMutation({
      mutationFn: (id: string) =>
        previewOr<ChannelPairing>(
          () => api.channels.pairing(id),
          () => ({ code: '305718', expires_at: new Date(Date.now() + 10 * 60_000).toISOString(), instructions: 'Send /pair 305718 to your bot.' })
        )
    }),
    setDeliver: useMutation({
      mutationFn: ({ id, chatId, deliver }: { id: string; chatId: string; deliver: boolean }) =>
        previewOr(
          () => api.channels.setDeliver(id, chatId, deliver),
          () => {
            const c = current(qc, id)
            return { ...c, paired: c.paired.map((p) => (p.chat_id === chatId ? { ...p, deliver } : p)) }
          }
        ),
      onMutate: ({ id, chatId, deliver }) => {
        qc.setQueryData<Channel[]>(channelKeys.all, (old) =>
          old?.map((c) => (c.id === id ? { ...c, paired: c.paired.map((p) => (p.chat_id === chatId ? { ...p, deliver } : p)) } : c))
        )
      },
      onSuccess,
      onError: () => void qc.invalidateQueries({ queryKey: channelKeys.all })
    }),
    removePaired: useMutation({
      mutationFn: ({ id, chatId }: { id: string; chatId: string }) =>
        previewOr(
          () => api.channels.removePaired(id, chatId),
          () => {
            const c = current(qc, id)
            return { ...c, paired: c.paired.filter((p) => p.chat_id !== chatId) }
          }
        ),
      onSuccess
    }),
    test: useMutation({
      mutationFn: ({ id, chatId }: { id: string; chatId?: string }) => previewOr<{ ok: boolean; error?: string }>(() => api.channels.test(id, chatId), () => ({ ok: true }))
    })
  }
}
