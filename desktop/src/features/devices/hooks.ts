/** React Query hooks for §13 devices. Live updates arrive via `node.updated` / `node.deleted` (lib/events.ts). */
import { useMutation, useQuery, useQueryClient, type QueryClient } from '@tanstack/react-query'
import { api, isNotImplemented } from '@/lib/api'
import type { DeviceInvokeResult, DeviceNode } from '@/lib/types'
import { previewLan, previewMode, previewNodes, previewOr, previewPairing, previewPhotoBase64 } from './preview'

export const deviceKeys = {
  all: ['nodes'] as const,
  lan: ['nodes', 'lan'] as const
}

export function upsertDevice(qc: QueryClient, node: DeviceNode) {
  qc.setQueryData<DeviceNode[]>(deviceKeys.all, (old) => {
    if (!old) return old
    const i = old.findIndex((n) => n.node_id === node.node_id)
    if (i === -1) return [...old, node]
    const next = old.slice()
    next[i] = { ...old[i], ...node }
    return next
  })
}

export function removeDevice(qc: QueryClient, nodeId: string) {
  qc.setQueryData<DeviceNode[]>(deviceKeys.all, (old) => old?.filter((n) => n.node_id !== nodeId))
}

export function useDevices() {
  return useQuery({ queryKey: deviceKeys.all, queryFn: () => previewOr(api.nodes.list, previewNodes) })
}

export function useLanInfo(enabled = true) {
  return useQuery({ queryKey: deviceKeys.lan, queryFn: () => previewOr(api.nodes.lan, previewLan), enabled })
}

export function useDeviceActions() {
  const qc = useQueryClient()
  return {
    rename: useMutation({
      mutationFn: ({ id, name }: { id: string; name: string }) =>
        previewOr(
          () => api.nodes.rename(id, name),
          () => ({ ...(qc.getQueryData<DeviceNode[]>(deviceKeys.all)?.find((n) => n.node_id === id) as DeviceNode), name })
        ),
      onSuccess: (node) => upsertDevice(qc, node)
    }),
    remove: useMutation({
      mutationFn: (id: string) => previewOr(() => api.nodes.remove(id), () => ({ ok: true })),
      onSuccess: (_r, id) => removeDevice(qc, id)
    }),
    invoke: useMutation({
      mutationFn: ({ id, capability, params }: { id: string; capability: string; params?: Record<string, unknown> }) =>
        previewOr<DeviceInvokeResult>(
          () => api.nodes.invoke(id, capability, params),
          () =>
            capability === 'camera.photo' || capability === 'screen.capture'
              ? { ok: true, data: { mime: 'image/svg+xml', base64: previewPhotoBase64 } }
              : capability === 'location.get'
                ? { ok: true, data: { lat: 18.5362, lon: 73.8939, accuracy_m: 18, label: 'Koregaon Park, Pune' } }
                : { ok: true }
        )
    }),
    pairing: useMutation({ mutationFn: () => previewOr(api.nodes.pairing, previewPairing) })
  }
}

/** True when the devices endpoints are not in this engine yet (and preview mode is off). */
export function devicesUnavailable(error: unknown): boolean {
  return isNotImplemented(error) && !previewMode()
}
