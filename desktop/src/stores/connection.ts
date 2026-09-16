/**
 * Engine lifecycle + live socket state.
 *
 *   const backend = useConnection((s) => s.backend)      // BackendStatus from the shell
 *   const socket = useConnection((s) => s.socket)        // 'open' | 'reconnecting' ...
 */
import { create } from 'zustand'
import { setConnection } from '@/lib/api'
import { getBridge } from '@/lib/bridge'
import { live, type SocketState } from '@/lib/ws'
import type { BackendStatus, Connection } from '@/types/bridge'

interface ConnectionState {
  backend: BackendStatus
  connection: Connection | null
  /** The engine has been ready at least once this session (splash is only shown before that). */
  everReady: boolean
  socket: SocketState
  engineVersion?: string
  assistantName?: string
  init: () => () => void
  restart: () => Promise<void>
}

export const useConnection = create<ConnectionState>((set, get) => ({
  backend: { state: 'starting' },
  connection: null,
  everReady: false,
  socket: 'idle',

  init: () => {
    const bridge = getBridge()

    const onStatus = async (status: BackendStatus) => {
      set({ backend: status })
      if (status.state === 'ready') {
        const conn = await bridge.getConnection()
        setConnection(conn)
        live.connect(conn)
        set({ connection: conn, everReady: true })
      }
    }

    const offStatus = bridge.onBackendStatus((s) => void onStatus(s))
    void bridge.getBackendStatus().then((s) => {
      // Don't clobber a newer pushed status with the initial snapshot.
      if (!get().everReady || s.state === 'ready') void onStatus(s)
    })

    const offSocket = live.onState((socket) => set({ socket }))
    const offHello = live.onChat((e) => {
      if (e.type === 'hello') set({ engineVersion: e.version, assistantName: e.assistant })
    })

    return () => {
      offStatus()
      offSocket()
      offHello()
    }
  },

  restart: async () => {
    set({ backend: { state: 'starting', message: 'Restarting…' } })
    await getBridge().restartBackend()
  }
}))
