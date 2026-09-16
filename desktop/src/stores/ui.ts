/**
 * UI preferences and overlay state. Theme/accent mirror `config.ui` (the source of
 * truth) and are cached locally so the first paint uses the right colors.
 */
import { create } from 'zustand'
import { createJSONStorage, persist } from 'zustand/middleware'
import { applyTheme, normalizeAccent } from '@/lib/theme'
import type { AccentName, ThemePreference } from '@/lib/types'

interface UIState {
  sidebarCollapsed: boolean
  theme: ThemePreference
  accent: AccentName
  paletteOpen: boolean
  notificationsOpen: boolean
  toggleSidebar: () => void
  setSidebarCollapsed: (v: boolean) => void
  setTheme: (theme: ThemePreference) => void
  setAccent: (accent: string) => void
  setPaletteOpen: (open: boolean) => void
  setNotificationsOpen: (open: boolean) => void
}

export const useUI = create<UIState>()(
  persist(
    (set, get) => ({
      sidebarCollapsed: false,
      theme: 'dark',
      accent: 'sentient',
      paletteOpen: false,
      notificationsOpen: false,
      toggleSidebar: () => set({ sidebarCollapsed: !get().sidebarCollapsed }),
      setSidebarCollapsed: (sidebarCollapsed) => set({ sidebarCollapsed }),
      setTheme: (theme) => {
        applyTheme(theme, get().accent)
        set({ theme })
      },
      setAccent: (accent) => {
        const a = normalizeAccent(accent)
        applyTheme(get().theme, a)
        set({ accent: a })
      },
      setPaletteOpen: (paletteOpen) => set({ paletteOpen }),
      setNotificationsOpen: (notificationsOpen) => set({ notificationsOpen })
    }),
    {
      name: 'sentient.ui',
      storage: createJSONStorage(() => localStorage),
      partialize: (s) => ({ sidebarCollapsed: s.sidebarCollapsed, theme: s.theme, accent: s.accent }),
      onRehydrateStorage: () => (state) => {
        if (state) applyTheme(state.theme, state.accent)
      }
    }
  )
)
