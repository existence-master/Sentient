import { QueryClientProvider } from '@tanstack/react-query'
import { MotionConfig } from 'motion/react'
import { useEffect } from 'react'
import { HashRouter, Navigate, Outlet, Route, Routes, useLocation, useNavigate } from 'react-router'
import { Toaster } from 'sonner'
import { AppShell } from '@/components/shell/AppShell'
import { EngineError, Splash } from '@/components/shell/EngineScreens'
import { TooltipProvider } from '@/components/ui'
import { ChannelsPage } from '@/features/channels/ChannelsPage'
import { ChatPage } from '@/features/chat/ChatPage'
import { holdScreenShare } from '@/features/chat/screenShare'
import { DevicesPage } from '@/features/devices/DevicesPage'
import { IntegrationsPage } from '@/features/integrations/IntegrationsPage'
import { MemoryPage } from '@/features/memory/MemoryPage'
import { NotificationsPage } from '@/features/notifications/NotificationsPage'
import { OnboardingPage } from '@/features/onboarding/OnboardingPage'
import { SettingsPage } from '@/features/settings/SettingsPage'
import { SkillsPage } from '@/features/skills/SkillsPage'
import { TasksPage } from '@/features/tasks/TasksPage'
import { AboutPage } from '@/features/usermodel/AboutPage'
import { VoiceMode } from '@/features/voice/VoiceMode'
import { installWakeListener, setAlwaysListening, useWakeStore } from '@/features/voice/wake'
import { useBootstrap } from '@/hooks/core'
import { useHotkey } from '@/hooks/useHotkey'
import { useSmokeReady } from '@/hooks/useSmokeReady'
import { getBridge } from '@/lib/bridge'
import { installDemoData } from '@/lib/demo'
import { installDomainEvents } from '@/lib/events'
import { queryClient } from '@/lib/queryClient'
import { resolveTheme } from '@/lib/theme'
import { installChatEvents } from '@/stores/chat'
import { useConnection } from '@/stores/connection'
import { useNotificationStore } from '@/stores/notifications'
import { useUI } from '@/stores/ui'

export function App() {
  useEffect(() => {
    const offs = [
      useConnection.getState().init(),
      installChatEvents(),
      installDomainEvents(queryClient),
      installDemoData(queryClient),
      installWakeListener()
    ]
    return () => offs.forEach((off) => off())
  }, [])

  useAlwaysListeningShell()

  return (
    <QueryClientProvider client={queryClient}>
      <MotionConfig reducedMotion="user" transition={{ duration: 0.18, ease: [0.2, 0.8, 0.2, 1] }}>
        <TooltipProvider delayDuration={450} skipDelayDuration={200}>
          <HashRouter>
            <GlobalCommands />
            <Routes>
              <Route element={<EngineGate />}>
                <Route element={<BootstrapGate />}>
                  <Route path="/onboarding/:step?" element={<OnboardingPage />} />
                  <Route path="/voice" element={<VoiceMode />} />
                  <Route element={<AppShell />}>
                    <Route index element={<Navigate to="/chat" replace />} />
                    <Route path="/chat" element={<ChatPage />} />
                    <Route path="/chat/:sessionId" element={<ChatPage />} />
                    <Route path="/tasks" element={<TasksPage />} />
                    <Route path="/tasks/:taskId" element={<TasksPage />} />
                    <Route path="/memory" element={<MemoryPage />} />
                    <Route path="/integrations" element={<IntegrationsPage />} />
                    <Route path="/skills" element={<SkillsPage />} />
                    <Route path="/devices/:tab?" element={<DevicesPage />} />
                    <Route path="/channels" element={<ChannelsPage />} />
                    <Route path="/settings/:section?" element={<SettingsPage />} />
                    <Route path="/notifications" element={<OpenNotifications />} />
                    <Route path="/about" element={<AboutPage />} />
                    <Route path="/about/:tab" element={<AboutPage />} />
                    <Route path="*" element={<Navigate to="/chat" replace />} />
                  </Route>
                </Route>
              </Route>
            </Routes>
          </HashRouter>
          <ThemedToaster />
        </TooltipProvider>
      </MotionConfig>
    </QueryClientProvider>
  )
}

/**
 * Keeps the wake-word switch (features/voice/wake.ts) and the shell in sync: the shell keeps the
 * renderer awake while hidden and shows a tray checkbox; hearing the wake word brings the window forward.
 */
function useAlwaysListeningShell() {
  useEffect(() => {
    const bridge = getBridge()
    if (!bridge.isDesktop) return
    const apply = (on: boolean) => {
      if (on !== useWakeStore.getState().enabled) setAlwaysListening(on)
    }
    void bridge.getAlwaysListening().then((on) => {
      if (on === null) void bridge.setAlwaysListening(useWakeStore.getState().enabled)
      else apply(on)
    })
    const offShell = bridge.onAlwaysListeningChange(apply)
    const offStore = useWakeStore.subscribe((s, prev) => {
      if (s.enabled !== prev.enabled) void bridge.setAlwaysListening(s.enabled)
    })
    const onHash = () => {
      if (/^#\/voice\?(.*&)?wake=1\b/.test(window.location.hash)) void bridge.notifyWake()
    }
    window.addEventListener('hashchange', onHash)
    return () => {
      offShell()
      offStore()
      window.removeEventListener('hashchange', onHash)
    }
  }, [])
}

function ThemedToaster() {
  const theme = useUI((s) => s.theme)
  return (
    <Toaster
      theme={resolveTheme(theme)}
      position="bottom-right"
      offset={16}
      gap={8}
      toastOptions={{
        classNames: {
          toast: '!bg-overlay !border-border-strong !text-fg !shadow-pop !rounded-xl !font-sans',
          description: '!text-fg-muted',
          actionButton: '!bg-accent !text-accent-fg',
          cancelButton: '!bg-active !text-fg'
        }
      }}
    />
  )
}

/** Splash until the engine is ready the first time; error screen when it gives up. */
function EngineGate() {
  const backend = useConnection((s) => s.backend)
  const everReady = useConnection((s) => s.everReady)
  const connection = useConnection((s) => s.connection)
  if (backend.state === 'failed') return <EngineError status={backend} />
  if (!everReady || !connection) return <Splash status={backend} />
  return <Outlet />
}

function BootstrapGate() {
  const { data, error, isLoading, refetch } = useBootstrap()
  const location = useLocation()
  const setTheme = useUI((s) => s.setTheme)
  const setAccent = useUI((s) => s.setAccent)

  useEffect(() => {
    if (!data) return
    const ui = useUI.getState()
    if (data.ui.theme !== ui.theme) setTheme(data.ui.theme)
    if (data.ui.accent !== ui.accent) setAccent(data.ui.accent)
    useNotificationStore.getState().setUnread(data.unread_notifications)
    void getBridge().syncPrefs({ minimizeToTray: data.ui.minimize_to_tray, proactivityEnabled: data.features.proactivity })
  }, [data, setTheme, setAccent])

  if (isLoading) return <Splash status={{ state: 'starting', message: 'Loading your assistant…' }} />
  if (error || !data) return <EngineError error={error} onRetry={() => void refetch()} />
  if (!data.assistant.onboarding_complete && !location.pathname.startsWith('/onboarding')) {
    return <Navigate to="/onboarding" replace />
  }
  return (
    <>
      <SmokeReporter />
      <Outlet />
    </>
  )
}

function SmokeReporter() {
  useSmokeReady(true)
  return null
}

function OpenNotifications() {
  return <NotificationsPage />
}

/** Global shortcuts + commands from the shell (tray, global shortcut, notification clicks). */
function GlobalCommands() {
  const navigate = useNavigate()

  useEffect(
    () =>
      getBridge().onCommand((cmd) => {
        switch (cmd.type) {
          case 'navigate':
            navigate(cmd.route)
            break
          case 'new-chat':
            navigate('/chat', { state: { fresh: Date.now() } })
            break
          case 'voice-mode':
            navigate(cmd.wake ? '/voice?wake=1' : '/voice')
            break
          case 'open-settings':
            navigate(`/settings/${cmd.section ?? ''}`)
            break
          case 'share-screen':
            // a new chat with the picture attached; the user asks and sends (nothing is sent before that)
            holdScreenShare(cmd.share)
            navigate('/chat', { state: { fresh: Date.now() } })
            break
        }
      }),
    [navigate]
  )

  useHotkey('mod+k', () => useUI.getState().setPaletteOpen(!useUI.getState().paletteOpen))
  useHotkey('mod+n', () => navigate('/chat', { state: { fresh: Date.now() } }))
  useHotkey('mod+,', () => navigate('/settings'))
  useHotkey('mod+b', () => useUI.getState().toggleSidebar())
  return null
}
