import { create } from "zustand"
import { devtools, persist } from "zustand/middleware"
import { taskSubSteps } from "@lib/tour-steps"

// --- UI Store ---
export const useUIStore = create(
	devtools(
		(set) => ({
			isSearchOpen: false,
			isMobileNavOpen: false,
			openSearch: () => set({ isSearchOpen: true }),
			closeSearch: () => set({ isSearchOpen: false }),
			openMobileNav: () => set({ isMobileNavOpen: true }),
			closeMobileNav: () => set({ isMobileNavOpen: false })
		}),
		{ name: "UIStore" }
	)
)

// --- User Store ---
const userStoreInitialState = {
	user: null,
	onboardingComplete: false,
	isLoading: true,
	error: null
}
export const useUserStore = create(
	devtools(
		(set) => ({
			...userStoreInitialState,
			fetchUserData: async () => {
				set({ isLoading: true, error: null })
				try {
					const res = await fetch("/api/user/data", {
						method: "POST"
					})
					if (!res.ok)
						throw new Error("Could not verify user status.")
					const data = await res.json()

					const profileRes = await fetch("/api/user/profile")
					if (!profileRes.ok)
						throw new Error("Could not fetch user profile.")
					const profileData = await profileRes.json()

					const userData = { ...data.data, ...profileData }

					set({
						user: userData,
						onboardingComplete: userData.onboardingComplete,
						isLoading: false
					})
					return userData
				} catch (error) {
					console.error("Error fetching user data:", error)
					set({ error: error.message, isLoading: false })
					return null
				}
			}
		}),
		{ name: "UserStore" }
	)
)
// Add getInitialState to the store for easy resetting in tests
useUserStore.getInitialState = () => userStoreInitialState

// --- Notification Store ---
export const useNotificationStore = create(
	devtools(
		(set) => ({
			unreadCount: 0,
			isNotificationsOpen: false,
			notifRefreshKey: 0,
			incrementUnreadCount: () =>
				set((state) => ({ unreadCount: state.unreadCount + 1 })),
			refreshNotifications: () =>
				set((state) => ({
					notifRefreshKey: state.notifRefreshKey + 1
				})),
			openNotifications: () =>
				set({ isNotificationsOpen: true, unreadCount: 0 }),
			closeNotifications: () => set({ isNotificationsOpen: false })
		}),
		{ name: "NotificationStore" }
	)
)

// --- Tour Store ---
const tourInitialState = {
	isActive: false,
	step: 0,
	subStep: 0,
	phase: null,
	isWaitingForAction: false,
	isHighlightPaused: false
}
export const useTourStore = create(
	devtools(
		(set, get) => ({
			...tourInitialState,
			chatActionsRef: { current: null },
			startTour: () => set({ ...tourInitialState, isActive: true }),
			skipTour: () => set(tourInitialState),
			finishTour: () => set(tourInitialState),
			nextStep: () =>
				set((state) => {
					const newStep = state.step + 1
					return {
						...state,
						step: newStep,
						subStep: 0,
						phase: newStep === 5 ? "list" : null,
						isWaitingForAction: false,
						isHighlightPaused: false
					}
				}),
			nextSubStep: () => set((state) => ({ subStep: state.subStep + 1 })),
			setHighlightPaused: (isPaused) =>
				set({ isHighlightPaused: isPaused }),
			setPhase: (phase) => set({ phase }),
			handleCustomAction: () => {
				set((state) => {
					get().setHighlightPaused(true)
					if (state.step === 5) {
						const isMobile = () =>
							typeof window !== "undefined" &&
							window.innerWidth < 768
						if (isMobile()) {
							if (state.phase === "list")
								return { ...state, phase: "panel" }
							const isLastSubStep =
								state.subStep >= taskSubSteps.length - 1
							if (isLastSubStep) {
								const newStep = state.step + 1
								return {
									...state,
									step: newStep,
									subStep: 0,
									phase: null
								}
							}
							return {
								...state,
								subStep: state.subStep + 1,
								phase: "list"
							}
						} else {
							const isLastSubStep =
								state.subStep >= taskSubSteps.length - 1
							if (isLastSubStep) {
								const newStep = state.step + 1
								return {
									...state,
									step: newStep,
									subStep: 0,
									phase: null
								}
							}
							return { ...state, subStep: state.subStep + 1 }
						}
					}
					const newStep = state.step + 1
					return { ...state, step: newStep }
				})
				setTimeout(() => get().setHighlightPaused(false), 500)
			},
			setChatActionsRef: (ref) => set({ chatActionsRef: ref })
		}),
		{ name: "TourStore" }
	)
)

// --- Chat Store ---
export const useChatStore = create(
	devtools(
		(set) => ({
			isVoiceMode: false,
			connectionStatus: "disconnected",
			isMuted: false,
			voiceStatusText: "Click to start call",
			audioLevel: 0,
			setVoiceMode: (isVoiceMode) => set({ isVoiceMode }),
			setConnectionStatus: (status) => set({ connectionStatus: status }),
			setIsMuted: (muted) => set({ isMuted: muted }),
			setVoiceStatusText: (text) => set({ voiceStatusText: text }),
			setAudioLevel: (level) => set({ audioLevel: level }),
			endVoiceCall: () =>
				set({
					connectionStatus: "disconnected",
					voiceStatusText: "Click to start call",
					isMuted: false,
					isVoiceMode: false
				})
		}),
		{ name: "ChatStore" }
	)
)

// --- Task Store ---
export const useTaskStore = create(
	devtools(
		(set) => ({
			view: "tasks",
			searchQuery: "",
			isComposerOpen: false,
			composerInitialData: null,
			setView: (view) => set({ view }),
			setSearchQuery: (query) => set({ searchQuery: query }),
			openComposer: (initialData = null) =>
				set({ isComposerOpen: true, composerInitialData: initialData }),
			closeComposer: () =>
				set({ isComposerOpen: false, composerInitialData: null })
		}),
		{ name: "TaskStore" }
	)
)

// --- Memory Store ---
export const useMemoryStore = create(
	devtools(
		persist(
			(set) => ({
				view: "graph",
				activeTopic: "All",
				selectedMemory: null,
				isInfoPanelOpen: false,
				isCreateModalOpen: false,
				setView: (view) => set({ view }),
				setActiveTopic: (topic) => set({ activeTopic: topic }),
				setSelectedMemory: (memory) => set({ selectedMemory: memory }),
				openInfoPanel: () => set({ isInfoPanelOpen: true }),
				closeInfoPanel: () => set({ isInfoPanelOpen: false }),
				openCreateModal: () => set({ isCreateModalOpen: true }),
				closeCreateModal: () => set({ isCreateModalOpen: false })
			}),
			{
				name: "memory-store-storage", // name of the item in the storage (must be unique)
				partialize: (state) => ({
					view: state.view,
					activeTopic: state.activeTopic
				}) // only persist these fields
			}
		),
		{ name: "MemoryStore" }
	)
)

// --- Integration Store (for integrations page) ---
export const useIntegrationStore = create(
	devtools(
		(set) => ({
			searchQuery: "",
			activeCategory: "Core", // Set a default category
			privacyModalService: null,
			disconnectingIntegration: null,
			setSearchQuery: (query) => set({ searchQuery: query }),
			setActiveCategory: (category) => set({ activeCategory: category }),
			openPrivacyModal: (service) => set({ privacyModalService: service }),
			closePrivacyModal: () => set({ privacyModalService: null }),
			setDisconnectingIntegration: (integration) =>
				set({ disconnectingIntegration: integration }),

			// Deprecated modal state, kept for backward compatibility during refactor
			isIntegrationModalOpen: false,
			openIntegrationModal: () => set({ isIntegrationModalOpen: true }),
			closeIntegrationModal: () => set({ isIntegrationModalOpen: false })
		}),
		{ name: "IntegrationStore" }
	)
)
