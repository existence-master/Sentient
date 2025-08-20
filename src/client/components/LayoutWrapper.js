// src/client/components/LayoutWrapper.js
"use client"
import React, {
	useState,
	useEffect,
	useCallback,
	useRef,
	createContext
} from "react"
// Import useSearchParams
import { usePathname, useRouter, useSearchParams } from "next/navigation"
import { AnimatePresence } from "framer-motion"
import NotificationsOverlay from "@components/NotificationsOverlay"
import { IconMenu2, IconLoader, IconX } from "@tabler/icons-react"
import Sidebar from "@components/Sidebar"
import CommandPalette from "./CommandPallete"
import GlobalSearch from "./GlobalSearch"
import { useGlobalShortcuts } from "@hooks/useGlobalShortcuts"
import { cn } from "@utils/cn"
import toast from "react-hot-toast"
import { useUser } from "@auth0/nextjs-auth0"
import { usePostHog } from "posthog-js/react"

// ... (keep the rest of your imports and context creation)
export const PlanContext = createContext({
	plan: "free",
	isPro: false,
	isLoading: true
})
import { subscribeUser } from "@app/actions"

// ... (keep your urlBase64ToUint8Array function)
function urlBase64ToUint8Array(base64String) {
	const padding = "=".repeat((4 - (base64String.length % 4)) % 4)
	const base64 = (base64String + padding)
		.replace(/-/g, "+")
		.replace(/_/g, "/")
	const rawData = atob(base64)
	const outputArray = new Uint8Array(rawData.length)
	for (let i = 0; i < rawData.length; ++i) {
		outputArray[i] = rawData.charCodeAt(i)
	}
	return outputArray
}

export default function LayoutWrapper({ children }) {
	// ... (keep all your existing state declarations)
	const [isNotificationsOpen, setNotificationsOpen] = useState(false)
	const [isSearchOpen, setSearchOpen] = useState(false)
	const [isCommandPaletteOpen, setCommandPaletteOpen] = useState(false)
	const [isSidebarCollapsed, setSidebarCollapsed] = useState(true)
	const [isMobileNavOpen, setMobileNavOpen] = useState(false)
	const [unreadCount, setUnreadCount] = useState(0)
	const [notifRefreshKey, setNotifRefreshKey] = useState(0)
	const [userDetails, setUserDetails] = useState(null)
	const wsRef = useRef(null)
	const pathname = usePathname()
	const router = useRouter()
	const searchParams = useSearchParams() // Hook to read URL query parameters
	const posthog = usePostHog()

	const { user, error: authError, isLoading: isAuthLoading } = useUser()

	const [isLoading, setIsLoading] = useState(true)
	const [isAllowed, setIsAllowed] = useState(false)

	const showNav = !["/", "/onboarding"].includes(pathname)

	useEffect(() => {
		if (user && posthog) {
			posthog.identify(user.sub, {
				name: user.name,
				email: user.email
			})
		}
	}, [user, posthog])

	useEffect(() => {
		const paymentStatus = searchParams.get("payment_status")
		const needsRefresh = searchParams.get("refresh_session")

		if (paymentStatus === "success" && posthog) {
			posthog.capture("plan_upgraded", {
				plan_name: "pro"
				// MRR and billing_cycle are not available on the client
			})
		}
		// Check for either trigger
		if (paymentStatus === "success" || needsRefresh === "true") {
			// CRITICAL FIX: Clean the URL synchronously *before* doing anything else.
			// This prevents the refresh loop.
			window.history.replaceState(null, "", pathname)
			const toastId = toast.loading("Updating your session...", {
				duration: 4000
			})

			// const refreshSession = async () => {
			// 	const toastId = toast.loading("Updating your session...", {
			// 		duration: 4000
			// 	})
			// 	try {
			// 		// Call the API to get a new session cookie
			// 		const res = await fetch("/api/auth/refresh-session")
			// 		if (!res.ok) {
			// 			const errorData = await res.json()
			// 			throw new Error(
			// 				errorData.error || "Session refresh failed."
			// 			)
			// 		}

			// 		// Now that the cookie is updated and the URL is clean, reload the page.
			// 		// This will re-run server components and hooks with the new session data.
			// 		window.location.reload()
			// 	} catch (error) {
			// 		toast.error(
			// 			`Failed to refresh session: ${error.message}. Please log in again to see your new plan.`,
			// 			{ id: toastId }
			// 		)
			// 	}
			// }
			// refreshSession()

			const logoutUrl = new URL("/auth/logout", window.location.origin)
			logoutUrl.searchParams.set(
				"returnTo",
				process.env.NEXT_PUBLIC_APP_BASE_URL
			)
			window.location.assign(logoutUrl.toString())
		}
	}, [searchParams, router, pathname]) // Dependencies are correct

	// ... (keep the rest of your useEffects and functions exactly as they were)
	useEffect(() => {
		if (!showNav) {
			setIsLoading(false)
			setIsAllowed(true)
			return
		}

		if (isAuthLoading) return

		if (authError) {
			toast.error("Session error. Redirecting to login.")
			router.push("/auth/login")
			return
		}

		if (!user) {
			router.push("/auth/login")
			return
		}

		const checkStatus = async () => {
			setIsLoading(true)
			try {
				const res = await fetch("/api/user/data")
				if (!res.ok) throw new Error("Could not verify user status.")
				const data = await res.json()
				if (data?.data?.onboardingComplete) {
					setIsAllowed(true)
				} else {
					toast.error("Please complete onboarding first.")
					router.push("/onboarding")
				}
			} catch (error) {
				toast.error(error.message)
				router.push("/")
			} finally {
				setIsLoading(false)
			}
		}

		checkStatus()
	}, [pathname, router, showNav, user, authError, isAuthLoading])

	// ... (keep the rest of your useEffects and functions exactly as they were)
	useEffect(() => {
		if (!user?.sub) return

		const connectWebSocket = async () => {
			if (wsRef.current && wsRef.current.readyState < 2) return

			try {
				const tokenResponse = await fetch("/api/auth/token")
				if (!tokenResponse.ok) {
					setTimeout(connectWebSocket, 5000)
					return
				}
				const { accessToken } = await tokenResponse.json()
				const wsProtocol =
					window.location.protocol === "https:" ? "wss" : "ws"
				const serverUrlHttp =
					process.env.NEXT_PUBLIC_APP_SERVER_URL ||
					"http://localhost:5000"
				const serverHost = serverUrlHttp.replace(/^https?:\/\//, "")
				const wsUrl = `${wsProtocol}://${serverHost}/api/ws/notifications`

				const ws = new WebSocket(wsUrl)
				ws.isCleaningUp = false
				wsRef.current = ws

				ws.onopen = () =>
					ws.send(
						JSON.stringify({ type: "auth", token: accessToken })
					)
				ws.onmessage = (event) => {
					const data = JSON.parse(event.data)
					if (data.type === "new_notification") {
						setUnreadCount((prev) => prev + 1)
						setNotifRefreshKey((prev) => prev + 1)
						toast(
							(t) => (
								<div className="flex items-center gap-3">
									<span className="flex-1">
										{data.notification.message}
									</span>
									<button
										onClick={() => {
											handleNotificationsOpen()
											toast.dismiss(t.id)
										}}
										className="py-1 px-3 rounded-md bg-brand-orange text-black text-sm font-semibold"
									>
										View
									</button>
									<button
										onClick={() => toast.dismiss(t.id)}
										className="p-1.5 rounded-full hover:bg-neutral-700"
									>
										<IconX size={16} />
									</button>
								</div>
							),
							{
								duration: 6000
							}
						)
					} else if (data.type === "task_progress_update") {
						// Dispatch a custom event that the tasks page can listen for
						window.dispatchEvent(
							new CustomEvent("taskProgressUpdate", {
								detail: data.payload
							})
						)
					} else if (data.type === "task_list_updated") {
						window.dispatchEvent(
							new CustomEvent("tasksUpdatedFromBackend")
						)
					}
				}
				ws.onclose = () => {
					if (!ws.isCleaningUp) {
						wsRef.current = null
						setTimeout(connectWebSocket, 5000)
					}
				}
				ws.onerror = () => ws.close()
			} catch (error) {
				setTimeout(connectWebSocket, 5000)
			}
		}
		connectWebSocket()

		return () => {
			if (wsRef.current) {
				wsRef.current.isCleaningUp = true
				wsRef.current.close()
				wsRef.current = null
			}
		}
	}, [user?.sub])

	const handleNotificationsOpen = useCallback(() => {
		setNotificationsOpen(true)
		setUnreadCount(0)
	}, [])

	useGlobalShortcuts(handleNotificationsOpen, () =>
		setCommandPaletteOpen((prev) => !prev)
	)

	// PWA Update Handler

	useEffect(() => {
		// This effect runs only on the client side to register the service worker.
		// It's enabled for all environments to allow testing in development.
		if ("serviceWorker" in navigator) {
			// The 'load' event ensures that SW registration doesn't delay page rendering.
			window.addEventListener("load", function () {
				navigator.serviceWorker.register("/sw.js").then(
					function (registration) {
						console.log(
							"ServiceWorker registration successful with scope: ",
							registration.scope
						)
					},
					function (err) {
						console.log("ServiceWorker registration failed: ", err)
					}
				)
			})
		}
	}, [])

	// PWA Update Handler
	useEffect(() => {
		if (
			typeof window !== "undefined" &&
			"serviceWorker" in navigator &&
			window.workbox !== undefined
		) {
			const wb = window.workbox

			const promptNewVersionAvailable = (event) => {
				if (!event.wasWaitingBeforeRegister) {
					toast(
						(t) => (
							<div className="flex flex-col items-center gap-2 text-white">
								<span>A new version is available!</span>
								<div className="flex gap-2">
									<button
										className="py-1 px-3 rounded-md bg-green-600 hover:bg-green-500 text-white text-sm font-medium"
										onClick={() => {
											wb.addEventListener(
												"controlling",
												() => {
													window.location.reload()
												}
											)
											wb.messageSkipWaiting()
											toast.dismiss(t.id)
										}}
									>
										Refresh
									</button>
									<button
										className="py-1 px-3 rounded-md bg-neutral-600 hover:bg-neutral-500 text-white text-sm font-medium"
										onClick={() => toast.dismiss(t.id)}
									>
										Dismiss
									</button>
								</div>
							</div>
						),
						{ duration: Infinity }
					)
				}
			}

			wb.addEventListener("waiting", promptNewVersionAvailable)
			return () => {
				wb.removeEventListener("waiting", promptNewVersionAvailable)
			}
		}
	}, [])

	const subscribeToPushNotifications = useCallback(async () => {
		if (!("serviceWorker" in navigator) || !("PushManager" in window)) {
			console.log("Push notifications not supported by this browser.")
			return
		}
		if (!process.env.NEXT_PUBLIC_VAPID_PUBLIC_KEY) {
			console.warn(
				"VAPID public key not configured. Skipping push subscription."
			)
			return
		}

		try {
			const registration = await navigator.serviceWorker.ready
			let subscription = await registration.pushManager.getSubscription()

			if (subscription === null) {
				const permission = await window.Notification.requestPermission()
				if (permission !== "granted") {
					console.log("Notification permission not granted.")
					return
				}

				console.log("Permission granted. Subscribing...")
				const applicationServerKey = urlBase64ToUint8Array(
					process.env.NEXT_PUBLIC_VAPID_PUBLIC_KEY
				)

				subscription = await registration.pushManager.subscribe({
					userVisibleOnly: true,
					applicationServerKey
				})

				console.log("New subscription created:", subscription)
				const serializedSub = JSON.parse(JSON.stringify(subscription))
				await subscribeUser(serializedSub)
				toast.success("Subscribed to push notifications!")
			} else {
				console.log("User is already subscribed.")
			}
		} catch (error) {
			console.error("Error during push notification subscription:", error)
			toast.error("Failed to subscribe to push notifications.")
		}
	}, [])

	useEffect(() => {
		if (showNav && userDetails?.sub) subscribeToPushNotifications()
	}, [showNav, userDetails, subscribeToPushNotifications])

	if (isLoading || isAuthLoading) {
		return (
			<div className="flex-1 flex h-screen bg-black text-white overflow-hidden justify-center items-center">
				<IconLoader className="w-10 h-10 animate-spin text-brand-orange" />
			</div>
		)
	}

	if (!isAllowed) {
		return (
			<div className="flex-1 flex h-screen bg-black text-white overflow-hidden justify-center items-center">
				<IconLoader className="w-10 h-10 animate-spin text-brand-orange" />
			</div>
		)
	}

	console.log(user?.[`${process.env.NEXT_PUBLIC_AUTH0_NAMESPACE}/roles`])

	// ... (rest of the component is unchanged)
	return (
		<PlanContext.Provider
			value={{
				plan: (
					user?.[
						`${process.env.NEXT_PUBLIC_AUTH0_NAMESPACE}/roles`
					] || []
				).includes("Pro")
					? "pro"
					: "free",
				isPro: (
					user?.[
						`${process.env.NEXT_PUBLIC_AUTH0_NAMESPACE}/roles`
					] || []
				).includes("Pro"),
				isLoading: isAuthLoading
			}}
		>
			{showNav && (
				<>
					<Sidebar
						isCollapsed={isSidebarCollapsed}
						onToggle={() =>
							setSidebarCollapsed(!isSidebarCollapsed)
						}
						onNotificationsOpen={handleNotificationsOpen}
						onSearchOpen={() => setSearchOpen(true)}
						unreadCount={unreadCount}
						isMobileOpen={isMobileNavOpen}
						onMobileClose={() => setMobileNavOpen(false)}
						user={user}
					/>
					<button
						onClick={() => setMobileNavOpen(true)}
						className="md:hidden fixed top-4 left-4 z-30 p-2 rounded-full bg-neutral-800/50 backdrop-blur-sm text-white"
					>
						<IconMenu2 size={20} />
					</button>
				</>
			)}
			{showNav && (
				<CommandPalette
					open={isCommandPaletteOpen}
					setOpen={setCommandPaletteOpen}
				/>
			)}
			<div
				className={cn(
					"flex-1 transition-[padding-left] duration-300 ease-in-out",
					showNav &&
						(isSidebarCollapsed ? "md:pl-20" : "md:pl-[260px]")
				)}
			>
				{children}
			</div>
			<AnimatePresence>
				{isNotificationsOpen && (
					<NotificationsOverlay
						notifRefreshKey={notifRefreshKey}
						onClose={() => setNotificationsOpen(false)}
					/>
				)}
			</AnimatePresence>
			<AnimatePresence>
				{isSearchOpen && (
					<GlobalSearch onClose={() => setSearchOpen(false)} />
				)}
			</AnimatePresence>
		</PlanContext.Provider>
	)
}
