"use client"
import React, { useState, useEffect, useCallback, useMemo, useRef } from "react"
import { usePathname, useRouter, useSearchParams } from "next/navigation"
import { AnimatePresence, motion } from "framer-motion"
import NotificationsOverlay from "@components/ui/NotificationsOverlay"
import { IconMenu2, IconLoader, IconX } from "@tabler/icons-react"
import Sidebar from "@components/layout/Sidebar"
import GlobalSearch from "@components/ui/GlobalSearch"
import { useGlobalShortcuts } from "@hooks/useGlobalShortcuts"
import { cn } from "@utils/cn"
import toast from "react-hot-toast"
import { useUser } from "@auth0/nextjs-auth0"
import { usePostHog } from "posthog-js/react"
import {
	useUIStore,
	useUserStore,
	useNotificationStore,
	useTourStore
} from "@stores/app-stores"
import { tourSteps, chatSubSteps, taskSubSteps } from "@lib/tour-steps"
import { subscribeUser } from "@app/actions"
import { useMutation } from "@tanstack/react-query"

import { Drawer } from "@components/ui/drawer"
const isMobile = () => typeof window !== "undefined" && window.innerWidth < 768

// --- Guided Tour Component ---
const GuidedTour = () => {
	const {
		isActive: isTourActive, // Renaming for clarity and to avoid conflict
		step,
		subStep,
		phase,
		isHighlightPaused,
		nextStep,
		skipTour,
		finishTour,
		handleCustomAction,
		chatActionsRef
	} = useTourStore()
	const { openMobileNav, closeMobileNav } = useUIStore()
	const router = useRouter()
	const pathname = usePathname()
	const searchParams = useSearchParams()
	const [targetRect, setTargetRect] = useState(null)
	const [targetElement, setTargetElement] = useState(null)
	const [tooltipContent, setTooltipContent] = useState({
		title: "",
		body: "",
		instruction: ""
	})
	const [modalContent, setModalContent] = useState({
		title: "",
		body: "",
		buttons: []
	})
	const [isWaiting, setIsWaiting] = useState(false)

	const { positionStyle, isPositionedBelow, hideArrow } = useMemo(() => {
		if (!targetRect)
			return {
				positionStyle: {},
				isPositionedBelow: true,
				hideArrow: false
			}

		const { innerWidth, innerHeight } = window
		const margin = 10
		const tooltipMaxWidth = 320
		const isFullScreenModal =
			isMobile() && targetRect.height > innerHeight * 0.8

		if (isFullScreenModal) {
			return {
				positionStyle: {
					bottom: "120px",
					left: "50%",
					transform: "translateX(-50%)",
					width: `calc(100vw - ${margin * 2}px)`,
					maxWidth: tooltipMaxWidth
				},
				isPositionedBelow: false,
				hideArrow: true
			}
		}

		const tooltipHeightEstimate = 180
		const spaceBelow = innerHeight - targetRect.bottom
		const spaceAbove = targetRect.top
		const positionBelow =
			spaceBelow > tooltipHeightEstimate || spaceAbove < spaceBelow
		const top = positionBelow ? targetRect.bottom + margin : "auto"
		const bottom = positionBelow
			? "auto"
			: innerHeight - targetRect.top + margin
		const availableWidth = innerWidth - margin * 2
		const finalTooltipWidth = Math.min(tooltipMaxWidth, availableWidth)
		let left =
			targetRect.left + targetRect.width / 2 - finalTooltipWidth / 2
		if (left < margin) left = margin
		if (left + finalTooltipWidth > innerWidth - margin) {
			left = innerWidth - finalTooltipWidth - margin
		}

		return {
			positionStyle: { top, bottom, left, width: finalTooltipWidth },
			isPositionedBelow: positionBelow,
			hideArrow: false
		}
	}, [targetRect])

	useEffect(() => {
		if (!isTourActive || isHighlightPaused) {
			setTargetRect(null)
			setTargetElement(null)
			return
		}

		let currentStepConfig = { ...tourSteps[step] }
		if (!currentStepConfig) {
			finishTour()
			return
		}

		if (step === 5) {
			const subStepConfig = taskSubSteps[subStep]
			if (subStepConfig) {
				if (isMobile()) {
					const isPanelPhase = phase === "panel"
					currentStepConfig.selector = isPanelPhase
						? "[data-tour-id='task-details-panel']"
						: "[data-tour-id='demo-task-card']"
					currentStepConfig.title = isPanelPhase
						? "Task Details"
						: subStepConfig.title
					currentStepConfig.body = isPanelPhase
						? subStepConfig.body_panel
						: subStepConfig.body_list
					currentStepConfig.custom_button = isPanelPhase
						? "Continue Simulation"
						: "View Details"
				} else {
					currentStepConfig.selector =
						"[data-tour-id='demo-task-card']"
					currentStepConfig.title = subStepConfig.title
					currentStepConfig.body = subStepConfig.body_list
					currentStepConfig.custom_button = subStepConfig.button
				}
			}
		}

		if (isMobile()) {
			const isSidebarStep =
				currentStepConfig.selector?.includes("sidebar-")
			isSidebarStep ? openMobileNav() : closeMobileNav()
		}

		const { action } = currentStepConfig
		if (action?.type === "navigate" && pathname === action.targetPath) {
			nextStep()
			return
		}
		if (currentStepConfig.path && pathname !== currentStepConfig.path) {
			router.push(currentStepConfig.path)
			return
		}

		const waitCondition = currentStepConfig.wait_for
		if (waitCondition) {
			setIsWaiting(true)
			if (waitCondition === "integration_success") {
				const successParam = searchParams.get("integration_success")
				if (successParam) {
					router.replace(pathname, { scroll: false })
					nextStep()
					return
				}
			}
		} else {
			setIsWaiting(false)
		}

		if (currentStepConfig.type === "modal") {
			setModalContent(currentStepConfig)
			setTargetElement(null)
			setTargetRect(null)
		} else if (
			currentStepConfig.type === "tooltip" &&
			currentStepConfig.selector
		) {
			setTooltipContent(currentStepConfig)
			setTargetElement(null)
			setTargetRect(null)

			const findAndSetTarget = (retries = 5, delay = 200) => {
				if (!useTourStore.getState().isActive) return
				const target = document.querySelector(
					currentStepConfig.selector
				)
				if (target) {
					target.scrollIntoView({
						behavior: "auto",
						block: "center",
						inline: "center"
					})
					requestAnimationFrame(() => {
						setTargetElement(target)
						setTargetRect(target.getBoundingClientRect())
					})
				} else if (retries > 0) {
					setTimeout(
						() => findAndSetTarget(retries - 1, delay),
						delay
					)
				}
			}
			setTimeout(findAndSetTarget, currentStepConfig.initialDelay || 100)
		}

		if (step === 1 && !isHighlightPaused) {
			const subStepConfig = chatSubSteps[subStep]
			if (subStepConfig) {
				const attemptToSetInput = (retries = 5, delay = 100) => {
					if (chatActionsRef.current?.setInput) {
						chatActionsRef.current.setInput(subStepConfig.prefill)
					} else if (retries > 0) {
						setTimeout(
							() => attemptToSetInput(retries - 1, delay),
							delay
						)
					}
				}
				attemptToSetInput()
				setTooltipContent((prev) => ({
					...prev,
					instruction:
						subStepConfig.instruction ||
						"Now, let's create a workflow. Click send."
				}))
			}
		} else if (step === 4 && !isHighlightPaused) {
			setTooltipContent({
				...currentStepConfig,
				title: "Step 4/7: Describe Your Goal",
				body: "I've pre-filled the composer with a goal. I will analyze this and create a plan to achieve it.",
				instruction: "Click 'Create Task' to see me get to work.",
				selector: "[data-tour-id='task-composer-create-button']",
				isWaitingForAction: true
			})
		}
	}, [
		isTourActive,
		step,
		subStep,
		phase,
		isHighlightPaused,
		pathname,
		router,
		finishTour,
		chatActionsRef,
		searchParams,
		nextStep,
		openMobileNav,
		closeMobileNav
	])

	useEffect(() => {
		if (targetElement) {
			const originalPosition = targetElement.style.position
			const originalZIndex = targetElement.style.zIndex
			targetElement.style.position = "relative"
			targetElement.style.zIndex = "1001"
			return () => {
				targetElement.style.position = originalPosition
				targetElement.style.zIndex = originalZIndex
			}
		}
	}, [targetElement])

	if (!isTourActive) return null

	const currentStepConfig = tourSteps[step]
	if (!currentStepConfig) return null

	const renderContent = () => {
		if (currentStepConfig.type === "modal") {
			return (
				<motion.div
					initial={{ opacity: 0, scale: 0.9 }}
					animate={{ opacity: 1, scale: 1 }}
					exit={{ opacity: 0, scale: 0.9 }}
					className="bg-neutral-900/90 backdrop-blur-xl p-6 rounded-2xl shadow-2xl w-full max-w-md border border-neutral-700 flex flex-col text-center"
					style={{ pointerEvents: "auto" }}
				>
					<h2 className="text-xl font-bold text-white mb-2">
						{modalContent.title}
					</h2>
					<p className="text-neutral-300 mb-6">{modalContent.body}</p>
					<div className="flex justify-center gap-3">
						{modalContent.buttons.map((button) => (
							<button
								key={button.label}
								onClick={() => {
									if (button.label === "Start Tour")
										nextStep()
									if (button.label === "Skip for now")
										skipTour()
									if (button.label === "Finish Tour")
										finishTour()
								}}
								className={cn(
									"py-2 px-5 rounded-lg text-sm font-medium",
									button.primary
										? "bg-brand-orange text-brand-black"
										: "bg-neutral-700 hover:bg-neutral-600"
								)}
							>
								{button.label}
							</button>
						))}
					</div>
				</motion.div>
			)
		}

		if (currentStepConfig.type === "tooltip" && targetRect) {
			return (
				<motion.div
					initial={{ opacity: 0, y: 10 }}
					animate={{ opacity: 1, y: 0 }}
					style={{
						position: "fixed",
						zIndex: 1002,
						...positionStyle
					}}
					className="bg-neutral-900/90 backdrop-blur-xl p-4 rounded-lg shadow-2xl border border-neutral-700 pointer-events-auto"
				>
					<div
						className={cn(
							"absolute w-3 h-3 bg-neutral-900/90 transform rotate-45",
							isPositionedBelow ? "-top-1.5" : "-bottom-1.5",
							hideArrow && "hidden"
						)}
						style={{
							left: hideArrow
								? 0
								: targetRect.left -
									positionStyle.left +
									targetRect.width / 2 -
									6
						}}
					/>
					<h3 className="font-bold text-white mb-1">
						{tooltipContent.title}
					</h3>
					<p className="text-sm text-neutral-300 mb-3">
						{tooltipContent.body}
					</p>
					<p
						className="text-sm text-neutral-400 italic mb-4"
						dangerouslySetInnerHTML={{
							__html: tooltipContent.instruction
						}}
					/>
					<div className="flex justify-end gap-2">
						<button
							onClick={skipTour}
							className="text-xs text-neutral-400 hover:text-white"
						>
							Skip
						</button>
						{!isWaiting &&
							!tooltipContent.custom_button &&
							!currentStepConfig.isWaitingForAction &&
							!currentStepConfig.action && (
								<button
									onClick={nextStep}
									className="py-1 px-3 text-sm rounded-md bg-brand-orange text-brand-black font-semibold"
								>
									Next
								</button>
							)}
						{tooltipContent.custom_button && (
							<button
								onClick={() => handleCustomAction()}
								className="py-1 px-3 text-sm rounded-md bg-brand-orange text-brand-black font-semibold"
							>
								{tooltipContent.custom_button}
							</button>
						)}
					</div>
				</motion.div>
			)
		}
		return null
	}

	return (
		<div className="fixed inset-0 z-[1000] pointer-events-none">
			<AnimatePresence>
				<motion.div
					initial={{ opacity: 0 }}
					animate={{ opacity: 1 }}
					exit={{ opacity: 0 }}
					className={cn(
						"absolute inset-0 pointer-events-none",
						(currentStepConfig.type === "modal" ||
							!isHighlightPaused) &&
							"bg-black/50",
						currentStepConfig.type === "modal" &&
							"flex items-center justify-center"
					)}
				>
					{renderContent()}
				</motion.div>
			</AnimatePresence>
			{targetRect && (
				<div
					className="absolute rounded-lg border-2 border-brand-orange border-dashed shadow-2xl pointer-events-none"
					style={{
						left: targetRect.left - 4,
						top: targetRect.top - 4,
						width: targetRect.width + 8,
						height: targetRect.height + 8,
						transition: "all 0.3s ease-in-out"
					}}
				/>
			)}
		</div>
	)
}

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
	const { user, error: authError, isLoading: isAuthLoading } = useUser()
	const {
		isSearchOpen,
		isMobileNavOpen,
		openSearch,
		closeSearch,
		openMobileNav,
		closeMobileNav
	} = useUIStore()
	const {
		user: storedUser,
		isLoading: isUserLoading,
		error: userError,
		fetchUserData
	} = useUserStore()
	const {
		unreadCount,
		isNotificationsOpen,
		notifRefreshKey,
		incrementUnreadCount,
		refreshNotifications,
		openNotifications,
		closeNotifications
	} = useNotificationStore()
	const { isActive: isTourActive } = useTourStore()
	const wsRef = useRef(null)
	const searchParams = useSearchParams()
	const posthog = usePostHog()
	const pathname = usePathname()
	const router = useRouter()

	const [isLoading, setIsLoading] = useState(true)
	const [isAllowed, setIsAllowed] = useState(false)

	const showNav = !["/", "/onboarding"].includes(pathname)

	useEffect(() => {
		if (user && posthog) {
			posthog.identify(user.sub, {
				name: user.name,
				email: user.email
			})

			const fetchAndSetUserGroups = async () => {
				try {
					const res = await fetch("/api/user/properties")
					if (!res.ok) {
						console.error(
							"Failed to fetch user properties for PostHog, status:",
							res.status
						)
						return
					}
					const properties = await res.json()
					posthog.group("plan", properties.plan_type, {
						name:
							properties.plan_type.charAt(0).toUpperCase() +
							properties.plan_type.slice(1)
					})
					posthog.group(
						"insider_status",
						properties.is_insider ? "insider" : "not_insider",
						{
							name: properties.is_insider
								? "Insiders"
								: "Not Insiders"
						}
					)
					posthog.setPersonProperties({
						plan_type: properties.plan_type,
						is_insider: properties.is_insider
					})
				} catch (error) {
					console.error(
						"Error fetching/setting user properties for PostHog:",
						error
					)
				}
			}
			fetchAndSetUserGroups()
		}
	}, [user, posthog])

	useEffect(() => {
		const paymentStatus = searchParams.get("payment_status")
		const needsRefresh = searchParams.get("refresh_session")

		if (paymentStatus === "success" || needsRefresh === "true") {
			window.history.replaceState(null, "", pathname)
			toast.loading("Session updated. Redirecting...", { duration: 5000 })
			const logoutUrl = new URL("/auth/logout", window.location.origin)
			logoutUrl.searchParams.set(
				"returnTo",
				process.env.NEXT_PUBLIC_APP_BASE_URL
			)
			window.location.assign(logoutUrl.toString())
		}
	}, [searchParams, router, pathname, posthog])

	useEffect(() => {
		if (user && !isAuthLoading && showNav && !storedUser) {
			fetchUserData()
		}

		if (!showNav) {
			setIsLoading(false)
			setIsAllowed(true)
			return
		}

		if (isAuthLoading || isUserLoading) {
			setIsLoading(true)
			return
		}

		if (authError) {
			toast.error("Session error. Redirecting to login.", {
				id: "auth-error"
			})
			router.push("/auth/login")
			return
		}

		if (!user) {
			router.push("/auth/login")
			return
		}

		if (userError) {
			toast.error(userError)
			router.push("/")
			return
		}

		if (storedUser && !storedUser.onboardingComplete) {
			toast.error("Please complete onboarding first.", {
				id: "onboarding-check"
			})
			router.push("/onboarding")
		} else {
			setIsAllowed(true)
			setIsLoading(false)
		}
	}, [
		showNav,
		user,
		authError,
		isAuthLoading,
		storedUser,
		isUserLoading,
		userError,
		fetchUserData,
		router
	])

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
						incrementUnreadCount()
						refreshNotifications()
						toast(
							(t) => (
								<div className="flex items-center gap-3">
									<span className="flex-1">
										{data.notification.message}
									</span>
									<button
										onClick={() => {
											openNotifications()
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
	}, [
		user?.sub,
		incrementUnreadCount,
		refreshNotifications,
		openNotifications
	])

	useEffect(() => {
		if ("serviceWorker" in navigator) {
			window.addEventListener("load", function () {
				navigator.serviceWorker.register("/sw.js").then(
					function (registration) {},
					function (err) {
						console.error(
							"ServiceWorker registration failed: ",
							err
						)
					}
				)
			})
		}
	}, [])

	const subscribeMutation = useMutation({
		mutationFn: subscribeUser,
		onSuccess: () => {
			toast.success("Subscribed to push notifications!")
		},
		onError: (error) => {
			console.error("Error during push notification subscription:", error)
			toast.error(
				`Failed to subscribe: ${error.message || "Unknown error"}`
			)
		}
	})

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
		if (!process.env.NEXT_PUBLIC_VAPID_PUBLIC_KEY) {
			console.warn(
				"VAPID public key not configured. Skipping push subscription."
			)
			return
		}
		if (!("serviceWorker" in navigator) || !("PushManager" in window)) {
			toast.error("Push notifications are not supported by your browser.")
			return
		}
		try {
			const registration = await navigator.serviceWorker.ready
			let subscription = await registration.pushManager.getSubscription()
			if (subscription === null) {
				const permission = await window.Notification.requestPermission()
				if (permission !== "granted") {
					toast.error(
						"Permission for push notifications was not granted."
					)
					return
				}
				const applicationServerKey = urlBase64ToUint8Array(
					process.env.NEXT_PUBLIC_VAPID_PUBLIC_KEY
				)
				subscription = await registration.pushManager.subscribe({
					userVisibleOnly: true,
					applicationServerKey
				})
				const serializedSub = JSON.parse(JSON.stringify(subscription))
				subscribeMutation.mutate(serializedSub)
			}
		} catch (error) {
			subscribeMutation.onError(error)
		}
	}, [subscribeMutation])

	useEffect(() => {
		if (showNav && user?.sub) subscribeToPushNotifications()
	}, [showNav, user, subscribeToPushNotifications])

	useGlobalShortcuts(openNotifications, openSearch)

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

	return (
		<>
			{showNav && (
				<>
					<Sidebar
						onNotificationsOpen={openNotifications}
						onSearchOpen={openSearch}
						unreadCount={unreadCount}
						isMobileOpen={isMobileNavOpen}
						onMobileClose={closeMobileNav}
						user={user}
						isTourActive={isTourActive}
					/>
					<button
						onClick={openMobileNav}
						className="md:hidden fixed top-4 left-4 z-30 p-2 rounded-full bg-neutral-800/50 backdrop-blur-sm text-white"
					>
						<IconMenu2 size={20} />
					</button>
				</>
			)}
			<div
				className={cn(
					"flex-1 transition-[padding-left] duration-300 ease-in-out",
					showNav && "md:pl-[260px]"
				)}
			>
				{children}
			</div>
			<Drawer isOpen={isNotificationsOpen} onClose={closeNotifications}>
				<NotificationsOverlay
					notifRefreshKey={notifRefreshKey}
					onClose={closeNotifications}
				/>
			</Drawer>
			<AnimatePresence>
				{isSearchOpen && <GlobalSearch onClose={closeSearch} />}
			</AnimatePresence>
			<GuidedTour />
		</>
	)
}
