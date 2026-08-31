// src/client/components/LayoutWrapper.js
"use client"
import React, { // eslint-disable-line
	useState,
	useEffect,
	useCallback,
	useMemo,
	useRef,
	createContext,
	useContext
} from "react"
// Import useSearchParams
import { usePathname, useRouter, useSearchParams } from "next/navigation"
import { AnimatePresence } from "framer-motion"
import NotificationsOverlay from "@components/NotificationsOverlay"
import { IconMenu2, IconLoader, IconX } from "@tabler/icons-react"
import Sidebar from "@components/Sidebar"
import GlobalSearch from "./GlobalSearch"
import { useGlobalShortcuts } from "@hooks/useGlobalShortcuts"
import { cn } from "@utils/cn"
import toast from "react-hot-toast"
import { useUser } from "@auth0/nextjs-auth0"
import { usePostHog } from "posthog-js/react"
import { motion } from "framer-motion"

const isMobile = () => typeof window !== "undefined" && window.innerWidth < 768

// --- Context Creation ---
export const PlanContext = createContext({
	plan: "free",
	isPro: false,
	isLoading: true
})
export const TourContext = createContext(null)
export const useTour = () => useContext(TourContext)
import { subscribeUser } from "@app/actions"

function usePrevious(value) {
	const ref = useRef()
	useEffect(() => {
		ref.current = value
	})
	return ref.current
}

// --- Guided Tour Component ---
const GuidedTour = ({
	tourSteps,
	chatSubSteps,
	taskSubSteps,
	setMobileNavOpen
}) => {
	const tour = useTour()
	const { tourState, nextStep, skipTour, finishTour } = tour
	const router = useRouter()
	const pathname = usePathname()
	const tourRef = useRef(tour) // Ref to hold the latest tour object
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
	const prevStep = usePrevious(tourState.step)
	const prevPhase = usePrevious(tourState.phase)

	// Keep a ref to the tour object to avoid stale closures in useEffect
	useEffect(() => {
		tourRef.current = tour
	}, [tour])

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

		// NEW: Special positioning for full-screen mobile modals
		const isFullScreenModal =
			isMobile() && targetRect.height > innerHeight * 0.8

		if (isFullScreenModal) {
			return {
				positionStyle: {
					bottom: "120px", // Position above the panel's footer buttons
					left: "50%",
					transform: "translateX(-50%)",
					width: `calc(100vw - ${margin * 2}px)`,
					maxWidth: tooltipMaxWidth
				},
				isPositionedBelow: false,
				hideArrow: true // The arrow is confusing on a full-screen overlay
			}
		}

		// --- Existing logic for non-fullscreen elements ---
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
		if (left < margin) {
			left = margin
		}
		if (left + finalTooltipWidth > innerWidth - margin) {
			left = innerWidth - finalTooltipWidth - margin
		}

		return {
			positionStyle: {
				top,
				bottom,
				left,
				width: finalTooltipWidth
			},
			isPositionedBelow: positionBelow,
			hideArrow: false
		}
	}, [targetRect])

	useEffect(() => {
		if (!tourState.isActive || tourState.isHighlightPaused) {
			setTargetRect(null)
			setTargetElement(null)
			return
		}

		let baseConfig = tourSteps[tourState.step]
		if (!baseConfig) {
			finishTour()
			return
		}

		// Create a mutable config for this render cycle that we can adjust dynamically.
		let currentStepConfig = { ...baseConfig }

		// --- DYNAMIC CONFIGURATION FOR STEP 5 (Task Simulation) ---
		// This block adjusts the tour's target and content *before* it tries to render.
		if (tourState.step === 5) {
			const subStepConfig = taskSubSteps[tourState.subStep]
			if (subStepConfig) {
				if (isMobile()) {
					const isPanelPhase = tourState.phase === "panel"
					// Dynamically set the correct selector based on the phase
					currentStepConfig.selector = isPanelPhase
						? "[data-tour-id='task-details-panel']"
						: "[data-tour-id='demo-task-card']"

					// Update content to match the phase
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
					// Desktop logic remains simple
					currentStepConfig.selector =
						"[data-tour-id='demo-task-card']"
					currentStepConfig.title = subStepConfig.title
					currentStepConfig.body = subStepConfig.body_list
					currentStepConfig.custom_button = subStepConfig.button
				}
			}
		}

		if (!currentStepConfig) {
			finishTour()
			return
		}

		// --- NEW LOGIC for mobile sidebar ---
		if (isMobile()) {
			const isSidebarStep =
				currentStepConfig.selector?.includes("sidebar-")
			if (isSidebarStep) {
				setMobileNavOpen(true) // Open sidebar for these steps
			} else {
				setMobileNavOpen(false) // For other steps, ensure it's closed
			}
		}
		// --- END NEW LOGIC ---

		// --- REVISED CRITICAL FIX: Synchronize tour with navigation ---
		// 1. If the step has a navigation action and the user has completed it (by changing the URL),
		// then we can safely advance to the next step. This must be checked FIRST.
		const { action } = currentStepConfig
		if (action?.type === "navigate" && pathname === action.targetPath) {
			nextStep()
			return
		}

		// 2. If the action wasn't completed, check if we are on the correct page for the current step.
		// If not, navigate to it.
		if (currentStepConfig.path && pathname !== currentStepConfig.path) {
			router.push(currentStepConfig.path)
			return
		}
		// --- END CRITICAL FIX ---

		// Handle waiting conditions
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
			// Set the content from our dynamically adjusted config
			setTooltipContent(currentStepConfig)
			setTargetElement(null)
			setTargetRect(null) // Reset rect before finding new one

			const findAndSetTarget = (retries = 5, delay = 200) => {
				// Stop if tour is no longer active
				if (!tourState.isActive) return

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
				} else {
					console.warn(
						`Tour step ${tourState.step}: Target element "${currentStepConfig.selector}" not found on page "${pathname}".`
					)
					setTargetElement(null)
				}
			}
			const isSidebarStepOnMobile =
				isMobile() && currentStepConfig.selector?.includes("sidebar-")
			const animationDelay = isSidebarStepOnMobile ? 400 : 0

			// Detect the specific transition from composer (step 4) to task list (step 5)
			const isTransitioningToTaskList =
				tourState.step === 5 && prevStep === 4
			const taskCreationDelay = isTransitioningToTaskList ? 400 : 0 // Wait for composer to close and list to animate in.

			// Add a new delay for the panel opening animation on mobile
			const isPanelOpening =
				tourState.step === 5 &&
				tourState.phase === "panel" &&
				prevPhase === "list"
			const panelAnimationDelay = isPanelOpening ? 400 : 0

			const initialDelay =
				(currentStepConfig.initialDelay || 100) +
				animationDelay +
				taskCreationDelay +
				panelAnimationDelay
			setTimeout(findAndSetTarget, initialDelay)
		}

		// Handle dynamic content for specific steps
		if (tourState.step === 1 && !tourState.isHighlightPaused) {
			const subStepConfig = chatSubSteps[tourState.subStep]
			if (subStepConfig) {
				const attemptToSetInput = (retries = 5, delay = 100) => {
					if (tour.chatActionsRef.current?.setInput) {
						tour.chatActionsRef.current.setInput(
							subStepConfig.prefill
						)
					} else if (retries > 0) {
						setTimeout(
							() => attemptToSetInput(retries - 1, delay),
							delay
						)
					}
				}
				attemptToSetInput()
				// Update tooltip content dynamically for chat steps
				setTooltipContent((prev) => ({
					...prev,
					instruction:
						subStepConfig.instruction ||
						"Now, let's create a workflow. Click send."
				}))
			}
		} else if (tourState.step === 4 && !tourState.isHighlightPaused) {
			// Special handling for composer step
			setTooltipContent({
				...currentStepConfig,
				title: "Step 4/7: Describe Your Goal",
				body: "I've pre-filled the composer with a goal. I will analyze this and create a plan to achieve it.",
				instruction: "Click 'Create Task' to see me get to work.",
				selector: "[data-tour-id='task-composer-create-button']",
				// This step doesn't have a next button, it waits for the composer to close
				// which is handled by the useEffect in tasks/page.js triggering nextStep
				isWaitingForAction: true
			})
		}
	}, [
		tourState.isActive,
		tourState.step,
		tourState.subStep,
		tourState.phase,
		tourState.isHighlightPaused,
		pathname,
		router,
		finishTour,
		tour.chatActionsRef, // It's a ref, but ESLint might want it.
		tour.setTourState,
		searchParams,
		nextStep,
		prevStep,
		prevPhase,
		setMobileNavOpen
		// Note: `tour` object itself is not a dependency as it causes loops.
		// We depend on its granular state properties instead.
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

	if (!tourState.isActive) return null

	const currentStepConfig = tourSteps[tourState.step]
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
								onClick={button.onClick}
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
							// When the arrow is hidden (e.g., on a full-screen modal),
							// we provide a safe default value to prevent a NaN error from
							// trying to perform calculations with percentage-based CSS values.
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
							!tooltipContent.custom_button && // prettier-ignore
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
								onClick={() => tour.handleCustomAction()}
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
							!tourState.isHighlightPaused) &&
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
	// --- Tour Configuration ---

	// ... (keep all your existing state declarations)
	const [isNotificationsOpen, setNotificationsOpen] = useState(false)
	const [isSearchOpen, setSearchOpen] = useState(false)
	const [isMobileNavOpen, setMobileNavOpen] = useState(false)
	const [unreadCount, setUnreadCount] = useState(0)
	const [user, setUser] = useState(null) // Unified user state
	const [notifRefreshKey, setNotifRefreshKey] = useState(0)
	const [userDetails, setUserDetails] = useState(null)
	const wsRef = useRef(null)
	const searchParams = useSearchParams() // Hook to read URL query parameters
	const posthog = usePostHog()

	// --- Guided Tour State ---
	const [tourState, setTourState] = useState({
		isActive: false,
		step: 0,
		subStep: 0, // For multi-part steps like the task simulation
		phase: null, // new state: null | 'list' | 'panel'
		isWaitingForAction: false,
		isHighlightPaused: false
	})
	const chatActionsRef = useRef(null)
	const pathname = usePathname() // eslint-disable-line
	const router = useRouter()

	const skipTour = useCallback(() => {
		setTourState({
			isActive: false,
			step: 0,
			subStep: 0,
			phase: null,
			isWaitingForAction: false,
			isHighlightPaused: false
		})
	}, [setTourState])

	const finishTour = useCallback(() => {
		setTourState({
			isActive: false,
			step: 0,
			subStep: 0,
			phase: null,
			isWaitingForAction: false,
			isHighlightPaused: false
		})
		// Potentially set a flag in localStorage or user profile to not show again
	}, [setTourState])

	const startTour = useCallback(() => {
		// Always reset to the beginning of the tour
		setTourState({
			isActive: true,
			step: 0,
			subStep: 0,
			phase: null,
			isWaitingForAction: false,
			isHighlightPaused: false
		})
		// If not on the chat page, navigate there. The tour's useEffect will handle the rest.
		if (pathname !== "/chat") {
			router.push("/chat")
		}
	}, [setTourState, pathname, router])

	const nextStep = useCallback(() => {
		setTourState((prev) => {
			const newStep = prev.step + 1
			return {
				...prev,
				step: newStep,
				subStep: 0,
				phase: newStep === 5 ? "list" : null, // Set initial phase for step 5
				isWaitingForAction: false,
				isHighlightPaused: false // Ensure highlight is active on new step
			}
		})
	}, [setTourState])

	const nextSubStep = useCallback(() => {
		setTourState((prev) => ({
			...prev,
			subStep: prev.subStep + 1
		}))
	}, [setTourState])

	const startTaskDemo = useCallback(() => {
		setTourState({
			isActive: true,
			step: 3,
			subStep: 0,
			phase: null,
			isWaitingForAction: false,
			isHighlightPaused: false
		})
	}, [setTourState])

	const setHighlightPaused = useCallback((isPaused) => {
		setTourState((prev) => ({
			...prev,
			isHighlightPaused: isPaused
		}))
	}, [])

	const tourSteps = [
		// Step 0: Welcome Mat (Modal)
		{
			type: "modal",
			title: "Welcome to BUYASOUL! Let's see your AI in action.",
			body: "This quick, interactive tour will show you how I handle everything from simple commands to complex projects. You'll get to see the full lifecycle of an automated task.",
			buttons: [
				{
					// This button will trigger the navigation to the /chat page for the next step
					label: "Start Tour",
					onClick: () => nextStep(),
					primary: true
				},
				{ label: "Skip for now", onClick: skipTour }
			]
		},
		// Step 1: Send Chat Message (was step 2)
		{
			type: "tooltip",
			path: "/chat",
			selector: "[data-tour-id='chat-input-area']", // This will stay highlighted for the whole chat sequence
			title: "Step 1/7: Give a Command",
			body: "For quick actions, you can tell me what to do. I'll handle it and reply here. After you send the message, I'll prepare the next one.",
			instruction:
				"Let's start with a simple greeting. Click the send button.",
			isWaitingForAction: true, // This will hide the 'Next' button
			// The prefill action is handled inside the useEffect logic
			prefill:
				"Send a 'Hello World' email to existence.sentient@gmail.com"
		},
		// Step 2: Go to Tasks Page (was step 3)
		{
			type: "tooltip",
			path: "/chat",
			selector: "[data-tour-id='sidebar-tasks-icon']",
			title: "Step 2/7: Delegating Complex Work",
			body: "That was a simple task. For bigger goals with multiple steps, I create a project on the Tasks page that you can track. Let's see a simulation of how that works.",
			instruction: "Click the Tasks icon to continue.",
			// NEW: This action tells the tour to wait for a navigation to /tasks before proceeding
			action: { type: "navigate", targetPath: "/tasks" }
		},
		// Step 3: Create Task Button
		{
			type: "tooltip",
			path: "/tasks",
			selector: "[data-tour-id='create-task-button']",
			title: "Step 3/7: Creating a Complex Task",
			body: "Let's create a more complex, multi-step task. You can start by describing your goal in plain English.",
			instruction: "Click the 'Create Task' button to open the composer.",
			isWaitingForAction: true,
			initialDelay: 500
		},
		// Step 4: Composer (placeholder for useEffect logic)
		{
			type: "tooltip",
			path: "/tasks",
			// Selector is overridden in useEffect, but we need a placeholder
			selector: "[data-tour-id='task-composer']",
			title: "Step 4/7: Describe Your Goal",
			body: "I've pre-filled the composer with a goal. I will analyze this and create a plan to achieve it.",
			instruction: "Click 'Create Task' to see me get to work.",
			isWaitingForAction: true,
			initialDelay: 500
		},
		// Step 5: Task Lifecycle Simulation
		{
			type: "tooltip",
			path: "/tasks",
			selector: "[data-tour-id='demo-task-card']",
			title: "Step 5/7: Task Lifecycle",
			body: "This is a simulation of a long-form task. Follow the steps to see how I handle complex goals.",
			// Instruction is provided by taskSubSteps
			instruction: "",
			initialDelay: 500
		},
		// Step 6: Go to Integrations Page
		{
			type: "tooltip",
			path: "/tasks", // from tasks page
			selector: "[data-tour-id='sidebar-integrations-icon']",
			title: "Step 6/7: Connect Your Apps",
			body: "None of this works without connecting your apps. This is where you can manage connections to services like Gmail, Calendar, and more.",
			instruction: "Click the Integrations icon to see.",
			action: { type: "navigate", targetPath: "/integrations" }
		},
		// Step 7: Tour Complete
		{
			type: "modal",
			title: "You're Ready to Go!",
			body: "You've now seen how BUYASOUL can handle immediate commands, orchestrate complex projects, and automate your work with workflows. You can replay the task simulation anytime from the Help menu.",
			buttons: [
				{ label: "Finish Tour", onClick: finishTour, primary: true }
			]
		}
	]

	const chatSubSteps = [
		// subStep 0
		{
			prefill: "Hi BUYASOUL!",
			instruction:
				"Let's start with a simple greeting. Click the send button."
		},
		{
			// subStep 1
			prefill: "Send an email to existence.sentient@gmail.com",
			instruction:
				"Great! Now, let's ask it to perform an action. Click send."
		},
		{
			// subStep 2
			prefill:
				"Send me a daily brief of my unread emails on whatsapp every morning at 8"
		}
	]

	const taskSubSteps = [
		// subStep 0: Planning
		{
			title: "Step 5/7: Planning",
			body_list:
				"The task has been created and is now in the 'Planning' stage. I'm breaking down your goal into a series of steps.",
			body_panel:
				"In the details panel, you can see the execution log. It shows that I'm currently in the planning phase.",
			button: "Simulate Next Step" // for desktop
		},
		{
			// subStep 1: Email sub-task
			title: "Step 5/7: Taking Action",
			body_list:
				"I've created the first sub-task: to email Kabeer. The main task status is now 'Processing'.",
			body_panel:
				"I'll search for Kabeer's contact details and send the email automatically. The new sub-task appears here.",
			button: "Simulate Next Step"
		},
		{
			// subStep 2: Waiting
			title: "Step 5/7: Waiting Intelligently",
			body_list:
				"Now, I'll wait for Kabeer to reply. The main task status is now 'Waiting'. I won't waste resources; I'll pause and check back later.",
			body_panel:
				"The log shows I'm waiting. This could be for a few hours in a real task. Let's fast-forward time.",
			button: "Simulate Next Step"
		},
		{
			// subStep 3: Checking
			title: "Step 5/7: Following Up",
			body_list:
				"The waiting period is over. I've created a new sub-task to check the email thread for a response. Let's assume Kabeer replied.",
			body_panel:
				"The new sub-task to check the email thread is now visible. Once this is complete, I'll know what to do next.",
			button: "Simulate Next Step"
		},
		{
			// subStep 4: Scheduling
			title: "Step 5/7: Finalizing the Goal",
			body_list:
				"Kabeer suggested a time. I'm now creating the final sub-task to schedule the event in your calendar.",
			body_panel:
				"The final sub-task to create the calendar event has been added. The goal is almost complete.",
			button: "Simulate Next Step"
		},
		{
			// subStep 5: Completed
			title: "Step 5/7: Task Completed!",
			body_list:
				"Success! All sub-tasks are done, and the main goal is achieved. The task is now marked as 'Completed'.",
			body_panel:
				"The task is complete. You can review the full history of sub-tasks and logs at any time.",
			button: "Next"
		}
	]

	const {
		user: auth0User,
		error: authError,
		isLoading: isAuthLoading
	} = useUser()

	const [isLoading, setIsLoading] = useState(true)
	const [isAllowed, setIsAllowed] = useState(false)

	const handleCustomAction = useCallback(() => {
		setHighlightPaused(true)
		setTourState((prev) => {
			if (prev.step === 5) {
				if (isMobile()) {
					// On mobile, toggle between list and panel
					if (prev.phase === "list") {
						// Was showing list, now show panel for the same sub-step
						return { ...prev, phase: "panel" }
					} else {
						// phase === 'panel'
						// Was showing panel.
						const isLastSubStep =
							prev.subStep >= taskSubSteps.length - 1
						if (isLastSubStep) {
							// If it's the end of the simulation, move to the next main step
							return {
								...prev,
								step: prev.step + 1,
								subStep: 0,
								phase: null
							}
						}
						// Otherwise, advance simulation and go back to list view
						return {
							...prev,
							subStep: prev.subStep + 1,
							phase: "list"
						}
					}
				} else {
					// On desktop, panel is always open, just advance the sub-step
					const isLastSubStep =
						prev.subStep >= taskSubSteps.length - 1
					if (isLastSubStep) {
						return {
							...prev,
							step: prev.step + 1,
							subStep: 0,
							phase: null
						}
					}
					return { ...prev, subStep: prev.subStep + 1 }
				}
			}
			// Default action for other custom buttons if any
			return { ...prev, step: prev.step + 1 }
		})
		// Re-enable highlight after a delay to allow UI to update
		setTimeout(() => setHighlightPaused(false), 500)
	}, [setTourState, setHighlightPaused, taskSubSteps])

	const tourValue = useMemo(
		() => ({
			tourState,
			setTourState,
			startTour,
			nextStep,
			nextSubStep,
			setHighlightPaused,
			skipTour,
			finishTour,
			startTaskDemo,
			handleCustomAction,
			chatActionsRef,
			tourSteps,
			taskSubSteps
		}),
		[
			tourState,
			setTourState,
			startTour,
			nextStep,
			nextSubStep,
			setHighlightPaused,
			skipTour,
			finishTour,
			startTaskDemo,
			handleCustomAction,
			tourSteps,
			taskSubSteps
		]
	)

	const showNav = !["/", "/onboarding"].includes(pathname)

	useEffect(() => {
		if (auth0User && posthog) {
			posthog.identify(auth0User.sub, {
				name: auth0User.name,
				email: auth0User.email
			})

			// --- NEW: Fetch custom properties and set PostHog groups ---
			const fetchAndSetUserGroups = async () => {
				try {
					// This is a new client-side API route that proxies to the main server
					const res = await fetch("/api/user/properties")
					if (!res.ok) {
						console.error(
							"Failed to fetch user properties for PostHog, status:",
							res.status
						)
						return // Don't proceed if the call fails
					}

					const properties = await res.json()

					// Set groups in PostHog
					// This allows creating cohorts based on group properties
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

					// Also set as person properties for easier direct filtering on users
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
			// --- END NEW ---
		}
	}, [auth0User, posthog])

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
	}, [searchParams, router, pathname, posthog]) // Dependencies are correct

	// ... (keep the rest of your useEffects and functions exactly as they were)
	useEffect(() => {
		if (!showNav) {
			setIsLoading(false)
			setIsAllowed(true)
			return
		}

		const checkStatus = async () => {
			// --- SELF-HOST AUTH LOGIC ---
			if (process.env.NEXT_PUBLIC_ENVIRONMENT === "selfhost") {
				try {
					const res = await fetch("/api/user/profile")
					if (res.ok) {
						const selfHostUser = await res.json()
						setUser(selfHostUser) // Set the unified user state
						setIsAllowed(true)
					} else {
						throw new Error(
							"Failed to fetch self-host user profile."
						)
					}
				} catch (error) {
					toast.error(error.message)
					setIsAllowed(false)
				} finally {
					setIsLoading(false)
				}
				return
			}

			// --- AUTH0 AUTH LOGIC ---
			if (isAuthLoading) return

			if (authError) {
				toast.error(
					`Session error: ${authError.message}. Redirecting to login.`,
					{ id: "auth-error" }
				)
				router.push("/api/auth/login")
				return
			}

			if (!auth0User) {
				router.push("/api/auth/login")
				return
			}

			setUser(auth0User) // Set the unified user state

			try {
				const res = await fetch("/api/user/data", { method: "POST" })
				if (!res.ok) throw new Error("Could not verify user status.")
				const result = await res.json()
				if (!result?.data?.onboardingComplete) {
					router.push("/onboarding")
				} else {
					setIsAllowed(true)
				}
			} catch (error) {
				toast.error(error.message)
				router.push("/")
			} finally {
				setIsLoading(false)
			}
		}

		checkStatus()
	}, [showNav, auth0User, isAuthLoading, authError, router])

	const handleNotificationsOpen = useCallback(() => {
		setNotificationsOpen(true)
		setUnreadCount(0)
	}, [])

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
	}, [user?.sub, handleNotificationsOpen])

	// PWA Update Handler

	useEffect(() => {
		// This effect runs only on the client side to register the service worker.
		// It's enabled for all environments to allow testing in development.
		if ("serviceWorker" in navigator) {
			// The 'load' event ensures that SW registration doesn't delay page rendering.
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

				const applicationServerKey = urlBase64ToUint8Array(
					process.env.NEXT_PUBLIC_VAPID_PUBLIC_KEY
				)

				subscription = await registration.pushManager.subscribe({
					userVisibleOnly: true,
					applicationServerKey
				})

				const serializedSub = JSON.parse(JSON.stringify(subscription))
				await subscribeUser(serializedSub)
				toast.success("Subscribed to push notifications!")
			}
		} catch (error) {
			console.error("Error during push notification subscription:", error)
			toast.error("Failed to subscribe to push notifications.")
		}
	}, [])

	useEffect(() => {
		if (showNav && user?.sub) subscribeToPushNotifications()
	}, [showNav, user, subscribeToPushNotifications])

	// Define shortcuts after all their callback dependencies are defined
	useGlobalShortcuts(
		handleNotificationsOpen,
		() => setSearchOpen(true) // New: Pass search open function
		// Removed: Command palette toggle is no longer needed
	)

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

	// ... (rest of the component is unchanged)
	return (
		<PlanContext.Provider
			value={{
				plan: (
					auth0User?.[
						`${process.env.NEXT_PUBLIC_AUTH0_NAMESPACE}/roles`
					] || []
				).includes("Pro")
					? "pro"
					: "free",
				isPro: (
					auth0User?.[
						`${process.env.NEXT_PUBLIC_AUTH0_NAMESPACE}/roles`
					] || []
				).includes("Pro"),
				isLoading: isAuthLoading
			}}
		>
			<TourContext.Provider value={tourValue}>
				{showNav && (
					<>
						<Sidebar
							onNotificationsOpen={handleNotificationsOpen}
							onSearchOpen={() => setSearchOpen(true)}
							unreadCount={unreadCount}
							isMobileOpen={isMobileNavOpen}
							onMobileClose={() => setMobileNavOpen(false)}
							user={user}
							isTourActive={tourState.isActive}
						/>
						<button
							onClick={() => setMobileNavOpen(true)}
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
				<GuidedTour
					tourSteps={tourSteps}
					chatSubSteps={chatSubSteps}
					taskSubSteps={taskSubSteps}
					setMobileNavOpen={setMobileNavOpen}
				/>
			</TourContext.Provider>
		</PlanContext.Provider>
	)
}
