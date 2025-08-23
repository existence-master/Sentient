"use client"

import React, {
	IconClock,
	useState,
	useEffect,
	useCallback,
	Suspense,
	useMemo
} from "react"
import { useRouter, useSearchParams } from "next/navigation"
import { format, isSameDay, parseISO } from "date-fns"
import {
	IconLoader,
	IconX,
	IconSparkles,
	IconCheck,
	IconPlus
} from "@tabler/icons-react"
import { AnimatePresence, motion } from "framer-motion"
import toast from "react-hot-toast"
import { Tooltip } from "react-tooltip"
import { calculateNextRun } from "@utils/taskUtils"

import TaskDetailsPanel from "@components/tasks/TaskDetailsPanel"
import TaskViewSwitcher from "@components/tasks/TaskViewSwitcher"
import ListView from "@components/tasks/ListView"
import CalendarView from "@components/tasks/CalendarView"
import TaskComposer from "@components/tasks/TaskComposer" // New component
import InteractiveNetworkBackground from "@components/ui/InteractiveNetworkBackground"
import DayDetailView from "@components/tasks/DayDetailView"
import { usePlan } from "@hooks/usePlan"

const proPlanFeatures = [
	{ name: "Text Chat", limit: "100 messages per day" },
	{ name: "Voice Chat", limit: "10 minutes per day" },
	{ name: "One-Time Tasks", limit: "20 async tasks per day" },
	{ name: "Recurring Tasks", limit: "10 active recurring workflows" },
	{ name: "Triggered Tasks", limit: "10 triggered workflows" },
	{
		name: "Parallel Agents",
		limit: "5 complex tasks per day with 50 sub agents"
	},
	{ name: "File Uploads", limit: "20 files per day" },
	{ name: "Memories", limit: "Unlimited memories" },
	{
		name: "Other Integrations",
		limit: "Notion, GitHub, Slack, Discord, Trello"
	}
]

const UpgradeToProModal = ({ isOpen, onClose }) => {
	if (!isOpen) return null

	const handleUpgrade = () => {
		const dashboardUrl = process.env.NEXT_PUBLIC_LANDING_PAGE_URL
		if (dashboardUrl) {
			window.location.href = `${dashboardUrl}/dashboard`
		}
		onClose()
	}

	return (
		<AnimatePresence>
			{isOpen && (
				<motion.div
					initial={{ opacity: 0 }}
					animate={{ opacity: 1 }}
					exit={{ opacity: 0 }}
					className="fixed inset-0 bg-black/70 backdrop-blur-md z-[100] flex items-center justify-center p-4"
					onClick={onClose}
				>
					<motion.div
						initial={{ scale: 0.95, y: 20 }}
						animate={{ scale: 1, y: 0 }}
						exit={{ scale: 0.95, y: -20 }}
						transition={{ duration: 0.2, ease: "easeInOut" }}
						onClick={(e) => e.stopPropagation()}
						className="relative bg-neutral-900/90 backdrop-blur-xl p-6 rounded-2xl shadow-2xl w-full max-w-lg border border-neutral-700 flex flex-col"
					>
						<header className="text-center mb-4">
							<h2 className="text-2xl font-bold text-white flex items-center justify-center gap-2">
								<IconSparkles className="text-brand-orange" />
								Upgrade to Pro
							</h2>
							<p className="text-neutral-400 mt-2">
								Unlock Parallel Agents and other powerful
								features.
							</p>
						</header>
						<main className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-4 my-4">
							{proPlanFeatures.map((feature) => (
								<div
									key={feature.name}
									className="flex items-start gap-2.5"
								>
									<IconCheck
										size={18}
										className="text-green-400 flex-shrink-0 mt-0.5"
									/>
									<div>
										<p className="text-white text-sm font-medium">
											{feature.name}
										</p>
										<p className="text-neutral-400 text-xs">
											{feature.limit}
										</p>
									</div>
								</div>
							))}
						</main>
						<footer className="mt-4 flex flex-col gap-2">
							<button
								onClick={handleUpgrade}
								className="w-full py-2.5 px-5 rounded-lg bg-brand-orange hover:bg-brand-orange/90 text-brand-black font-semibold transition-colors"
							>
								Upgrade to Pro - $9/month
							</button>
							<button
								onClick={onClose}
								className="w-full py-2 px-5 rounded-lg hover:bg-neutral-800 text-sm font-medium text-neutral-400"
							>
								Not now
							</button>
						</footer>
					</motion.div>
				</motion.div>
			)}
		</AnimatePresence>
	)
}

function TasksPageContent() {
	const router = useRouter()
	const searchParams = useSearchParams()

	// Raw tasks from API
	const [allTasks, setAllTasks] = useState([])
	// Processed tasks for different views

	const [integrations, setIntegrations] = useState([])
	const [allTools, setAllTools] = useState([])
	const [isLoading, setIsLoading] = useState(true)

	const [view, setView] = useState("list") // 'list' or 'calendar'
	const [isMobile, setIsMobile] = useState(false)

	// --- NEW STATE MANAGEMENT FOR SMART PANEL & MODAL ---
	const [rightPanelContent, setRightPanelContent] = useState({
		type: "composer",
		data: null
	})
	const [isModalOpen, setIsModalOpen] = useState(false)
	// --- END NEW STATE ---

	const [currentCalendarDate, setCurrentCalendarDate] = useState(new Date())
	const [searchQuery, setSearchQuery] = useState("")
	const [isUpgradeModalOpen, setUpgradeModalOpen] = useState(false)
	const { isPro } = usePlan()

	// Processed tasks for different views are derived from allTasks
	const {
		oneTimeTasks,
		recurringTasks,
		triggeredTasks,
		swarmTasks,
		longFormTasks,
		recurringInstances
	} = useMemo(() => {
		const oneTime = []
		const recurring = []
		const triggered = []
		const swarm = []
		const instances = []
		const longForm = []

		allTasks.forEach((task) => {
			if (task.task_type === "swarm") {
				swarm.push(task)
			} else if (task.task_type === "long_form") {
				longForm.push(task)
			} else if (task.schedule?.type === "recurring") {
				recurring.push(task)
				// Process past runs from `runs` array
				if (task.runs && Array.isArray(task.runs)) {
					task.runs.forEach((run) => {
						const runDate = run.execution_start_time
							? parseISO(run.execution_start_time)
							: null
						if (runDate) {
							instances.push({
								...task,
								status: run.status, // Use the run's specific status
								scheduled_date: runDate,
								instance_id: `${task.task_id}-${run.run_id}`
							})
						}
					})
				}
				// Add next upcoming run
				const nextRunDate = calculateNextRun(
					task.schedule,
					task.created_at,
					task.runs
				)
				if (nextRunDate) {
					instances.push({
						...task,
						status: "pending", // An upcoming run is pending
						scheduled_date: nextRunDate,
						instance_id: `${task.task_id}-next`
					})
				}
			} else if (task.schedule?.type === "triggered") {
				triggered.push(task)
			} else {
				// One-time tasks
				const scheduledDate = task.schedule?.run_at
					? parseISO(task.schedule.run_at)
					: parseISO(task.created_at)
				oneTime.push({
					...task,
					scheduled_date: scheduledDate,
					instance_id: task.task_id
				})
			}
		})

		return {
			oneTimeTasks: oneTime,
			recurringTasks: recurring,
			triggeredTasks: triggered,
			swarmTasks: swarm,
			longFormTasks: longForm,
			recurringInstances: instances
		}
	}, [allTasks])

	// --- NEW: Handle responsive layout and initial panel state ---
	useEffect(() => {
		const checkMobile = () => window.innerWidth < 768
		setIsMobile(checkMobile())

		if (checkMobile()) {
			setRightPanelContent({ type: "hidden", data: null })
		} else {
			const taskId = searchParams.get("taskId")
			if (!taskId) {
				setRightPanelContent({ type: "composer", data: null })
			}
		}

		const handleResize = () => {
			const mobile = checkMobile()
			if (mobile !== isMobile) {
				setIsMobile(mobile)
				// If switching to mobile, hide the panel
				if (mobile) {
					setRightPanelContent({ type: "hidden", data: null })
				} else {
					// If switching to desktop and no task is selected, show composer
					if (
						rightPanelContent.type === "hidden" ||
						rightPanelContent.type === "composer"
					) {
						setRightPanelContent({ type: "composer", data: null })
					}
				}
			}
		}

		window.addEventListener("resize", handleResize)
		return () => window.removeEventListener("resize", handleResize)
	}, []) // Run only once on mount

	// --- NEW: Sync panel with URL ---
	useEffect(() => {
		const taskId = searchParams.get("taskId")
		if (taskId) {
			const task = allTasks.find((t) => t.task_id === taskId)
			if (task) {
				setRightPanelContent({ type: "details", data: task })
				if (isMobile) setIsModalOpen(true)
			}
		} else {
			// If URL is cleared, show composer on desktop, hide on mobile
			if (!isMobile) {
				setRightPanelContent({ type: "composer", data: null })
			} else {
				// Don't automatically open the composer modal on mobile when URL clears
				setIsModalOpen(false)
			}
		}
	}, [searchParams, allTasks, isMobile])

	const fetchTasks = useCallback(async () => {
		setIsLoading(true)
		try {
			const tasksRes = await fetch("/api/tasks", { method: "POST" })
			if (!tasksRes.ok) throw new Error("Failed to fetch tasks")
			const tasksData = await tasksRes.json()
			const rawTasks = Array.isArray(tasksData.tasks)
				? tasksData.tasks
				: []
			setAllTasks(rawTasks)

			const integrationsRes = await fetch("/api/settings/integrations", {
				method: "POST"
			})
			if (!integrationsRes.ok)
				throw new Error("Failed to fetch integrations")
			const integrationsData = await integrationsRes.json()
			const tools = integrationsData.integrations.map((i) => ({
				name: i.name,
				display_name: i.display_name
			}))
			setAllTools(tools)
			setIntegrations(integrationsData.integrations || [])
		} catch (error) {
			toast.error(`Error fetching data: ${error.message}`)
		} finally {
			setIsLoading(false)
		}
	}, [])

	useEffect(() => {
		fetchTasks()
	}, [fetchTasks])

	useEffect(() => {
		const handleBackendUpdate = () => {
			console.log("Received task_list_updated event, fetching tasks...")
			toast.success("Task list updated from backend.")
			fetchTasks()
		}
		window.addEventListener("task_list_updated", handleBackendUpdate)
		return () => {
			window.removeEventListener("task_list_updated", handleBackendUpdate)
		}
	}, [fetchTasks])

	const handleAction = useCallback(
		async (actionFn, successMessage, ...args) => {
			const toastId = toast.loading("Processing...")
			try {
				const response = await actionFn(...args)
				if (!response.ok) {
					const errorData = await response.json()
					throw new Error(errorData.error || "Action failed")
				}
				toast.success(successMessage, { id: toastId })
				await fetchTasks()
			} catch (error) {
				toast.error(`Error: ${error.message}`, { id: toastId })
			}
		},
		[fetchTasks]
	)

	const handleAnswerClarifications = async (taskId, answers) => {
		await handleAction(
			() =>
				fetch("/api/tasks/answer-clarifications", {
					method: "POST",
					headers: { "Content-Type": "application/json" },
					body: JSON.stringify({ taskId, answers })
				}),
			"Answers submitted successfully. The task will now resume."
		)
		handleClosePanel()
	}

	const handleAnswerLongFormClarification = async (
		taskId,
		requestId,
		answer
	) => {
		await handleAction(
			() =>
				fetch(`/api/tasks/${taskId}/answer-clarification`, {
					method: "POST",
					headers: { "Content-Type": "application/json" },
					body: JSON.stringify({ requestId, answer })
				}),
			"Answer submitted. The task will now resume."
		)
	}

	// --- REVISED: handleAddTask is now handleCreateTask and takes a payload ---
	const handleCreateTask = async (payload) => {
		const toastId = toast.loading("Creating task...")
		try {
			const response = await fetch("/api/tasks/add", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify(payload)
			})
			if (!response.ok) {
				const errorData = await response.json().catch(() => ({}))
				const error = new Error(errorData.error || "Failed to add task")
				error.status = response.status
				throw error
			}
			const data = await response.json()

			toast.success(data.message || "Task created!", { id: toastId })
			if (isMobile)
				setIsModalOpen(false) // Close modal on success
			else setRightPanelContent({ type: "composer", data: null }) // Reset composer
			await fetchTasks() // Refresh tasks list
		} catch (error) {
			if (error.status === 429) {
				toast.error(
					error.message || "You've reached your daily task limit.",
					{ id: toastId }
				)
				if (!isPro) setUpgradeModalOpen(true)
			} else {
				toast.error(`Error: ${error.message}`, { id: toastId })
			}
		}
	}
	// --- END REVISED ---

	const handleUpdateTask = async (updatedTask) => {
		await handleAction(
			() =>
				fetch("/api/tasks/update", {
					method: "POST",
					headers: { "Content-Type": "application/json" },
					body: JSON.stringify({
						...updatedTask,
						taskId: updatedTask.task_id
					})
				}),
			"Task updated!"
		)
	}

	// --- REVISED: handleSelectItem now updates URL and panel state ---
	const handleSelectItem = (item) => {
		const taskId = item.task_id
		router.push(`/tasks?taskId=${taskId}`, { scroll: false })
		setRightPanelContent({ type: "details", data: item })
		if (isMobile) {
			setIsModalOpen(true)
		}
	}
	// --- END REVISED ---

	const handleShowMoreClick = (date) => {
		const tasksForDay = filteredCalendarTasks.filter(
			(task) => isSameDay(task.scheduled_date, date) // prettier-ignore
		)
		setRightPanelContent({
			type: "day",
			data: { date, tasks: tasksForDay }
		})
		if (isMobile) setIsModalOpen(true)
	}

	// --- REVISED: handleCloseRightPanel now handles both mobile and desktop ---
	const handleClosePanel = () => {
		router.push("/tasks", { scroll: false }) // Clear URL param
		if (isMobile) {
			setIsModalOpen(false)
		} else {
			setRightPanelContent({ type: "composer", data: null })
		}
	}
	// --- END REVISED ---

	const handleCreateTaskFromEvent = async (event) => {
		const prompt = `Help me prepare for this event from my calendar:

Title: ${event.summary}
Time: ${event.start} to ${event.end}
Attendees: ${(event.attendees || []).join(", ")}
Description: ${event.description || "No description."}`

		await handleAction(
			() =>
				fetch("/api/tasks/add", {
					method: "POST",
					headers: { "Content-Type": "application/json" },
					body: JSON.stringify({ prompt, assignee: "ai" })
				}),
			"Task created from calendar event!"
		)
		handleClosePanel() // Close the panel after creating the task
	}

	const handleAddTaskForDay = (date) => {
		// Set the panel to show the composer with the default date
		setRightPanelContent({
			type: "composer",
			data: { defaultDate: date }
		})
		// If on mobile, open the modal
		if (isMobile) {
			setIsModalOpen(true)
		}
	}

	const filteredOneTimeTasks = useMemo(() => {
		if (!searchQuery.trim()) {
			return oneTimeTasks
		}
		return oneTimeTasks.filter(
			(task) =>
				task.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
				(task.description &&
					task.description
						.toLowerCase()
						.includes(searchQuery.toLowerCase()))
		)
	}, [oneTimeTasks, searchQuery])

	const filteredActiveWorkflows = useMemo(() => {
		const allWorkflows = [
			...swarmTasks,
			...recurringTasks,
			...triggeredTasks
		]
		if (!searchQuery.trim()) {
			return allWorkflows
		}
		return allWorkflows.filter(
			(task) =>
				task.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
				(task.description &&
					task.description
						.toLowerCase()
						.includes(searchQuery.toLowerCase()))
		)
    }, [swarmTasks, recurringTasks, triggeredTasks, searchQuery])

	const filteredLongFormTasks = useMemo(() => {
		if (!searchQuery.trim()) {
			return longFormTasks
		}
		return longFormTasks.filter(
			(task) =>
				task.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
				(task.description &&
					task.description
						.toLowerCase()
						.includes(searchQuery.toLowerCase()))
		)
	}, [longFormTasks, searchQuery])
	const filteredCalendarTasks = useMemo(() => {
		const allCalendarTasks = [...oneTimeTasks, ...recurringInstances]
		if (!searchQuery.trim()) {
			return allCalendarTasks
		}
		return allCalendarTasks.filter((task) =>
			(task.name || task.summary)
				?.toLowerCase()
				.includes(searchQuery.toLowerCase())
		)
	}, [oneTimeTasks, recurringInstances, searchQuery])

	// --- NEW: Render logic for panel/modal ---
	const renderPanelContent = () => {
		switch (rightPanelContent.type) {
			case "composer":
				return (
					<TaskComposer
						onTaskCreated={handleCreateTask}
						isPro={isPro}
						composerData={rightPanelContent.data}
						onUpgradeClick={() => setUpgradeModalOpen(true)}
						onClose={isMobile ? () => setIsModalOpen(false) : null}
					/>
				)
			case "details":
				return (
					<TaskDetailsPanel
						task={rightPanelContent.data}
						allTools={allTools}
						integrations={integrations}
						onClose={handleClosePanel}
						onSave={handleUpdateTask}
						onAnswerClarifications={handleAnswerClarifications}
						onAnswerLongFormClarification={
							handleAnswerLongFormClarification
						}
						onSelectTask={handleSelectItem}
						onDelete={(taskId) =>
							handleAction(
								() =>
									fetch(`/api/tasks/delete`, {
										method: "POST",
										body: JSON.stringify({ taskId }),
										headers: {
											"Content-Type": "application/json"
										}
									}),
								"Task deleted."
							)
						}
						onApprove={(taskId) =>
							handleAction(
								() =>
									fetch(`/api/tasks/approve`, {
										method: "POST",
										body: JSON.stringify({ taskId }),
										headers: {
											"Content-Type": "application/json"
										}
									}),
								"Task approved."
							)
						}
						onRerun={(taskId) =>
							handleAction(
								() =>
									fetch("/api/tasks/rerun", {
										method: "POST",
										headers: {
											"Content-Type": "application/json"
										},
										body: JSON.stringify({ taskId })
									}),
								"Task re-run initiated."
							)
						}
						onArchiveTask={(taskId) =>
							handleAction(
								() =>
									fetch(`/api/tasks/update`, {
										method: "POST",
										body: JSON.stringify({
											taskId,
											status: "archived"
										}),
										headers: {
											"Content-Type": "application/json"
										}
									}),
								"Task archived."
							)
						}
						onSendChatMessage={(taskId, message) =>
							handleAction(
								() =>
									fetch(`/api/tasks/chat`, {
										method: "POST",
										body: JSON.stringify({
											taskId,
											message
										}),
										headers: {
											"Content-Type": "application/json"
										}
									}),
								"Message sent."
							)
						}
					/>
				)
			case "day":
				return (
					<DayDetailView
						date={rightPanelContent.data.date}
						tasks={rightPanelContent.data.tasks}
						onSelectTask={handleSelectItem}
						onClose={handleClosePanel}
					/>
				)
			default:
				return null
		}
	}
	// --- END NEW ---

	return (
		<div className="flex-1 flex h-full text-white overflow-hidden">
			<Tooltip
				id="tasks-tooltip"
				place="right"
				style={{ zIndex: 9999 }}
			/>
			<UpgradeToProModal
				isOpen={isUpgradeModalOpen}
				onClose={() => setUpgradeModalOpen(false)}
			/>
			<div className="flex-1 flex overflow-hidden relative">
				<div className="absolute inset-0 z-[-1] network-grid-background">
					<InteractiveNetworkBackground />
				</div>
				{/* Main Content Panel */}
				<main className="flex-1 flex flex-col overflow-hidden relative">
					<div className="absolute -top-[250px] left-1/2 -translate-x-1/2 w-[800px] h-[500px] bg-brand-orange/10 rounded-full blur-3xl -z-10" />
					<header className="p-6 pt-20 md:pt-6 flex-shrink-0 flex items-center justify-between bg-transparent">
						<h1 className="text-3xl font-bold text-white">Tasks</h1>
						<div className="absolute top-6 left-1/2 -translate-x-1/2">
							<TaskViewSwitcher view={view} setView={setView} />
						</div>
						{/* --- NEW: Desktop "+" button --- */}
						<div className="hidden md:block">
							<button
								onClick={() => {
									router.push("/tasks", { scroll: false })
									setRightPanelContent({
										type: "composer",
										data: null
									})
								}}
								className="p-2 rounded-full bg-neutral-800/50 hover:bg-neutral-700/80 text-white"
							>
								<IconPlus size={20} />
							</button>
						</div>
					</header>

					<div className="flex-1 overflow-y-auto custom-scrollbar">
						{isLoading ? (
							<div className="flex justify-center items-center h-full">
								<IconLoader className="w-8 h-8 animate-spin text-sentient-blue" />
							</div>
						) : (
							<AnimatePresence mode="wait">
								<motion.div
									key={view}
									initial={{ opacity: 0 }}
									animate={{ opacity: 1 }}
									exit={{ opacity: 0 }}
									transition={{ duration: 0.3 }}
									className="h-full"
								>
									{view === "list" ? (
										<ListView
											oneTimeTasks={filteredOneTimeTasks}
											activeWorkflows={
												filteredActiveWorkflows
											}
											longFormTasks={
												filteredLongFormTasks
											}
											onSelectTask={handleSelectItem}
											searchQuery={searchQuery}
											onSearchChange={setSearchQuery}
										/>
									) : (
										<CalendarView
											tasks={filteredCalendarTasks}
											onSelectTask={handleSelectItem}
											onAddTaskForDay={
												handleAddTaskForDay
											}
											onShowMoreClick={
												handleShowMoreClick
											}
											onMonthChange={
												setCurrentCalendarDate
											}
										/>
									)}
								</motion.div>
							</AnimatePresence>
						)}
					</div>

					{/* --- REMOVED CreateTaskInput --- */}
				</main>

				{/* --- NEW: Right Panel for Desktop --- */}
				<aside className="hidden md:flex w-[500px] lg:w-[550px] bg-brand-black/50 backdrop-blur-sm border-l border-brand-gray flex-shrink-0 flex-col">
					<AnimatePresence mode="wait">
						<motion.div
							key={rightPanelContent.type}
							initial={{ opacity: 0, x: 50 }}
							animate={{ opacity: 1, x: 0 }}
							exit={{ opacity: 0, x: -50 }}
							transition={{ duration: 0.3 }}
							className="h-full"
						>
							{renderPanelContent()}
						</motion.div>
					</AnimatePresence>
				</aside>
				{/* --- END NEW --- */}
			</div>

			{/* --- NEW: Floating Action Button for Mobile --- */}
			<button
				onClick={() => {
					setRightPanelContent({ type: "composer", data: null })
					setIsModalOpen(true)
				}}
				className="md:hidden fixed bottom-6 right-6 z-40 p-4 bg-brand-orange text-black rounded-full shadow-lg hover:bg-brand-orange/90 transition-transform hover:scale-105"
			>
				<IconPlus size={24} strokeWidth={2.5} />
			</button>
			{/* --- END NEW --- */}

			{/* --- NEW: Modal for Mobile --- */}
			<AnimatePresence>
				{isMobile && isModalOpen && (
					<motion.div
						initial={{ opacity: 0 }}
						animate={{ opacity: 1 }}
						exit={{ opacity: 0 }}
						className="fixed inset-0 bg-black/70 z-[70] md:hidden"
					>
						<motion.div
							initial={{ y: "100%" }}
							animate={{ y: "0%" }}
							exit={{ y: "100%" }}
							transition={{
								type: "spring",
								stiffness: 300,
								damping: 30
							}}
							className="absolute inset-0"
						>
							{renderPanelContent()}
						</motion.div>
					</motion.div>
				)}
			</AnimatePresence>
			{/* --- END NEW --- */}
		</div>
	)
}

export default function TasksPage() {
	return (
		<Suspense
			fallback={
				<div className="flex-1 flex h-full bg-black text-white overflow-hidden justify-center items-center">
					<IconLoader className="w-10 h-10 animate-spin text-[var(--color-accent-blue)]" />
				</div>
			}
		>
			<TasksPageContent />
		</Suspense>
	)
}
