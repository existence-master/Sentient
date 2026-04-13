"use client"

import { cn } from "@utils/cn"
import React, { useRef, useState, useMemo, useEffect, Suspense } from "react" // eslint-disable-line
import { useRouter, useSearchParams } from "next/navigation"
import {
	IconLoader,
	IconSparkles,
	IconPlus
} from "@tabler/icons-react"
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { AnimatePresence, motion } from "framer-motion"
import toast from "react-hot-toast"
import { Tooltip } from "react-tooltip"

import TaskDetails from "@components/tasks/TaskDetails"
import TaskViewSwitcher from "@components/tasks/TaskViewSwitcher"
import ListView from "@components/tasks/ListView"
import TaskComposer from "@components/tasks/TaskComposer"
import InteractiveNetworkBackground from "@components/ui/InteractiveNetworkBackground"
import { Drawer } from "@components/ui/drawer"
import { Button } from "@components/ui/button"
import apiClient from "@lib/apiClient"
import { useTaskStore, useTourStore } from "@stores/app-stores"

function usePrevious(value) {
	const ref = useRef()
	useEffect(() => {
		ref.current = value
	})
	return ref.current
}

function TasksPageContent() {
	const router = useRouter()
	const searchParams = useSearchParams()
	const queryClient = useQueryClient()
	const {
		view,
		searchQuery,
		isComposerOpen,
		composerInitialData,
		setView,
		setSearchQuery,
		openComposer,
		closeComposer
	} = useTaskStore()
	const { isActive: isTourActive, step, subStep, phase, nextStep } = useTourStore()
	const [isMobile, setIsMobile] = useState(false)
	const [isModalOpen, setIsModalOpen] = useState(false)
	const prevTourState = usePrevious({ isActive: isTourActive, step, subStep })

	const { data, isLoading, isError } = useQuery({
		queryKey: ["tasksPageData"],
		queryFn: async () => {
			if (isTourActive) {
				return { tasks: [], integrations: [] }
			}
			const [tasksRes, integrationsRes] = await Promise.all([
				fetch("/api/tasks", { method: "POST" }),
				fetch("/api/settings/integrations", { method: "POST" })
			])
			if (!tasksRes.ok) throw new Error("Failed to fetch tasks")
			if (!integrationsRes.ok)
				throw new Error("Failed to fetch integrations")
			const tasksData = await tasksRes.json()
			const integrationsData = await integrationsRes.json()
			return {
				tasks: Array.isArray(tasksData.tasks) ? tasksData.tasks : [],
				integrations: integrationsData.integrations || []
			}
		},
		staleTime: 1000 * 60 // 1 minute
	})

	const allTasks = data?.tasks || []
	const integrations = data?.integrations || []
	const allTools = integrations.map((i) => ({
		name: i.name,
		display_name: i.display_name
	}))
	const selectedTaskId = searchParams.get("taskId")

	const demoWorkflow = useMemo(() => {
		if (!isTourActive || step < 6) return null

		return {
			task_id: "demo-workflow-123",
			name: "Daily Email Briefing",
			description:
				"A simulated workflow to summarize unread emails daily.",
			status: "active",
			task_type: "recurring",
			isDemoWorkflow: true,
			created_at: new Date().toISOString()
		}
	}, [isTourActive, step])

	const demoTask = useMemo(() => {
		if (!isTourActive || step < 5) return null

		let status = "planning"
		let subTasks = []

		const baseSubTasks = [
			{
				task_id: "demo-sub-1",
				name: "Email Kabeer to ask for availability",
				status: "completed"
			},
			{
				task_id: "demo-sub-2",
				name: "Check email thread for reply from Kabeer",
				status: "completed"
			},
			{
				task_id: "demo-sub-3",
				name: "Schedule calendar event for Monday 5 PM with Kabeer",
				status: "completed"
			}
		]

		if (subStep === 1) {
			status = "processing"
			subTasks = [baseSubTasks[0]]
		} else if (subStep === 2) {
			status = "waiting"
			subTasks = [baseSubTasks[0]]
		} else if (subStep === 3) {
			status = "processing"
			subTasks = [baseSubTasks[0], baseSubTasks[1]]
		} else if (subStep === 4) {
			status = "processing"
			subTasks = [baseSubTasks[0], baseSubTasks[1], baseSubTasks[2]]
		} else if (subStep >= 5) {
			status = "completed"
			subTasks = baseSubTasks
		}

		return {
			task_id: "demo-task-123",
			name: "Coordinate with Kabeer to set up a meeting next week.",
			description:
				"A simulated task to demonstrate the lifecycle of an automated project.",
			status: status,
			task_type: "long_form",
			isDemoTask: true,
			created_at: new Date().toISOString(),
			subTasks: subTasks,
			runs: [
				{
					run_id: "demo-run-1",
					status: status,
					progress_updates: [
						{
							message: {
								type: "info",
								content: `Simulating step: ${status}`
							},
							timestamp: new Date().toISOString()
						}
					]
				}
			]
		}
	}, [isTourActive, step, subStep])

	const handleClosePanel = () => {
		if (isTourActive && step >= 3) {
			// Don't close panel during tour simulation
		} else {
			router.push("/tasks", { scroll: false }) // Clear URL param
		}
		setIsModalOpen(false)
	}

	// Effect to sync UI state with the tour state
	useEffect(() => {
		if (isTourActive) {
			if (step === 3) {
				// This is the "Create Task" button step.
				if (isComposerOpen) {
					// If user clicks the button, composer opens, and we advance.
					nextStep()
				} else {
					// Otherwise, ensure composer is closed.
					closeComposer()
				}
			} else if (step === 4) {
				// This is the composer step. If it's not open or doesn't have the
				// initial data yet, set it up. This prevents re-render loops.
				if (!isComposerOpen || !composerInitialData) {
					openComposer({
						prompt: "Coordinate with Kabeer to set up a meeting next week."
					})
				}
			}
			// For subsequent steps, ensure composer is closed.
			else if (step > 4 && isComposerOpen) {
				closeComposer()
			}
		}
	}, [
		isTourActive,
		step,
		isComposerOpen,
		nextStep,
		openComposer,
		closeComposer
	])

	useEffect(() => {
		const checkMobile = () => window.innerWidth < 768
		setIsMobile(checkMobile())

		// Open modal on mobile if a task is selected
		if (checkMobile() && selectedTaskId) {
			setIsModalOpen(true)
		}

		const handleResize = () => {
			const mobile = checkMobile()
			if (mobile !== isMobile) {
				setIsMobile(mobile)
			}
		}

		window.addEventListener("resize", handleResize)
		return () => window.removeEventListener("resize", handleResize)
	}, [isMobile, selectedTaskId])

	const selectedTask = useMemo(() => {
		return allTasks.find((t) => t.task_id === selectedTaskId) || null
	}, [allTasks, selectedTaskId])

	useEffect(() => {
		const taskId = searchParams.get("taskId")

		if (isMobile) {
			// Tour logic: Modal is open ONLY during step 5 AND when the phase is 'panel'.
			if (isTourActive && step === 5) {
				setIsModalOpen(phase === "panel")
			}
			// Regular logic: Open modal if a task is selected via URL and it exists.
			else if (taskId && selectedTask) {
				setIsModalOpen(true)
			}
			// Cleanup: If no task is selected (or tour ended), close the modal.
			else {
				setIsModalOpen(false)
			}
		} else {
			// On desktop, the modal is never used.
			setIsModalOpen(false)
		}
	}, [searchParams, isMobile, isTourActive, step, phase, selectedTask, handleClosePanel])

	const tasksWithDemo = useMemo(() => {
		// Filter(Boolean) removes null/undefined demo tasks
		return [demoTask, demoWorkflow, ...allTasks].filter(Boolean)
	}, [allTasks, demoTask, demoWorkflow])

	useEffect(() => {
		// When tour ends, refetch tasks
		if (!isTourActive && prevTourState?.isActive) {
			queryClient.invalidateQueries({ queryKey: ["tasksPageData"] })
		}
	}, [isTourActive, prevTourState?.isActive, queryClient])

	useEffect(() => {
		const handleBackendUpdate = () => {
			console.log(
				"Received tasksUpdatedFromBackend event, fetching tasks..."
			)
			toast.success("Task list updated from backend.")
			queryClient.invalidateQueries({ queryKey: ["tasksPageData"] })
		}
		window.addEventListener("tasksUpdatedFromBackend", handleBackendUpdate)
		return () => {
			window.removeEventListener(
				"tasksUpdatedFromBackend",
				handleBackendUpdate
			)
		}
	}, [queryClient])

	const useTaskMutation = (mutationFn, { successMessage, errorMessage }) => {
		return useMutation({
			mutationFn,
			onSuccess: () => {
				toast.success(successMessage)
				queryClient.invalidateQueries({ queryKey: ["tasksPageData"] })
			},
			onError: (error) => {
				toast.error(error.message || errorMessage)
			}
		})
	}

	const answerClarificationsMutation = useTaskMutation(
		({ taskId, answers }) =>
			fetch("/api/tasks/answer-clarifications", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ taskId, answers })
			}).then((res) => {
				if (!res.ok) throw new Error("Failed to submit answers.")
				return res.json()
			}),
		{
			successMessage: "Answers submitted. The task will now resume.",
			errorMessage: "Failed to submit answers."
		}
	)

	const answerLongFormClarificationMutation = useTaskMutation(
		({ taskId, requestId, answer }) =>
			fetch(`/api/tasks/${taskId}/answer-clarification`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ requestId, answer })
			}).then((res) => {
				if (!res.ok) throw new Error("Failed to submit answer.")
				return res.json()
			}),
		{
			successMessage: "Answer submitted. The task will now resume.",
			errorMessage: "Failed to submit answer."
		}
	)

	const createTaskMutation = useMutation({
		mutationFn: (payload) =>
			fetch("/api/tasks/add", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify(payload)
			}).then(async (res) => {
				if (!res.ok) {
					const errorData = await res.json().catch(() => ({}))
					const error = new Error(
						errorData.error || "Failed to add task"
					)
					error.status = res.status
					throw error
				}
				return res.json()
			}),
		onSuccess: (data) => {
			toast.success(
				data.message ||
					(view === "workflows"
						? "Workflow created!"
						: "Task created!")
			)
			queryClient.invalidateQueries({ queryKey: ["tasksPageData"] })
		},
		onError: (error) => {
			toast.error(`Error: ${error.message}`)
		}
	})

	const resumeTaskMutation = useTaskMutation(
		(taskId) =>
			fetch(`/api/tasks/${taskId}/action`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ action: "resume" })
			}).then((res) => {
				if (!res.ok) throw new Error("Failed to resume task.")
				return res.json()
			}),
		{
			successMessage: "Task resumed.",
			errorMessage: "Failed to resume task."
		}
	)

	const updateTaskMutation = useTaskMutation(
		(updatedTask) =>
			fetch("/api/tasks/update", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					...updatedTask,
					taskId: updatedTask.task_id
				})
			}).then((res) => {
				if (!res.ok) throw new Error("Failed to update task.")
				return res.json()
			}),
		{
			successMessage: "Task updated!",
			errorMessage: "Failed to update task."
		}
	)

	const deleteTaskMutation = useTaskMutation(
		(taskId) =>
			fetch(`/api/tasks/delete`, {
				method: "POST",
				body: JSON.stringify({ taskId }),
				headers: { "Content-Type": "application/json" }
			}).then((res) => {
				if (!res.ok) throw new Error("Failed to delete task.")
				return res.json()
			}),
		{
			successMessage: "Task deleted.",
			errorMessage: "Failed to delete task."
		}
	)

	const approveTaskMutation = useTaskMutation(
		(taskId) =>
			fetch(`/api/tasks/approve`, {
				method: "POST",
				body: JSON.stringify({ taskId }),
				headers: { "Content-Type": "application/json" }
			}).then((res) => {
				if (!res.ok) throw new Error("Failed to approve task.")
				return res.json()
			}),
		{
			successMessage: "Task approved.",
			errorMessage: "Failed to approve task."
		}
	)

	const rerunTaskMutation = useTaskMutation(
		(taskId) =>
			fetch("/api/tasks/rerun", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ taskId })
			}).then((res) => {
				if (!res.ok) throw new Error("Failed to re-run task.")
				return res.json()
			}),
		{
			successMessage: "Task re-run initiated.",
			errorMessage: "Failed to re-run task."
		}
	)

	const archiveTaskMutation = useTaskMutation(
		(taskId) =>
			fetch(`/api/tasks/update`, {
				method: "POST",
				body: JSON.stringify({ taskId, status: "archived" }),
				headers: { "Content-Type": "application/json" }
			}).then((res) => {
				if (!res.ok) throw new Error("Failed to archive task.")
				return res.json()
			}),
		{
			successMessage: "Task archived.",
			errorMessage: "Failed to archive task."
		}
	)

	const sendChatMessageMutation = useTaskMutation(
		({ taskId, message }) =>
			fetch(`/api/tasks/chat`, {
				method: "POST",
				body: JSON.stringify({ taskId, message }),
				headers: { "Content-Type": "application/json" }
			}).then((res) => {
				if (!res.ok) throw new Error("Failed to send message.")
				return res.json()
			}),
		{
			successMessage: "Message sent.",
			errorMessage: "Failed to send message."
		}
	)

	const selectedTaskOrDemo = useMemo(() => {
		// During the tour's task simulation step (step 5+), force the demo task data into the panel.
		if (isTourActive && step >= 5) {
			return demoTask
		}
		// Otherwise, use the task selected via the URL parameter for regular use.
		return selectedTask
	}, [isTourActive, step, demoTask, selectedTask])

	const handleAnswerClarifications = (taskId, answers) => {
		answerClarificationsMutation.mutate({ taskId, answers })
		handleClosePanel()
	}

	const handleAnswerLongFormClarification = (taskId, requestId, answer) => {
		answerLongFormClarificationMutation.mutate({
			taskId,
			requestId,
			answer
		})
	}

	const handleCreateTask = (payload) => {
		createTaskMutation.mutate(payload)
	}

	const handleResumeTask = (taskId) => {
		resumeTaskMutation.mutate(taskId)
	}

	const handleUpdateTask = (updatedTask) => {
		updateTaskMutation.mutate(updatedTask)
	}

	const handleDeleteTask = (taskId) => {
		deleteTaskMutation.mutate(taskId)
	}

	const handleApproveTask = (taskId) => {
		approveTaskMutation.mutate(taskId)
	}

	const handleRerunTask = (taskId) => {
		rerunTaskMutation.mutate(taskId)
	}

	const handleArchiveTask = (taskId) => {
		archiveTaskMutation.mutate(taskId)
	}

	const handleSendChatMessage = (taskId, message) => {
		sendChatMessageMutation.mutate({ taskId, message })
	}

	const handleSelectItem = (item) => {
		const taskId = item.task_id
		router.push(`/tasks?taskId=${taskId}`, { scroll: false })
		if (isMobile) {
			setIsModalOpen(true)
		}
	}

	const handleExampleClick = (example) => {
		if (example.type === "workflow") {
			setView("workflows")
		} else {
			setView("tasks")
		}
		openComposer(example)
	}

	const renderTaskDetails = (task) => (
		<TaskDetails
			task={task}
			allTools={allTools}
			integrations={integrations}
			onClose={handleClosePanel}
			onSave={handleUpdateTask}
			onAnswerClarifications={handleAnswerClarifications}
			onAnswerLongFormClarification={handleAnswerLongFormClarification}
			onResumeTask={handleResumeTask}
			onSelectTask={handleSelectItem}
			onDelete={handleDeleteTask}
			onApprove={handleApproveTask}
			onRerun={handleRerunTask}
			onArchiveTask={handleArchiveTask}
			onSendChatMessage={handleSendChatMessage}
		/>
	)

	return (
		<div className="flex-1 flex h-full text-white overflow-hidden">
			<Tooltip
				id="tasks-tooltip"
				place="right"
				style={{ zIndex: 9999 }}
			/>
			<div className="flex-1 flex overflow-hidden relative">
				<div className="absolute inset-0 z-[-1] network-grid-background">
					<InteractiveNetworkBackground />
				</div>
				{/* Main Content Panel */}
				<main className="flex-1 flex flex-col overflow-hidden relative md:pl-6 min-w-0">
					<div className="absolute -top-[250px] left-1/2 -translate-x-1/2 w-[800px] h-[500px] bg-brand-orange/10 rounded-full blur-3xl -z-10" />
					<header className="p-6 pt-20 md:pt-6 flex-shrink-0 flex items-center justify-between bg-transparent">
						<h1 className="text-3xl font-bold text-white">Tasks</h1>
						<div className="absolute top-6 left-1/2 -translate-x-1/2 z-10">
							<TaskViewSwitcher view={view} setView={setView} />
						</div>
					</header>

					<div
						className="flex-1 overflow-y-auto custom-scrollbar px-2 md:px-6 pb-24"
						style={{
							display: isMobile && isModalOpen ? "none" : "block"
						}}
						key="list-view-container"
					>
						{isLoading && !demoTask ? (
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
									className="h-full max-w-7xl mx-auto"
								>
									<ListView
										tasks={tasksWithDemo}
										view={view}
										onSelectTask={handleSelectItem}
										searchQuery={searchQuery}
										onSearchChange={setSearchQuery}
										onExampleClick={handleExampleClick}
									/>
								</motion.div>
							</AnimatePresence>
						)}
					</div>

					{/* Floating Task Composer */}
					<AnimatePresence>
						{isComposerOpen && (
							<TaskComposer
								view={view}
								onTaskCreated={(payload) => {
									if (isTourActive && step === 4) {
										closeComposer()
										nextStep() // Advance the tour
										return
									}
									// Otherwise, proceed with normal task creation.
									handleCreateTask(payload)
									closeComposer()
								}}
								onClose={() => {
									if (isTourActive && step === 4) {
										// If user closes manually, also advance the tour.
										nextStep()
									}
									closeComposer()
								}}
								initialData={composerInitialData}
							/>
						)}
					</AnimatePresence>

					{/* Floating Action Button */}
					<AnimatePresence>
						{!isComposerOpen && (
							<motion.div
								initial={{ opacity: 0, y: 50, scale: 0.8 }}
								animate={{ opacity: 1, y: 0, scale: 1 }}
								exit={{ opacity: 0, y: 50, scale: 0.8 }}
								transition={{
									duration: 0.3,
									ease: "easeInOut"
								}}
								className="absolute bottom-8 left-1/2 -translate-x-1/2 z-40"
							>
								<Button
									onClick={() => openComposer()}
			
									aria-label={
										view === "workflows"
											? "Create new workflow"
											: "Create new task"
									}
									data-tour-id="create-task-button"
									className="gap-2 rounded-xl px-6 py-3 font-semibold shadow-2xl transition-all duration-300 hover:scale-105 bg-brand-orange text-brand-black hover:bg-brand-orange/90"
								>
									<IconPlus size={20} />
									<span>
										{view === "workflows"
											? "Create Workflow"
											: "Create Task"}
									</span>
								</Button>
							</motion.div>
						)}
					</AnimatePresence>
				</main>

				{/* Desktop Drawer */}
				<Drawer
					isOpen={!isMobile && !!selectedTaskOrDemo}
					onClose={handleClosePanel}
				>
					{selectedTaskOrDemo &&
						renderTaskDetails(selectedTaskOrDemo)}
				</Drawer>
			</div>

			{/* Mobile Drawer */}
			<Drawer
				isOpen={isMobile && isModalOpen && !!selectedTaskOrDemo}
				onClose={handleClosePanel}
				side="bottom"
				className={cn(
					// Let the tour component control the overlay during the tour
					isTourActive && "!bg-transparent"
				)}
			>
				{selectedTaskOrDemo && renderTaskDetails(selectedTaskOrDemo)}
			</Drawer>
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
