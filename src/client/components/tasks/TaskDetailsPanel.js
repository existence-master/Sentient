"use client"

import React, { useState, useEffect, useMemo } from "react"
import {
	IconX,
	IconPencil,
	IconTrash,
	IconRepeat,
	IconDeviceFloppy,
	IconSquareX,
	IconArchive,
	IconCircleCheck,
	IconPlayerPlay,
	IconClock,
	IconClipboardList,
	IconUsersGroup
} from "@tabler/icons-react"
import { getDisplayName } from "@utils/taskUtils"
import RecurringTaskDetails from "./RecurringTaskDetails"
import ConnectToolButton from "./ConnectToolButton"
import { cn } from "@utils/cn"
import TaskDetailsContent from "./TaskDetailsContent"
import TriggeredTaskDetails from "./TriggeredTaskDetails"

const TaskDetailsPanel = ({
	task,
	allTools = [],
	integrations,
	onClose,
	onSave,
	onApprove,
	onDelete,
	onRerun,
	onArchiveTask,
	className,
	onSendChatMessage,
	onAnswerClarifications,
	onAnswerLongFormClarification,
	onSelectTask,
	onResumeTask
}) => {
	const [isEditing, setIsEditing] = useState(false)
	const [editableTask, setEditableTask] = useState(task)
	const scheduleType = task?.schedule?.type

	const missingTools = useMemo(() => {
		if (!task || !integrations) {
			return []
		}

		// The plan is in the latest run. Fallback to top-level for legacy tasks.
		const plan =
			task.runs && task.runs.length > 0
				? task.runs[task.runs.length - 1].plan
				: task.plan

		if (!plan || plan.length === 0) {
			return []
		}

		// FIX: Filter out any null, undefined, or empty tool names from the plan
		// to prevent errors when rendering the missing tools buttons.
		const requiredTools = new Set(
			plan.map((step) => step.tool).filter(Boolean)
		)

		const connectedTools = new Set(
			integrations
				.filter((i) => i.connected || i.auth_type === "builtin")
				.map((i) => i.name)
		)

		const missing = []
		for (const tool of requiredTools) {
			if (!connectedTools.has(tool)) {
				const toolDetails = integrations.find((i) => i.name === tool)
				missing.push({
					name: tool,
					displayName: toolDetails?.display_name || tool
				})
			}
		}
		return missing
	}, [task, integrations])

	useEffect(() => {
		setEditableTask(task)
		if (!task) {
			setIsEditing(false) // Reset editing state when task is closed/changed
		}
	}, [task])

	const handleStartEditing = () => {
		// When editing, we need to make sure we're editing the plan from the latest run.
		const latestRun =
			task.runs && task.runs.length > 0
				? task.runs[task.runs.length - 1]
				: {}
		const planForEditing = latestRun.plan || task.plan || [] // Fallback for different structures

		setEditableTask({ ...task, plan: planForEditing })
		setIsEditing(true)
	}

	const handleFieldChange = (field, value) =>
		setEditableTask((prev) => ({ ...prev, [field]: value }))
	const handleScheduleChange = (newSchedule) =>
		setEditableTask((prev) => ({ ...prev, schedule: newSchedule }))
	const handleAddStep = () =>
		setEditableTask((prev) => ({
			...prev,
			plan: [...(prev.plan || []), { tool: "", description: "" }]
		}))
	const handleRemoveStep = (index) =>
		setEditableTask((prev) => ({
			...prev,
			plan: prev.plan.filter((_, i) => i !== index)
		}))
	const handleStepChange = (index, field, value) =>
		setEditableTask((prev) => ({
			...prev,
			plan: prev.plan.map((step, i) =>
				i === index ? { ...step, [field]: value } : step
			)
		}))
	const handleSaveEdit = () => {
		onSave(editableTask)
		setIsEditing(false)
	}

	const ActionButton = ({
		onClick,
		icon,
		children,
		className = "",
		disabled = false
	}) => (
		<button
			onClick={onClick}
			disabled={disabled}
			className={cn(
				"flex items-center gap-2 text-sm px-3 py-2 rounded-lg transition-colors",
				className
			)}
		>
			{icon}
			{children}
		</button>
	)

	return (
		<aside
			className={cn(
				"w-full h-full bg-brand-black backdrop-blur-xl shadow-2xl md:border-l border-neutral-700/80 flex flex-col flex-shrink-0",
				className
			)}
		>
			{!task ? (
				<div className="flex flex-col items-center justify-center h-full text-center text-neutral-500 p-8">
					<IconClipboardList size={48} className="mb-4" />
					<h3 className="text-lg font-semibold text-neutral-400">
						Select a Task
					</h3>
					<p className="max-w-xs">
						Choose a task from the list to see its details, plan,
						and outcome here.
					</p>
				</div>
			) : (
				<>
					{/* --- HEADER --- */}
					<header className="flex items-start justify-between p-6 border-b border-neutral-700/50 flex-shrink-0">
						<div className="flex-1 pr-4">
							{isEditing ? (
								<input
									type="text"
									value={editableTask.description}
									onChange={(e) =>
										handleFieldChange(
											"description",
											e.target.value
										)
									}
									className="w-full bg-transparent text-2xl font-bold text-white focus:ring-0 focus:border-brand-orange border-b-2 border-transparent"
								/>
							) : (
								<h2 className="text-lg md:text-xl font-bold text-white leading-snug flex items-center gap-2">
									{task.task_type === "swarm" && (
										<span className="p-1.5 bg-blue-500/20 text-blue-300 rounded-md">
											<IconUsersGroup size={20} />
										</span>
									)}
									{task.task_type === "long_form" && (
										<span className="p-1.5 bg-purple-500/20 text-purple-300 rounded-md">
											<IconClock size={20} />
										</span>
									)}
									{getDisplayName(task)}
								</h2>
							)}
						</div>
						<button
							onClick={onClose}
							className="ml-4 p-2 rounded-full text-neutral-400 hover:bg-neutral-700 hover:text-white"
						>
							<IconX size={20} />
						</button>
					</header>

					{/* --- CONTENT --- */}
					<main className="flex-1 overflow-y-auto custom-scrollbar p-6">
						{isEditing ? (
							<TaskDetailsContent // Use the one-time editor for all types for now
								task={task}
								isEditing={isEditing}
								editableTask={editableTask}
								handleFieldChange={handleFieldChange}
								handleScheduleChange={handleScheduleChange}
								handleAddStep={handleAddStep}
								handleRemoveStep={handleRemoveStep}
								handleStepChange={handleStepChange}
								allTools={allTools}
								integrations={integrations}
							/>
						) : scheduleType === "recurring" ? (
							<RecurringTaskDetails
								task={task}
								onAnswerClarifications={onAnswerClarifications}
							/>
						) : scheduleType === "triggered" ? (
							<TriggeredTaskDetails task={task} />
						) : (
							<TaskDetailsContent
								task={task}
								onSendChatMessage={onSendChatMessage}
								onAnswerClarifications={onAnswerClarifications}
								onAnswerLongFormClarification={
									onAnswerLongFormClarification
								}
								onSelectTask={onSelectTask}
							/>
						)}
					</main>

					{/* --- FOOTER --- */}
					<footer className="p-4 border-t border-neutral-700/50 flex-shrink-0 bg-brand-gray/50">
						{isEditing ? (
							<>
								<div className="flex items-center gap-2">
									{/* Empty div for spacing */}
								</div>
								<div className="flex items-center gap-2">
									<ActionButton
										onClick={() => setIsEditing(false)}
										icon={<IconSquareX size={16} />}
										className="bg-neutral-700 text-neutral-200 hover:bg-neutral-600"
									>
										Cancel
									</ActionButton>
									<ActionButton
										onClick={handleSaveEdit}
										icon={<IconDeviceFloppy size={16} />}
										className="bg-brand-orange text-brand-black font-semibold hover:bg-brand-orange/90"
									>
										Save
									</ActionButton>
								</div>
							</>
						) : (
							<div className="flex flex-col gap-4">
								{/* Top section with minor actions and warnings */}
								<div className="flex justify-between items-start gap-2">
									<div className="flex items-center gap-2">
										<ActionButton
											onClick={handleStartEditing}
											icon={<IconPencil size={16} />}
											className="text-neutral-400 hover:bg-neutral-700 hover:text-white"
										>
											Edit
										</ActionButton>
										<ActionButton
											onClick={() =>
												onDelete(task.task_id)
											}
											icon={<IconTrash size={16} />} // prettier-ignore
											className="text-neutral-400 hover:bg-red-500/20 hover:text-red-400"
										>
											Delete
										</ActionButton>
										<ActionButton
											onClick={() =>
												onRerun(task.task_id)
											}
											icon={<IconRepeat size={16} />}
											className="text-neutral-400 hover:bg-neutral-700 hover:text-white"
										>
											Rerun
										</ActionButton>
									</div>
									{missingTools.length > 0 && (
										<div className="text-xs text-red-400 flex flex-col items-end gap-1 text-right">
											<span>
												Connect tools to approve:
											</span>
											<div className="flex gap-2 flex-wrap justify-end">
												{missingTools.map((tool) => (
													<ConnectToolButton
														key={tool.name}
														toolName={
															tool.displayName
														}
													/>
												))}
											</div>
										</div>
									)}
								</div>

								{/* Main action buttons */}
								<div className="flex flex-col sm:flex-row gap-2 w-full">
									{task.status === "approval_pending" && (
										<ActionButton
											onClick={() =>
												onApprove(task.task_id)
											}
											icon={<IconCircleCheck size={16} />}
											className="bg-green-600 text-white hover:bg-green-500 disabled:opacity-50 disabled:cursor-not-allowed w-full sm:w-auto flex-grow justify-center"
											disabled={missingTools.length > 0}
										>
											Approve & Run
										</ActionButton>
									)}
									{task.task_type === "long_form" &&
										task.orchestrator_state?.current_state ===
											"WAITING" && (
											<ActionButton
												onClick={() =>
													onResumeTask(task.task_id)
												}
												icon={<IconPlayerPlay size={16} />}
												className="bg-blue-600 text-white hover:bg-blue-500 w-full sm:w-auto flex-grow justify-center"
											>
												Resume Now
											</ActionButton>
									)}
									<ActionButton
										onClick={() =>
											onArchiveTask(task.task_id)
										}
										icon={<IconArchive size={16} />}
										className="bg-neutral-700 text-neutral-200 hover:bg-neutral-600 w-full sm:w-auto justify-center"
									>
										Archive
									</ActionButton>
								</div>
							</div>
						)}
					</footer>
				</>
			)}
		</aside>
	)
}

export default TaskDetailsPanel
