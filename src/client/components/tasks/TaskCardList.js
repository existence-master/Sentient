"use client"
import React, { useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import {
	IconUsersGroup,
	IconClock,
	IconChevronDown,
	IconChevronUp,
	IconSubtask
} from "@tabler/icons-react"
import { cn } from "@utils/cn"
import { taskStatusColors, priorityMap } from "./constants"
import { format, isToday } from "date-fns"
import { BorderTrail } from "@components/ui/border-trail"
import { getDisplayName } from "@utils/taskUtils"

const StatusBadge = ({ status, taskType, orchestratorState }) => {
	let displayStatus = status
	if (taskType === "long_form" && orchestratorState) {
		// Map orchestrator state to a display status that taskStatusColors can understand
		const state = orchestratorState.toLowerCase()
		switch (state) {
			case "created":
			case "planning":
				displayStatus = "planning"
				break
			case "active": // A waiting task is still actively processing/monitoring
				displayStatus = "processing"
				break
			case "waiting":
				displayStatus = "waiting"
				break
			case "suspended":
				// Suspended means it's waiting for user input.
				displayStatus = "clarification_pending"
				break
			case "failed":
				displayStatus = "error"
				break
			case "completed":
				displayStatus = "completed"
				break
			default:
				displayStatus = state.toLowerCase()
		}
	}
	const statusInfo =
		taskStatusColors[displayStatus] || taskStatusColors.default

	return (
		<div
			className={cn(
				"px-2 py-0.5 rounded-full text-xs font-medium flex items-center gap-1.5",
				statusInfo.bgColor,
				statusInfo.textColor
			)}
		>
			<statusInfo.icon size={12} />
			{statusInfo.label}
		</div>
	)
}

const SubTaskItem = ({ task, onSelectTask }) => {
	const statusInfo = taskStatusColors[task.status] || taskStatusColors.default
	return (
		<div
			onClick={(e) => {
				e.stopPropagation()
				onSelectTask(task)
			}}
			className="flex items-center justify-between p-2 mx-2 rounded-md hover:bg-neutral-800 cursor-pointer transition-colors group"
		>
			<div className="flex items-center gap-2 overflow-hidden">
				<IconSubtask
					size={14}
					className="text-neutral-500 group-hover:text-neutral-300 flex-shrink-0"
				/>
				<p className="text-xs text-neutral-300 font-sans truncate">
					{getDisplayName(task)}
				</p>
			</div>
			<div
				className={cn(
					"px-1.5 py-0.5 rounded-full text-[10px] font-medium flex items-center gap-1 flex-shrink-0",
					statusInfo.bgColor,
					statusInfo.textColor
				)}
			>
				<statusInfo.icon size={10} />
				<span>{statusInfo.label}</span>
			</div>
		</div>
	)
}

const TaskCardList = ({ task, onSelectTask }) => {
	const [isExpanded, setIsExpanded] = useState(false)
	const subTasks = task.subTasks || []
	const hasSubTasks = subTasks && subTasks.length > 0

	let dateText = ""
	if (task.scheduled_date) {
		try {
			const date = new Date(task.scheduled_date) // Ensure it's a Date object
			dateText = format(date, "MMM d")
		} catch (e) {
			// ignore invalid date
		}
	}

	const inProgress = [
		"processing",
		"planning",
		"clarification_answered"
	].includes(task.status)

	const cardVariants = {
		hidden: { opacity: 0, y: -20, scale: 0.95 },
		visible: { opacity: 1, y: 0, scale: 1 }
	}

	return (
		<motion.div
			layout
			variants={cardVariants}
			exit={{ opacity: 0, transition: { duration: 0.1 } }}
			className="bg-neutral-900/50 rounded-lg border border-zinc-700 hover:border-brand-orange/60 transition-all relative"
		>
			{inProgress && (
				<BorderTrail size={80} className="bg-brand-yellow" />
			)}
			<div
				onClick={() => onSelectTask(task)}
				className="p-4 cursor-pointer"
			>
				<div className="flex bg-transparent p-1 transition-all justify-between items-start gap-4">
					<p className="font-sans font-semibold text-brand-white flex-1 text-sm line-clamp-2 flex items-center gap-2">
						{task.task_type === "swarm" && (
							<span
								data-tooltip-id="tasks-tooltip"
								data-tooltip-content="Swarm Task"
							>
								<IconUsersGroup
									size={16}
									className="text-blue-400"
								/>
							</span>
						)}
						{task.task_type === "long_form" && (
							<span
								data-tooltip-id="tasks-tooltip"
								data-tooltip-content="Long-Form Task"
							>
								<IconClock
									size={16}
									className="text-purple-400"
								/>
							</span>
						)}
						{getDisplayName(task)}
					</p>
					<StatusBadge
						status={task.status}
						taskType={task.task_type}
						orchestratorState={
							task.orchestrator_state?.current_state
						}
					/>
				</div>
				<div className="flex items-center justify-between mt-3 pt-3 border-t border-neutral-800 text-xs text-neutral-400 font-mono">
					{dateText ? <span>{dateText}</span> : <span />}
					{hasSubTasks && (
						<button
							onClick={(e) => {
								e.stopPropagation()
								setIsExpanded(!isExpanded)
							}}
							className="flex items-center gap-1 text-xs text-neutral-500 hover:text-white z-10 relative"
						>
							{isExpanded ? (
								<IconChevronUp size={14} />
							) : (
								<IconChevronDown size={14} />
							)}
							<span>
								{subTasks.length} Sub-task
								{subTasks.length > 1 ? "s" : ""}
							</span>
						</button>
					)}
				</div>
			</div>
			<AnimatePresence>
				{isExpanded && hasSubTasks && (
					<motion.div
						initial={{ opacity: 0, height: 0 }}
						animate={{ opacity: 1, height: "auto" }}
						exit={{ opacity: 0, height: 0 }}
						transition={{ duration: 0.2 }}
						className="pb-2"
					>
						{subTasks.map((subTask) => (
							<SubTaskItem
								key={subTask.task_id}
								task={subTask}
								onSelectTask={onSelectTask}
							/>
						))}
					</motion.div>
				)}
			</AnimatePresence>
		</motion.div>
	)
}

export default TaskCardList
