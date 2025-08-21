"use client"
import React from "react"
import { motion } from "framer-motion"
import { IconUsersGroup, IconClock } from "@tabler/icons-react"
import { cn } from "@utils/cn"
import { taskStatusColors, priorityMap } from "./constants"
import { format, isToday } from "date-fns"
import { BorderTrail } from "@components/ui/border-trail"
import { getDisplayName } from "@utils/taskUtils"

const StatusBadge = ({ status }) => {
	const statusInfo = taskStatusColors[status] || taskStatusColors.default

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

const TaskCardList = ({ task, onSelectTask }) => {
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
			onClick={() => onSelectTask(task)}
			className="bg-neutral-900/50 p-4 rounded-lg border border-zinc-700 hover:border-brand-orange transition-all cursor-pointer relative"
		>
			{inProgress && (
				<BorderTrail size={80} className="bg-brand-yellow" />
			)}
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
							<IconClock size={16} className="text-purple-400" />
						</span>
					)}
					{getDisplayName(task)}
				</p>
				<StatusBadge status={task.status} />
			</div>
			<div className="flex items-center justify-between mt-3 pt-3 border-t border-neutral-800 text-xs text-neutral-400 font-mono">
				{dateText && <span>{dateText}</span>}
			</div>
		</motion.div>
	)
}

export default TaskCardList
