"use client"

import React, { useState, useEffect, useRef } from "react"
import {
	IconPlayerPlay,
	IconRepeat,
	IconBolt,
	IconInfoCircle,
	IconSparkles,
	IconMail,
	IconCalendarEvent
} from "@tabler/icons-react"
import { motion, AnimatePresence } from "framer-motion"
import { cn } from "@utils/cn"
import toast from "react-hot-toast"
import useClickOutside from "@hooks/useClickOutside"
import { TextLoop } from "@components/ui/TextLoop"

import { Button } from "@components/ui/button"
import { Textarea } from "@components/ui/textarea"
import { Select } from "@components/ui/select"
const workflowTabs = [
	{
		id: "recurring",
		label: "Recurring",
		icon: <IconRepeat size={16} />
	},
	{
		id: "triggered",
		label: "Triggered",
		icon: <IconBolt size={16} />
	}
]

const tasksPlaceholders = [
	"Draft a follow-up email to the client about the new proposal.",
	"Find the top 3 restaurants near me for a team lunch tomorrow.",
	"Schedule a meeting with John for next Tuesday.",
	"Every morning, summarize my unread emails from the past 24 hours.",
	"Read Leads List Sheet from Google Drive and mail each one.",
	"Research 20 different topics and prepare a report on each one."
]

const triggers = [
	{
		id: "gmail",
		label: "New Email in Gmail",
		icon: <IconMail size={20} />,
		source: "gmail",
		event: "new_email"
	},
	{
		id: "gcalendar",
		label: "New Google Calendar Event",
		icon: <IconCalendarEvent size={20} />,
		source: "gcalendar",
		event: "new_event"
	}
]

const TaskComposer = ({
	view,
	onTaskCreated,
	onClose,
	initialData
}) => {
	const [goalInput, setGoalInput] = useState("")
	const [workflowTab, setWorkflowTab] = useState("recurring")
	const [autoApprove, setAutoApprove] = useState(false)

	// State for recurring workflows
	const [recurringFrequency, setRecurringFrequency] = useState("daily")
	const [recurringDays, setRecurringDays] = useState([])
	const [recurringTime, setRecurringTime] = useState("09:00")

	// State for triggered workflows
	const textareaRef = useRef(null)
	const [selectedTrigger, setSelectedTrigger] = useState(null)

	const composerRef = useRef(null)
	useClickOutside(composerRef, onClose)

	useEffect(() => {
		if (initialData) {
			setGoalInput(initialData.prompt || "")
			if (initialData.type === "workflow" && initialData.params) {
				setWorkflowTab(initialData.params.workflowTab || "recurring")
				setRecurringFrequency(
					initialData.params.recurringFrequency || "daily"
				)
				setRecurringTime(initialData.params.recurringTime || "09:00")
				setRecurringDays(initialData.params.recurringDays || [])
			}
		} else {
			// Reset all state for a fresh composer instance
			setGoalInput("")
			setWorkflowTab("recurring")
			setRecurringFrequency("daily")
			setRecurringDays([])
			setRecurringTime("09:00")
			setSelectedTrigger(null)
		}
	}, [initialData])

	const handleDayToggle = (day) => {
		setRecurringDays((prev) =>
			prev.includes(day) ? prev.filter((d) => d !== day) : [...prev, day]
		)
	}

	const handleGoalInputChange = (e) => {
		setGoalInput(e.target.value)
		// Auto-resize logic
		if (textareaRef.current) {
			textareaRef.current.style.height = "auto"
			const newHeight = Math.min(textareaRef.current.scrollHeight, 150) // Max height 150px
			textareaRef.current.style.height = `${newHeight}px`
		}
	}

	const handleCreateTask = async () => {
		if (!goalInput.trim()) {
			toast.error("Please describe the task you want to create.")
			return
		}

		let prompt = goalInput.trim()
		if (view === "workflows") {
			if (workflowTab === "recurring") {
				if (
					recurringFrequency === "weekly" &&
					recurringDays.length === 0
				) {
					toast.error(
						"Please select at least one day for a weekly workflow."
					)
					return
				}
				const dayPart =
					recurringFrequency === "weekly"
						? ` on ${recurringDays.join(", ")}`
						: ""
				prompt = `${prompt} This is a recurring task. It should run ${recurringFrequency}${dayPart} at ${recurringTime}.`
			} else if (workflowTab === "triggered") {
				if (!selectedTrigger) {
					toast.error("Please select a trigger for the workflow.")
					return
				}
				prompt = `When a ${selectedTrigger.label}, do the following: ${prompt}`
			}
		}

		const payload = { prompt, auto_approve_subtasks: autoApprove }

		onTaskCreated(payload)
		setGoalInput("") // Clear input after creation
	}

	const Switch = ({ checked, onChange }) => (
		<button
			type="button"
			onClick={() => onChange(!checked)}
			className={cn(
				"relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-brand-orange focus:ring-offset-2 focus:ring-offset-neutral-900",
				checked ? "bg-brand-orange" : "bg-neutral-700"
			)}
			aria-pressed={checked}
		>
			<motion.span
				animate={{ x: checked ? "100%" : "0%" }}
				transition={{ type: "spring", stiffness: 700, damping: 30 }}
				className="pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out"
			/>
		</button>
	)

	return (
		<motion.div
			ref={composerRef}
			initial={{ opacity: 0, y: 100, scale: 0.9 }}
			animate={{ opacity: 1, y: 0, scale: 1 }}
			exit={{ opacity: 0, y: 100, scale: 0.9 }}
			transition={{ type: "spring", stiffness: 300, damping: 30 }}
			data-tour-id="task-composer"
			className="absolute bottom-8 w-[90vw] max-w-3xl left-1/2 -translate-x-1/2 bg-neutral-900/50 backdrop-blur-lg border border-neutral-700 rounded-2xl shadow-2xl z-50"
		>
			<div className="relative p-4">
				{view === "tasks" ? (
					<div className="space-y-3">
						<div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3">
							<div className="relative flex-1">
								<Textarea
									ref={textareaRef}
									value={goalInput}
									onChange={handleGoalInputChange}
									placeholder=" "
									onKeyDown={(e) => {
										if (e.key === "Enter" && !e.shiftKey) {
											e.preventDefault()
											handleCreateTask()
										}
									}}
									className="p-3 bg-transparent resize-none custom-scrollbar relative z-10 h-20 sm:h-12"
									style={{ maxHeight: "150px" }}
								/>
								{!goalInput && (
									<div className="absolute top-1/2 left-4 -translate-y-1/2 text-neutral-500 pointer-events-none z-0 overflow-hidden">
										<TextLoop>
											{tasksPlaceholders.map((p) => (
												<span key={p}>{p}</span>
											))}
										</TextLoop>
									</div>
								)}
							</div>
							<Button
								onClick={handleCreateTask}
								data-tour-id="task-composer-create-button"
								className="w-full sm:w-auto bg-brand-orange hover:bg-brand-orange/90 text-brand-black font-semibold"
							>
								Create Task
							</Button>
						</div>
						<div className="flex items-center gap-2 text-xs text-neutral-400 pl-1">
							<Switch
								checked={autoApprove}
								onChange={setAutoApprove}
							/>
							<label
								onClick={() => setAutoApprove(!autoApprove)}
								className="cursor-pointer select-none"
							>
								Auto-approve sub-tasks
							</label>
							<IconInfoCircle
								size={14}
								data-tooltip-id="tasks-tooltip"
								data-tooltip-content="When enabled, the AI will automatically approve and run the sub-tasks it creates to achieve your goal."
							/>
						</div>
					</div>
				) : (
					<div className="space-y-3">
						<div className="flex items-center">
							<div className="flex items-center gap-1 bg-neutral-800 p-1 rounded-lg">
								{workflowTabs.map((tab) => (
									<button
										key={tab.id}
										onClick={() => setWorkflowTab(tab.id)}
										className={cn(
											"flex items-center gap-2 px-3 py-1 rounded-md text-sm font-medium transition-colors",
											workflowTab === tab.id
												? "bg-neutral-700 text-white"
												: "text-neutral-400 hover:bg-neutral-700/50"
										)}
									>
										{tab.icon}
										{tab.label}
									</button>
								))}
							</div>
						</div>

						<AnimatePresence mode="wait">
							<motion.div
								key={workflowTab}
								initial={{ opacity: 0, y: 10 }}
								animate={{ opacity: 1, y: 0 }}
								exit={{ opacity: 0, y: -10 }}
								transition={{ duration: 0.2 }}
								className="space-y-3"
							>
								{workflowTab === "recurring" && (
									<div className="grid grid-cols-1 md:grid-cols-3 gap-3">
										<Select
											value={recurringFrequency}
											onChange={(e) =>
												setRecurringFrequency(
													e.target.value
												)
											}
											className="w-full p-2 h-auto"
										>
											<option value="daily">Daily</option>
											<option value="weekly">
												Weekly
											</option>
										</Select>
										{recurringFrequency === "weekly" && (
											<div className="flex gap-1 bg-neutral-800 border border-neutral-700 rounded-md p-1 md:col-span-2">
												{[
													"Mon",
													"Tue",
													"Wed",
													"Thu",
													"Fri",
													"Sat",
													"Sun"
												].map((day) => (
													<button
														key={day}
														onClick={() =>
															handleDayToggle(day)
														}
														className={cn(
															"p-1.5 rounded text-xs w-full font-semibold",
															recurringDays.includes(
																day
															)
																? "bg-brand-orange text-black"
																: "hover:bg-neutral-700"
														)}
													>
														{day}
													</button>
												))}
											</div>
										)}
										<input
											type="time"
											value={recurringTime}
											onChange={(e) =>
												setRecurringTime(e.target.value)
											}
											className="w-full p-2 bg-neutral-800 border border-neutral-700 rounded-md h-10"
										/>
									</div>
								)}

								{workflowTab === "triggered" && (
									<div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
										{triggers.map((trigger) => (
											<button
												key={trigger.id}
												onClick={() =>
													setSelectedTrigger(trigger)
												}
												className={cn(
													"flex items-center gap-3 text-left p-3 rounded-md border-2 transition-all",
													selectedTrigger?.id ===
														trigger.id
														? "bg-brand-orange/10 border-brand-orange text-white"
														: "bg-neutral-800 border-transparent hover:border-neutral-600 text-neutral-300"
												)}
											>
												<span
													className={cn(
														"flex-shrink-0",
														selectedTrigger?.id ===
															trigger.id
															? "text-brand-orange"
															: "text-neutral-400"
													)}
												>
													{trigger.icon}
												</span>
												<span className="font-semibold text-sm">
													{trigger.label}
												</span>
											</button>
										))}
									</div>
								)}

								<Textarea
									ref={textareaRef}
									value={goalInput}
									onChange={handleGoalInputChange}
									placeholder="Describe the goal of the workflow..."
									className="w-full p-3 resize-none custom-scrollbar"
									rows={2}
									style={{ maxHeight: "150px" }}
								/>
							</motion.div>
						</AnimatePresence>
						<div className="flex justify-end pt-2">
							<Button
								onClick={handleCreateTask}
								data-tour-id="task-composer-create-button"
								className="bg-brand-orange hover:bg-brand-orange/90 text-brand-black font-semibold text-sm"
							>
								{view === "workflows"
									? "Create Workflow"
									: "Create Task"}
							</Button>
						</div>
					</div>
				)}
			</div>
		</motion.div>
	)
}

export default TaskComposer
