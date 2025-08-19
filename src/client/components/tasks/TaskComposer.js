"use client"

import React, { useState, useEffect } from "react"
import {
	IconPlayerPlay,
	IconRepeat,
	IconBolt,
	IconUsersGroup,
	IconX,
	IconSparkles,
	IconMail,
	IconCalendarEvent
} from "@tabler/icons-react"
import { motion, AnimatePresence } from "framer-motion"
import { cn } from "@utils/cn"
import toast from "react-hot-toast"
import { format } from "date-fns"
import { TextLoop } from "@components/ui/TextLoop"

const tabs = [
	{
		id: "once",
		label: "Run Once",
		description: "For tasks that happen now or at a specific future time.",
		icon: <IconPlayerPlay size={18} />,
		isProFeature: false
	},
	{
		id: "recurring",
		label: "Recurring",
		description:
			"For tasks that repeat on a schedule (e.g., daily, weekly).",
		icon: <IconRepeat size={18} />,
		isProFeature: false // Available on free plan with limits
	},
	{
		id: "triggered",
		label: "Triggered",
		description: "Automate actions based on events from connected apps.",
		icon: <IconBolt size={18} />,
		isProFeature: false // Available on free plan with limits
	},
	{
		id: "swarm",
		label: "Swarm",
		description:
			"For complex projects that require a larger team of AI workers.",
		icon: <IconUsersGroup size={18} />,
		isProFeature: true // Not available on free plan
	}
]

const oncePlaceholders = [
	"Draft a follow-up email to the client about the new proposal.",
	"Find the top 3 restaurants near me for a team lunch tomorrow.",
	"Schedule a meeting with John for next Tuesday."
]

const recurringPlaceholders = [
	"Every morning, summarize my unread emails from the past 24 hours.",
	"On Fridays, create a report of this week's completed tasks.",
	"On the 1st of every month, draft an invoice for Client X.",
	"Every weekday at 5 PM, give me a summary of my day."
]

const swarmPlaceholders = [
	"Read Leads List Sheet from Google Drive and mail each one.",
	"Research 20 different topics and prepare a report on each one.",
	"Research all the different case-laws mentioned in this list."
]

// Plan limits mirrored from src/server/main/plans.py
const PLAN_LIMITS = {
	free: {
		recurring_tasks_active: 2,
		triggered_tasks_active: 1
	}
}

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
	onTaskCreated,
	isPro,
	plan, // 'free', 'pro', or 'selfhost'
	recurringTaskCount,
	triggeredTaskCount,
	onUpgradeClick,
	onClose,
	composerData
}) => {
	const [activeTab, setActiveTab] = useState("once")
	const [goalInput, setGoalInput] = useState("")

	// State for each tab
	const [runOnceType, setRunOnceType] = useState("now")
	const [runOnceDateTime, setRunOnceDateTime] = useState("")
	const [recurringFrequency, setRecurringFrequency] = useState("daily")
	const [recurringDays, setRecurringDays] = useState([])
	const [recurringTime, setRecurringTime] = useState("09:00")
	const [selectedTrigger, setSelectedTrigger] = useState(null)

	useEffect(() => {
		const defaultDate = composerData?.defaultDate
		if (defaultDate) {
			// Set the composer to the correct state for a scheduled task
			setActiveTab("once")
			setRunOnceType("later")

			// Format the date and set a default time (e.g., 09:00)
			const formattedDateTime = `${format(
				defaultDate,
				"yyyy-MM-dd"
			)}T09:00`
			setRunOnceDateTime(formattedDateTime)
		}
	}, [composerData])

	useEffect(() => {
		setGoalInput("")
		setSelectedTrigger(null)
	}, [activeTab])

	const handleDayToggle = (day) => {
		setRecurringDays((prev) =>
			prev.includes(day) ? prev.filter((d) => d !== day) : [...prev, day]
		)
	}

	const getRecurringSummary = () => {
		if (recurringFrequency === "daily") {
			return `Runs daily at ${recurringTime}.`
		}
		if (recurringFrequency === "weekly") {
			if (recurringDays.length === 0) {
				return "Select days of the week to see a summary."
			}
			const dayOrder = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
			const sortedDays = dayOrder.filter((day) =>
				recurringDays.includes(day)
			)
			return `Runs weekly on ${sortedDays.join(
				", "
			)} at ${recurringTime}.`
		}
		return ""
	}

	const handleCreateTask = async () => {
		let payload = {}

		if (activeTab === "swarm") {
			if (!isPro) {
				onUpgradeClick()
				return
			}
			payload = { prompt: goalInput.trim(), task_type: "swarm" }
		} else {
			if (!goalInput.trim()) {
				toast.error("Please provide a goal for the task.")
				return
			}

			payload = {
				prompt: goalInput.trim(),
				task_type: "single",
				schedule: {}
			}

			if (activeTab === "once") {
				payload.schedule.type = "once"
				if (runOnceType === "later") {
					if (!runOnceDateTime) {
						toast.error(
							"Please select a date and time for the scheduled task."
						)
						return
					}
					payload.schedule.run_at = new Date(
						runOnceDateTime
					).toISOString()
				} else {
					payload.schedule.run_at = null
				}
			} else if (activeTab === "recurring") {
				if (
					plan === "free" &&
					recurringTaskCount >=
						PLAN_LIMITS.free.recurring_tasks_active
				) {
					toast.error(
						`You've reached your limit of ${PLAN_LIMITS.free.recurring_tasks_active} active recurring tasks. Please upgrade to Pro for more.`
					)
					onUpgradeClick()
					return
				}
				payload.schedule.type = "recurring"
				payload.schedule.frequency = recurringFrequency
				payload.schedule.time = recurringTime
				if (recurringFrequency === "weekly") {
					if (recurringDays.length === 0) {
						toast.error(
							"Please select at least one day for the weekly recurring task."
						)
						return
					}
					payload.schedule.days = recurringDays
				}
			} else if (activeTab === "triggered") {
				if (
					plan === "free" &&
					triggeredTaskCount >=
						PLAN_LIMITS.free.triggered_tasks_active
				) {
					toast.error(
						`You've reached your limit of ${PLAN_LIMITS.free.triggered_tasks_active} active triggered task. Please upgrade to Pro for more.`
					)
					onUpgradeClick()
					return
				}
				if (!selectedTrigger) {
					toast.error("Please select a trigger for the workflow.")
					return
				}
				if (!goalInput.trim()) {
					toast.error(
						"Please describe what the task should do when triggered."
					)
					return
				}
				payload.schedule.type = "triggered"
				payload.schedule.source = selectedTrigger.source
				payload.schedule.event = selectedTrigger.event
				payload.schedule.filter = {}
			}
		}

		onTaskCreated(payload)
		setGoalInput("") // Clear input after creation
		setSelectedTrigger(null)
	}

	const currentTabInfo = tabs.find((t) => t.id === activeTab)

	return (
		<div className="flex flex-col h-full bg-neutral-900/80 backdrop-blur-lg text-white">
			<header className="flex items-center justify-between p-4 border-b border-neutral-800 flex-shrink-0">
				<h2 className="text-lg font-semibold flex items-center gap-2">
					<IconSparkles className="text-brand-orange" />
					<span>Task Composer</span>
				</h2>
				{onClose && (
					<button
						onClick={onClose}
						className="p-1.5 rounded-full text-neutral-400 hover:bg-neutral-700 hover:text-white"
					>
						<IconX size={18} />
					</button>
				)}
			</header>

			<main className="flex-1 overflow-y-auto p-4 space-y-6 custom-scrollbar">
				{/* Improved Tab Navigation */}
				<div className="relative grid grid-cols-2 sm:flex items-center gap-1.5 bg-neutral-800/50 p-1 rounded-xl">
					{tabs.map((tab) => (
						<button
							key={tab.id}
							onClick={() => setActiveTab(tab.id)}
							className={cn(
								"relative sm:flex-1 flex items-center justify-center p-2 rounded-lg text-sm font-medium transition-colors",
								activeTab === tab.id
									? "text-brand-black font-semibold"
									: "text-neutral-400 hover:text-white"
							)}
						>
							<span className="relative z-10 flex items-center gap-2">
								{tab.icon}
								<span>{tab.label}</span>
								{tab.isProFeature && !isPro && (
									<span
										className={cn(
											"px-1.5 py-0.5 text-xs font-semibold rounded-full",
											activeTab === tab.id
												? "bg-black/10"
												: "bg-yellow-500/20 text-yellow-300"
										)}
									>
										Pro
									</span>
								)}
							</span>
							{activeTab === tab.id && (
								<motion.div
									layoutId="active-task-tab-indicator"
									className="absolute inset-0 bg-brand-orange rounded-lg"
									transition={{
										type: "spring",
										stiffness: 300,
										damping: 25
									}}
								/>
							)}
						</button>
					))}
				</div>

				{/* Tab Description */}
				<AnimatePresence mode="wait">
					<motion.p
						key={activeTab}
						initial={{ opacity: 0, y: -5 }}
						animate={{ opacity: 1, y: 0 }}
						exit={{ opacity: 0, y: 5 }}
						transition={{ duration: 0.2 }}
						className="text-center text-sm text-neutral-400"
					>
						{currentTabInfo.description}
					</motion.p>
				</AnimatePresence>

				{/* Panels */}
				<AnimatePresence mode="wait">
					<motion.div
						key={activeTab}
						initial={{ opacity: 0, y: 10 }}
						animate={{ opacity: 1, y: 0 }}
						exit={{ opacity: 0, y: -10 }}
						transition={{ duration: 0.2 }}
						className="space-y-4 p-4 bg-neutral-800/50 rounded-lg border border-neutral-700/50"
					>
						{activeTab !== "triggered" && (
							<div className="relative w-full bg-neutral-900 border border-neutral-700 rounded-md focus-within:ring-2 focus-within:ring-brand-orange">
								<textarea
									value={goalInput}
									onChange={(e) =>
										setGoalInput(e.target.value)
									}
									placeholder=" "
									className={cn(
										"w-full p-2 bg-transparent border-none focus:ring-0 relative z-10 resize-none custom-scrollbar",
										activeTab === "swarm" ? "h-32" : "h-24"
									)}
								/>
								{!goalInput && (
									<div
										className={cn(
											"absolute top-0 left-0 right-0 text-neutral-500 pointer-events-none z-0 p-2 overflow-hidden",
											activeTab === "swarm"
												? "h-32"
												: "h-24"
										)}
									>
										<TextLoop>
											{(activeTab === "once"
												? oncePlaceholders
												: activeTab === "recurring"
													? recurringPlaceholders
													: swarmPlaceholders
											).map((p) => (
												<span key={p}>{p}</span>
											))}
										</TextLoop>
									</div>
								)}
							</div>
						)}
						{activeTab === "once" && (
							<div className="space-y-4">
								<div className="flex items-center gap-4">
									<label className="flex items-center gap-2 cursor-pointer">
										<input
											type="radio"
											name="runOnce"
											value="now"
											checked={runOnceType === "now"}
											onChange={() =>
												setRunOnceType("now")
											}
											className="accent-brand-orange"
										/>
										Run Immediately
									</label>
									<label className="flex items-center gap-2 cursor-pointer">
										<input
											type="radio"
											name="runOnce"
											value="later"
											checked={runOnceType === "later"}
											onChange={() =>
												setRunOnceType("later")
											}
											className="accent-brand-orange"
										/>
										Schedule for Later
									</label>
								</div>
								{runOnceType === "later" && (
									<input
										type="datetime-local"
										value={runOnceDateTime}
										min={new Date()
											.toISOString()
											.slice(0, 16)}
										onChange={(e) =>
											setRunOnceDateTime(e.target.value)
										}
										className="w-full p-2 bg-neutral-900 border border-neutral-700 rounded-md"
									/>
								)}
							</div>
						)}
						{activeTab === "recurring" && (
							<div className="space-y-4">
								<div className="grid grid-cols-2 gap-4">
									<div>
										<label className="text-sm text-neutral-400">
											Frequency
										</label>
										<select
											value={recurringFrequency}
											onChange={(e) =>
												setRecurringFrequency(
													e.target.value
												)
											}
											className="w-full p-2 bg-neutral-900 border border-neutral-700 rounded-md mt-1"
										>
											<option value="daily">Daily</option>
											<option value="weekly">
												Weekly
											</option>
										</select>
									</div>
									<div>
										<label className="text-sm text-neutral-400">
											Time
										</label>
										<input
											type="time"
											value={recurringTime}
											onChange={(e) =>
												setRecurringTime(e.target.value)
											}
											className="w-full p-2 bg-neutral-900 border border-neutral-700 rounded-md mt-1"
										/>
									</div>
								</div>
								{recurringFrequency === "weekly" && (
									<div>
										<label className="text-sm text-neutral-400">
											Days of the Week
										</label>
										<div className="flex gap-2 mt-1">
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
														"p-2 rounded-md text-xs w-full font-semibold",
														recurringDays.includes(
															day
														)
															? "bg-brand-orange text-black"
															: "bg-neutral-700 hover:bg-neutral-600"
													)}
												>
													{day}
												</button>
											))}
										</div>
									</div>
								)}
								<div className="p-2 bg-neutral-900 rounded-md text-sm text-neutral-300 font-mono">
									{getRecurringSummary()}
								</div>
							</div>
						)}
						{activeTab === "triggered" && (
							<div className="space-y-4">
								{/* WHEN Section */}
								<div className="p-3 bg-neutral-900 rounded-lg border border-neutral-700 space-y-3">
									<h4 className="font-medium text-neutral-400">
										WHEN...
									</h4>
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
								</div>

								{/* THEN Section - appears after selecting a trigger */}
								<AnimatePresence>
									{selectedTrigger && (
										<motion.div
											initial={{ opacity: 0, y: -10 }}
											animate={{ opacity: 1, y: 0 }}
											exit={{ opacity: 0, y: -10 }}
											className="p-3 bg-neutral-900 rounded-lg border border-neutral-700 space-y-3"
										>
											<h4 className="font-medium text-neutral-400">
												THEN...
											</h4>
											<div className="relative w-full bg-neutral-800 border border-neutral-700 rounded-md focus-within:ring-2 focus-within:ring-brand-orange">
												<textarea
													value={goalInput}
													onChange={(e) =>
														setGoalInput(
															e.target.value
														)
													}
													placeholder="Describe the action to perform..."
													className="w-full p-2 bg-transparent border-none focus:ring-0 relative z-10 resize-none custom-scrollbar h-24"
												/>
											</div>
										</motion.div>
									)}
								</AnimatePresence>
							</div>
						)}
					</motion.div>
				</AnimatePresence>
			</main>

			<footer className="p-4 border-t border-neutral-700/50 flex-shrink-0">
				<button
					onClick={handleCreateTask}
					className="w-full py-2.5 px-5 rounded-lg bg-brand-orange hover:bg-brand-orange/90 text-brand-black font-semibold transition-colors"
				>
					Create Task
				</button>
			</footer>
		</div>
	)
}

export default TaskComposer
