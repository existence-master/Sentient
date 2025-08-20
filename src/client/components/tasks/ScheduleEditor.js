"use client"

import React from "react"
import { cn } from "@utils/cn"
import { Tooltip } from "react-tooltip"

const ScheduleEditor = ({ schedule, setSchedule }) => {
	const handleTypeChange = (type) => {
		const baseSchedule = { ...schedule, type: type }
		if (type === "once") {
			delete baseSchedule.frequency
			delete baseSchedule.days
			delete baseSchedule.time
		} else if (type === "triggered") {
			delete baseSchedule.run_at
			delete baseSchedule.frequency
			delete baseSchedule.days
			delete baseSchedule.time
		} else {
			delete baseSchedule.run_at
		}
		setSchedule(baseSchedule)
	}

	const handleRunAtChange = (e) => {
		const localDateTimeString = e.target.value
		if (localDateTimeString) {
			// The input value is a string like "2024-07-26T09:00".
			// new Date() will parse this as 9 AM in the browser's local timezone.
			const localDate = new Date(localDateTimeString)
			// .toISOString() converts it to a UTC string, which is what we want to store.
			setSchedule({ ...schedule, run_at: localDate.toISOString() })
		} else {
			setSchedule({ ...schedule, run_at: null })
		}
	}

	const getLocalDateTimeString = (isoString) => {
		if (!isoString) return ""
		const date = new Date(isoString)
		const tzOffset = date.getTimezoneOffset() * 60000 // offset in milliseconds
		const localISOTime = new Date(date.getTime() - tzOffset)
			.toISOString()
			.slice(0, 16)
		return localISOTime
	}

	const handleDayToggle = (day) => {
		const currentDays = schedule.days || []
		const newDays = currentDays.includes(day)
			? currentDays.filter((d) => d !== day)
			: [...currentDays, day]
		setSchedule({ ...schedule, days: newDays })
	}

	return (
		<div className="bg-neutral-800/50 p-4 rounded-lg space-y-4 border border-neutral-700/80">
			<div className="flex items-center gap-2">
				{[
					{ label: "Run Once", value: "once" },
					{ label: "Recurring", value: "recurring" },
					{ label: "Triggered", value: "triggered" }
				].map(({ label, value }) => (
					<button
						key={value}
						type="button"
						onClick={() => handleTypeChange(value)}
						className={cn(
							"px-4 py-1.5 rounded-full text-sm font-medium transition-colors",
							(schedule.type || "once") === value
								? "bg-brand-orange text-brand-black font-semibold"
								: "bg-neutral-700 hover:bg-neutral-600"
						)}
					>
						{label}
					</button>
				))}
			</div>

			{(schedule.type === "once" || !schedule.type) && (
				<div>
					<label className="text-xs text-gray-400 block mb-1">
						Run At (optional, local time)
					</label>
					<input
						type="datetime-local"
						value={getLocalDateTimeString(schedule.run_at)}
						step="1" // Ensures seconds are included in the value
						onChange={handleRunAtChange}
						className="w-full p-2 bg-neutral-700 border border-neutral-600 rounded-md focus:border-brand-orange focus:ring-brand-orange/50 [color-scheme:dark]"
					/>
					<p className="text-xs text-gray-500 mt-1">
						If left blank, the task will be planned immediately.
					</p>
				</div>
			)}

			{schedule.type === "triggered" && (
				<div className="space-y-4">
					<div>
						<label className="text-xs text-gray-400 block mb-1">
							Source Service
						</label>
						<select
							value={schedule.source || "gmail"}
							onChange={(e) =>
								setSchedule({
									...schedule,
									source: e.target.value
								})
							}
							className="w-full p-2 bg-neutral-700 border border-neutral-600 rounded-md focus:border-brand-orange focus:ring-brand-orange/50"
						>
							<option value="gmail">Gmail</option>
							<option value="slack">Slack</option>
							<option value="gcalendar">Google Calendar</option>
						</select>
					</div>
					<div>
						<label className="text-xs text-gray-400 block mb-1">
							Trigger Event
						</label>
						<input
							type="text"
							value={schedule.event || "new_email"}
							onChange={(e) =>
								setSchedule({
									...schedule,
									event: e.target.value
								})
							}
							placeholder="e.g., new_email, new_message"
							className="w-full p-2 bg-neutral-700 border border-neutral-600 rounded-md focus:border-brand-orange focus:ring-brand-orange/50"
						/>
					</div>
					<div>
						<label className="text-xs text-gray-400 block mb-1">
							Filter Conditions (JSON)
						</label>
						<textarea
							value={JSON.stringify(
								schedule.filter || {},
								null,
								2
							)}
							onChange={(e) => {
								try {
									const parsed = JSON.parse(e.target.value)
									setSchedule({ ...schedule, filter: parsed })
								} catch (err) {
									// Silently ignore invalid JSON for now
								}
							}}
							rows={4}
							className="w-full p-2 bg-neutral-700 border border-neutral-600 rounded-md focus:border-brand-orange focus:ring-brand-orange/50 font-mono text-xs"
							placeholder={`{\n  "from": "boss@example.com",\n  "subject_contains": "Urgent"\n}`}
						/>
						<p className="text-xs text-gray-500 mt-1">
							Example: {`{"from": "user@example.com"}`}
						</p>
					</div>
				</div>
			)}

			{schedule.type === "recurring" && (
				<div className="grid grid-cols-1 md:grid-cols-2 gap-4">
					<div>
						<label className="text-xs text-gray-400 block mb-1">
							Frequency
						</label>
						<select
							value={schedule.frequency || "daily"}
							onChange={(e) =>
								setSchedule({
									...schedule,
									frequency: e.target.value
								})
							}
							className="w-full p-2 bg-neutral-700 border border-neutral-600 rounded-md focus:border-brand-orange focus:ring-brand-orange/50"
						>
							<option value="daily">Daily</option>
							<option value="weekly">Weekly</option>
						</select>
					</div>
					<div>
						<label className="text-xs text-gray-400 block mb-1">
							Time (Local)
						</label>
						<input
							type="time"
							value={schedule.time || "09:00"}
							onChange={(e) =>
								setSchedule({
									...schedule,
									time: e.target.value
								})
							}
							className="w-full p-2 bg-neutral-700 border border-neutral-600 rounded-md focus:border-brand-orange focus:ring-brand-orange/50 [color-scheme:dark]"
						/>
					</div>
					{schedule.frequency === "weekly" && (
						<div className="md:col-span-2">
							<label className="text-xs text-gray-400 block mb-2">
								Days
							</label>
							<div className="flex flex-wrap gap-2">
								{[
									"Monday",
									"Tuesday",
									"Wednesday",
									"Thursday",
									"Friday",
									"Saturday",
									"Sunday"
								].map((day) => (
									<button
										type="button"
										key={day}
										onClick={() => handleDayToggle(day)}
										className={cn(
											"px-3 py-1.5 rounded-full text-xs font-semibold transition-colors",
											(schedule.days || []).includes(day)
												? "bg-brand-orange text-brand-black"
												: "bg-neutral-700 hover:bg-neutral-600"
										)}
									>
										{day.substring(0, 3)}
									</button>
								))}
							</div>
						</div>
					)}
				</div>
			)}
		</div>
	)
}

export default ScheduleEditor
