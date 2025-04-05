"use client"
import { useState } from "react"
import { BellDot } from "lucide-react"

import { cn } from "@utils/cn"

const reminders = ["Lunch with John", "Presentation due tomorrow at 11 AM"]

export default function ReminderWidget() {
	const [reminder, setReminder] = useState(reminders)

	const handleCheckboxChange = (data) => {
		setReminder((prev) =>
			prev.includes(data)
				? prev.filter((remind) => remind !== data)
				: [...prev, data]
		)
	}

	return (
		<div
			className={cn("group min-h-fit w-52 rounded-3xl bg-zinc-900 p-4")}
			style={{ zIndex: 100 }}
		>
			<div className="text-md flex items-center justify-between gap-2 font-bold text-blue-400">
				<div className="flex items-center gap-2">
					<BellDot color="#60a5fa" size={15} />
					<p className="text-xl">Reminders</p>
				</div>
				<div className="flex h-4 w-4 items-center justify-center rounded-full bg-zinc-700 text-xs text-blue-400">
					{reminder.length}
				</div>
			</div>
			<div className="mt-1 overflow-hidden">
				{reminders.map((data) => (
					<label
						key={`item-${data}`}
						className="flex cursor-pointer items-center gap-3 border-b border-gray-700 py-1"
					>
						<input
							type="checkbox"
							checked={!reminder.includes(data)}
							onChange={() => handleCheckboxChange(data)}
							className="h-3 w-3 appearance-none rounded-full border-2 border-gray-700 checked:bg-blue-500"
						/>
						<span className="bg-transparent text-xs font-semibold capitalize text-white">
							{data}
						</span>
					</label>
				))}
			</div>
		</div>
	)
}
