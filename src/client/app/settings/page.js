"use client"

import toast from "react-hot-toast"
import {
	// ICONS
	IconLoader,
	IconDeviceLaptop,
	IconUser,
	IconClock,
	IconHeart,
	IconBriefcase,
	IconX,
	IconPlus,
	IconHelpCircle,
	IconKeyboard,
	IconFlask,
	IconMapPin,
	IconBrandWhatsapp,
	IconRefresh
} from "@tabler/icons-react"
import { useState, useEffect, useCallback } from "react"
import { Tooltip } from "react-tooltip"
import { cn } from "@utils/cn"
import InteractiveNetworkBackground from "@components/ui/InteractiveNetworkBackground"
import CollapsibleSection from "@components/tasks/CollapsibleSection"
import { sendNotificationToCurrentUser } from "@app/actions"

const handleTestPush = async () => {
	const toastId = toast.loading("Sending test push notification...")
	try {
		const result = await sendNotificationToCurrentUser({
			title: "Test Push Notification",
			body: "This is a test push notification from Sentient.",
			data: { url: "/tasks" } // Example data
		})
		if (result.success) {
			toast.success(
				result.message || "Push notification sent successfully!",
				{ id: toastId }
			)
		} else {
			toast.error(`Failed to send: ${result.error}`, { id: toastId })
		}
	} catch (error) {
		toast.error(`Error: ${error.message}`, { id: toastId })
	}
}

const questionSections = {
	essentials: {
		title: "The Essentials",
		icon: <IconDeviceLaptop />
	},
	context: {
		title: "About You",
		icon: <IconUser />
	}
}

const handleTestInApp = async () => {
	const toastId = toast.loading("Sending test in-app notification...")
	try {
		const response = await fetch("/api/testing/notification", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ type: "in-app" })
		})
		const result = await response.json()
		if (!response.ok) {
			throw new Error(result.detail || "Failed to send notification.")
		}
		// The notification will arrive via WebSocket, so no success toast here.
		// The LayoutWrapper will show the toast. I'll just dismiss the loading one.
		toast.dismiss(toastId)
		toast("In-app notification sent. It should appear shortly.")
	} catch (error) {
		toast.error(`Error: ${error.message}`, { id: toastId })
	}
}

const questions = [
	{
		id: "user-name",
		question: "What should I call you?",
		type: "text-input",
		required: true,
		placeholder: "e.g., Alex",
		section: "essentials",
		icon: <IconUser />
	},
	{
		id: "timezone",
		question: "What's your timezone?",
		type: "select",
		required: true,
		options: [
			{ value: "", label: "Select your timezone...", disabled: true },
			{ value: "America/New_York", label: "Eastern Time (US & Canada)" },
			{ value: "America/Chicago", label: "Central Time (US & Canada)" },
			{ value: "America/Denver", label: "Mountain Time (US & Canada)" },
			{
				value: "America/Los_Angeles",
				label: "Pacific Time (US & Canada)"
			},
			{
				value: "America/St_Johns",
				label: "Newfoundland (NDT)"
			},
			{ value: "Europe/London", label: "London, Dublin (GMT/BST)" },
			{ value: "Europe/Berlin", label: "Berlin, Paris (CET)" },
			{ value: "Asia/Kolkata", label: "India (IST)" },
			{ value: "Asia/Singapore", label: "Singapore (SGT)" },
			{ value: "UTC", label: "Coordinated Universal Time (UTC)" }
		],
		section: "essentials",
		icon: <IconClock />
	},
	{
		id: "location",
		question: "Where are you located?",
		type: "location",
		section: "context",
		icon: <IconMapPin />
	},
	{
		id: "professional-context",
		question: "What's your professional world like?",
		type: "textarea",
		placeholder:
			"e.g., I'm a software developer at a startup, aiming to become a team lead...",
		section: "context",
		icon: <IconBriefcase />
	},
	{
		id: "personal-context",
		question: "What about your personal life and hobbies?",
		type: "textarea",
		placeholder:
			"e.g., I enjoy hiking on weekends, I'm learning to play the guitar...",
		section: "context",
		icon: <IconHeart />
	}
]

const WhatsAppSettings = () => {
	// State for System Notifications
	const [notificationNumber, setNotificationNumber] = useState("")
	const [isNotifLoading, setIsNotifLoading] = useState(true)
	const [isSaving, setIsSaving] = useState(false)

	const fetchNotificationSettings = useCallback(async () => {
		setIsNotifLoading(true)
		try {
			const response = await fetch("/api/settings/whatsapp-notifications") // prettier-ignore
			if (!response.ok)
				throw new Error(
					"Failed to fetch WhatsApp notification settings."
				)
			const data = await response.json()
			setNotificationNumber(data.whatsapp_notifications_number || "")
		} catch (error) {
			toast.error(error.message)
		} finally {
			setIsNotifLoading(false)
		}
	}, [])

	useEffect(() => {
		fetchNotificationSettings()
	}, [fetchNotificationSettings])

	const handleSaveNotifNumber = async () => {
		setIsSaving(true)
		try {
			const response = await fetch(
				"/api/settings/whatsapp-notifications",
				{
					method: "POST",
					headers: { "Content-Type": "application/json" },
					body: JSON.stringify({
						whatsapp_notifications_number: notificationNumber
					})
				}
			)
			const data = await response.json()
			if (!response.ok)
				throw new Error(data.detail || "Failed to save number.")
			toast.success("Notifications enabled for this number!")
		} catch (error) {
			toast.error(error.message)
		} finally {
			setIsSaving(false)
		}
	}

	const handleRemoveNotifNumber = async () => {
		setIsSaving(true)
		try {
			const response = await fetch(
				"/api/settings/whatsapp-notifications",
				{
					method: "POST",
					headers: { "Content-Type": "application/json" },
					body: JSON.stringify({ whatsapp_notifications_number: "" })
				}
			)
			if (!response.ok) throw new Error("Failed to remove number.")
			setNotificationNumber("")
			toast.success("Notification number removed.")
		} catch (error) {
			toast.error(error.message)
		} finally {
			setIsSaving(false)
		}
	}

	const hasNotifNumber =
		notificationNumber && notificationNumber.trim() !== ""

	return (
		<section>
			<h2 className="text-2xl font-bold mb-2 text-white flex items-center gap-3">
				<IconBrandWhatsapp />
				WhatsApp Notifications
			</h2>
			<p className="text-neutral-400 mb-6">
				Receive important notifications, task updates, and reminders on WhatsApp.
			</p>
			<div className="bg-neutral-900/50 p-6 rounded-2xl border border-neutral-800">
				<div className="space-y-4">
					<p className="text-neutral-400 text-sm">
						Receive important notifications, task updates, and reminders on WhatsApp. We're in the process of getting an official number, so for now, messages will come from our co-founder Sarthak (+91827507823), who may also occasionally reach out for feedback. Please enter your number with the country code.
					</p>
					{isNotifLoading ? (
						<div className="flex justify-center mt-4">
							<IconLoader className="w-6 h-6 animate-spin text-brand-orange" />
						</div>
					) : (
						<div className="space-y-4">
							<div className="flex flex-col sm:flex-row gap-2">
								<div className="relative flex-grow">
									<IconBrandWhatsapp
										className="absolute left-3 top-1/2 -translate-y-1/2 text-green-500"
										size={20}
									/>
									<input
										type="tel"
										value={notificationNumber}
										onChange={(e) =>
											setNotificationNumber(
												e.target.value
											)
										}
										placeholder="+14155552671"
										className="w-full pl-10 pr-4 bg-neutral-800/50 border font-mono border-neutral-700 rounded-lg py-2 text-white placeholder-neutral-500 focus:outline-none focus:ring-2 focus:ring-brand-orange"
									/>
								</div>
								<div className="flex gap-2 justify-end">
									<button
										onClick={handleSaveNotifNumber}
										disabled={
											isSaving ||
											!notificationNumber.trim()
										}
										className="flex items-center py-2 px-4 rounded-lg bg-brand-orange hover:bg-brand-orange/70 text-white font-medium transition-colors disabled:opacity-50"
									>
										{isSaving ? (
											<IconLoader className="w-4 h-4 mr-2 animate-spin" />
										) : (
											<IconPlus className="w-4 h-4 mr-2" />
										)}{" "}
										{hasNotifNumber ? "Update" : "Save"}
									</button>
									{hasNotifNumber && (
										<button
											onClick={handleRemoveNotifNumber}
											disabled={isSaving}
											className="flex items-center py-2 px-4 rounded-lg bg-red-600/80 hover:bg-red-600 text-white font-medium transition-colors"
										>
											<IconX className="w-4 h-4 mr-2" />{" "}
											Remove
										</button>
									)}
								</div>
							</div>
						</div>
					)}
				</div>
			</div>
		</section>
	)
}

const ShortcutsSettings = () => {
	const shortcuts = {
		Global: [
			{ keys: ["Ctrl", "M"], description: "Open Chat" },
			{ keys: ["Ctrl", "B"], description: "Toggle Notifications" },
			{ keys: ["Esc"], description: "Close Modal / Chat" },
			{ keys: ["Ctrl", "K"], description: "Open Command Palette" }
		],
		Navigation: [
			{ keys: ["Ctrl", "H"], description: "Go to Chat" },
			{ keys: ["Ctrl", "J"], description: "Go to Notes" },
			{ keys: ["Ctrl", "A"], description: "Go to Tasks" },
			{ keys: ["Ctrl", "I"], description: "Go to Integrations" },
			{ keys: ["Ctrl", "S"], description: "Go to Settings" }
		]
	}

	return (
		<section>
			<h2 className="text-2xl font-bold mb-6 text-white flex items-center gap-3">
				<IconKeyboard />
				Keyboard Shortcuts
			</h2>

			<div className="bg-neutral-900/50 p-6 rounded-2xl border border-neutral-800">
				<div className="grid grid-cols-1 md:grid-cols-2 gap-8">
					{Object.entries(shortcuts).map(([category, list]) => (
						<div key={category}>
							<h3 className="text-lg font-semibold text-brand-orange mb-4">
								{category}
							</h3>
							<div className="space-y-3">
								{list.map((shortcut) => (
									<div
										key={shortcut.description}
										className="flex justify-between items-center text-sm"
									>
										<span className="text-neutral-300">
											{shortcut.description}
										</span>
										<div className="flex items-center gap-2">
											{shortcut.keys.map((key) => (
												<kbd
													key={key}
													className="px-2 py-1.5 text-xs font-semibold text-neutral-300 bg-neutral-700 border border-neutral-600 rounded-md"
												>
													{key}
												</kbd>
											))}
										</div>
									</div>
								))}
							</div>
						</div>
					))}
				</div>
			</div>
		</section>
	)
}

const TestingTools = () => {
	const [serviceName, setServiceName] = useState("gmail")
	const [eventData, setEventData] = useState(
		'{\n  "subject": "Project Alpha Kick-off",\n  "body": "Hi team, let\'s schedule a meeting for next Tuesday to discuss the Project Alpha kick-off. John, please prepare the presentation."\n}'
	)
	const [isSubmitting, setIsSubmitting] = useState(false)
	const [isReprocessing, setIsReprocessing] = useState(false)

	const handleSubmit = async (e) => {
		e.preventDefault()
		setIsSubmitting(true)
		let parsedData
		try {
			parsedData = JSON.parse(eventData)
		} catch (error) {
			toast.error("Invalid JSON in event data.")
			setIsSubmitting(false)
			return
		}

		try {
			const response = await fetch("/api/testing/inject-context", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					service_name: serviceName,
					event_data: parsedData
				})
			})
			const result = await response.json()
			if (!response.ok) {
				throw new Error(result.error || "Failed to inject event.")
			}
			toast.success(
				`Event injected successfully! Event ID: ${result.event_id}`
			)
		} catch (error) {
			toast.error(error.message)
		} finally {
			setIsSubmitting(false)
		}
	}

	const handleServiceChange = (e) => {
		const newService = e.target.value
		setServiceName(newService)
		if (newService === "gmail") {
			setEventData(
				'{\n  "subject": "Project Alpha Kick-off",\n  "body": "Hi team, let\'s schedule a meeting for next Tuesday to discuss the Project Alpha kick-off. John, please prepare the presentation."\n}'
			)
		} else if (newService === "gcalendar") {
			setEventData(
				'{\n  "summary": "Finalize Q3 report",\n  "description": "Need to finalize the Q3 sales report with Sarah before the end of the week."\n}'
			)
		}
	}

	const handleReprocessOnboarding = async () => {
		setIsReprocessing(true)
		const toastId = toast.loading(
			"Queueing onboarding data for memory reprocessing..."
		)
		try {
			const response = await fetch("/api/testing/reprocess-onboarding", {
				method: "POST"
			})
			const result = await response.json()
			if (!response.ok) {
				throw new Error(
					result.detail || "Failed to trigger reprocessing."
				)
			}
			toast.success(result.message, { id: toastId })
		} catch (error) {
			toast.error(`Error: ${error.message}`, { id: toastId })
		} finally {
			setIsReprocessing(false)
		}
	}
	const [isTriggeringScheduler, setIsTriggeringScheduler] = useState(false)
	const [isTriggeringPoller, setIsTriggeringPoller] = useState(false)

	const [whatsAppNumber, setWhatsAppNumber] = useState("")
	const [isVerifying, setIsVerifying] = useState(false)
	const [isSending, setIsSending] = useState(false)
	const [verificationResult, setVerificationResult] = useState({
		status: null,
		message: ""
	})

	const handleTriggerScheduler = async () => {
		setIsTriggeringScheduler(true)
		try {
			const response = await fetch("/api/testing/trigger-scheduler", {
				method: "POST"
			})
			const result = await response.json()
			if (!response.ok) {
				throw new Error(result.detail || "Failed to trigger scheduler.")
			}
			toast.success(result.message)
		} catch (error) {
			toast.error(error.message)
		} finally {
			setIsTriggeringScheduler(false)
		}
	}
	const handleTriggerPoller = async () => {
		setIsTriggeringPoller(true)
		try {
			const response = await fetch("/api/testing/trigger-poller", {
				method: "POST"
			})
			const result = await response.json()
			if (!response.ok) {
				throw new Error(result.detail || "Failed to trigger poller.")
			}
			toast.success(result.message)
		} catch (error) {
			toast.error(error.message)
		} finally {
			setIsTriggeringPoller(false)
		}
	}

	const handleVerifyWhatsApp = async () => {
		if (!whatsAppNumber) {
			toast.error("Please enter a phone number to verify.")
			return
		}
		setIsVerifying(true)
		setVerificationResult({ status: null, message: "Verifying..." })
		try {
			const response = await fetch("/api/testing/whatsapp/verify", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ phone_number: whatsAppNumber })
			})
			const result = await response.json()
			if (!response.ok) {
				throw new Error(result.detail || "Verification request failed.")
			}

			if (result.numberExists) {
				toast.success("Verification successful!")
				setVerificationResult({
					status: "success",
					message: `Number is valid. Chat ID: ${result.chatId}`
				})
			} else {
				toast.error("Number not found on WhatsApp.")
				setVerificationResult({
					status: "failure",
					message:
						"This phone number does not appear to be registered on WhatsApp."
				})
			}
		} catch (error) {
			toast.error(error.message)
			setVerificationResult({ status: "failure", message: error.message })
		} finally {
			setIsVerifying(false)
		}
	}

	const handleSendTestWhatsApp = async () => {
		if (!whatsAppNumber) {
			toast.error("Please enter a phone number to send a message to.")
			return
		}
		setIsSending(true)
		try {
			const response = await fetch("/api/testing/whatsapp", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ phone_number: whatsAppNumber })
			})
			const result = await response.json()
			if (!response.ok) {
				throw new Error(
					result.detail || "Failed to send test notification."
				)
			}
			toast.success("Test notification sent successfully!")
		} catch (error) {
			toast.error(`Send failed: ${error.message}`)
		} finally {
			setIsSending(false)
		}
	}

	return (
		<section>
			<h2 className="text-2xl font-bold mb-6 text-white flex items-center gap-3">
				<IconFlask />
				Developer Tools
			</h2>

			<div className="bg-neutral-900/50 p-6 rounded-2xl border border-neutral-800">
				<h3 className="font-semibold text-lg text-white mb-2">
					Inject Context Event
				</h3>
				<p className="text-gray-400 text-sm mb-4">
					Simulate a polling event by manually injecting context data
					into the processing pipeline. This is useful for testing the
					extractor and planner workers.
				</p>
				<form onSubmit={handleSubmit} className="space-y-4">
					<div>
						<label
							htmlFor="serviceName"
							className="block text-sm font-medium text-gray-300 mb-1"
						>
							Service
						</label>
						<select
							id="serviceName"
							value={serviceName}
							onChange={handleServiceChange}
							className="w-full bg-neutral-800/50 border font-mono border-neutral-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-brand-orange"
						>
							<option value="gmail">Gmail</option>
							<option value="gcalendar">Google Calendar</option>
						</select>
					</div>
					<div>
						<label
							htmlFor="eventData"
							className="block text-sm font-medium text-gray-300 mb-1"
						>
							Event Data (JSON)
						</label>
						<textarea
							id="eventData"
							value={eventData}
							onChange={(e) => setEventData(e.target.value)}
							rows={8}
							className="w-full font-mono text-xs bg-neutral-800/50 border border-neutral-700 rounded-lg px-3 py-2 text-white placeholder-neutral-500 focus:outline-none focus:ring-2 focus:ring-brand-orange"
							placeholder='e.g., { "subject": "Hello", "body": "World" }'
						/>
					</div>
					<div className="flex justify-end">
						<button
							type="submit"
							disabled={isSubmitting}
							className="flex items-center py-2 px-4 rounded-lg bg-brand-orange hover:bg-brand-orange/70 text-white font-medium transition-colors disabled:opacity-50"
						>
							{isSubmitting ? (
								<IconLoader className="w-5 h-5 animate-spin" />
							) : (
								"Inject Event"
							)}
						</button>
					</div>
				</form>
			</div>
			{/* Reprocess Onboarding Data */}
			<div className="bg-neutral-900/50 p-6 rounded-2xl border border-neutral-800 mt-6">
				<h3 className="font-semibold text-lg text-white mb-2">
					Reprocess Onboarding Data
				</h3>
				<p className="text-gray-400 text-sm mb-4">
					Manually trigger the Celery worker to process your saved
					onboarding answers and add them to your long-term memory.
					This is useful for testing memory functions without
					re-onboarding.
				</p>
				<div className="flex justify-end">
					<button
						onClick={handleReprocessOnboarding}
						disabled={isReprocessing}
						className="flex items-center py-2 px-4 rounded-md bg-purple-600 hover:bg-purple-500 text-white font-medium transition-colors disabled:opacity-50"
					>
						{isReprocessing ? (
							<IconLoader className="w-5 h-5 animate-spin" />
						) : (
							<IconRefresh className="w-5 h-5" />
						)}{" "}
						Run Reprocessing
					</button>
				</div>
			</div>
			{/* WhatsApp Test Tools */}
			<div className="bg-neutral-900/50 p-6 rounded-2xl border border-neutral-800 mt-6">
				<h3 className="font-semibold text-lg text-white mb-2">
					Test WhatsApp Integration
				</h3>
				<p className="text-gray-400 text-sm mb-4">
					Verify a phone number's existence on WhatsApp and then send
					a test message. The number must include the country code
					(e.g., +14155552671).
				</p>
				<div className="flex flex-col sm:flex-row gap-2">
					<div className="relative flex-grow">
						<IconBrandWhatsapp
							className="absolute left-3 top-1/2 -translate-y-1/2 text-green-500"
							size={20}
						/>
						<input
							type="tel"
							value={whatsAppNumber}
							onChange={(e) => {
								setWhatsAppNumber(e.target.value)
								setVerificationResult({
									status: null,
									message: ""
								}) // Reset on change
							}}
							placeholder="Enter WhatsApp Number with country code"
							className="w-full pl-10 pr-4 bg-neutral-800/50 border font-mono border-neutral-700 rounded-lg py-2 text-white placeholder-neutral-500 focus:outline-none focus:ring-2 focus:ring-brand-orange"
						/>
					</div>
					<div className="flex gap-2 justify-end">
						<button
							onClick={handleVerifyWhatsApp}
							disabled={isVerifying || isSending}
							className="flex items-center justify-center py-2 px-4 rounded-md bg-purple-600 hover:bg-purple-500 text-white font-medium transition-colors disabled:opacity-50"
						>
							{isVerifying ? (
								<IconLoader className="w-5 h-5 animate-spin" />
							) : (
								"Verify"
							)}
						</button>
						<button
							onClick={handleSendTestWhatsApp}
							disabled={isSending || isVerifying}
							className="flex items-center justify-center py-2 px-4 rounded-lg bg-brand-orange hover:bg-brand-orange/70 text-white font-medium transition-colors disabled:opacity-50"
						>
							{isSending ? (
								<IconLoader className="w-5 h-5 animate-spin" />
							) : (
								"Send Test"
							)}
						</button>
					</div>
				</div>
				{verificationResult.message && (
					<p
						className={cn(
							"text-sm mt-3",
							verificationResult.status === "success"
								? "text-green-400"
								: "text-red-400"
						)}
					>
						{verificationResult.message}
					</p>
				)}
			</div>
			{/* Notification Test Tools */}
			<div className="bg-neutral-900/50 p-6 rounded-2xl border border-neutral-800 mt-6">
				<h3 className="font-semibold text-lg text-white mb-2">
					Test Notifications
				</h3>
				<p className="text-gray-400 text-sm mb-4">
					Send test notifications to verify your setup. In-app
					notifications appear as toasts, while push notifications are
					sent to your subscribed devices.
				</p>
				<div className="flex flex-col sm:flex-row gap-4">
					<button
						onClick={handleTestInApp}
						className="flex items-center justify-center py-2 px-4 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-medium transition-colors"
					>
						Test In-App Notification
					</button>
					<button
						onClick={handleTestPush}
						className="flex items-center justify-center py-2 px-4 rounded-lg bg-green-600 hover:bg-green-500 text-white font-medium transition-colors"
					>
						Test Push Notification
					</button>
				</div>
			</div>
			{/* Poller Test Tool */}
			<div className="bg-neutral-900/50 p-6 rounded-2xl border border-neutral-800 mt-6">
				<h3 className="font-semibold text-lg text-white mb-2">
					Trigger Proactive Poller
				</h3>
				<p className="text-gray-400 text-sm mb-4">
					Manually run the Celery Beat scheduler task
					(`schedule_all_polling`) to immediately check for any users
					who are due for a Gmail or Google Calendar poll. This is
					useful for testing the proactive pipeline without waiting
					for the hourly interval.
				</p>
				<div className="flex justify-end">
					<button
						onClick={handleTriggerPoller}
						disabled={isTriggeringPoller}
						className="flex items-center py-2 px-4 rounded-md bg-purple-600 hover:bg-purple-500 text-white font-medium transition-colors disabled:opacity-50"
					>
						{isTriggeringPoller ? (
							<IconLoader className="w-5 h-5 animate-spin" />
						) : (
							"Run Poller Now"
						)}
					</button>
				</div>
			</div>
			{/* Scheduler Test Tool */}
			<div className="bg-neutral-900/50 p-6 rounded-2xl border border-neutral-800 mt-6">
				<h3 className="font-semibold text-lg text-white mb-2">
					Trigger Task Scheduler
				</h3>
				<p className="text-gray-400 text-sm mb-4">
					Manually run the Celery Beat scheduler task
					(`run_due_tasks`) to immediately check for and execute any
					due scheduled or recurring tasks. This is useful for testing
					without waiting for the 5-minute interval.
				</p>
				<div className="flex justify-end">
					<button
						onClick={handleTriggerScheduler}
						disabled={isTriggeringScheduler}
						className="flex items-center py-2 px-4 rounded-md bg-purple-600 hover:bg-purple-500 text-white font-medium transition-colors disabled:opacity-50"
					>
						{isTriggeringScheduler ? (
							<IconLoader className="w-5 h-5 animate-spin" />
						) : (
							"Run Scheduler Now"
						)}
					</button>
				</div>
			</div>
		</section>
	)
}

const ProfileSettings = ({ initialData, onSave, isSaving }) => {
	const [formData, setFormData] = useState(initialData || {})

	useEffect(() => {
		setFormData(initialData || {})
	}, [initialData])

	const handleAnswer = (questionId, answer) => {
		setFormData((prev) => ({ ...prev, [questionId]: answer }))
	}

	return (
		<section>
			<div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-6">
				<div>
					<h2 className="text-2xl font-bold text-white flex items-center gap-3">
						<IconUser />
						Your Profile
					</h2>
					<p className="text-neutral-400 mt-1">
						This information helps me personalize my responses and
						actions for you.
					</p>
				</div>
				<button
					onClick={() => onSave(formData)}
					disabled={isSaving}
					className="mt-4 sm:mt-0 py-2 px-5 rounded-lg bg-brand-orange hover:bg-brand-orange/70 text-white font-medium transition-colors flex items-center gap-2 disabled:opacity-50"
				>
					{isSaving ? (
						<IconLoader className="w-5 h-5 animate-spin" />
					) : (
						"Save Profile"
					)}
				</button>
			</div>

			<div className="space-y-10 bg-neutral-900/50 p-6 rounded-2xl border border-neutral-800">
				{Object.entries(questionSections).map(
					([key, { title, icon }]) => (
						<CollapsibleSection
							key={key}
							title={
								<h3 className="text-lg font-semibold text-neutral-200 flex items-center gap-3">
									{icon} {title}
								</h3>
							}
							defaultOpen={true}
						>
							<div className="space-y-6 mt-4 pl-4 md:pl-8 border-l-2 border-neutral-800">
								{questions
									.filter((q) => q.section === key)
									.map((q) => (
										<div
											key={q.id}
											className="min-h-[68px]"
										>
											<label className="block text-sm font-medium text-neutral-300 mb-2 font-sans">
												{q.question}
											</label>
											{(() => {
												switch (q.type) {
													case "text-input":
														return (
															// eslint-disable-line
															<input
																type="text"
																value={
																	formData[
																		q.id
																	] || ""
																}
																onChange={(e) =>
																	handleAnswer(
																		q.id,
																		e.target
																			.value
																	)
																}
																className="w-full bg-neutral-800/50 border font-mono border-neutral-700 rounded-lg px-3 py-2 text-white placeholder-neutral-500 focus:outline-none focus:ring-2 focus:ring-brand-orange"
																placeholder={
																	q.placeholder
																}
															/>
														)
													case "textarea":
														return (
															// eslint-disable-line
															<textarea
																value={
																	formData[
																		q.id
																	] || ""
																}
																onChange={(e) =>
																	handleAnswer(
																		q.id,
																		e.target
																			.value
																	)
																}
																rows={4}
																className="w-full bg-neutral-800/50 border font-mono border-neutral-700 rounded-lg px-3 py-2 text-white placeholder-neutral-500 focus:outline-none focus:ring-2 focus:ring-brand-orange"
																placeholder={
																	q.placeholder
																}
															/>
														)
													case "select":
														return (
															// eslint-disable-line
															<select
																value={
																	formData[
																		q.id
																	] || ""
																}
																onChange={(e) =>
																	handleAnswer(
																		q.id,
																		e.target
																			.value
																	)
																}
																className="w-full bg-neutral-800/50 border font-mono border-neutral-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-brand-orange appearance-none"
															>
																{q.options.map(
																	(opt) => (
																		<option
																			key={
																				opt.value
																			}
																			value={
																				opt.value
																			}
																		>
																			{
																				opt.label
																			}
																		</option>
																	)
																)}
															</select>
														)
													case "location": // Simplified for now
														const locationValue =
															formData[q.id]
														const isGpsLocation =
															typeof locationValue ===
																"object" &&
															locationValue !==
																null &&
															locationValue.latitude

														if (isGpsLocation) {
															return (
																<div className="flex items-center gap-2 font-mono">
																	<p className="w-full bg-neutral-800/50 border border-neutral-700 rounded-lg px-3 py-2 text-neutral-300">
																		{`Lat: ${locationValue.latitude?.toFixed(
																			4
																		)}, Lon: ${locationValue.longitude?.toFixed(
																			4
																		)} (Detected)`}
																	</p>
																	<button
																		onClick={() =>
																			handleAnswer(
																				q.id,
																				""
																			)
																		}
																		className="p-2 text-neutral-400 hover:text-white hover:bg-neutral-700 rounded-md"
																		title="Clear and enter manually"
																	>
																		<IconX
																			size={
																				18
																			}
																		/>
																	</button>
																</div>
															)
														}
														return (
															<input
																type="text"
																value={
																	formData[
																		q.id
																	] || ""
																}
																onChange={
																	(
																		e
																	) =>
																		handleAnswer(
																			q.id,
																			e
																				.target
																				.value
																		) // prettier-ignore
																}
																className="w-full bg-neutral-800/50 border font-mono border-neutral-700 rounded-lg px-3 py-2 text-white placeholder-neutral-500 focus:outline-none focus:ring-2 focus:ring-brand-orange"
																placeholder="City, Country"
															/>
														)
													default:
														return null
												}
											})()}
										</div>
									))}
							</div>
						</CollapsibleSection>
					)
				)}
			</div>
		</section>
	)
}

export default function SettingsPage() {
	const [profileData, setProfileData] = useState(null)
	const [isSavingProfile, setIsSavingProfile] = useState(false)

	const handleSaveProfile = async (newOnboardingData) => {
		setIsSavingProfile(true)
		try {
			const payload = {
				onboardingAnswers: newOnboardingData,
				personalInfo: {
					name: newOnboardingData["user-name"],
					location: newOnboardingData["location"],
					timezone: newOnboardingData["timezone"]
				},
				preferences: {}
			}

			const response = await fetch("/api/settings/profile", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify(payload)
			})

			if (!response.ok) {
				const errorData = await response.json()
				throw new Error(errorData.error || "Failed to save profile")
			}
			toast.success("Profile updated successfully!")
			fetchData() // Refresh data to show changes
		} catch (error) {
			toast.error(`Error saving profile: ${error.message}`)
		} finally {
			setIsSavingProfile(false)
		}
	}

	const fetchData = useCallback(async () => {
		try {
			const [response, profileResponse] = await Promise.all([
				fetch("/api/user/data"),
				fetch("/api/user/profile")
			])
			if (!response.ok) {
				const errorData = await response.json()
				throw new Error(
					errorData.message || "Failed to fetch user data"
				)
			}
			if (!profileResponse.ok)
				throw new Error("Failed to fetch user profile")
			const result = await response.json()
			const profile = await profileResponse.json()
			if (result.data) {
				setProfileData({ ...profile, ...result.data })
			}
		} catch (error) {
			toast.error(`Failed to fetch user data: ${error.message}`)
		}
	}, [])

	useEffect(() => {
		fetchData()
	}, [fetchData])

	return (
		<div className="flex-1 flex h-screen text-white overflow-x-hidden">
			<Tooltip
				id="page-help-tooltip"
				place="right-start"
				style={{ zIndex: 9999 }}
			/>
			<div className="flex-1 flex flex-col overflow-hidden relative w-full pt-16 md:pt-0">
				<div className="absolute inset-0 z-[-1] network-grid-background">
					<InteractiveNetworkBackground />
				</div>
				<div className="absolute -top-[250px] left-1/2 -translate-x-1/2 w-[800px] h-[500px] bg-brand-orange/10 rounded-full blur-3xl -z-10" />

				<header className="flex items-center justify-between p-4 sm:p-6 md:px-8 md:py-6 bg-transparent border-b border-neutral-800 shrink-0">
					<div>
						<h1 className="text-3xl lg:text-4xl font-bold text-white">
							Settings
						</h1>
						<p className="text-neutral-400 mt-1">
							Manage your profile, notifications, and developer
							tools.
						</p>
					</div>
				</header>

				<main className="flex-1 overflow-y-auto px-4 sm:px-6 md:px-10 pb-4 sm:pb-6 md:pb-10 custom-scrollbar">
					<div className="w-full max-w-4xl mx-auto space-y-12 pt-8">
						<ProfileSettings
							initialData={profileData?.onboardingAnswers}
							onSave={handleSaveProfile}
							isSaving={isSavingProfile}
						/>
						<WhatsAppSettings />
						<ShortcutsSettings />
						{process.env.NEXT_PUBLIC_ENVIRONMENT !== "prod" &&
							process.env.NEXT_PUBLIC_ENVIRONMENT !== "stag" && (
								<TestingTools />
							)}
					</div>
				</main>
			</div>
		</div>
	)
}
