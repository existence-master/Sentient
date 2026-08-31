"use client"

import React from "react"
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
	IconRefresh,
	IconUsers
} from "@tabler/icons-react"
import { useState, useCallback } from "react"
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { Tooltip } from "react-tooltip"
import { cn } from "@utils/cn"
import { useUserStore } from "@stores/app-stores"
import InteractiveNetworkBackground from "@components/ui/InteractiveNetworkBackground"
import { sendNotificationToCurrentUser } from "@app/actions"
import { Button } from "@components/ui/button"
import {
	Accordion,
	AccordionContent,
	AccordionItem,
	AccordionTrigger
} from "@components/ui/accordion"
import apiClient from "@lib/apiClient"

const handleTestPush = async () => {
	const toastId = toast.loading("Sending test push notification...")
	try {
		const result = await sendNotificationToCurrentUser({
			title: "Test Push Notification",
			body: "This is a test push notification from BUYASOUL.",
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
		await apiClient("/api/testing/notification", {
			method: "POST",
			body: JSON.stringify({ type: "in-app" })
		})
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
			{ value: "", label: "Select your timezone..." },
			{ value: "UTC", label: "(GMT+00:00) Coordinated Universal Time" },
			{
				value: "America/New_York",
				label: "(GMT-04:00) Eastern Time (US & Canada)"
			},
			{
				value: "America/Chicago",
				label: "(GMT-05:00) Central Time (US & Canada)"
			},
			{
				value: "America/Denver",
				label: "(GMT-06:00) Mountain Time (US & Canada)"
			},
			{
				value: "America/Los_Angeles",
				label: "(GMT-07:00) Pacific Time (US & Canada)"
			},
			{ value: "America/Anchorage", label: "(GMT-08:00) Alaska" },
			{ value: "America/Phoenix", label: "(GMT-07:00) Arizona" },
			{ value: "Pacific/Honolulu", label: "(GMT-10:00) Hawaii" },
			{ value: "America/Sao_Paulo", label: "(GMT-03:00) Brasilia" },
			{
				value: "America/Buenos_Aires",
				label: "(GMT-03:00) Buenos Aires"
			},
			{
				value: "Europe/London",
				label: "(GMT+01:00) London, Dublin, Lisbon"
			},
			{
				value: "Europe/Berlin",
				label: "(GMT+02:00) Amsterdam, Berlin, Paris, Rome"
			},
			{
				value: "Europe/Helsinki",
				label: "(GMT+03:00) Helsinki, Kyiv, Riga, Sofia"
			},
			{
				value: "Europe/Moscow",
				label: "(GMT+03:00) Moscow, St. Petersburg"
			},
			{ value: "Africa/Cairo", label: "(GMT+02:00) Cairo" },
			{ value: "Africa/Johannesburg", label: "(GMT+02:00) Johannesburg" },
			{ value: "Asia/Dubai", label: "(GMT+04:00) Abu Dhabi, Muscat" },
			{ value: "Asia/Kolkata", label: "(GMT+05:30) India Standard Time" },
			{
				value: "Asia/Shanghai",
				label: "(GMT+08:00) Beijing, Hong Kong, Shanghai"
			},
			{ value: "Asia/Singapore", label: "(GMT+08:00) Singapore" },
			{ value: "Asia/Tokyo", label: "(GMT+09:00) Tokyo, Seoul" },
			{
				value: "Australia/Sydney",
				label: "(GMT+10:00) Sydney, Melbourne"
			},
			{ value: "Australia/Brisbane", label: "(GMT+10:00) Brisbane" },
			{ value: "Australia/Adelaide", label: "(GMT+09:30) Adelaide" },
			{ value: "Australia/Perth", label: "(GMT+08:00) Perth" },
			{
				value: "Pacific/Auckland",
				label: "(GMT+12:00) Auckland, Wellington"
			}
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
		question: "Tell me about your professional background",
		type: "textarea",
		placeholder: "e.g., I'm a software developer at a startup...",
		section: "context",
		icon: <IconBriefcase />
	},
	{
		id: "working-hours",
		question: "What are your usual working hours?",
		type: "text-input",
		placeholder: "e.g., Mon-Fri, 9 AM to 6 PM",
		section: "context",
		icon: <IconClock />
	},
	{
		id: "key-people",
		question: "Who are the key people in your life I should remember?",
		type: "textarea",
		placeholder: "e.g., Jane Doe - spouse, John Smith - assistant",
		section: "context",
		icon: <IconUsers />
	},
	{
		id: "personal-context",
		question: "Any personal details you'd like me to remember?",
		type: "textarea",
		placeholder:
			"e.g., My anniversary is on June 5th. I love Italian food.",
		section: "context",
		icon: <IconHeart />
	}
]

const WhatsAppSettings = () => {
	const queryClient = useQueryClient()
	const [notificationNumber, setNotificationNumber] = useState("")

	const { data: settings, isLoading: isNotifLoading } = useQuery({
		queryKey: ["whatsappNotificationSettings"],
		queryFn: async () => {
			const response = await fetch("/api/settings/whatsapp-notifications")
			if (!response.ok)
				throw new Error(
					"Failed to fetch WhatsApp notification settings."
				)
			const data = await response.json()
			setNotificationNumber(data.whatsapp_notifications_number || "")
			return data
		}
	})

	const notificationsEnabled = settings?.notifications_enabled || false

	const saveNumberMutation = useMutation({
		mutationFn: (number) =>
			fetch("/api/settings/whatsapp-notifications", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					whatsapp_notifications_number: number
				})
			}).then(async (res) => {
				if (!res.ok) {
					const data = await res.json()
					throw new Error(data.detail || "Failed to save number.")
				}
				return res.json()
			}),
		onSuccess: () => {
			toast.success("Notifications enabled for this number!")
			queryClient.invalidateQueries({
				queryKey: ["whatsappNotificationSettings"]
			})
		},
		onError: (error) => toast.error(error.message)
	})

	const toggleNotificationsMutation = useMutation({
		mutationFn: (enabled) =>
			fetch("/api/settings/whatsapp-notifications/toggle", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ enabled })
			}).then(async (res) => {
				if (!res.ok) {
					const data = await res.json()
					throw new Error(
						data.detail || "Failed to update preference."
					)
				}
				return res.json()
			}),
		onSuccess: (data) => {
			toast.success(data.message)
			queryClient.invalidateQueries({
				queryKey: ["whatsappNotificationSettings"]
			})
		},
		onError: (error) => {
			toast.error(error.message)
			// Revert optimistic update on error by invalidating
			queryClient.invalidateQueries({
				queryKey: ["whatsappNotificationSettings"]
			})
		}
	})

	const hasNotifNumber =
		notificationNumber && notificationNumber.trim() !== ""

	return (
		<section>
			<h2 className="text-2xl font-bold mb-2 text-white flex items-center gap-3">
				<IconBrandWhatsapp />
				WhatsApp Notifications
			</h2>
			<p className="text-neutral-400 mb-6">
				Receive important notifications, task updates, and reminders on
				WhatsApp.
			</p>
			<div className="bg-neutral-900/50 p-6 rounded-2xl border border-neutral-800">
				<div className="space-y-4">
					<p className="text-neutral-400 text-sm">
						Receive important notifications, task updates, and
						reminders on WhatsApp. We're in the process of getting
						an official number, so for now, messages will come from
						our co-founder Sarthak (+91827507823), who may also
						occasionally reach out for feedback. Please enter your
						number with the country code.
					</p>
					{isNotifLoading ? (
						<div className="flex justify-center mt-4">
							<IconLoader className="w-6 h-6 animate-spin text-brand-orange" />
						</div>
					) : (
						<div className="space-y-4">
							<div className="flex items-center justify-between p-3 bg-neutral-800/30 rounded-lg">
								<label
									htmlFor="whatsapp-toggle"
									className="font-medium text-neutral-200"
								>
									Enable Notifications
								</label>
								<button
									id="whatsapp-toggle"
									onClick={() =>
										toggleNotificationsMutation.mutate(
											!notificationsEnabled
										)
									}
									disabled={!hasNotifNumber}
									className={cn(
										"relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-brand-orange focus:ring-offset-2 focus:ring-offset-neutral-900",
										"disabled:opacity-50 disabled:cursor-not-allowed",
										notificationsEnabled
											? "bg-green-500"
											: "bg-neutral-600"
									)}
								>
									<span
										aria-hidden="true"
										className={cn(
											"pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out",
											notificationsEnabled
												? "translate-x-5"
												: "translate-x-0"
										)}
									/>
								</button>
							</div>
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
									<Button
										onClick={() =>
											saveNumberMutation.mutate(
												notificationNumber
											)
										}
										disabled={
											saveNumberMutation.isPending ||
											!notificationNumber.trim()
										}
										className="gap-2 bg-brand-orange hover:bg-brand-orange/70 text-white font-medium"
									>
										{saveNumberMutation.isPending ? (
											<IconLoader className="w-4 h-4 mr-2 animate-spin" />
										) : (
											<IconPlus className="w-4 h-4 mr-2" />
										)}{" "}
										{hasNotifNumber ? "Update" : "Save"}
									</Button>
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
		General: [
			{ keys: ["Ctrl", "K"], description: "Open Search" },
			{
				keys: ["Ctrl", "Shift", "E"],
				description: "Toggle Notifications"
			},
			{ keys: ["Esc"], description: "Close Modal / Popup" }
		],
		Navigation: [
			{ keys: ["Ctrl", "Shift", "1"], description: "Go to Chat" },
			{ keys: ["Ctrl", "Shift", "2"], description: "Go to Tasks" },
			{ keys: ["Ctrl", "Shift", "3"], description: "Go to Memories" },
			{ keys: ["Ctrl", "Shift", "4"], description: "Go to Integrations" },
			{ keys: ["Ctrl", "Shift", "5"], description: "Go to Settings" }
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
			const result = await apiClient("/api/testing/inject-context", {
				method: "POST",
				body: JSON.stringify({
					service_name: serviceName,
					event_data: parsedData
				})
			})
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
			const result = await apiClient(
				"/api/testing/reprocess-onboarding",
				{
					method: "POST"
				}
			)
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
			const result = await apiClient("/api/testing/trigger-scheduler", {
				method: "POST"
			})
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
			const result = await apiClient("/api/testing/trigger-poller", {
				method: "POST"
			})
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
			const result = await apiClient("/api/testing/whatsapp/verify", {
				method: "POST",
				body: JSON.stringify({ phone_number: whatsAppNumber })
			})

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
			await apiClient("/api/testing/whatsapp", {
				method: "POST",
				body: JSON.stringify({ phone_number: whatsAppNumber })
			})
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
						<Button
							type="submit"
							disabled={isSubmitting}
							className="bg-brand-orange hover:bg-brand-orange/70 text-white font-medium"
						>
							{isSubmitting ? (
								<IconLoader className="w-5 h-5 animate-spin" />
							) : (
								"Inject Event"
							)}
						</Button>
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
					<Button
						onClick={handleReprocessOnboarding}
						disabled={isReprocessing}
						className="gap-2 bg-purple-600 hover:bg-purple-500 text-white font-medium"
					>
						{isReprocessing ? (
							<IconLoader className="w-5 h-5 animate-spin" />
						) : (
							<IconRefresh className="w-5 h-5" />
						)}{" "}
						Run Reprocessing
					</Button>
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
						<Button
							onClick={handleVerifyWhatsApp}
							disabled={isVerifying || isSending}
							className="bg-purple-600 hover:bg-purple-500 text-white font-medium"
						>
							{isVerifying ? (
								<IconLoader className="w-5 h-5 animate-spin" />
							) : (
								"Verify"
							)}
						</Button>
						<Button
							onClick={handleSendTestWhatsApp}
							disabled={isSending || isVerifying}
							className="bg-brand-orange hover:bg-brand-orange/70 text-white font-medium"
						>
							{isSending ? (
								<IconLoader className="w-5 h-5 animate-spin" />
							) : (
								"Send Test"
							)}
						</Button>
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
					<Button
						onClick={handleTestInApp}
						className="justify-center bg-blue-600 hover:bg-blue-500"
					>
						Test In-App Notification
					</Button>
					<Button
						onClick={handleTestPush}
						className="justify-center bg-green-600 hover:bg-green-500"
					>
						Test Push Notification
					</Button>
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
					<Button
						onClick={handleTriggerPoller}
						disabled={isTriggeringPoller}
						className="bg-purple-600 hover:bg-purple-500 text-white font-medium"
					>
						{isTriggeringPoller ? (
							<IconLoader className="w-5 h-5 animate-spin" />
						) : (
							"Run Poller Now"
						)}
					</Button>
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
					<Button
						onClick={handleTriggerScheduler}
						disabled={isTriggeringScheduler}
						className="bg-purple-600 hover:bg-purple-500 text-white font-medium"
					>
						{isTriggeringScheduler ? (
							<IconLoader className="w-5 h-5 animate-spin" />
						) : (
							"Run Scheduler Now"
						)}
					</Button>
				</div>
			</div>
		</section>
	)
}

const ProfileSettings = ({ initialData, onSave, isSaving }) => {
	const [formData, setFormData] = useState({})

	// biome-ignore lint/correctness/useExhaustiveDependencies: We only want this to run when initialData changes
	React.useEffect(() => {
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
				<Button
					onClick={() => onSave(formData)}
					disabled={isSaving}
					className="mt-4 sm:mt-0 gap-2 bg-brand-orange hover:bg-brand-orange/70 text-white font-medium"
				>
					{isSaving ? (
						<IconLoader className="w-5 h-5 animate-spin" />
					) : (
						"Save Profile"
					)}
				</Button>
			</div>

			<div className="space-y-10 bg-neutral-900/50 p-3 rounded-2xl border border-neutral-800">
				{Object.entries(questionSections).map(
					([key, { title, icon }], index) => (
						<Accordion
							key={key}
							type="single"
							collapsible
							defaultValue={`item-${index}`}
						>
							<AccordionItem value={`item-${index}`}>
								<AccordionTrigger>
									<h3 className="text-lg font-semibold text-neutral-200 flex items-center gap-3">
										{icon} {title}
									</h3>
								</AccordionTrigger>
								<AccordionContent>
									<div className="space-y-6 mt-4">
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
																				q
																					.id
																			] ||
																			""
																		}
																		onChange={(
																			e
																		) =>
																			handleAnswer(
																				q.id,
																				e
																					.target
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
																				q
																					.id
																			] ||
																			""
																		}
																		onChange={(
																			e
																		) =>
																			handleAnswer(
																				q.id,
																				e
																					.target
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
															case "select": {
																let options =
																	q.options
																if (
																	q.id ===
																	"timezone"
																) {
																	const savedTimezone =
																		formData[
																			q.id
																		]
																	const isTimezoneInOptions =
																		q.options.some(
																			(
																				opt
																			) =>
																				opt.value ===
																				savedTimezone
																		)
																	if (
																		savedTimezone &&
																		!isTimezoneInOptions
																	) {
																		// Clone to avoid mutating the original questions array
																		options =
																			[
																				...q.options
																			]
																		options.unshift(
																			{
																				value: savedTimezone,
																				label: savedTimezone.replace(
																					/_/g,
																					" "
																				)
																			}
																		)
																	}
																}

																return (
																	// eslint-disable-line
																	<select
																		value={
																			formData[
																				q
																					.id
																			] ||
																			""
																		}
																		onChange={(
																			e
																		) =>
																			handleAnswer(
																				q.id,
																				e
																					.target
																					.value
																			)
																		}
																		className="w-full bg-neutral-800/50 border font-mono border-neutral-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-brand-orange appearance-none"
																	>
																		{options.map(
																			(
																				opt
																			) => (
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
															}
															case "location": // Simplified for now
																const locationValue =
																	formData[
																		q.id
																	]
																const isGpsLocation =
																	typeof locationValue ===
																		"object" &&
																	locationValue !==
																		null &&
																	locationValue.latitude

																if (
																	isGpsLocation
																) {
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
																				q
																					.id
																			] ||
																			""
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
								</AccordionContent>
							</AccordionItem>
						</Accordion>
					)
				)}
			</div>
		</section>
	)
}

export default function SettingsPage() {
	const queryClient = useQueryClient()

	const {
		user: profileData,
		isLoading: isProfileLoading,
		fetchUserData
	} = useUserStore()

	const saveProfileMutation = useMutation({
		mutationFn: (newOnboardingData) => {
			return fetch("/api/settings/profile", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					onboardingAnswers: newOnboardingData,
					personalInfo: {
						name: newOnboardingData["user-name"],
						location: newOnboardingData["location"],
						timezone: newOnboardingData["timezone"]
					},
					preferences: {}
				})
			}).then(async (res) => {
				if (!res.ok) {
					const errorData = await res.json()
					throw new Error(errorData.error || "Failed to save profile")
				}
				return res.json()
			})
		},
		onSuccess: () => {
			toast.success("Profile updated successfully!")
			queryClient.invalidateQueries({ queryKey: ["userProfileData"] })
		},
		onError: (error) =>
			toast.error(`Error saving profile: ${error.message}`)
	})

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

				<header className="flex items-center justify-between p-4 sm:p-6 md:px-8 md:py-6 bg-transparent shrink-0">
					<div>
						<h1 className="text-3xl lg:text-4xl font-bold text-white">
							Settings
						</h1>
					</div>
				</header>

				<main className="flex-1 overflow-y-auto px-4 sm:px-6 md:px-10 pb-4 sm:pb-6 md:pb-10 custom-scrollbar">
					<div className="w-full max-w-4xl mx-auto space-y-12">
						{isProfileLoading ? (
							<div className="flex justify-center items-center h-64">
								<IconLoader className="animate-spin text-brand-orange" />
							</div>
						) : (
							<ProfileSettings
								initialData={profileData?.onboardingAnswers}
								onSave={(data) =>
									saveProfileMutation.mutate(data)
								}
								isSaving={saveProfileMutation.isPending}
							/>
						)}
						<div data-tour-id="notifications-section">
							<WhatsAppSettings />
						</div>
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
