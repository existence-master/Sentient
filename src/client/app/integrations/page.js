"use client"

import React, { useState, useCallback, useMemo, useRef, useEffect } from "react"
import SparkleEffect from "@components/ui/SparkleEffect"
import { BorderTrail } from "@components/ui/border-trail"
import toast from "react-hot-toast"
import {
	IconLoader,
	IconSettingsCog,
	IconBrandGoogleDrive,
	IconBrandSlack,
	IconBrandDiscord,
	IconBrandNotion,
	IconPlugConnected,
	IconPlugOff,
	IconPlus,
	IconCloud,
	IconBrandTrello,
	IconChartPie,
	IconBrain,
	IconBrandGithub,
	IconNews,
	IconFileText,
	IconFile,
	IconPresentation,
	IconTable,
	IconMapPin,
	IconShoppingCart,
	IconX,
	IconMail,
	IconBrandWhatsapp,
	IconUsers,
	IconHelpCircle,
	IconCalendarEvent,
	IconWorldSearch,
	IconSearch,
	IconBolt,
	IconSparkles,
	IconAlertTriangle,
	IconEye,
	IconPlug,
	IconArrowUpCircle,
	IconCheck,
	IconListCheck
} from "@tabler/icons-react"
import { cn } from "@utils/cn"
import InteractiveNetworkBackground from "@components/ui/InteractiveNetworkBackground"
import { motion, AnimatePresence } from "framer-motion"
import {
	MorphingDialog,
	MorphingDialogTrigger,
	MorphingDialogContent,
	MorphingDialogTitle,
	MorphingDialogSubtitle,
	MorphingDialogClose,
	MorphingDialogDescription,
	MorphingDialogContainer
} from "@components/ui/morphing-dialog"
import { Tooltip } from "react-tooltip"
import {
	ModalDialog,
	ModalHeader,
	ModalTitle,
	ModalCloseButton,
	ModalBody,
	ModalFooter
} from "@components/ui/ModalDialog"
import { useRouter } from "next/navigation"
import { INTEGRATION_CAPABILITIES } from "@utils/integration-capabilities"
import { Button } from "@components/ui/button"
import { Card, CardContent, CardFooter, CardHeader } from "@components/ui/card"
import apiClient from "@lib/apiClient"
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import {
	useUIStore,
	useUserStore,
	useIntegrationStore
} from "@stores/app-stores"
import { usePostHog } from "posthog-js/react"

const integrationColorIcons = {
	gmail: IconMail,
	gcalendar: IconCalendarEvent,
	gpeople: IconUsers,
	internet_search: IconWorldSearch,
	gdrive: IconBrandGoogleDrive,
	gdocs: IconFileText,
	gtasks: IconListCheck,
	gslides: IconPresentation,
	gsheets: IconTable,
	gmaps: IconMapPin,
	slack: IconBrandSlack,
	notion: IconBrandNotion,
	accuweather: IconCloud,
	quickchart: IconChartPie,
	memory: IconBrain,
	google_search: IconWorldSearch,
	trello: IconBrandTrello,
	github: IconBrandGithub,
	news: IconNews,
	discord: IconBrandDiscord,
	whatsapp: IconBrandWhatsapp,
	file_management: IconFile
}

const IconPlaceholder = IconSettingsCog

const PRO_ONLY_INTEGRATIONS = [
	"notion",
	"github",
	"slack",
	"discord",
	"trello",
	"whatsapp",
	"gpeople",
	"gslides"
]

const proPlanFeatures = [
	{ name: "Text Chat", limit: "100 messages per day" },
	{ name: "Voice Chat", limit: "10 minutes per day" },
	{ name: "Async Tasks", limit: "100 tasks per month" },
	{ name: "Active Workflows", limit: "25 recurring & triggered" },
	{
		name: "Parallel Agents",
		limit: "5 complex tasks per day with 50 sub agents"
	},
	{ name: "File Uploads", limit: "20 files per day" },
	{ name: "Memories", limit: "Unlimited memories" },
	{
		name: "Other Integrations",
		limit: "Notion, GitHub, Slack, Discord, Trello"
	}
]

const UpgradeToProModal = ({ isOpen, onClose }) => {
	if (!isOpen) return null

	const handleUpgrade = () => {
		const dashboardUrl = process.env.NEXT_PUBLIC_LANDING_PAGE_URL
		if (dashboardUrl) {
			window.location.href = `${dashboardUrl}/dashboard`
		}
		onClose()
	}

	return (
		<AnimatePresence>
			{isOpen && (
				<motion.div
					initial={{ opacity: 0 }}
					animate={{ opacity: 1 }}
					exit={{ opacity: 0 }}
					className="fixed inset-0 bg-black/70 backdrop-blur-md z-[100] flex items-center justify-center p-4"
					onClick={onClose}
				>
					<motion.div
						initial={{ scale: 0.95, y: 20 }}
						animate={{ scale: 1, y: 0 }}
						exit={{ scale: 0.95, y: -20 }}
						transition={{ duration: 0.2, ease: "easeInOut" }}
						onClick={(e) => e.stopPropagation()}
						className="relative bg-neutral-900/90 backdrop-blur-xl p-6 rounded-2xl shadow-2xl w-full max-w-lg border border-neutral-700 flex flex-col"
					>
						<header className="text-center mb-4">
							<h2 className="text-2xl font-bold text-white flex items-center justify-center gap-2">
								<IconBolt className="text-yellow-400" />
								Unlock Pro Features
							</h2>
							<p className="text-neutral-400 mt-2">
								Unlock powerful features to conquer your day.
							</p>
						</header>
						<main className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-4 my-4">
							{proPlanFeatures.map((feature) => (
								<div
									key={feature.name}
									className="flex items-start gap-2.5"
								>
									<IconCheck
										size={18}
										className="text-green-400 flex-shrink-0 mt-0.5"
									/>
									<div>
										<p className="text-white text-sm font-medium">
											{feature.name}
										</p>
										<p className="text-neutral-400 text-xs">
											{feature.limit}
										</p>
									</div>
								</div>
							))}
						</main>
						<footer className="mt-4 flex flex-col gap-2">
							<Button
								onClick={handleUpgrade}
								className="w-full bg-brand-orange hover:bg-brand-orange/90 text-brand-black font-semibold"
							>
								Upgrade Now - $9/month
							</Button>
							<Button onClick={onClose} variant="ghost">
								Cancel
							</Button>
						</footer>
					</motion.div>
				</motion.div>
			)}
		</AnimatePresence>
	)
}

const WhatsAppDisclaimerModal = ({ isOpen, onAgree, onClose }) => {
	// This modal doesn't need to know if it's open, the parent handles it.
	// But we keep the prop for clarity and potential internal logic.
	if (!isOpen) return null

	return (
		<ModalDialog isOpen={isOpen} onClose={onClose}>
			<ModalHeader>
				<ModalTitle>
					<div className="flex items-center gap-2">
						<IconBrandWhatsapp />
						<span>WhatsApp Disclaimer</span>
					</div>
				</ModalTitle>
			</ModalHeader>
			<ModalBody>
				<p>
					Please review the following before connecting your WhatsApp
					account.
				</p>
				<div className="text-sm text-neutral-300 space-y-3 pt-4">
					<p>
						By proceeding, you acknowledge that you have read and
						agree to the{" "}
						<a
							href="https://www.whatsapp.com/legal/terms-of-service"
							target="_blank"
							rel="noopener noreferrer"
							className="text-blue-400 hover:underline"
						>
							WhatsApp Terms of Service by Meta
						</a>
						.
					</p>
					<p>
						Connecting this integration allows Sentient to act on
						your behalf to: read your messages, send messages, and
						manage your chats and contacts.
					</p>
				</div>
			</ModalBody>
			<ModalFooter>
				<Button variant="secondary" onClick={onClose}>
					Cancel
				</Button>
				<Button
					onClick={onAgree}
					className="bg-brand-orange hover:bg-brand-orange/90 text-brand-black font-semibold"
				>
					Agree and Connect
				</Button>
			</ModalFooter>
		</ModalDialog>
	)
}

const WhatsAppQRCodeModal = ({ onClose }) => {
	const [qrCode, setQrCode] = useState(null)
	const [status, setStatus] = useState("initiating") // initiating, scanning, working,const WhatsAppQRCodeModal = ({ onClose }) => {
	const [error, setError] = useState("")
	const intervalRef = useRef(null)

	const stopPolling = useCallback(() => {
		if (intervalRef.current) {
			clearInterval(intervalRef.current)
			intervalRef.current = null
		}
	}, [])

	const pollStatus = useCallback(async () => {
		try {
			const data = await apiClient(
				"/api/settings/integrations/whatsapp/status"
			)
			if (data.status === "WORKING") {
				setStatus("working")
				toast.success("WhatsApp connected successfully!")
				stopPolling()
				setTimeout(onClose, 1500)
			} else if (data.status === "FAILED") {
				setError("Connection failed. Please close this and try again.")
				setStatus("error")
				stopPolling()
			} else {
				setStatus("scanning")
			}
		} catch (err) {
			setError(err.message)
			setStatus("error")
			stopPolling()
		}
	}, [onClose, stopPolling])

	const initiateConnection = useCallback(async () => {
		setStatus("initiating")
		setError("")
		try {
			const data = await apiClient(
				"/api/settings/integrations/whatsapp/connect/initiate",
				{ method: "POST" }
			)
			setQrCode(data.data)
			setStatus("scanning")

			// Start polling for status if not already started
			if (!intervalRef.current) {
				intervalRef.current = setInterval(pollStatus, 3000)
			}
		} catch (err) {
			setError(err.message)
			setStatus("error")
		}
	}, [pollStatus])

	useEffect(() => {
		initiateConnection()
		return () => {
			stopPolling()
		}
	}, [initiateConnection, stopPolling])

	return (
		<motion.div
			initial={{ opacity: 0 }}
			animate={{ opacity: 1 }}
			exit={{ opacity: 0 }}
			className="fixed inset-0 bg-black/70 backdrop-blur-md z-[110] flex items-center justify-center p-4"
			onClick={onClose}
		>
			<motion.div
				initial={{ scale: 0.95, y: 20 }}
				animate={{ scale: 1, y: 0 }}
				exit={{ scale: 0.95, y: -20 }}
				transition={{ duration: 0.2, ease: "easeInOut" }}
				onClick={(e) => e.stopPropagation()}
				className="relative bg-neutral-900/90 backdrop-blur-xl p-6 rounded-2xl shadow-2xl w-full max-w-sm border border-neutral-700 flex flex-col items-center"
			>
				<header className="text-center mb-4">
					<h2 className="text-lg font-semibold text-white">
						Connect WhatsApp
					</h2>
					<p className="text-sm text-neutral-400 mt-1">
						Scan this QR code with your WhatsApp mobile app.
					</p>
				</header>
				<main className="w-64 h-64 bg-neutral-800 rounded-lg flex items-center justify-center">
					{status === "initiating" && (
						<IconLoader className="animate-spin text-brand-orange" />
					)}
					{status === "scanning" && qrCode && (
						<img
							src={`data:image/png;base64,${qrCode}`}
							alt="WhatsApp QR Code"
						/>
					)}
					{status === "working" && (
						<div className="flex flex-col items-center gap-2 text-green-400">
							<IconCheck size={48} />
							<p className="font-semibold">Connected!</p>
						</div>
					)}
					{status === "error" && (
						<div className="text-center text-red-400 p-4">
							<IconAlertTriangle
								size={32}
								className="mx-auto mb-2"
							/>
							<p className="text-sm">{error}</p>
						</div>
					)}
				</main>
				<footer className="mt-6 text-center">
					<Button onClick={onClose} variant="secondary">
						Cancel
					</Button>
				</footer>
			</motion.div>
		</motion.div>
	)
}

const FilterInputSection = ({
	title,
	description,
	items,
	onAdd,
	onDelete,
	placeholder
}) => {
	const [inputValue, setInputValue] = useState("")

	const handleAdd = () => {
		if (inputValue.trim()) {
			onAdd(inputValue)
			setInputValue("")
		}
	}

	return (
		<div className="bg-[var(--color-primary-surface)]/50 p-4 rounded-lg border border-[var(--color-primary-surface-elevated)]">
			<h4 className="text-md font-semibold text-gray-200 mb-1">
				{title}
			</h4>
			{description && (
				<p className="text-gray-400 text-xs mb-3">{description}</p>
			)}
			<div className="flex flex-col sm:flex-row gap-2 mb-4">
				<input
					type="text"
					value={inputValue}
					onChange={(e) => setInputValue(e.target.value)}
					onKeyDown={(e) => e.key === "Enter" && handleAdd()}
					placeholder={placeholder}
					className="flex-grow bg-[var(--color-primary-surface-elevated)] border border-neutral-600 rounded-md px-3 py-2 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-[var(--color-accent-blue)] w-full"
				/>
				<Button
					onClick={handleAdd}
					className="gap-2 bg-brand-orange hover:bg-brand-orange/80 text-brand-black font-semibold"
				>
					<IconPlus size={16} /> Add
				</Button>
			</div>
			<div className="flex flex-wrap gap-2">
				{items.length > 0 ? (
					items.map((item, index) => (
						<div
							key={index}
							className="flex items-center gap-2 bg-[var(--color-primary-surface-elevated)] rounded-full py-1.5 px-3 text-sm text-gray-200"
						>
							<span>{item}</span>
							<button onClick={() => onDelete(item)}>
								<IconX
									size={14}
									className="text-gray-500 hover:text-red-400"
								/>
							</button>
						</div>
					))
				) : (
					<p className="text-sm text-gray-500">
						No filters added yet.
					</p>
				)}
			</div>
		</div>
	)
}

const PrivacySettings = ({ serviceName }) => {
	const queryClient = useQueryClient()

	const { data: filters, isLoading } = useQuery({
		queryKey: ["privacyFilters", serviceName],
		queryFn: async () => {
			const response = await fetch(
				`/api/settings/privacy-filters?service=${serviceName}`
			)
			if (!response.ok) throw new Error("Failed to fetch filters.")
			const data = await response.json()
			return data.filters
		},
		initialData: { keywords: [], emails: [], labels: [] }
	})

	const updateFiltersMutation = useMutation({
		mutationFn: (updatedFilters) =>
			fetch("/api/settings/privacy-filters", {
				method: "POST",
				body: JSON.stringify({
					service: serviceName,
					filters: updatedFilters
				})
			}).then((res) => {
				if (!res.ok) throw new Error("Failed to save filters.")
				return res.json()
			}),
		onMutate: async (updatedFilters) => {
			await queryClient.cancelQueries({
				queryKey: ["privacyFilters", serviceName]
			})
			const previousFilters = queryClient.getQueryData([
				"privacyFilters",
				serviceName
			])
			queryClient.setQueryData(
				["privacyFilters", serviceName],
				updatedFilters
			)
			return { previousFilters }
		},
		onError: (err, newFilters, context) => {
			queryClient.setQueryData(
				["privacyFilters", serviceName],
				context.previousFilters
			)
			toast.error(err.message)
		},
		onSettled: () => {
			queryClient.invalidateQueries({
				queryKey: ["privacyFilters", serviceName]
			})
		},
		onSuccess: () => {
			toast.success("Privacy filters updated.")
		}
	})

	const handleAddItem = (type, value) => {
		if (!filters[type].includes(value)) {
			const updatedFilters = {
				...filters,
				[type]: [...filters[type], value]
			}
			updateFiltersMutation.mutate(updatedFilters)
		}
	}

	const handleDeleteItem = (type, value) => {
		const updatedFilters = {
			...filters,
			[type]: filters[type].filter((item) => item !== value)
		}
		updateFiltersMutation.mutate(updatedFilters)
	}

	if (isLoading) {
		return (
			<div className="flex justify-center p-8">
				<IconLoader className="w-6 h-6 animate-spin text-[var(--color-accent-blue)]" />
			</div>
		)
	}

	return (
		<div className="space-y-6">
			<FilterInputSection
				title="Keyword Filters"
				description="Emails or events containing these keywords will be ignored by the proactive memory pipeline."
				items={filters.keywords}
				onAdd={(value) => handleAddItem("keywords", value)}
				onDelete={(value) => handleDeleteItem("keywords", value)}
				placeholder="Add a new keyword..."
			/>
			{serviceName === "gmail" ? (
				<div className="grid grid-cols-1 md:grid-cols-2 gap-6">
					<FilterInputSection
						title="Blocked Senders"
						description="Emails from these addresses will be ignored."
						items={filters.emails}
						onAdd={(value) => handleAddItem("emails", value)}
						onDelete={(value) => handleDeleteItem("emails", value)}
						placeholder="Add an email..."
					/>
					<FilterInputSection
						title="Blocked Labels"
						description="Emails with these labels will be ignored."
						items={filters.labels}
						onAdd={(value) => handleAddItem("labels", value)}
						onDelete={(value) => handleDeleteItem("labels", value)}
						placeholder="Add a label..."
					/>
				</div>
			) : serviceName === "gcalendar" ? (
				<FilterInputSection
					title="Blocked Attendees"
					description="Events containing any of these attendees (by email) will be ignored."
					items={filters.emails}
					onAdd={(value) => handleAddItem("emails", value)}
					onDelete={(value) => handleDeleteItem("emails", value)}
					placeholder="Add an attendee's email..."
				/>
			) : null}
		</div>
	)
}

const PrivacySettingsModal = ({ serviceName, onClose }) => {
	const capitalizedServiceName =
		serviceName.charAt(0).toUpperCase() + serviceName.slice(1)

	return (
		<motion.div
			initial={{ opacity: 0 }}
			animate={{ opacity: 1 }}
			exit={{ opacity: 0 }}
			className="fixed inset-0 bg-black/70 backdrop-blur-md z-[110] flex items-center justify-center p-4"
			onClick={onClose}
		>
			<motion.div
				initial={{ scale: 0.95, y: 20 }}
				animate={{ scale: 1, y: 0 }}
				exit={{ scale: 0.95, y: -20 }}
				transition={{ duration: 0.2, ease: "easeInOut" }}
				onClick={(e) => e.stopPropagation()}
				className="relative bg-neutral-900/90 backdrop-blur-xl p-6 rounded-2xl shadow-2xl w-full max-w-2xl border border-neutral-700 max-h-[80vh] flex flex-col"
			>
				<header className="flex justify-between items-center mb-4 flex-shrink-0">
					<h2 className="text-lg font-semibold text-white">
						Privacy Filters for {capitalizedServiceName}
					</h2>
					<button
						onClick={onClose}
						className="p-1.5 rounded-full hover:bg-neutral-700"
					>
						<IconX size={18} />
					</button>
				</header>
				<main className="flex-1 overflow-y-auto custom-scrollbar pr-2">
					<PrivacySettings serviceName={serviceName} />
				</main>
				<footer className="mt-6 pt-4 border-t border-neutral-800 flex justify-end">
					<Button onClick={onClose} variant="secondary">
						Done
					</Button>
				</footer>
			</motion.div>
		</motion.div>
	)
}

const InfoPanel = ({ onClose, title, children }) => (
	<motion.div
		initial={{ opacity: 0, backdropFilter: "blur(0px)" }}
		animate={{ opacity: 1, backdropFilter: "blur(12px)" }}
		exit={{ opacity: 0, backdropFilter: "blur(0px)" }}
		className="fixed inset-0 bg-black/70 z-[60] flex items-center justify-center p-4 md:p-6"
		onClick={onClose}
	>
		<motion.div
			initial={{ opacity: 0, y: 20 }}
			animate={{ opacity: 1, y: 0 }}
			exit={{ opacity: 0, y: 20 }}
			transition={{ duration: 0.2, ease: "easeInOut" }}
			onClick={(e) => e.stopPropagation()}
			className="relative bg-neutral-900 backdrop-blur-lg p-6 rounded-2xl shadow-lg w-full max-w-3xl max-h-[80vh] md:max-h-[700px] border border-neutral-700 flex flex-col"
		>
			<header className="flex justify-between items-center mb-6 flex-shrink-0">
				<h2 className="text-lg font-semibold text-white flex items-center gap-2">
					{title}
				</h2>
				<button
					onClick={onClose}
					className="p-1.5 rounded-full hover:bg-neutral-700"
				>
					<IconX size={18} />
				</button>
			</header>
			<main className="flex-1 overflow-y-auto custom-scrollbar pr-2 text-left space-y-6">
				{children}
			</main>
		</motion.div>
	</motion.div>
)

const IntegrationHeader = ({
	searchQuery,
	onSearchChange,
	categoriesToShow,
	activeCategory,
	onCategoryChange
}) => {
	return (
		<div className="xs:mb-4 md:mb-8 mt-1 md:sticky md:top-0 bg-dark-surface/80 backdrop-blur-sm md:py-4 z-10">
			{/* Redesigned Search Bar */}
			<div className="relative">
				<IconSearch
					className="absolute left-4 top-1/2 -translate-y-1/2 text-neutral-500"
					size={20}
				/>
				<input
					type="text"
					placeholder="Search integrations..."
					value={searchQuery}
					onChange={(e) => onSearchChange(e.target.value)}
					className="w-full bg-neutral-900 border border-neutral-700 rounded-full pl-12 pr-4 py-3 text-white placeholder-neutral-500 focus:ring-2 focus:ring-brand-orange"
				/>
			</div>

			{/* Filter Pills */}
			<div className="mt-4 flex flex-wrap gap-2">
				{categoriesToShow.map((category) => (
					<Button
						key={category}
						onClick={() => onCategoryChange(category)}
						variant={
							activeCategory === category
								? "default"
								: "secondary"
						}
						className={cn(
							"rounded-full",
							activeCategory === category &&
								"bg-brand-orange text-black hover:bg-brand-orange/90",
							activeCategory !== category && "text-neutral-300"
						)}
					>
						{category}
					</Button>
				))}
			</div>
		</div>
	)
}

const IntegrationTag = ({ type }) => {
	const styles = {
		Native: "bg-green-500/20 text-green-300",
		"3rd Party": "bg-neutral-600/50 text-neutral-300"
	}
	return (
		<span
			className={cn(
				"px-2 py-0.5 rounded-full text-xs font-semibold",
				styles[type]
			)}
		>
			{type}
		</span>
	)
}

const IntegrationCard = ({
	integration,
	icon: Icon,
	isProFeature,
	isProUser,
	onUpgradeClick
}) => {
	const getTagType = (authType) => {
		if (authType === "builtin") return "Native"
		if (["oauth", "manual", "composio"].includes(authType))
			return "3rd Party"
		return null
	}

	const tagType = getTagType(integration.auth_type)

	const isConnectable = ["oauth", "manual", "composio"].includes(
		integration.auth_type
	)

	const isConnected =
		integration.connected || integration.auth_type === "builtin"

	const isDisabledForFree = isProFeature && !isProUser

	return (
		<Card className="transition-all duration-300 hover:border-brand-orange hover:-translate-y-1 flex flex-col text-left h-full">
			{/* Top Section */}
			<CardHeader className="flex-row items-start justify-between pb-4">
				<div className="flex items-center gap-3">
					<div className="w-10 h-10 flex items-center justify-center rounded-lg bg-brand-gray p-1.5 text-brand-orange">
						<Icon className="w-full h-full" />
					</div>
					<div>
						<h3 className="font-semibold text-white text-base">
							{integration.display_name}
						</h3>
						<span
							className={cn(
								"text-xs font-semibold",
								isConnected
									? "text-green-400"
									: "text-neutral-500"
							)}
						>
							{isConnected ? "Connected" : "Not Connected"}
						</span>
					</div>
				</div>
				<div className="flex flex-col items-end gap-1">
					{tagType && <IntegrationTag type={tagType} />}
					{isProFeature && (
						<div
							className={cn(
								"p-1.5 rounded-full text-xs font-semibold",
								"bg-yellow-500/20 text-yellow-300"
							)}
						>
							<IconBolt size={12} />
						</div>
					)}
				</div>
			</CardHeader>

			{/* Middle Section */}
			<CardContent className="flex-grow pt-0">
				<p className="text-sm text-gray-400 mt-1 line-clamp-3">
					{integration.description}
				</p>
			</CardContent>

			{/* Bottom Section */}
			{isConnectable && (
				<CardFooter className="pt-4 border-t border-neutral-800 flex justify-end">
					{isDisabledForFree ? (
						<Button
							onClick={onUpgradeClick}
							variant="link"
							className="text-brand-orange group-hover:text-yellow-300"
						>
							<IconArrowUpCircle size={16} className="mr-1.5" />
							Upgrade to Unlock
						</Button>
					) : (
						<span className="text-sm font-medium text-neutral-400 group-hover:text-white transition-colors">
							View Details →
						</span>
					)}
				</CardFooter>
			)}
		</Card>
	)
}

// Define the missing constant to prevent reference errors.
const MANUAL_INTEGRATION_CONFIGS = {}

const IntegrationsPage = () => {
	const queryClient = useQueryClient()
	const {
		searchQuery,
		activeCategory,
		privacyModalService,
		disconnectingIntegration,
		setSearchQuery,
		setActiveCategory,
		openPrivacyModal,
		closePrivacyModal,
		setDisconnectingIntegration
	} = useIntegrationStore()
	const { isUpgradeModalOpen, openUpgradeModal, closeUpgradeModal } =
		useUIStore()
	const { isPro } = useUserStore()
	const [processingIntegration, setProcessingIntegration] = useState(null)
	const [selectedIntegration, setSelectedIntegration] = useState(null)
	const [activeManualIntegration, setActiveManualIntegration] = useState(null)
	const [isWhatsAppDisclaimerOpen, setIsWhatsAppDisclaimerOpen] =
		useState(false)
	const [isWhatsAppQRModalOpen, setIsWhatsAppQRModalOpen] = useState(false)
	const [sparkleTrigger, setSparkleTrigger] = useState(0)
	const [isInfoPanelOpen, setIsInfoPanelOpen] = useState(false)
	const posthog = usePostHog()
	const router = useRouter()

	const googleServices = [
		"gmail",
		"gcalendar",
		"gdrive",
		"gdocs",
		"gslides",
		"gsheets",
		"gmaps",
		"gpeople"
	]

	const {
		data: integrationsData,
		isLoading: loading,
		isError
	} = useQuery({
		queryKey: ["integrations"],
		queryFn: async () => {
			const response = await fetch("/api/settings/integrations", {
				method: "POST",
				cache: "no-store"
			})
			if (!response.ok) throw new Error("Failed to fetch integrations")
			return response.json()
		}
	})

	const { userIntegrations, defaultTools } = useMemo(() => {
		if (!integrationsData) return { userIntegrations: [], defaultTools: [] }

		const integrationsWithIcons = (integrationsData.integrations || []).map(
			(ds) => ({
				...ds,
				icon: integrationColorIcons[ds.name] || IconSettingsCog
			})
		)

		const hiddenTools = [
			"google_search",
			"progress_updater",
			"chat_tools",
			"tasks"
		]
		const connectable = integrationsWithIcons.filter(
			(i) =>
				(i.auth_type === "oauth" ||
					i.auth_type === "manual" ||
					i.auth_type === "composio") &&
				!hiddenTools.includes(i.name)
		)
		const builtIn = integrationsWithIcons.filter(
			(i) => i.auth_type === "builtin" && !hiddenTools.includes(i.name)
		)
		return { userIntegrations: connectable, defaultTools: builtIn }
	}, [integrationsData])

	const handleUpgradeClick = () => {
		openUpgradeModal()
	}

	const handleConnect = async (integration) => {
		const isProFeature = PRO_ONLY_INTEGRATIONS.includes(integration.name)
		if (isProFeature && !isPro) {
			handleUpgradeClick()
			return
		}

		// If it's a pro feature, refresh the session cookie before redirecting
		// to ensure the backend gets a fresh token with the correct roles.
		if (isProFeature) {
			const toastId = toast.loading("Preparing secure connection...")
			try {
				await apiClient("/api/auth/refresh-session")
				toast.dismiss(toastId)
			} catch (e) {
				toast.error("Could not prepare connection. Please try again.", {
					id: toastId
				})
				return // Stop if refresh fails
			}
		}

		if (integration.auth_type === "composio") {
			handleComposioConnect(integration)
			return
		}

		if (integration.auth_type === "oauth") {
			const { name: serviceName, client_id: clientId } = integration
			if (!clientId) {
				toast.error(
					`Client ID for ${integration.display_name} is not configured.`
				)
				return
			}

			if (serviceName === "trello") {
				// Trello uses an implicit grant flow where the token is returned in the URL fragment.
				const returnUrl = `${window.location.origin}/integrations` // Redirect back to this page to handle the fragment.
				const scope = "read,write" // Request read and write permissions for creating cards.
				const authUrl = `https://trello.com/1/authorize?expiration=never&scope=${scope}&response_type=token&key=${clientId}&return_url=${encodeURIComponent(returnUrl)}&callback_method=fragment`
				window.location.href = authUrl
				return // Stop execution for Trello as it's a redirect.
			}

			const redirectUri = `${window.location.origin}/api/settings/integrations/connect/oauth/callback`
			let authUrl = ""
			const scopes = {
				gdrive: "https://www.googleapis.com/auth/drive",
				gcalendar: "https://www.googleapis.com/auth/calendar",
				gmail: "https://mail.google.com/",
				gdocs: "https://www.googleapis.com/auth/documents https://www.googleapis.com/auth/drive",
				gslides:
					"https://www.googleapis.com/auth/presentations https://www.googleapis.com/auth/drive",
				gsheets: "https://www.googleapis.com/auth/spreadsheets",
				gmaps: "https://www.googleapis.com/auth/cloud-platform",
				gpeople: "https://www.googleapis.com/auth/contacts",
				github: "repo user",
				notion: "read_content write_content insert_content", // This is not a scope, it's just for user to know. Notion doesn't use scopes in the URL.
				slack: "channels:history,channels:read,chat:write,users:read,reactions:write"
			}
			const scope =
				scopes[serviceName] ||
				"https://www.googleapis.com/auth/userinfo.email"
			if (googleServices.includes(serviceName)) {
				authUrl = `https://accounts.google.com/o/oauth2/v2/auth?client_id=${clientId}&redirect_uri=${encodeURIComponent(
					redirectUri
				)}&response_type=code&scope=${encodeURIComponent(scope)}&access_type=offline&prompt=consent&state=${serviceName}`
			} else if (serviceName === "github") {
				// For GitHub, it's safer to omit the redirect_uri and let it use the default
				// configured in the OAuth App settings to avoid mismatches.
				authUrl = `https://github.com/login/oauth/authorize?client_id=${clientId}&scope=${encodeURIComponent(scope)}&state=${serviceName}`
			} else if (serviceName === "slack") {
				authUrl = `https://slack.com/oauth/v2/authorize?client_id=${clientId}&user_scope=${encodeURIComponent(
					scope
				)}&redirect_uri=${encodeURIComponent(redirectUri)}&state=${serviceName}`
			} else if (serviceName === "notion") {
				// Notion's `owner` parameter is important
				authUrl = `https://api.notion.com/v1/oauth/authorize?client_id=${clientId}&redirect_uri=${encodeURIComponent(
					redirectUri
				)}&response_type=code&owner=user&state=${serviceName}`
			} else if (serviceName === "discord") {
				// Scopes for Discord: identify (read user info), guilds (list servers), bot (add bot to servers), applications.commands (for slash commands)
				const scope = "identify guilds bot applications.commands"
				// Permissions for the bot
				const permissions = "580851377359936"
				authUrl = `https://discord.com/api/oauth2/authorize?client_id=${clientId}&redirect_uri=${encodeURIComponent(
					redirectUri
				)}&response_type=code&scope=${encodeURIComponent(scope)}&permissions=${permissions}&state=${serviceName}`
			}
			if (authUrl) window.location.href = authUrl
			else
				toast.error(
					`OAuth flow for ${integration.display_name} is not implemented.`
				)
		} else if (integration.auth_type === "manual") {
			if (MANUAL_INTEGRATION_CONFIGS[integration.name]) {
				setActiveManualIntegration(integration)
			} else {
				toast.error(`UI for ${integration.display_name} not found.`)
			}
		}
	}

	const handleComposioConnect = async (integration) => {
		const { name: serviceName, auth_config_id: authConfigId } = integration
		if (!authConfigId) {
			toast.error(
				`Auth Config ID for ${integration.display_name} is not configured.`
			)
			return
		}
		setProcessingIntegration(serviceName)
		try {
			// Store service name for callback handling
			localStorage.setItem("composio_pending_service", serviceName)

			const data = await apiClient(
				"/api/settings/integrations/connect/composio/initiate",
				{
					method: "POST",
					body: JSON.stringify({ service_name: serviceName })
				}
			)
			window.location.href = data.redirect_url
		} catch (error) {
			toast.error(`Connection failed: ${error.message}`)
			setProcessingIntegration(null)
			localStorage.removeItem("composio_pending_service")
		}
	}

	const disconnectMutation = useMutation({
		mutationFn: ({ integrationName }) => {
			const apiEndpoint =
				integrationName === "whatsapp"
					? "/api/settings/integrations/whatsapp/disconnect"
					: "/api/settings/integrations/disconnect"

			const bodyPayload =
				integrationName === "whatsapp"
					? {}
					: { service_name: integrationName }

			return fetch(apiEndpoint, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify(bodyPayload)
			}).then((res) => {
				if (!res.ok) throw new Error("Failed to disconnect")
				return res.json()
			})
		},
		onSuccess: (data, { displayName, integrationName }) => {
			toast.success(`${displayName} disconnected.`)
			posthog?.capture("integration_disconnected", {
				integration_name: integrationName
			})
			queryClient.invalidateQueries({ queryKey: ["integrations"] })
		},
		onError: (error) => {
			toast.error(error.message)
		},
		onSettled: () => {
			setProcessingIntegration(null)
			setDisconnectingIntegration(null) // Close modal
		}
	})

	useEffect(() => {
		// --- Handle Composio OAuth Callback ---
		const urlParams = new URLSearchParams(window.location.search)
		const composioStatus = urlParams.get("status")
		const connectedAccountId = urlParams.get("connectedAccountId")
		const pendingService = localStorage.getItem("composio_pending_service")

		if (
			composioStatus === "success" &&
			connectedAccountId &&
			pendingService
		) {
			const finalizeConnection = async () => {
				const toastId = toast.loading(
					`Finalizing ${pendingService} connection...`
				)
				localStorage.removeItem("composio_pending_service") // Clean up immediately
				try {
					const data = await apiClient(
						"/api/settings/integrations/connect/composio/finalize",
						{
							method: "POST",
							body: JSON.stringify({
								service_name: pendingService,
								connectedAccountId: connectedAccountId
							})
						}
					)

					toast.success(data.message, { id: toastId })
					posthog?.capture("integration_connected", {
						integration_name: pendingService,
						auth_type: "composio"
					})
					queryClient.invalidateQueries({
						queryKey: ["integrations"]
					}) // Refresh the list
				} catch (error) {
					toast.error(`Error: ${error.message}`, { id: toastId })
				} finally {
					// Clean up URL
					window.history.replaceState(
						{},
						document.title,
						"/integrations"
					)
				}
			}
			finalizeConnection()
		}

		const success = urlParams.get("integration_success")
		const error = urlParams.get("integration_error")

		// Handle Trello's implicit grant flow which returns the token in the URL hash.
		// This must be handled on the client-side as the hash is not sent to the server.
		if (window.location.hash.includes("#token=")) {
			const hash = window.location.hash.substring(1) // remove #
			const params = new URLSearchParams(hash)
			const token = params.get("token")

			if (token) {
				// Immediately clear the hash from the URL bar for security
				window.history.replaceState({}, document.title, "/integrations")

				const saveTrelloToken = async (t) => {
					const toastId = toast.loading(
						"Finalizing Trello connection..."
					)
					try {
						// Use the manual connection endpoint to save the user's token
						await apiClient(
							"/api/settings/integrations/connect/manual",
							{
								method: "POST",
								body: JSON.stringify({
									service_name: "trello",
									credentials: { token: t } // Trello returns a single token
								})
							}
						)
						// This will trigger the success toast and state refresh below
						router.replace(
							"/integrations?integration_success=trello",
							{ scroll: false }
						)
					} catch (error) {
						toast.error(
							`Trello connection failed: ${error.message}`,
							{ id: toastId }
						)
					}
				}
				saveTrelloToken(token)
			}
		}

		if (success && !composioStatus) {
			// Avoid double-toasting for Composio
			const capitalized =
				success.charAt(0).toUpperCase() + success.slice(1)
			posthog?.capture("integration_connected", {
				integration_name: success,
				auth_type: "oauth_redirect"
			})
			toast.success(`Successfully connected to ${capitalized}!`)
			setSparkleTrigger((c) => c + 1)
			queryClient.invalidateQueries({ queryKey: ["integrations"] })
			window.history.replaceState({}, document.title, "/integrations")
		} else if (error && !composioStatus) {
			// Avoid showing generic error on composio callback
			toast.error(`Connection failed: ${error}`)
			queryClient.invalidateQueries({ queryKey: ["integrations"] }) // Fetch to show current state even on error
			window.history.replaceState({}, document.title, "/integrations")
		}
	}, [posthog, router, queryClient])

	const renderIntegrationGrid = (integrations) => (
		<motion.div
			className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
			variants={{
				hidden: { opacity: 0 },
				visible: {
					opacity: 1,
					transition: {
						staggerChildren: 0.05
					}
				}
			}}
			initial="hidden"
			animate="visible"
		>
			<AnimatePresence>
				{integrations.map((integration) => {
					const Icon =
						integrationColorIcons[integration.name] ||
						IconPlaceholder
					const isConnectable = [
						"oauth",
						"manual",
						"composio"
					].includes(integration.auth_type)
					const isProFeature = PRO_ONLY_INTEGRATIONS.includes(
						integration.name
					)
					const isDisabledForFree = isProFeature && !isPro

					const card = (
						<IntegrationCard
							integration={integration}
							icon={Icon}
							isProFeature={isProFeature}
							isProUser={isPro}
							onUpgradeClick={handleUpgradeClick}
						/>
					)

					const cardVariants = {
						hidden: {
							opacity: 0,
							y: -20
						},
						visible: {
							opacity: 1,
							y: 0
						}
					}

					if (isConnectable && !isDisabledForFree) {
						return (
							<motion.div
								key={integration.name}
								variants={cardVariants}
								className="h-full"
								data-tour-id={
									// Add tour ID for Gmail card
									integration.name === "gmail"
										? "gmail-card"
										: undefined
								}
							>
								<MorphingDialog
									transition={{
										type: "spring",
										bounce: 0.05,
										duration: 0.3
									}}
								>
									<MorphingDialogTrigger className="group h-full w-full">
										{card}
									</MorphingDialogTrigger>
									<MorphingDialogContainer>
										{renderIntegrationDialogContent(
											integration
										)}
									</MorphingDialogContainer>
								</MorphingDialog>
							</motion.div>
						)
					} else {
						return (
							<motion.div
								key={integration.name}
								variants={cardVariants}
								className="h-full"
							>
								<div
									className={cn(
										"h-full",
										isDisabledForFree && "opacity-70"
									)}
								>
									{card}
								</div>
							</motion.div>
						)
					}
				})}
			</AnimatePresence>
		</motion.div>
	)

	const allIntegrations = useMemo(
		() => [...userIntegrations, ...defaultTools],
		[userIntegrations, defaultTools]
	)
	const categoriesToShow = ["Core", "Inbuilt", "Advanced"]

	const displayedIntegrations = useMemo(() => {
		// This filter function is used when a search query is active.
		const searchFilter = (integration) => {
			return (
				integration.display_name
					.toLowerCase()
					.includes(searchQuery?.toLowerCase()) ||
				integration.description
					.toLowerCase()
					.includes(searchQuery?.toLowerCase())
			)
		}

		// If there's a search query, ignore category filters and search everything.
		if (searchQuery?.trim() !== "") {
			return allIntegrations.filter(searchFilter)
		}

		const coreServices = new Set([
			"gmail",
			"gcalendar",
			"gdocs",
			"gsheets",
			"gdrive",
			"gtasks",
			"notion",
			"whatsapp"
		])

		if (activeCategory === "Core") {
			return allIntegrations.filter((i) => coreServices.has(i.name))
		}
		if (activeCategory === "Inbuilt") {
			return allIntegrations.filter((i) => i.auth_type === "builtin")
		}
		if (activeCategory === "Advanced") {
			return allIntegrations.filter(
				(i) => !coreServices.has(i.name) && i.auth_type !== "builtin"
			)
		}

		// Fallback for default category before state update or if category is unknown
		return allIntegrations.filter((i) => coreServices.has(i.name))
	}, [activeCategory, allIntegrations, searchQuery])

	const renderIntegrationDialogContent = useCallback(
		(integration) => {
			const Icon =
				integrationColorIcons[integration.name] || IconPlaceholder
			const capabilities =
				INTEGRATION_CAPABILITIES[integration.name] ||
				"Capabilities for this integration will be detailed here soon."
			return (
				<MorphingDialogContent className="pointer-events-auto relative flex h-auto w-full flex-col overflow-hidden border border-neutral-700 bg-neutral-900 sm:w-[600px] rounded-2xl">
					<BorderTrail className="bg-brand-orange" />
					<div
						className="p-4 sm:p-6 overflow-y-auto custom-scrollbar"
						data-tour-id={
							integration.name === "gmail"
								? "gmail-card-modal"
								: undefined
						}
					>
						<div className="flex items-center gap-4 mb-4">
							<div className="w-10 h-10 flex items-center justify-center rounded-lg bg-brand-gray p-1.5 text-brand-orange">
								<Icon className="w-full h-full" />
							</div>
							<div>
								<MorphingDialogTitle className="text-xl sm:text-2xl font-bold text-white">
									{integration.display_name}
								</MorphingDialogTitle>
								<MorphingDialogSubtitle className="text-sm text-neutral-400">
									{integration.connected
										? "Connected"
										: "Not Connected"}
								</MorphingDialogSubtitle>
							</div>
						</div>
						<MorphingDialogDescription>
							<div className="bg-neutral-800/50 p-4 rounded-lg border border-neutral-700/50 mb-6">
								<h4 className="font-semibold text-white mb-2">
									What I can do with{" "}
									{integration.display_name}:
								</h4>
								<p className="text-sm text-neutral-300 leading-relaxed">
									{capabilities}
								</p>
							</div>
							{["gmail", "gcalendar"].includes(
								integration.name
							) && (
								<div className="my-4">
									<button
										onClick={() =>
											openPrivacyModal(integration.name)
										}
										className="w-full text-center text-sm text-neutral-400 hover:text-white hover:bg-neutral-700/50 py-2 rounded-lg transition-colors border border-neutral-700"
									>
										Manage Privacy Filters
									</button>
								</div>
							)}
							<div className="mt-6 pt-4 border-t border-neutral-800">
								{processingIntegration === integration.name ? (
									<div className="flex justify-center">
										<IconLoader className="w-6 h-6 animate-spin text-[var(--color-accent-blue)]" />
									</div>
								) : integration.connected ? (
									<button
										onClick={(e) => {
											e.stopPropagation()
											setDisconnectingIntegration(
												integration
											)
										}}
										className="flex items-center justify-center gap-2 w-full py-2 px-3 rounded-md bg-red-600/20 hover:bg-red-600/40 text-red-300 text-sm font-medium transition-colors"
									>
										<IconPlugOff size={16} />
										<span>Disconnect</span>
									</button>
								) : (
									<Button
										onClick={async (e) => {
											e.stopPropagation()
											if (
												integration.name === "whatsapp"
											) {
												// Show disclaimer first
												setIsWhatsAppDisclaimerOpen(
													true
												)
											} else if (
												integration.auth_type ===
												"composio"
											) {
												await handleComposioConnect(
													integration
												)
											} else {
												await handleConnect(integration)
											}
										}}
										className="w-full gap-2 bg-brand-orange hover:bg-brand-orange/90 text-brand-black font-semibold"
									>
										<IconSparkles size={16} />
										<span>Connect</span>
									</Button>
								)}
							</div>
						</MorphingDialogDescription>
					</div>
					<MorphingDialogClose className="text-white hover:bg-neutral-700 p-1 rounded-full" />
				</MorphingDialogContent>
			)
		},
		[processingIntegration, handleComposioConnect, handleConnect]
	)

	return (
		<div className="flex-1 flex h-screen text-white overflow-x-hidden">
			<Tooltip
				id="page-help-tooltip"
				place="right-start"
				style={{ zIndex: 9999 }}
			/>
			<UpgradeToProModal
				isOpen={isUpgradeModalOpen}
				onClose={closeUpgradeModal}
			/>
			<AnimatePresence>
				{disconnectingIntegration && (
					// The `isolate` class creates a new stacking context, and `z-[70]`
					// ensures this context is rendered above the MorphingDialog
					// and other modals.
					<div className="isolate z-[120]">
						<ModalDialog
							isOpen={!!disconnectingIntegration}
							onClose={() => setDisconnectingIntegration(null)}
						>
							<ModalHeader>
								<ModalTitle>
									<div className="flex items-center gap-2">
										<IconAlertTriangle className="text-yellow-400" />
										<span>{`Disconnect ${disconnectingIntegration.display_name}?`}</span>
									</div>
								</ModalTitle>
								<ModalCloseButton onClose={() => setDisconnectingIntegration(null)} />
							</ModalHeader>
							<ModalBody>
								<p className="text-neutral-300 text-sm">
									This will permanently delete all tasks that use this tool and any related polling data. This action cannot be undone.
								</p>
							</ModalBody>
							<ModalFooter>
								<button
									onClick={() => setDisconnectingIntegration(null)}
									className="px-4 py-2 rounded-md text-sm text-neutral-300 hover:text-white hover:bg-neutral-700 transition-colors"
								>
									Cancel
								</button>
								<button
									onClick={() => {
										if (disconnectingIntegration) {
											disconnectMutation.mutate({
												integrationName: disconnectingIntegration.name,
												displayName: disconnectingIntegration.display_name
											})
										}
									}}
									disabled={processingIntegration === disconnectingIntegration.name}
									className="px-4 py-2 rounded-md text-sm font-medium bg-red-600 hover:bg-red-700 text-white transition-colors disabled:opacity-50"
								>
									{disconnectMutation.isPending ? "Disconnecting..." : "Disconnect"}
								</button>
							</ModalFooter>
						</ModalDialog>
					</div>
				)}
			</AnimatePresence>
			<SparkleEffect trigger={sparkleTrigger} />
			<div className="flex-1 flex flex-col overflow-hidden relative w-full pt-16 md:pt-0">
				<div className="absolute inset-0 z-[-1] network-grid-background">
					<InteractiveNetworkBackground />
				</div>
				<div className="absolute -top-[250px] left-1/2 -translate-x-1/2 w-[800px] h-[500px] bg-brand-orange/10 rounded-full blur-3xl -z-10" />
				<header className="flex items-center justify-between p-4 sm:p-6 md:px-8 md:py-6 bg-transparent shrink-0">
					<div>
						<h1 className="text-3xl lg:text-4xl font-bold text-white flex items-center gap-3">
							<IconPlugConnected />
							Integrations
						</h1>
					</div>
				</header>
				<main className="flex-1 overflow-y-auto px-4 pb-4 sm:p-6 md:px-8 custom-scrollbar">
					{loading ? (
						<div className="flex justify-center items-center h-full">
							<IconLoader className="w-12 h-12 animate-spin text-brand-orange" />
						</div>
					) : (
						<div className="w-full max-w-7xl mx-auto">
							<IntegrationHeader
								searchQuery={searchQuery}
								onSearchChange={setSearchQuery}
								categoriesToShow={categoriesToShow}
								activeCategory={activeCategory}
								onCategoryChange={setActiveCategory}
							/>
							{renderIntegrationGrid(displayedIntegrations)}
						</div>
					)}
				</main>
			</div>
			<AnimatePresence>
				{isWhatsAppQRModalOpen && (
					<WhatsAppQRCodeModal
						onClose={() => {
							setIsWhatsAppQRModalOpen(false)
							queryClient.invalidateQueries({
								queryKey: ["integrations"]
							})
						}}
					/>
				)}
			</AnimatePresence>
			<AnimatePresence>
				{privacyModalService && (
					<PrivacySettingsModal
						serviceName={privacyModalService}
						onClose={closePrivacyModal}
					/>
				)}
			</AnimatePresence>
			<AnimatePresence>
				{isWhatsAppDisclaimerOpen && (
					<div className="isolate z-[120]">
						<WhatsAppDisclaimerModal
							isOpen={isWhatsAppDisclaimerOpen}
							onClose={() => setIsWhatsAppDisclaimerOpen(false)}
							onAgree={() => {
								setIsWhatsAppDisclaimerOpen(false)
								setIsWhatsAppQRModalOpen(true)
							}}
						/>
					</div>
				)}
			</AnimatePresence>
		</div>
	)
}

export default IntegrationsPage
