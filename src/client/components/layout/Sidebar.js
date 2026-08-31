"use client"
import React, { useState, useEffect, useRef } from "react"
import { usePathname, useRouter } from "next/navigation"
import Link from "next/link"
import {
	IconPlugConnected,
	IconUser,
	IconMessage,
	IconBell,
	IconBrain,
	IconSearch,
	IconLayoutSidebarLeftCollapse,
	IconLayoutSidebarLeftExpand,
	IconBolt,
	IconDots,
	IconAdjustments,
	IconLogout,
	IconX,
	IconDownload,
	IconHeadphones,
	IconBrandDiscord,
	IconPlayerPlay,
	IconSlideshow,
	IconSparkles,
	IconCheck,
	IconCode,
	IconMail,
	IconCalendarEvent,
	IconChecklist,
	IconBrandWhatsapp,
	IconBug,
	IconBulb
} from "@tabler/icons-react"
import { cn } from "@utils/cn"
import { motion, AnimatePresence } from "framer-motion"
import { usePostHog } from "posthog-js/react"
import useClickOutside from "@hooks/useClickOutside"
import { Tooltip } from "react-tooltip"
import { Button } from "@components/ui/button"
import {
	ModalDialog,
	ModalHeader,
	ModalTitle,
	ModalCloseButton
} from "@components/ui/ModalDialog"
import { useTourStore, useUserStore, useUIStore } from "@stores/app-stores"

const proPlanFeatures = [
	{ name: "Text Chat", limit: "100 messages per day" },
	{ name: "Voice Chat", limit: "10 minutes per day" },
	{ name: "One-Time Tasks", limit: "20 async tasks per day" },
	{ name: "Recurring Tasks", limit: "10 active recurring workflows" },
	{ name: "Triggered Tasks", limit: "10 triggered workflows" },
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
		if (dashboardUrl) window.location.href = `${dashboardUrl}/dashboard`
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
						transition={{ duration: 0.2, ease: "easeOut" }}
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
									className="flex items-start gap-3"
								>
									<IconCheck
										size={20}
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
								size="lg"
							>
								Upgrade Now - $9/month
							</Button>
							<Button
								onClick={onClose}
								variant="ghost"
								className="w-full text-neutral-400"
							>
								Not now
							</Button>
						</footer>
					</motion.div>
				</motion.div>
			)}
		</AnimatePresence>
	)
}

const comingSoonFeatures = [
	{
		name: "Autopilot Mode",
		icon: <IconSparkles />,
		description:
 "Let BUYASOUL proactively manage your digital life by monitoring your inbox and calendar to suggest and automate tasks before you even ask."
	},
	{
		name: "Multilingual Voice",
		icon: <IconHeadphones />,
		description:
 "Converse with BUYASOUL in multiple languages. Our advanced voice model will understand and respond to you in your preferred language."
	},
	{
		name: "Native Inbox Mirroring",
		icon: <IconMail />,
		description:
 "A dedicated, unified inbox within BUYASOUL that mirrors your emails, allowing for faster, AI-powered email management without leaving the app."
	},
	{
		name: "Native Calendar Mirroring",
		icon: <IconCalendarEvent />,
		description:
 "View and manage all your calendars directly within BUYASOUL. Let the AI schedule, reschedule, and find free slots for you seamlessly."
	},
	{
		name: "Inbuilt To-Do Lists",
		icon: <IconChecklist />,
		description:
 "A smart to-do list integrated with your AI assistant. Add tasks with natural language, and BUYASOUL will prioritize and execute them for you."
	},
	{
		name: "WhatsApp Automation",
		icon: <IconBrandWhatsapp />,
		description:
 "Allow BUYASOUL to manage your WhatsApp. It can read, reply to, and handle messages based on your instructions, turning conversations into actions."
	}
]

const ComingSoonModal = ({ isOpen, onClose }) => {
	if (!isOpen) return null

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
						className="relative bg-neutral-900/90 backdrop-blur-xl p-6 rounded-2xl shadow-2xl w-full max-w-2xl border border-neutral-700 flex flex-col max-h-[80vh]"
					>
						<header className="text-center mb-6 flex-shrink-0">
							<h2 className="text-2xl font-bold text-white flex items-center justify-center gap-2">
								<IconCode className="text-brand-orange" />
								Coming Soon
							</h2>
							<p className="text-neutral-400 mt-2">
								Here's a sneak peek at what we're building next.
							</p>
						</header>
						<main className="space-y-4 overflow-y-auto custom-scrollbar pr-2 flex-1">
							{comingSoonFeatures.map((feature) => (
								<div
									key={feature.name}
									className="flex items-start gap-4 p-4 bg-neutral-800/50 rounded-lg"
								>
									<div className="text-brand-orange flex-shrink-0 mt-1">
										{feature.icon}
									</div>
									<div>
										<p className="font-semibold text-white">
											{feature.name}
										</p>
										<p className="text-sm text-neutral-400">
											{feature.description}
										</p>
									</div>
								</div>
							))}
						</main>
						<footer className="mt-6 pt-4 border-t border-neutral-800 flex justify-end">
							<Button onClick={onClose} variant="secondary">
								Close
							</Button>
						</footer>
					</motion.div>
				</motion.div>
			)}
		</AnimatePresence>
	)
}
const UserProfileSection = ({ isCollapsed, user }) => {
	const [isUserMenuOpen, setUserMenuOpen] = useState(false)
	const userMenuRef = useRef(null)
	const { isPro } = useUserStore()
	const posthog = usePostHog()
	useClickOutside(userMenuRef, () => setUserMenuOpen(false))

	const planName = isPro ? "Pro" : "Basic"

	const dashboardUrl = process.env.NEXT_PUBLIC_LANDING_PAGE_URL
		? `${process.env.NEXT_PUBLIC_LANDING_PAGE_URL}/dashboard`
		: "#"

	return (
		<div
			className={cn(
				"flex items-center justify-between bg-neutral-800/40 border border-neutral-700/80 rounded-lg p-1.5",
				isCollapsed && "flex-col gap-2"
			)}
		>
			<div
				className={cn(
					"flex items-center gap-2 p-1 rounded-md flex-grow",
					isCollapsed ? "w-full justify-center" : ""
				)}
			>
				{user?.picture ? (
					<img
						src={user.picture}
						alt="User"
						className="w-7 h-7 rounded-full flex-shrink-0"
					/>
				) : (
					<div className="w-7 h-7 rounded-full bg-neutral-700 flex items-center justify-center flex-shrink-0">
						<IconUser size={16} />
					</div>
				)}
				<AnimatePresence>
					{!isCollapsed && (
						<motion.div
							initial={{ opacity: 0, width: 0 }}
							animate={{ opacity: 1, width: "auto" }}
							exit={{ opacity: 0, width: 0 }}
							className="overflow-hidden whitespace-nowrap"
						>
							<p className="font-semibold text-sm text-white truncate">
								{user?.given_name || user?.name || "User"}
							</p>
							<span
								className={cn(
									"text-xs flex items-center gap-1",
									isPro
										? "text-brand-orange"
										: "text-neutral-400"
								)}
							>
								{isPro && <IconBolt size={12} />}
								{planName}
							</span>
						</motion.div>
					)}
				</AnimatePresence>
			</div>
			{!isCollapsed && (
				<div className="relative" ref={userMenuRef}>
					<button
						onClick={() => setUserMenuOpen(!isUserMenuOpen)}
						className="p-2 rounded-md hover:bg-neutral-700 text-neutral-400 hover:text-white"
					>
						<IconDots size={16} />
					</button>
					<AnimatePresence>
						{isUserMenuOpen && (
							<motion.div
								initial={{ opacity: 0, y: 10, scale: 0.95 }}
								animate={{ opacity: 1, y: 0, scale: 1 }}
								exit={{ opacity: 0, y: 10, scale: 0.95 }}
								className="absolute bottom-full right-0 mb-2 w-48 bg-neutral-900/80 backdrop-blur-md border border-neutral-700 rounded-lg shadow-lg p-1 z-50"
							>
								<a
									href={dashboardUrl}
									rel="noopener noreferrer"
									className="w-full flex items-center gap-2 text-left px-3 py-2 text-sm rounded-md text-neutral-200 hover:bg-neutral-700/50 transition-colors"
								>
									<IconAdjustments size={16} />
									<span>Account & Billing</span>
								</a>
								<a
									href="/auth/logout"
									onClick={() => {
										posthog?.reset()
									}}
									className="w-full flex items-center gap-2 text-left px-3 py-2 text-sm rounded-md text-red-400 hover:bg-red-500/20 hover:text-red-300 transition-colors"
								>
									<IconLogout size={16} />
									<span>Logout</span>
								</a>
							</motion.div>
						)}
					</AnimatePresence>
				</div>
			)}
		</div>
	)
}
const HelpMenuModal = ({ onClose, onShowVideo, onShowDemo }) => {
	return (
		<ModalDialog
			isOpen={true}
			onClose={onClose}
			className="max-w-lg bg-neutral-900/90 backdrop-blur-xl p-0"
		>
			<div className="p-6">
				<ModalHeader className="p-0 border-none mb-6">
					<ModalTitle>Need Help?</ModalTitle>
					<ModalCloseButton onClose={onClose} />
				</ModalHeader>
				<main className="space-y-4">
					<button
						onClick={onShowDemo}
						className="w-full text-left p-4 rounded-lg bg-neutral-800 hover:bg-neutral-700/80 transition-colors flex items-center gap-4"
					>
						<IconSlideshow
							size={24}
							className="text-brand-orange flex-shrink-0"
						/>
						<div>
							<p className="font-semibold text-white">
								Interactive Demo
							</p>
							<p className="text-sm text-neutral-400">
								Get a hands-on tour of the main features.
							</p>
						</div>
					</button>
					<button
						onClick={onShowVideo}
						className="w-full text-left p-4 rounded-lg bg-neutral-800 hover:bg-neutral-700/80 transition-colors flex items-center gap-4"
					>
						<IconPlayerPlay
							size={24}
							className="text-brand-orange flex-shrink-0"
						/>
						<div>
							<p className="font-semibold text-white">
								Demo Video
							</p>
							<p className="text-sm text-neutral-400">
								Watch a quick video walkthrough.
							</p>
						</div>
					</button>
					<a
						href="https://forms.gle/7F4H3Pvy7fSdeeYm7"
						target="_blank"
						rel="noopener noreferrer"
						className="w-full text-left p-4 rounded-lg bg-neutral-800 hover:bg-neutral-700/80 transition-colors flex items-center gap-4"
					>
						<IconBug
							size={24}
							className="text-brand-orange flex-shrink-0"
						/>
						<div>
							<p className="font-semibold text-white">
								Report a Bug
							</p>
							<p className="text-sm text-neutral-400">
								Encountered an issue? Let us know so we can fix
								it.
							</p>
						</div>
					</a>
					<a
						href="https://forms.gle/82toLJG7rysZEBRi6"
						target="_blank"
						rel="noopener noreferrer"
						className="w-full text-left p-4 rounded-lg bg-neutral-800 hover:bg-neutral-700/80 transition-colors flex items-center gap-4"
					>
						<IconBulb
							size={24}
							className="text-brand-orange flex-shrink-0"
						/>
						<div>
							<p className="font-semibold text-white">
								Request a Feature
							</p>
							<p className="text-sm text-neutral-400">
								Have a great idea? We'd love to hear it.
							</p>
						</div>
					</a>
					<a
						href="https://wa.me/918275017823"
						target="_blank"
						rel="noopener noreferrer"
						className="w-full text-left p-4 rounded-lg bg-neutral-800 hover:bg-neutral-700/80 transition-colors flex items-center gap-4"
					>
						<IconBrandWhatsapp
							size={24}
							className="text-brand-orange flex-shrink-0"
						/>
						<div>
							<p className="font-semibold text-white">
								Talk to a Co-Founder
							</p>
							<p className="text-sm text-neutral-400">
								Get direct support on WhatsApp.
							</p>
						</div>
					</a>
				</main>
			</div>
		</ModalDialog>
	)
}
const SidebarContent = ({
	isCollapsed,
	onNotificationsOpen,
	onSearchOpen,
	unreadCount,
	isMobile = false,
	onMobileClose = () => {},
	installPrompt,
	handleInstallClick,
	user,
	isTourActive
}) => {
	const pathname = usePathname()
	const [isHelpMenuOpen, setHelpMenuOpen] = useState(false)
	const [isVideoModalOpen, setVideoModalOpen] = useState(false)
	const [isComingSoonModalOpen, setComingSoonModalOpen] = useState(false)
	const { isUpgradeModalOpen, openUpgradeModal, closeUpgradeModal } =
		useUIStore()
	const { isPro } = useUserStore()
	const { startTour } = useTourStore()
	const router = useRouter()

	const fadeInUp = {
		hidden: { opacity: 0, y: 10 },
		visible: { opacity: 1, y: 0, transition: { duration: 0.3 } }
	}

	// CHANGED: Use the environment variable for the namespace

	const handleStartDemo = () => {
		setHelpMenuOpen(false)
		startTour()
	}

	const handleShowVideo = () => {
		setHelpMenuOpen(false)
		setVideoModalOpen(true)
	}

	const navLinks = [
		{ title: "Chat", href: "/chat", icon: <IconMessage size={20} /> },
		{
			title: "Tasks",
			href: "/tasks",
			icon: <IconChecklist size={20} />,
			tourId: "sidebar-tasks-icon"
		},
		{
			title: "Memories",
			href: "/memories",
			icon: <IconBrain size={20} />
		},
		{
			title: "Integrations",
			href: "/integrations",
			icon: <IconPlugConnected size={20} />,
			tourId: "sidebar-integrations-icon"
		},
		{
			title: "Settings",
			href: "/settings",
			icon: <IconAdjustments size={20} />
		}
	]

	return (
		<div className="flex flex-col h-full w-full overflow-y-auto custom-scrollbar">
			<Tooltip id="sidebar-tooltip" />
			<UpgradeToProModal
				isOpen={isUpgradeModalOpen}
				onClose={closeUpgradeModal}
			/>
			<ComingSoonModal
				isOpen={isComingSoonModalOpen}
				onClose={() => setComingSoonModalOpen(false)}
			/>
			<AnimatePresence>
				{isVideoModalOpen && (
					// Assuming HelpVideoModal exists and is imported
					// <HelpVideoModal onClose={() => setVideoModalOpen(false)} />
					<div className="fixed inset-0 bg-black/70 backdrop-blur-md z-[100] flex items-center justify-center p-4">
						<div className="bg-neutral-900/90 p-6 rounded-2xl">
							<p className="text-white">
								Help Video Modal Placeholder
							</p>
							<button onClick={() => setVideoModalOpen(false)}>
								Close
							</button>
						</div>
					</div>
				)}
			</AnimatePresence>
			<AnimatePresence>
				{isHelpMenuOpen && (
					<HelpMenuModal
						onClose={() => setHelpMenuOpen(false)}
						onShowDemo={handleStartDemo}
						onShowVideo={handleShowVideo}
					/>
				)}
			</AnimatePresence>
			<div
				className={cn(
					"flex items-center justify-between mb-4 px-2",
					isCollapsed && "justify-center"
				)}
			>
				<Link
					href="/chat"
					className="flex items-center gap-2"
					onClick={isMobile ? onMobileClose : undefined}
				>
					<img
						src="/images/half-logo-dark.svg"
						alt="Logo"
						className="w-7 h-7"
					/>
					<AnimatePresence>
						{!isCollapsed && (
							<motion.div
								initial={{ opacity: 0, width: 0 }}
								animate={{ opacity: 1, width: "auto" }}
								exit={{ opacity: 0, width: 0 }}
								className="flex items-center gap-2 overflow-hidden"
							>
								<span className="font-bold text-lg text-white whitespace-nowrap">
									BUYASOUL
								</span>
								<span
									data-tooltip-id="sidebar-tooltip"
									data-tooltip-content="Sorry if anything breaks"
									className="px-1.5 py-0.5 text-xs font-semibold text-neutral-900 bg-yellow-400 rounded-md"
								>
									Beta
								</span>
							</motion.div>
						)}
					</AnimatePresence>
				</Link>
				<AnimatePresence>
					{isMobile && (
						<button
							onClick={onMobileClose}
							className="p-1.5 rounded-full hover:bg-neutral-800"
						>
							<IconX size={18} />
						</button>
					)}
				</AnimatePresence>
			</div>

			<button
				onClick={() => {
					onSearchOpen()
					if (isMobile) onMobileClose()
				}}
				className={cn(
					"w-full flex items-center gap-3 bg-neutral-800/40 border border-neutral-700/80 rounded-lg p-2.5 text-left mb-4 hover:bg-neutral-800/80 transition-colors",
					isCollapsed && "justify-center"
				)}
			>
				<IconSearch
					size={18}
					className="text-neutral-300 flex-shrink-0"
				/>
				{!isCollapsed && (
					<span className="text-sm font-medium text-neutral-200">
						Search...
					</span>
				)}
			</button>

			{!isPro && (
				<button
					onClick={openUpgradeModal}
					className={cn(
						"w-full bg-neutral-800/40 border border-neutral-700/80 rounded-lg p-2.5 text-left mb-2 hover:bg-neutral-800/80 transition-colors",
						isCollapsed && "flex justify-center"
					)}
				>
					<div className="flex items-center gap-3">
						<div className="bg-neutral-700/80 p-1 rounded-full text-brand-orange">
							<IconBolt size={18} />
						</div>
						<AnimatePresence>
							{!isCollapsed && (
								<motion.div
									initial={{ opacity: 0, width: 0 }}
									animate={{ opacity: 1, width: "auto" }}
									exit={{ opacity: 0, width: 0 }}
									className="overflow-hidden whitespace-nowrap"
								>
									<p className="font-semibold text-sm text-white">
										Unlock Pro
									</p>
									<p className="text-xs text-neutral-400">
										Unlock all features
									</p>
								</motion.div>
							)}
						</AnimatePresence>
					</div>
				</button>
			)}

			<nav className="flex flex-col gap-1 flex-grow overflow-y-auto min-h-[200px] custom-scrollbar">
				{navLinks.map((link) => {
					const isActive = pathname.startsWith(link.href)
					return (
						<Link
							href={link.href}
							key={link.title}
							data-tour-id={link.tourId}
							onClick={
								isMobile && !isTourActive
									? onMobileClose
									: undefined
							}
							className={cn(
								"flex items-center gap-3 rounded-md p-2 transition-colors duration-200 text-sm",
								isActive
									? "text-white bg-neutral-800"
									: "text-neutral-400 hover:text-white hover:bg-neutral-800/50",
								isCollapsed && "justify-center"
							)}
						>
							{link.icon}
							{!isCollapsed && (
								<span className="font-medium whitespace-nowrap">
									{link.title}
								</span>
							)}
						</Link>
					)
				})}
				<hr className="border-neutral-800 my-2" />
			</nav>
			<div className="flex-shrink-0 space-y-2 max-h-[calc(100vh-300px)] overflow-y-auto custom-scrollbar">
				<button
					onClick={onNotificationsOpen}
					className={cn(
						"w-full flex items-center gap-3 rounded-md p-2 transition-colors duration-200 text-sm",
						"text-neutral-400 hover:text-white hover:bg-neutral-800/50",
						isCollapsed && "justify-center"
					)}
				>
					<div className="relative">
						<IconBell size={20} />
						{unreadCount > 0 && (
							<div className="absolute top-0 right-0 h-2 w-2 bg-red-500 rounded-full" />
						)}
					</div>
					{!isCollapsed && (
						<span className="font-medium">Notifications</span>
					)}
				</button>

				<div className="flex flex-col gap-2 mt-2">
					{installPrompt && (
						<button
							onClick={handleInstallClick}
							className={cn(
								"w-full flex items-center gap-3 bg-green-600/20 border border-green-500/50 text-green-300 rounded-lg p-2 text-left text-sm hover:bg-green-600/40 transition-colors",
								isCollapsed && "justify-center"
							)}
						>
							<IconDownload size={20} className="flex-shrink-0" />
							{!isCollapsed && (
								<span className="font-medium whitespace-nowrap">
									Install App
								</span>
							)}
						</button>
					)}

					{isMobile && (
						<button
							onClick={onMobileClose}
							className="flex items-center gap-3 rounded-md p-2 transition-colors duration-200 text-sm text-neutral-400 hover:text-white hover:bg-neutral-800/50"
						>
							<IconLayoutSidebarLeftCollapse size={20} />
							<span className="font-medium whitespace-nowrap">
								Collapse
							</span>
						</button>
					)}
					<button
						onClick={() => setComingSoonModalOpen(true)}
						className={cn(
							"w-full flex items-center gap-3 bg-neutral-800/40 border border-neutral-700/80 rounded-lg p-2 text-left text-sm hover:bg-neutral-800/80 transition-colors",
							isCollapsed && "justify-center"
						)}
					>
						<IconCode
							size={20}
							className="text-neutral-400 flex-shrink-0"
						/>
						{!isCollapsed && (
							<span className="font-medium text-neutral-300 whitespace-nowrap">
								Coming Soon
							</span>
						)}
					</button>
					<button
						onClick={() => setHelpMenuOpen(true)}
						className={cn(
							"w-full flex items-center gap-3 bg-neutral-800/40 border border-neutral-700/80 rounded-lg p-2 text-left text-sm hover:bg-neutral-800/80 transition-colors",
							isCollapsed && "justify-center"
						)}
					>
						<IconHeadphones
							size={20}
							className="text-neutral-400 flex-shrink-0"
						/>
						{!isCollapsed && (
							<span className="font-medium text-neutral-300 whitespace-nowrap">
								Need help?
							</span>
						)}
					</button>
					<a
						href="https://discord.gg/YwXdEvjKGe"
						target="_blank"
						rel="noopener noreferrer"
						className={cn(
							"w-full flex items-center gap-3 bg-indigo-600/20 border border-indigo-500/50 text-indigo-300 rounded-lg p-2 text-left text-sm hover:bg-indigo-600/40 transition-colors",
							isCollapsed && "justify-center"
						)}
					>
						<IconBrandDiscord size={20} className="flex-shrink-0" />
						{!isCollapsed && (
							<span className="font-medium whitespace-nowrap">
								Join Community
							</span>
						)}
					</a>
					<UserProfileSection isCollapsed={isCollapsed} user={user} />
				</div>
			</div>
		</div>
	)
}
const Sidebar = ({
	onNotificationsOpen,
	onSearchOpen,
	unreadCount,
	isMobileOpen,
	onMobileClose,
	user,
	isTourActive
}) => {
	const [installPrompt, setInstallPrompt] = useState(null)

	useEffect(() => {
		const handleBeforeInstallPrompt = (e) => {
			e.preventDefault()
			setInstallPrompt(e)
		}

		window.addEventListener(
			"beforeinstallprompt",
			handleBeforeInstallPrompt
		)

		return () => {
			window.removeEventListener(
				"beforeinstallprompt",
				handleBeforeInstallPrompt
			)
		}
	}, [])

	const handleInstallClick = async () => {
		if (!installPrompt) return
		const result = await installPrompt.prompt()
		console.log(`Install prompt was: ${result.outcome}`)
		setInstallPrompt(null)
	}

	return (
		<>
			{/* Mobile Sidebar */}
			<AnimatePresence>
				{isMobileOpen && (
					<>
						<motion.div
							className="fixed inset-0 bg-black/60 z-40 md:hidden"
							initial={{ opacity: 0 }}
							animate={{ opacity: 1 }}
							exit={{ opacity: 0 }}
							onClick={onMobileClose}
						/>
						<motion.div
							className="fixed top-0 left-0 h-screen w-[260px] bg-black p-3 text-neutral-200 border-r border-neutral-800/50 z-50 md:hidden"
							initial={{ x: "-100%" }}
							animate={{ x: 0 }}
							exit={{ x: "-100%" }}
							transition={{
								type: "spring",
								stiffness: 300,
								damping: 30
							}}
						>
							<SidebarContent
								isCollapsed={false}
								onMobileClose={onMobileClose}
								onNotificationsOpen={onNotificationsOpen}
								onSearchOpen={onSearchOpen}
								unreadCount={unreadCount}
								isMobile={true}
								installPrompt={installPrompt}
								handleInstallClick={handleInstallClick}
								isTourActive={isTourActive}
								user={user}
							/>
						</motion.div>
					</>
				)}
			</AnimatePresence>

			{/* Desktop Sidebar */}
			<div
				style={{ width: 260 }}
				className="hidden md:flex fixed top-0 left-0 h-screen bg-black flex-col p-3 text-neutral-200 border-r border-neutral-800/50 z-40"
			>
				<SidebarContent
					isCollapsed={false}
					onNotificationsOpen={onNotificationsOpen}
					onSearchOpen={onSearchOpen}
					unreadCount={unreadCount}
					installPrompt={installPrompt}
					handleInstallClick={handleInstallClick}
					isTourActive={isTourActive}
					user={user}
				/>
			</div>
		</>
	)
}
export default Sidebar
